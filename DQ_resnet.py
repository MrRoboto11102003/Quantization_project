import torch
import torch.nn as nn
import torch.nn.functional as F

BIT_OPTIONS = [3, 4, 5, 6, 7]

class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bits):
        min_val, max_val = x.min(), x.max()
        scale = (max_val - min_val) / (2**bits - 1)
        scale = torch.clamp(scale, min=1e-8)
        return torch.round((x - min_val) / scale) * scale + min_val

    @staticmethod
    def backward(ctx, grad):
        return grad, None

class DynamicQuantConv2d(nn.Module):
    def __init__(self, *args, bit_options=BIT_OPTIONS, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(*args, **kwargs)
        self.bit_options = bit_options

    def forward(self, x, soft_bits):
        outputs = []
        for b in self.bit_options:
            w_q = STEQuantize.apply(self.conv.weight, b)
            x_q = STEQuantize.apply(x, b)
            outputs.append(F.conv2d(x_q, w_q, self.conv.bias, self.conv.stride, self.conv.padding))
        weights = soft_bits.view(-1, len(self.bit_options), 1, 1, 1)
        stacked = torch.stack(outputs, dim=1)
        return (weights * stacked).sum(dim=1)

def select_bits(logits, temperature=1.0, training=True):
    if training:
        return F.gumbel_softmax(logits, tau=temperature, hard=False)
    return F.one_hot(torch.argmax(logits, dim=-1), logits.size(-1)).float()

class BitController(nn.Module):
    def __init__(self, num_layers, num_bits_options=3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.heads = nn.ModuleList([nn.Linear(32, num_bits_options) for _ in range(num_layers)])

    def forward(self, x):
        feat = self.shared(x)
        return [head(feat) for head in self.heads]

class DQBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bit_options=BIT_OPTIONS):
        super().__init__()
        self.conv1 = DynamicQuantConv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False, bit_options=bit_options)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = DynamicQuantConv2d(planes, planes, 3, stride=1, padding=1, bias=False, bit_options=bit_options)
        self.bn2 = nn.BatchNorm2d(planes)
        self.has_shortcut = (stride != 1 or in_planes != planes)
        if self.has_shortcut:
            self.shortcut_conv = DynamicQuantConv2d(in_planes, planes, 1, stride=stride, bias=False, bit_options=bit_options)
            self.shortcut_bn = nn.BatchNorm2d(planes)

    def forward(self, x, soft_bits):
        out = F.relu(self.bn1(self.conv1(x, soft_bits)))
        out = self.bn2(self.conv2(out, soft_bits))
        if self.has_shortcut:
            out += self.shortcut_bn(self.shortcut_conv(x, soft_bits))
        else:
            out += x
        return F.relu(out)

class DQResNet(nn.Module):
    def __init__(self, block, num_blocks, bit_options=BIT_OPTIONS, num_classes=10):
        super().__init__()
        self.in_planes = 16
        self.bit_options = bit_options

        self.conv1 = DynamicQuantConv2d(3, 16, 3, stride=1, padding=1, bias=False, bit_options=bit_options)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, bit_options=bit_options)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, bit_options=bit_options)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, bit_options=bit_options)
        self.linear = nn.Linear(64, num_classes)

        num_controlled = 1 + sum(num_blocks)
        self.controller = BitController(num_controlled, len(bit_options))
        self.layer_flops = self._compute_layer_flops(num_blocks)

    def _make_layer(self, block, planes, num_blocks, stride, bit_options):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = nn.ModuleList()
        for s in strides:
            layers.append(block(self.in_planes, planes, s, bit_options=bit_options))
            self.in_planes = planes * block.expansion
        return layers

    @staticmethod
    def _compute_layer_flops(num_blocks):
        flops = [3 * 16 * 9 * 32 * 32]
        cfg = [(32, 16, 16, num_blocks[0]), (16, 32, 32, num_blocks[1]), (8, 64, 64, num_blocks[2])]
        prev_c = 16
        for h, c_out, _, n in cfg:
            for i in range(n):
                c_in = prev_c if i == 0 else c_out
                block_f = c_in * c_out * 9 * h * h + c_out * c_out * 9 * h * h
                if c_in != c_out:
                    block_f += c_in * c_out * 1 * h * h
                flops.append(block_f)
            prev_c = c_out
        return flops

    def forward(self, x, temperature=1.0):
        training = self.training
        bit_logits = self.controller(x)
        soft_bits_list = [select_bits(l, temperature, training) for l in bit_logits]

        idx = 0
        out = F.relu(self.bn1(self.conv1(x, soft_bits_list[idx])))
        idx += 1
        for layer in [self.layer1, self.layer2, self.layer3]:
            for block in layer:
                out = block(out, soft_bits_list[idx])
                idx += 1

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, soft_bits_list

def compute_bitops(soft_bits_list, layer_flops, bit_options=BIT_OPTIONS):
    bit_sq = torch.tensor([b * b for b in bit_options], dtype=torch.float, device=soft_bits_list[0].device)
    total = torch.zeros(soft_bits_list[0].size(0), device=soft_bits_list[0].device)
    for soft, flops in zip(soft_bits_list, layer_flops):
        total = total + (soft * bit_sq).sum(dim=-1) * flops
    return total.mean()

def get_mean_bits(soft_bits_list, bit_options=BIT_OPTIONS):
    bit_tensor = torch.tensor(bit_options, dtype=torch.float, device=soft_bits_list[0].device)
    return torch.stack([(s * bit_tensor).sum(dim=-1) for s in soft_bits_list], dim=-1)

def max_bitops(layer_flops, bit_options=BIT_OPTIONS):
    max_b = float(max(bit_options))
    return sum(max_b * max_b * f for f in layer_flops)

def dq_resnet20(bit_options=BIT_OPTIONS):
    return DQResNet(DQBasicBlock, [3, 3, 3], bit_options=bit_options)
