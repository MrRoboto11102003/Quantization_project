import torch
import torch.nn as nn
import torch.nn.functional as F

BIT_OPTIONS = [2, 4, 8]

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
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(*args, **kwargs)

    def forward(self, x, bits):
        unique_bits = torch.unique(bits)
        chunks, chunk_idx = [], []
        for b in unique_bits:
            mask = (bits == b)
            idx = mask.nonzero(as_tuple=True)[0]
            w_q = STEQuantize.apply(self.conv.weight, b)
            x_q = STEQuantize.apply(x[idx], b)
            chunks.append(F.conv2d(x_q, w_q, self.conv.bias, self.conv.stride, self.conv.padding))
            chunk_idx.append(idx)
        all_idx = torch.cat(chunk_idx)
        all_out = torch.cat(chunks)
        _, sort_idx = all_idx.sort()
        return all_out[sort_idx]

def select_bits(logits, bit_options, temperature=1.0, training=True):
    bit_tensor = torch.tensor(bit_options, dtype=torch.float, device=logits.device)
    if training:
        soft = F.gumbel_softmax(logits, tau=temperature, hard=True)
        return (soft * bit_tensor).sum(dim=-1)
    return bit_tensor[torch.argmax(logits, dim=-1)]

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

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = DynamicQuantConv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = DynamicQuantConv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.has_shortcut = (stride != 1 or in_planes != planes)
        if self.has_shortcut:
            self.shortcut_conv = DynamicQuantConv2d(in_planes, planes, 1, stride=stride, bias=False)
            self.shortcut_bn = nn.BatchNorm2d(planes)

    def forward(self, x, bits):
        out = F.relu(self.bn1(self.conv1(x, bits)))
        out = self.bn2(self.conv2(out, bits))
        if self.has_shortcut:
            out += self.shortcut_bn(self.shortcut_conv(x, bits))
        else:
            out += x
        return F.relu(out)

class DQResNet(nn.Module):
    def __init__(self, block, num_blocks, bit_options=BIT_OPTIONS, num_classes=10):
        super().__init__()
        self.in_planes = 16
        self.bit_options = bit_options

        self.conv1 = DynamicQuantConv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        num_controlled = 1 + sum(num_blocks)
        self.controller = BitController(num_controlled, len(bit_options))
        self.layer_flops = self._compute_layer_flops(num_blocks)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = nn.ModuleList()
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
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
        bits_list = [select_bits(l, self.bit_options, temperature, training) for l in bit_logits]

        idx = 0
        out = F.relu(self.bn1(self.conv1(x, bits_list[idx])))
        idx += 1
        for layer in [self.layer1, self.layer2, self.layer3]:
            for block in layer:
                out = block(out, bits_list[idx])
                idx += 1

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, bits_list

def compute_bitops(bits_list, layer_flops):
    total = torch.zeros_like(bits_list[0])
    for bits, flops in zip(bits_list, layer_flops):
        total = total + bits * bits * flops
    return total.mean()

def max_bitops(layer_flops):
    return sum(32.0 * 32.0 * f for f in layer_flops)

def dq_resnet20(bit_options=BIT_OPTIONS):
    return DQResNet(DQBasicBlock, [3, 3, 3], bit_options=bit_options)
