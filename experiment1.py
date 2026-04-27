import torch
import torch.nn as nn
import torch.nn.functional as F

BIT_OPTIONS = [4, 8]

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

class GlobalDynamicQuantConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(*args, **kwargs)

    def forward(self, x, routing_info):
        """
        routing_info can be:
        1. A tensor of soft probabilities [batch_size, 2] during training.
        2. An integer (4 or 8) during hard inference where the batch is pre-split.
        """
        if isinstance(routing_info, int):
            w_q = STEQuantize.apply(self.conv.weight, routing_info)
            x_q = STEQuantize.apply(x, routing_info)
            return F.conv2d(x_q, w_q, self.conv.bias, self.conv.stride, self.conv.padding)
        else:
            w4 = STEQuantize.apply(self.conv.weight, 4)
            x4 = STEQuantize.apply(x, 4)
            out4 = F.conv2d(x4, w4, self.conv.bias, self.conv.stride, self.conv.padding)
            
            w8 = STEQuantize.apply(self.conv.weight, 8)
            x8 = STEQuantize.apply(x, 8)
            out8 = F.conv2d(x8, w8, self.conv.bias, self.conv.stride, self.conv.padding)
            
            # routing_info is [batch_size, 2]
            w = routing_info.view(-1, 2, 1, 1, 1)
            stacked = torch.stack([out4, out8], dim=1)
            return (w * stacked).sum(dim=1)

class GlobalRouter(nn.Module):
    def __init__(self):
        super().__init__()
        # Logistic regression on the image
        self.linear = nn.Linear(3 * 32 * 32, 2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)

class GlobalDQBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = GlobalDynamicQuantConv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = GlobalDynamicQuantConv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.has_shortcut = (stride != 1 or in_planes != planes)
        if self.has_shortcut:
            self.shortcut_conv = GlobalDynamicQuantConv2d(in_planes, planes, 1, stride=stride, bias=False)
            self.shortcut_bn = nn.BatchNorm2d(planes)

    def forward(self, x, routing_info):
        out = F.relu(self.bn1(self.conv1(x, routing_info)))
        out = self.bn2(self.conv2(out, routing_info))
        if self.has_shortcut:
            out += self.shortcut_bn(self.shortcut_conv(x, routing_info))
        else:
            out += x
        return F.relu(out)

class GlobalDQResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 16

        self.conv1 = GlobalDynamicQuantConv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.router = GlobalRouter()
        self.layer_flops = self._compute_layer_flops(num_blocks)
        
        self.bit_options = BIT_OPTIONS

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = nn.ModuleList()
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return layers

    @staticmethod
    def _compute_layer_flops(num_blocks):
        # Initial conv
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

    def _forward_network(self, x, routing_info):
        out = F.relu(self.bn1(self.conv1(x, routing_info)))
        for layer in [self.layer1, self.layer2, self.layer3]:
            for block in layer:
                out = block(out, routing_info)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def forward(self, x, temperature=1.0):
        logits = self.router(x)
        
        if self.training:
            # Soft routing
            soft_bits = F.gumbel_softmax(logits, tau=temperature, hard=False)
            out = self._forward_network(x, soft_bits)
            return out, soft_bits
        else:
            # Hard inference: split the batch for maximum latency efficiency
            hard_idx = torch.argmax(logits, dim=1)
            
            mask4 = (hard_idx == 0)
            mask8 = (hard_idx == 1)
            
            x4 = x[mask4]
            x8 = x[mask8]
            
            out = torch.empty((x.size(0), self.linear.out_features), device=x.device)
            
            if x4.size(0) > 0:
                out4 = self._forward_network(x4, routing_info=4)
                out[mask4] = out4
            if x8.size(0) > 0:
                out8 = self._forward_network(x8, routing_info=8)
                out[mask8] = out8
                
            # For inference, we return the output and the hard selections (0 for INT4, 1 for INT8)
            return out, hard_idx

def compute_global_bitops(soft_bits, layer_flops):
    # soft_bits is [batch_size, 2] containing probabilities for [INT4, INT8]
    # Total FLOPs for the backbone
    total_flops = sum(layer_flops)
    
    # Cost per bit option
    cost_4 = 4 * 4 * total_flops
    cost_8 = 8 * 8 * total_flops
    
    # Expected cost
    expected_cost = soft_bits[:, 0] * cost_4 + soft_bits[:, 1] * cost_8
    return expected_cost.mean()

def max_global_bitops(layer_flops):
    return 8 * 8 * sum(layer_flops)

def global_dq_resnet20():
    return GlobalDQResNet(GlobalDQBasicBlock, [3, 3, 3])
