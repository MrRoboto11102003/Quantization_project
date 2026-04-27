import torch
import torch.nn as nn
import torch.nn.functional as F

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

class FixedQuantConv2d(nn.Module):
    def __init__(self, *args, bits=8, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(*args, **kwargs)
        self.bits = bits

    def forward(self, x):
        w_q = STEQuantize.apply(self.conv.weight, self.bits)
        x_q = STEQuantize.apply(x, self.bits)
        return F.conv2d(x_q, w_q, self.conv.bias, self.conv.stride, self.conv.padding)

class EarlyExitINT8BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = FixedQuantConv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False, bits=8)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = FixedQuantConv2d(planes, planes, 3, stride=1, padding=1, bias=False, bits=8)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.has_shortcut = (stride != 1 or in_planes != planes)
        if self.has_shortcut:
            self.shortcut_conv = FixedQuantConv2d(in_planes, planes, 1, stride=stride, bias=False, bits=8)
            self.shortcut_bn = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.has_shortcut:
            out += self.shortcut_bn(self.shortcut_conv(x))
        else:
            out += x
        return F.relu(out)

class EarlyExitINT8ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 16

        self.conv1 = FixedQuantConv2d(3, 16, 3, stride=1, padding=1, bias=False, bits=8)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        
        # Early Exit Head (placed after layer2, where channels = 32)
        self.early_pool = nn.AdaptiveAvgPool2d(1)
        self.early_linear = nn.Linear(32, num_classes)
        
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.final_linear = nn.Linear(64, num_classes)

        # Flops tracking
        self.flops_base, self.flops_layer3 = self._compute_stage_flops(num_blocks)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = nn.ModuleList()
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    @staticmethod
    def _compute_stage_flops(num_blocks):
        # Initial conv
        flops_base = 3 * 16 * 9 * 32 * 32
        
        cfg = [(32, 16, 16, num_blocks[0]), (16, 32, 32, num_blocks[1]), (8, 64, 64, num_blocks[2])]
        prev_c = 16
        
        flops_layer3 = 0
        
        for stage_idx, (h, c_out, _, n) in enumerate(cfg):
            for i in range(n):
                c_in = prev_c if i == 0 else c_out
                block_f = c_in * c_out * 9 * h * h + c_out * c_out * 9 * h * h
                if c_in != c_out:
                    block_f += c_in * c_out * 1 * h * h
                
                if stage_idx < 2:
                    flops_base += block_f
                else:
                    flops_layer3 += block_f
                
            prev_c = c_out
            
        return flops_base, flops_layer3

    def forward(self, x, entropy_threshold=0.5):
        if self.training:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            
            early_feat = self.early_pool(out).view(out.size(0), -1)
            early_logits = self.early_linear(early_feat)
            
            out = self.layer3(out)
            out = F.avg_pool2d(out, out.size()[3])
            out = out.view(out.size(0), -1)
            final_logits = self.final_linear(out)
            
            return early_logits, final_logits
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            
            early_feat = self.early_pool(out).view(out.size(0), -1)
            early_logits = self.early_linear(early_feat)
            
            probs = F.softmax(early_logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            
            confident_mask = entropy < entropy_threshold
            unsure_mask = ~confident_mask
            
            final_output = torch.empty_like(early_logits)
            
            if confident_mask.any():
                final_output[confident_mask] = early_logits[confident_mask]
                
            if unsure_mask.any():
                unsure_out = out[unsure_mask]
                unsure_out = self.layer3(unsure_out)
                unsure_out = F.avg_pool2d(unsure_out, unsure_out.size()[3])
                unsure_out = unsure_out.view(unsure_out.size(0), -1)
                final_output[unsure_mask] = self.final_linear(unsure_out)
                
            return final_output, confident_mask

def early_exit_int8_resnet20():
    return EarlyExitINT8ResNet(EarlyExitINT8BasicBlock, [3, 3, 3])
