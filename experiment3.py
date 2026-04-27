import torch
import torch.nn as nn
import torch.nn.functional as F

class ActivityGatedBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.has_shortcut = (stride != 1 or in_planes != planes)
        if self.has_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        else:
            self.shortcut = nn.Sequential()
            
        # Store FLOPs for conv1 and conv2 separately
        self.flops_conv1 = 0
        self.flops_conv2 = 0

    def forward(self, x, threshold=0.0):
        # We compute conv1 -> bn1 -> relu (always executed)
        out1 = F.relu(self.bn1(self.conv1(x)))
        
        # Add FLOPs for conv1 for the whole batch
        b_size = x.size(0)
        flops_this_pass = self.flops_conv1 * b_size
        
        if self.has_shortcut or threshold <= 0.0:
            # Always execute fully for downsample blocks or if threshold is <= 0
            out2 = self.bn2(self.conv2(out1))
            flops_this_pass += self.flops_conv2 * b_size
            return F.relu(out2 + self.shortcut(x)), flops_this_pass
            
        # Activity Gating Logic (Zero-Cost Metric)
        activity = (out1 > 0).float().mean(dim=(1, 2, 3))
        active_mask = activity >= threshold
        aborted_mask = ~active_mask
        
        final_out = torch.empty_like(out1)
        
        if aborted_mask.any():
            # For aborted images, we skip conv2 and just add identity
            final_out[aborted_mask] = x[aborted_mask]
            
        if active_mask.any():
            active_out1 = out1[active_mask]
            active_out2 = self.bn2(self.conv2(active_out1))
            final_out[active_mask] = active_out2 + x[active_mask]
            
            # Add FLOPs only for the images that executed conv2
            flops_this_pass += self.flops_conv2 * active_mask.sum().item()
            
        return F.relu(final_out), flops_this_pass

class ActivityGatedResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)
        
        # Pre-compute FLOPs for each block
        self.flops_base_conv1 = 3 * 16 * 9 * 32 * 32
        self._assign_block_flops()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = nn.ModuleList()
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return layers

    def _assign_block_flops(self):
        # Calculate FLOPs for Conv1 and Conv2 separately for each block
        for layer_idx, layer in enumerate([self.layer1, self.layer2, self.layer3]):
            for block_idx, block in enumerate(layer):
                # Shape depends on layer output
                if layer_idx == 0: h = 32
                elif layer_idx == 1: h = 16
                else: h = 8
                
                c_out = block.conv1.out_channels
                c_in = block.conv1.in_channels
                
                # Conv1 FLOPs
                flops_c1 = c_in * c_out * 9 * h * h
                
                # Conv2 FLOPs
                flops_c2 = c_out * c_out * 9 * h * h
                
                if block.has_shortcut:
                    # add shortcut conv FLOPs to conv1
                    flops_c1 += c_in * c_out * 1 * h * h
                
                block.flops_conv1 = flops_c1
                block.flops_conv2 = flops_c2

    def forward(self, x, threshold=0.0):
        total_flops = self.flops_base_conv1 * x.size(0)
        
        out = F.relu(self.bn1(self.conv1(x)))
        
        for layer in [self.layer1, self.layer2, self.layer3]:
            for block in layer:
                out, block_flops = block(out, threshold)
                total_flops += block_flops
                
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        total_flops += 64 * self.linear.out_features * x.size(0)
        
        return out, total_flops

def activity_gated_resnet20():
    return ActivityGatedResNet(ActivityGatedBasicBlock, [3, 3, 3])
