import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation块"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, 
            stride, padding, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class BasicBlock(nn.Module):
    """基础残差块"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, use_se=False, use_ds=False):
        super().__init__()
        ConvLayer = DepthwiseSeparableConv if use_ds else nn.Conv2d
        
        self.conv1 = ConvLayer(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = ConvLayer(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )
        
        self.se = SEBlock(out_channels) if use_se else nn.Identity()
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class FastCIFARNet(nn.Module):
    """高效CIFAR-10分类模型"""
    def __init__(self, block, num_blocks, num_classes=10, use_se=True, use_ds=True):
        super().__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, use_se=use_se, use_ds=use_ds)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, use_se=use_se, use_ds=use_ds)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, use_se=use_se, use_ds=use_ds)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256 * block.expansion, num_classes)
        
        # 通道重排
        self.channel_shuffle = nn.ChannelShuffle(groups=4)
    
    def _make_layer(self, block, out_channels, num_blocks, stride, use_se, use_ds):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(
                self.in_channels, out_channels, stride, use_se, use_ds
            ))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 渐进式图像尺寸训练 - 初始阶段使用较小尺寸
        if self.training and x.size(2) > 32:
            x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.channel_shuffle(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def create_model():
    """创建模型实例（类似ResNet-20的架构）"""
    return FastCIFARNet(BasicBlock, [3, 3, 3])
