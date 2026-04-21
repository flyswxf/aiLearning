import torch
import torch.nn as nn
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    """
    ResNet 的基础残差块 (BasicBlock)，主要用于 ResNet-18 和 ResNet-34
    """
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        # 第一个卷积层，可能改变通道数和减半特征图尺寸 (strides=2)
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.bn1 = nn.BatchNorm2d(num_channels)
        
        # 第二个卷积层，保持尺寸和通道数不变
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
        
        # 1x1 卷积层，用于匹配输入和输出的形状 (通道数或分辨率不同时)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
            
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        # 残差连接
        Y += X
        return F.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    """
    构建一个 ResNet 的 stage (包含多个残差块)
    first_block: 是否是第一个阶段。第一个阶段的第一个块不需要减半高宽。
    """
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            # 除第一个阶段外，每个阶段的第一个残差块将高宽减半
            blk.append(ResidualBlock(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(ResidualBlock(num_channels, num_channels))
    return nn.Sequential(*blk)

class ResNet(nn.Module):
    """
    以 ResNet-18 为例的经典 ResNet 实现
    """
    def __init__(self, num_classes=1000):
        super().__init__()
        # 前置处理 (与 GoogLeNet 类似)
        self.b1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 主体残差阶段 (ResNet-18 每个阶段包含 2 个残差块)
        self.b2 = resnet_block(64, 64, 2, first_block=True)
        self.b3 = resnet_block(64, 128, 2)
        self.b4 = resnet_block(128, 256, 2)
        self.b5 = resnet_block(256, 512, 2)
        
        # 尾部处理 (全局平均池化 + 全连接层)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.head(x)
        return x
