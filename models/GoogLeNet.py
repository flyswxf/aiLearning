import torch
import torch.nn as nn
import torch.nn.functional as F

class Inception(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        c1: int,             # 线路1: 1x1 卷积输出通道
        c2: tuple,           # 线路2: (1x1 降维输出通道, 3x3 卷积输出通道)
        c3: tuple,           # 线路3: (1x1 降维输出通道, 5x5 卷积输出通道)
        c4: int              # 线路4: 最大池化后的 1x1 卷积输出通道
    ) -> None:
        super().__init__()
        
        # 线路1: 单独的 1x1 卷积
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        # 线路2: 1x1 卷积降维 -> 3x3 卷积提取特征
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, c2[0], kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 线路3: 1x1 卷积降维 -> 5x5 卷积提取特征
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, c3[0], kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        
        # 线路4: 3x3 最大池化 -> 1x1 卷积降维
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, c4, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        # 在通道维度 (dim=1) 上拼接所有线路的输出
        return torch.cat((b1, b2, b3, b4), dim=1)


# =========================================================================
# 2. 核心 GoogLeNet 模型类 (Inception v1)
# -------------------------------------------------------------------------
# 特点：
# 1. 由多个 Inception 模块堆叠而成，网络极深但参数极少。
# 2. 同样抛弃了全连接层，改用全局平均池化 (Global Average Pooling)。
# 3. 本代码为简化的基础版 GoogLeNet (无辅助分类器 AuxLogits)。
# =========================================================================
class GoogLeNet(nn.Module):
    """
    标准 GoogLeNet (Inception v1) 模型模板
    
    默认输入尺寸建议为 224x224（ImageNet 标准尺寸）。
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1000) -> None:
        super().__init__()
        
        # Block 1: 类似 AlexNet 的前端卷积，快速降低空间分辨率
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.b2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.b3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),
            Inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.b4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (144, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.b5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.classifier(x)
        return x

