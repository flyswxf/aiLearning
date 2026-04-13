import torch
import torch.nn as nn

def nin_block(in_channels: int, out_channels: int, kernel_size: int, strides: int, padding: int) -> nn.Sequential:
    """
    构建一个标准的 NiN 块
    
    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 空间卷积的卷积核大小
        strides: 空间卷积的步长
        padding: 空间卷积的填充
    返回:
        nn.Sequential 包含一个空间卷积和两个 1x1 卷积（各自带有 ReLU 激活）
    """
    return nn.Sequential(
        # 第一层：常规的空间特征提取卷积层
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(inplace=True),
        
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(inplace=True)
    )


class NiN(nn.Module):
    """
    标准 NiN 模型模板
    
    默认输入尺寸建议为 224x224（ImageNet 标准尺寸）。
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 10, init_weights: bool = True) -> None:
        super().__init__()
        
        # 1. 特征提取网络（由多个 NiN 块和最大池化层交替组成）
        self.features = nn.Sequential(
            nin_block(in_channels, 96, kernel_size=11, strides=4, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nin_block(96, 256, kernel_size=5, strides=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nin_block(256, 384, kernel_size=3, strides=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Dropout(0.5),
            
            nin_block(384, num_classes, kernel_size=3, strides=1, padding=1),
        )
        
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.flatten = nn.Flatten()
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_avgpool(x)
        x = self.flatten(x)
        return x

    def _initialize_weights(self) -> None:
        """Kaiming 权重初始化策略"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
