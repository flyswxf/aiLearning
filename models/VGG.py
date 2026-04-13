import torch
import torch.nn as nn
from typing import List, Union, Dict, cast

# =========================================================================
# 1. VGG 网络结构配置字典 (可自由扩展)
# -------------------------------------------------------------------------
# 数字: 表示卷积层的输出通道数 (kernel_size 固定为 3, padding 固定为 1)
# 'M':  表示 MaxPooling 最大池化层 (kernel_size 为 2, stride 为 2)
# =========================================================================
vgg_cfgs: Dict[str, List[Union[int, str]]] = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# =========================================================================
# 2. 核心 VGG 模型类
# =========================================================================
class VGG(nn.Module):
    """
    标准 VGG 模型模板
    
    支持: 
    1. 动态生成卷积特征提取层
    2. 自定义输入通道数、分类数
    3. 支持 Dropout 和自适应池化层
    4. 内置标准的权重初始化 (Kaiming Init)
    """
    def __init__(
        self, 
        features: nn.Module, 
        num_classes: int = 10, 
        init_weights: bool = True,
        dropout: float = 0.5
    ) -> None:
        super().__init__()
        
        # 1. 卷积特征提取层 (由外部工厂函数动态构建并传入)
        self.features = features
        
        # 2. 自适应池化层 (AdaptiveAvgPool2d)
        # 作用: 无论输入图像的尺寸是多少，强制将其池化为固定的 7x7 大小
        # 这使得 VGG 可以处理任意大于 32x32 的图像，而不用担心与后面的全连接层维度不匹配
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # 3. 分类器全连接层
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            
            nn.Linear(4096, num_classes),
        )
        
        # 4. 初始化权重 (有助于深层网络的稳定收敛)
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)    # 提取特征
        x = self.avgpool(x)     # 统一尺寸为 7x7
        x = self.classifier(x)  # 展平并分类
        return x

    def _initialize_weights(self) -> None:
        """Kaiming 初始化策略，针对 ReLU 激活函数优化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# =========================================================================
# 3. 辅助工厂函数：动态构建特征提取层
# =========================================================================
def make_layers(cfg: List[Union[int, str]], in_channels: int = 3, batch_norm: bool = False) -> nn.Sequential:
    """
    根据给定的配置列表，自动组装所有的卷积层、BN层和池化层
    """
    layers: List[nn.Module] = []
    current_channels = in_channels
    
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            # VGG 经典设定：3x3 卷积核，步长 1，填充 1
            conv2d = nn.Conv2d(current_channels, v, kernel_size=3, padding=1)
            
            # 引入 BatchNorm (BN) 可以极大加快 VGG 训练速度
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            
            current_channels = v
            
    return nn.Sequential(*layers)


# =========================================================================
# 4. 快速实例化函数 (提供对外的 API)
# =========================================================================
def vgg16(in_channels: int = 3, num_classes: int = 10, batch_norm: bool = False, **kwargs) -> VGG:
    """快速构建一个 VGG-16 模型"""
    cfg = vgg_cfgs['vgg16']
    features = make_layers(cfg, in_channels=in_channels, batch_norm=batch_norm)
    return VGG(features, num_classes=num_classes, **kwargs)

def vgg19(in_channels: int = 3, num_classes: int = 10, batch_norm: bool = False, **kwargs) -> VGG:
    """快速构建一个 VGG-19 模型"""
    cfg = vgg_cfgs['vgg19']
    features = make_layers(cfg, in_channels=in_channels, batch_norm=batch_norm)
    return VGG(features, num_classes=num_classes, **kwargs)


