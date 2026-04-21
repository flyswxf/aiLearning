import torch.nn as nn
from typing import List

vgg_cfgs: List[str|int] = [
    64,
    "M",
    128,
    "M",
    256,
    256,
    "M",
    512,
    512,
    "M",
    512,
    512,
    "M",
]


def make_layer(cfg: list, in_channels: int = 3) -> nn.Sequential:
    layers = []
    cur = in_channels
    for v in cfg:
        if isinstance(v, str):
            layers += [nn.MaxPool2d(2, 2)]
        else:
            layers += [
                nn.Conv2d(
                    in_channels=cur, out_channels=v, kernel_size=3, stride=1, padding=1
                )
            ]
            layers += [nn.ReLU()]
            cur = v
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, features: nn.Module, num_classes: int = 10):
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


def vgg_11(in_channels: int = 3, num_classes: int = 10) -> VGG:
    features = make_layer(vgg_cfgs, in_channels=in_channels)
    return VGG(features, num_classes=num_classes)
