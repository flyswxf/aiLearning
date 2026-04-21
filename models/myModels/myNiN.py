import torch.nn as nn


def nin_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=1),
        nn.ReLU(),
    )


class NiN(nn.Module):
    def __init__(self,in_channels:int=3,num_classes:int=10 ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nin_block(in_channels,),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nin_block(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nin_block(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(),
            nin_block(),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Flatten()
