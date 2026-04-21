import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, use_1x1: bool = False, stride=1
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )
        self.norm2 = nn.BatchNorm2d(out_channels)
        if use_1x1:
            self.conv3 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
            )
        else:
            self.conv3 = None

    def forward(self, X):
        Y = F.relu(self.norm1(self.conv1(X)))
        Y = self.norm2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


def res_block(in_channels, out_channels, num_residuals, first_block: bool = False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk += [
                ResidualBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    use_1x1=True,
                    stride=2,
                )
            ]
        else:
            blk += [ResidualBlock(in_channels=out_channels, out_channels=out_channels)]

    return nn.Sequential(*blk)


class ResNet(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.b2 = res_block(64, 64, 2, first_block=True)
        self.b3 = res_block(64, 128, 2)
        self.b4 = res_block(128, 256, 2)
        self.b5 = res_block(256, 512, 2)

        self.head = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)), nn.Flatten(), nn.Linear(512, num_classes)
        )

    def forward(self, X):
        X = self.b1(X)
        X = self.b2(X)
        X = self.b3(X)
        X = self.b4(X)
        X = self.b5(X)
        X = self.head(X)
        return X
