from turtle import forward
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, i,o,use_1x1,strides) -> None:
        super().__init__()
        self.conv1=nn.Conv2d(i,o,3,strides,1)
        self.norm1=nn.BatchNorm2d(o)
        self.conv2=nn.Conv2d(o,o,3,1,1)
        self.norm2=nn.BatchNorm2d(o)
        
        if use_1x1:
            self.conv3=nn.Conv2d(i,o,1,strides,1)
        else:
            self.conv3=None
            
    def forward(self,X):
        Y=F.relu(self.norm1(self.conv1(X)))
        Y=self.norm2(self.conv2(Y))
        if self.conv3:
            X=self.conv3(X)
        Y+=X
        return F.relu(Y)