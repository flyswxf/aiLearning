import os
import sys

# Add the project root directory to sys.path so we can import from the 'models' package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.AlexNet import AlexNet
import torch


model=AlexNet()

X=torch.randn(size=(1,1,224,224),dtype=torch.float32)
for layer in model.layers:
    X=layer(X)
    print(layer.__class__.__name__,"Output Shape :\t",X.shape)
    
