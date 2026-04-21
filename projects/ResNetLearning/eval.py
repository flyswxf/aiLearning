import os
import sys
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(PROJECT_ROOT)

from models.myModels.myResNet import ResNet
from dataset import test_loader
from config import num_classes

model = ResNet(num_classes=num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.load_state_dict(torch.load(f"{PROJECT_ROOT}/checkpoints/resnet_latest.pth"))

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        y_hat = model(X)
        _, predicted = torch.max(y_hat.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")
