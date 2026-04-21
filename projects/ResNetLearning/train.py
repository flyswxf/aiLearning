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
from config import num_classes, epochs
from dataset import train_loader


model = ResNet(num_classes=num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

os.makedirs(f"{PROJECT_ROOT}/checkpoints", exist_ok=True)

for epoch in range(epochs):
    model.train()
    for i, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat = model(X)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}")

    if (epoch + 1) % 10 == 0:
        torch.save(
            model.state_dict(), f"{PROJECT_ROOT}/checkpoints/resnet_epoch_{epoch+1}.pth"
        )

torch.save(model.state_dict(), f"{PROJECT_ROOT}/checkpoints/resnet_latest.pth")
