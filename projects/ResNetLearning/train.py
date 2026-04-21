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
from dataset import train_loader, val_loader
from utils.latest_checkpoint import get_latest_checkpoint

os.makedirs(f"{PROJECT_ROOT}/checkpoints", exist_ok=True)


model = ResNet(num_classes=num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

checkpoint_path, start_epoch = get_latest_checkpoint(
    f"{PROJECT_ROOT}/checkpoints", epochs
)
if checkpoint_path:
    print(f"Resuming training from {checkpoint_path}, starting at epoch {start_epoch}")
    model.load_state_dict(torch.load(checkpoint_path))
else:
    print("No checkpoint found. Starting from scratch.")
    start_epoch = 0

# 如果有多张显卡，使用 DataParallel 包装模型
if torch.cuda.device_count() > 1:
    print(f"use {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


for epoch in range(start_epoch, epochs):
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
        # 验证
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for val_X, val_y in val_loader:
                val_X, val_y = val_X.to(device), val_y.to(device)
                val_y_hat = model(val_X)
                _, predicted = torch.max(val_y_hat.data, 1)
                total += val_y.size(0)
                correct += (predicted == val_y).sum().item()
            val_acc = 100 * correct / total
            print(f"Epoch {epoch+1} Validation Accuracy: {val_acc:.2f}%")

        # 如果使用了 DataParallel，保存 model.module 的 state_dict
        state_dict = (
            model.module.state_dict()
            if isinstance(model, nn.DataParallel)
            else model.state_dict()
        )
        torch.save(state_dict, f"{PROJECT_ROOT}/checkpoints/resnet_epoch_{epoch+1}.pth")
        print(f"Epoch {epoch+1} model saved")

# 保存最后一次的模型权重
state_dict = (
    model.module.state_dict()
    if isinstance(model, nn.DataParallel)
    else model.state_dict()
)
torch.save(state_dict, f"{PROJECT_ROOT}/checkpoints/resnet_latest.pth")
print("Latest model saved")
