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
from config import num_classes, epochs, batch_size, image_size
from dataset import train_loader, val_loader
from utils.latest_checkpoint import get_latest_checkpoint
import swanlab
import datetime
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score

os.makedirs(f"{PROJECT_ROOT}/checkpoints", exist_ok=True)

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# 初始化 SwanLab 实验
swanlab.init(
    project="ResNetLearning",
    experiment_name=f"resnet_{current_time}",
    config={
        "dataset": "mini-ImageNet",
        "epochs": epochs,
        "batch_size": batch_size,
        "image_size": image_size,
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "loss_fn": "CrossEntropyLoss",
    },
)


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

# 初始化 TorchMetrics 评估指标，并放到正确的 device 上
val_acc_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)
val_prec_metric = Precision(
    task="multiclass", num_classes=num_classes, average="macro"
).to(device)
val_rec_metric = Recall(task="multiclass", num_classes=num_classes, average="macro").to(
    device
)
val_f1_metric = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(
    device
)

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
            # 记录训练 Loss
            swanlab.log({"Train/Loss": loss.item()})

        # 验证
        model.eval()
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_X, val_y = val_X.to(device), val_y.to(device)
                val_y_hat = model(val_X)
                _, predicted = torch.max(val_y_hat.data, 1)

                # 每个 batch，让 metrics 更新自己的状态
                val_acc_metric.update(predicted, val_y)
                val_prec_metric.update(predicted, val_y)
                val_rec_metric.update(predicted, val_y)
                val_f1_metric.update(predicted, val_y)

            # 一个 epoch 结束后，计算最终结果
            val_acc = val_acc_metric.compute().item()
            val_prec = val_prec_metric.compute().item()
            val_rec = val_rec_metric.compute().item()
            val_f1 = val_f1_metric.compute().item()

            print(
                f"Epoch {epoch+1} Validation - Acc: {val_acc:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}"
            )

            # 记录所有验证集指标到 SwanLab
            swanlab.log(
                {
                    "Val/Accuracy": val_acc,
                    "Val/Precision": val_prec,
                    "Val/Recall": val_rec,
                    "Val/F1": val_f1,
                },
                step=epoch + 1,
            )

            # 记录完之后，务必清空 metrics 的状态，以免影响下一个 epoch
            val_acc_metric.reset()
            val_prec_metric.reset()
            val_rec_metric.reset()
            val_f1_metric.reset()
            
    if (epoch + 1) % 10 == 0:
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
