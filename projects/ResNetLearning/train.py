import os
import sys
import datetime
import warnings

import torch
import torch.nn as nn
import swanlab
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(PROJECT_ROOT)

from models.myModels.myResNet import ResNet
from config import (
    num_classes,
    epochs,
    batch_size,
    image_size,
    learning_rate,
    momentum,
    weight_decay,
    lr_step_size,
    lr_gamma,
)
from dataset import create_train_val_dataloaders
from utils.latest_checkpoint import get_latest_checkpoint

# 忽略 PIL 读取图片时由于 EXIF 信息损坏导致的警告
warnings.filterwarnings("ignore", "(?s).*Corrupt EXIF data.*", category=UserWarning)


def main() -> None:
    os.makedirs(f"{PROJECT_ROOT}/checkpoints", exist_ok=True)

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    swanlab.init(
        project="ResNetLearning",
        experiment_name=f"resnet_{current_time}",
        config={
            "dataset": "mini-ImageNet",
            "epochs": epochs,
            "batch_size": batch_size,
            "image_size": image_size,
            "learning_rate": learning_rate,
            "optimizer": "SGD(momentum)",
            "scheduler": "StepLR",
            "lr_step_size": lr_step_size,
            "lr_gamma": lr_gamma,
            "weight_decay": weight_decay,
            "loss_fn": "CrossEntropyLoss",
        },
    )

    train_loader, val_loader = create_train_val_dataloaders()
    model = ResNet(num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    train_steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_step_size, gamma=lr_gamma
    )

    checkpoint_path, start_epoch = get_latest_checkpoint(
        f"{PROJECT_ROOT}/checkpoints", epochs
    )
    global_step = start_epoch * train_steps_per_epoch
    if checkpoint_path:
        print(
            f"Resuming training from {checkpoint_path}, starting at epoch {start_epoch}"
        )
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", start_epoch))
        global_step = int(
            checkpoint.get("global_step", start_epoch * train_steps_per_epoch)
        )

    else:
        print("No checkpoint found. Starting from scratch.")
        start_epoch = 0
        global_step = 0

    # 如果有多张显卡，使用 DataParallel 包装模型
    if torch.cuda.device_count() > 1:
        print(f"use {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()

    val_acc_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    val_prec_metric = Precision(
        task="multiclass", num_classes=num_classes, average="macro"
    ).to(device)
    val_rec_metric = Recall(
        task="multiclass", num_classes=num_classes, average="macro"
    ).to(device)
    val_f1_metric = F1Score(
        task="multiclass", num_classes=num_classes, average="macro"
    ).to(device)

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        seen_samples = 0
        for i, (X, y) in enumerate(train_loader):
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            y_hat = model(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            batch_size_now = X.size(0)
            running_loss += loss.item() * batch_size_now
            seen_samples += batch_size_now
            global_step += 1

            if i % 100 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}")
                swanlab.log(
                    {
                        "Train/Loss": loss.item(),
                        "Train/LearningRate": optimizer.param_groups[0]["lr"],
                    },
                    step=global_step,
                )

        # 验证
        model.eval()
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_X = val_X.to(device, non_blocking=True)
                val_y = val_y.to(device, non_blocking=True)
                val_y_hat = model(val_X)
                predicted = val_y_hat.argmax(dim=1)

                # 每个 batch，让 metrics 更新自己的状态
                val_acc_metric.update(predicted, val_y)
                val_prec_metric.update(predicted, val_y)
                val_rec_metric.update(predicted, val_y)
                val_f1_metric.update(predicted, val_y)

            val_acc = val_acc_metric.compute().item()
            val_prec = val_prec_metric.compute().item()
            val_rec = val_rec_metric.compute().item()
            val_f1 = val_f1_metric.compute().item()

            print(
                f"Epoch {epoch+1} Validation - Acc: {val_acc:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}"
            )

            swanlab.log(
                {
                    "Val/Accuracy": val_acc,
                    "Val/Precision": val_prec,
                    "Val/Recall": val_rec,
                    "Val/F1": val_f1,
                },
                step=epoch + 1,
            )

            val_acc_metric.reset()
            val_prec_metric.reset()
            val_rec_metric.reset()
            val_f1_metric.reset()

        epoch_train_loss = running_loss / max(1, seen_samples)
        current_lr = optimizer.param_groups[0]["lr"]
        swanlab.log(
            {
                "Train/EpochLoss": epoch_train_loss,
                "Train/EpochLearningRate": current_lr,
            },
            step=epoch + 1,
        )
        print(
            f"Epoch {epoch+1} - TrainLoss: {epoch_train_loss:.4f}, Learning Rate: {current_lr:.6f}"
        )
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            # 如果使用了 DataParallel，保存 model.module 的 state_dict
            state_dict = (
                model.module.state_dict()
                if isinstance(model, nn.DataParallel)
                else model.state_dict()
            )
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "model_state_dict": state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }
            torch.save(
                checkpoint, f"{PROJECT_ROOT}/checkpoints/resnet_epoch_{epoch+1}.pth"
            )
            print(f"Epoch {epoch+1} model saved")

    state_dict = (
        model.module.state_dict()
        if isinstance(model, nn.DataParallel)
        else model.state_dict()
    )
    checkpoint = {
        "epoch": epochs,
        "global_step": global_step,
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    torch.save(checkpoint, f"{PROJECT_ROOT}/checkpoints/resnet_latest.pth")
    print("Latest model saved")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
