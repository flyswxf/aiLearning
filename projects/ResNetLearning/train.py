import os
import sys
import datetime
import gc
import warnings
import psutil

import torch
import torch.nn as nn
import swanlab
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(PROJECT_ROOT)

from models.myModels.myResNet import ResNet
from config import num_classes, epochs, batch_size, image_size
from dataset import create_train_val_dataloaders
from utils.latest_checkpoint import get_latest_checkpoint

# 忽略 PIL 读取图片时由于 EXIF 信息损坏导致的警告
warnings.filterwarnings("ignore", "(?s).*Corrupt EXIF data.*", category=UserWarning)


def print_ram(stage: str):
    """打印当前进程及子进程的物理内存占用（RSS）"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    ram_gb = mem_info.rss / (1024**3)
    print(f"[Memory Monitor] {stage} - CPU RAM: {ram_gb:.3f} GB")


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
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "loss_fn": "CrossEntropyLoss",
        },
    )

    # train_loader, val_loader = create_train_val_dataloaders()
    model = ResNet(num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    checkpoint_path, start_epoch = get_latest_checkpoint(
        f"{PROJECT_ROOT}/checkpoints", epochs
    )
    if checkpoint_path:
        print(
            f"Resuming training from {checkpoint_path}, starting at epoch {start_epoch}"
        )
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("No checkpoint found. Starting from scratch.")
        start_epoch = 0

    # 如果有多张显卡，使用 DataParallel 包装模型
    if torch.cuda.device_count() > 1:
        print(f"use {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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

    print_ram("Before Training Loop Starts")

    for epoch in range(start_epoch, epochs):
        model.train()
        print_ram(f"Epoch {epoch} Start")
        for i in range(1000): 
            X, y = torch.randn(64, 3, 224, 224, device=device), torch.randint(0, 100, (64, ), device=device)
        # for i, (X, y) in enumerate(train_loader):
            # X = X.to(device, non_blocking=True)
            # y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            y_hat = model(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}")
                swanlab.log({"Train/Loss": loss.item()})
                if i > 0 and i % 500 == 0:
                    print_ram(f"Epoch {epoch} - After {i} Train Batches")

        print_ram(f"Epoch {epoch} - After Train Loop")

        # 验证
        # model.eval()
        # with torch.no_grad():
        #     for val_X, val_y in val_loader:
        #         val_X = val_X.to(device, non_blocking=True)
        #         val_y = val_y.to(device, non_blocking=True)
        #         val_y_hat = model(val_X)
        #         predicted = val_y_hat.argmax(dim=1)

        #         # 每个 batch，让 metrics 更新自己的状态
        #         val_acc_metric.update(predicted, val_y)
        #         val_prec_metric.update(predicted, val_y)
        #         val_rec_metric.update(predicted, val_y)
        #         val_f1_metric.update(predicted, val_y)

        #     val_acc = val_acc_metric.compute().item()
        #     val_prec = val_prec_metric.compute().item()
        #     val_rec = val_rec_metric.compute().item()
        #     val_f1 = val_f1_metric.compute().item()

        #     print(
        #         f"Epoch {epoch+1} Validation - Acc: {val_acc:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}"
        #     )

        #     swanlab.log(
        #         {
        #             "Val/Accuracy": val_acc,
        #             "Val/Precision": val_prec,
        #             "Val/Recall": val_rec,
        #             "Val/F1": val_f1,
        #         },
        #         step=epoch + 1,
        #     )

        #     val_acc_metric.reset()
        #     val_prec_metric.reset()
        #     val_rec_metric.reset()
        #     val_f1_metric.reset()

        # gc.collect()
        # print_ram(f"Epoch {epoch} - After Validation & GC")

        if (epoch + 1) % 10 == 0:
            # 如果使用了 DataParallel，保存 model.module 的 state_dict
            state_dict = (
                model.module.state_dict()
                if isinstance(model, nn.DataParallel)
                else model.state_dict()
            )
            torch.save(
                state_dict, f"{PROJECT_ROOT}/checkpoints/resnet_epoch_{epoch+1}.pth"
            )
            print(f"Epoch {epoch+1} model saved")

    state_dict = (
        model.module.state_dict()
        if isinstance(model, nn.DataParallel)
        else model.state_dict()
    )
    torch.save(state_dict, f"{PROJECT_ROOT}/checkpoints/resnet_latest.pth")
    print("Latest model saved")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
