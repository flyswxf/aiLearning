import os
import sys
import gc
import torch
import torch.nn as nn
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score
from sklearn.metrics import classification_report
import warnings

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(PROJECT_ROOT)

from models.myModels.myResNet import ResNet
from dataset import create_test_dataloader
from config import num_classes

warnings.filterwarnings("ignore")
# 忽略 PIL 读取图片时由于 EXIF 信息损坏导致的警告
warnings.filterwarnings("ignore", "(?s).*Corrupt EXIF data.*", category=UserWarning)


def main() -> None:
    checkpoint_path = f"{PROJECT_ROOT}/checkpoints/resnet_latest.pth"
    assert os.path.exists(checkpoint_path), "No checkpoint found."

    test_loader = create_test_dataloader()
    model = ResNet(num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    # 如果有多张显卡，在加载完权重后再用 DataParallel 包装模型
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    test_acc_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    test_prec_metric = Precision(
        task="multiclass", num_classes=num_classes, average="macro"
    ).to(device)
    test_rec_metric = Recall(
        task="multiclass", num_classes=num_classes, average="macro"
    ).to(device)
    test_f1_metric = F1Score(
        task="multiclass", num_classes=num_classes, average="macro"
    ).to(device)

    model.eval()
    with torch.no_grad():
        all_preds = []
        all_targets = []
        for X, y in test_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_hat = model(X)
            predicted = y_hat.argmax(dim=1)

            test_acc_metric.update(predicted, y)
            test_prec_metric.update(predicted, y)
            test_rec_metric.update(predicted, y)
            test_f1_metric.update(predicted, y)

            # 这里依然保留收集所有预测值，为了最后给 scikit-learn 画详细分类报告
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

        test_acc = test_acc_metric.compute().item()
        test_prec = test_prec_metric.compute().item()
        test_rec = test_rec_metric.compute().item()
        test_f1 = test_f1_metric.compute().item()

        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Precision (Macro): {test_prec:.4f}")
        print(f"Test Recall (Macro): {test_rec:.4f}")
        print(f"Test F1-score (Macro): {test_f1:.4f}")

        print("\n" + "=" * 50)
        print("Detailed Classification Report:")
        print("=" * 50)
        print(classification_report(all_targets, all_preds, digits=4, zero_division=0))
        gc.collect()


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
