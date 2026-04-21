import os
import sys
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torchvision import transforms


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(PROJECT_ROOT)

from utils.download_datasets import download_mini_imagenet
from config import batch_size, image_size


class MiniImageNetTorchDataset(TorchDataset):
    def __init__(self, hf_split: Dataset, transform: Optional[Callable] = None) -> None:
        super().__init__()
        self.hf_split = hf_split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.hf_split)

    def __getitem__(self, index: int):
        sample = self.hf_split[index]
        image = sample["image"].convert("RGB")
        label = int(sample["label"])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)
train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ]
)
test_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ]
)

dataset_dict: DatasetDict = download_mini_imagenet()

train_dataset = MiniImageNetTorchDataset(
    hf_split=dataset_dict["train"], transform=train_transform
)
test_dataset = MiniImageNetTorchDataset(
    hf_split=dataset_dict["test"], transform=test_transform
)
val_dataset = MiniImageNetTorchDataset(
    hf_split=dataset_dict["val"], transform=test_transform
)

pin_memory = torch.cuda.is_available()

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=pin_memory,
    num_workers=2,  # 降低多进程数量，避免内存 OOM
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=pin_memory,
    num_workers=2,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=pin_memory,
    num_workers=2,
)

if __name__ == "__main__":
    X, y = train_dataset[0]
    print(X.shape)
    print(y)
