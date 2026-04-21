import os
import sys
from typing import Callable, Optional, Tuple

import torch
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torchvision import transforms


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(PROJECT_ROOT)

from utils.download_datasets import download_mini_imagenet
from config import batch_size, image_size, num_workers


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


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )
    return train_transform, eval_transform


def build_datasets(
    cache_dir: Optional[str] = None,
) -> Tuple[
    MiniImageNetTorchDataset, MiniImageNetTorchDataset, MiniImageNetTorchDataset
]:
    dataset_dict: DatasetDict = download_mini_imagenet(cache_dir=cache_dir)
    train_transform, eval_transform = build_transforms()

    train_dataset = MiniImageNetTorchDataset(
        hf_split=dataset_dict["train"], transform=train_transform
    )
    test_dataset = MiniImageNetTorchDataset(
        hf_split=dataset_dict["test"], transform=eval_transform
    )
    val_dataset = MiniImageNetTorchDataset(
        hf_split=dataset_dict["validation"], transform=eval_transform
    )
    return train_dataset, test_dataset, val_dataset


def _is_kaggle_runtime() -> bool:
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.exists("/kaggle")


def _build_loader(
    dataset: TorchDataset, shuffle: bool, data_batch_size: int, data_num_workers: int
) -> DataLoader:
    effective_num_workers = data_num_workers
    use_pin_memory = torch.cuda.is_available()

    # Kaggle 上使用 HF datasets + PIL 解码时，多 worker 容易把 RAM 顶得过高。
    if _is_kaggle_runtime():
        effective_num_workers = min(data_num_workers, 1)
        use_pin_memory = False

    loader_kwargs = {
        "batch_size": data_batch_size,
        "shuffle": shuffle,
        "num_workers": effective_num_workers,
        "pin_memory": use_pin_memory,
    }

    if effective_num_workers > 0:
        loader_kwargs["prefetch_factor"] = 1
        loader_kwargs["persistent_workers"] = False

    return DataLoader(dataset, **loader_kwargs)


def create_dataloaders(
    data_batch_size: int = batch_size,
    data_num_workers: int = num_workers,
    cache_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset, test_dataset, val_dataset = build_datasets(cache_dir=cache_dir)
    train_loader = _build_loader(
        train_dataset,
        shuffle=True,
        data_batch_size=data_batch_size,
        data_num_workers=data_num_workers,
    )
    test_loader = _build_loader(
        test_dataset,
        shuffle=False,
        data_batch_size=data_batch_size,
        data_num_workers=data_num_workers,
    )
    val_loader = _build_loader(
        val_dataset,
        shuffle=False,
        data_batch_size=data_batch_size,
        data_num_workers=data_num_workers,
    )
    return train_loader, test_loader, val_loader


def create_train_val_dataloaders(
    data_batch_size: int = batch_size,
    data_num_workers: int = num_workers,
    cache_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    dataset_dict: DatasetDict = download_mini_imagenet(cache_dir=cache_dir)
    train_transform, eval_transform = build_transforms()

    train_dataset = MiniImageNetTorchDataset(
        hf_split=dataset_dict["train"], transform=train_transform
    )
    val_dataset = MiniImageNetTorchDataset(
        hf_split=dataset_dict["validation"], transform=eval_transform
    )

    train_loader = _build_loader(
        train_dataset,
        shuffle=True,
        data_batch_size=data_batch_size,
        data_num_workers=data_num_workers,
    )
    val_loader = _build_loader(
        val_dataset,
        shuffle=False,
        data_batch_size=data_batch_size,
        data_num_workers=data_num_workers,
    )
    return train_loader, val_loader


def create_test_dataloader(
    data_batch_size: int = batch_size,
    data_num_workers: int = num_workers,
    cache_dir: Optional[str] = None,
) -> DataLoader:
    dataset_dict: DatasetDict = download_mini_imagenet(cache_dir=cache_dir)
    _, eval_transform = build_transforms()

    test_dataset = MiniImageNetTorchDataset(
        hf_split=dataset_dict["test"], transform=eval_transform
    )
    return _build_loader(
        test_dataset,
        shuffle=False,
        data_batch_size=data_batch_size,
        data_num_workers=data_num_workers,
    )


if __name__ == "__main__":
    train_dataset, _, _ = build_datasets()
    X, y = train_dataset[0]
    print(X.shape)
    print(y)
