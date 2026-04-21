import os

# 设置 Hugging Face 镜像端点，必须在导入 datasets 之前设置
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from typing import Optional

from datasets import DatasetDict, load_dataset


def download_mini_imagenet(cache_dir: Optional[str] = None) -> DatasetDict:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if cache_dir is None:
        cache_dir = os.path.join(project_root, "data", "mini-imagenet")

    os.makedirs(cache_dir, exist_ok=True)

    dataset = load_dataset("timm/mini-imagenet", cache_dir=cache_dir)
    return dataset


if __name__ == "__main__":
    dataset = download_mini_imagenet()
    for split_name, split_dataset in dataset.items():
        print(f"{split_name}: {len(split_dataset)} samples")
