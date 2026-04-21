import os
import re
import torch
from typing import Tuple, Optional


def get_latest_checkpoint(
    checkpoints_dir: str, total_epochs: int
) -> Tuple[Optional[str], int]:
    """
    扫描 checkpoints 文件夹，返回最新的 checkpoint 路径和已经完成的 epoch 数。
    如果找到 resnet_latest.pth，认为已完成所有 epochs。
    否则找到数字最大的 resnet_epoch_{X}.pth。
    """
    if not os.path.exists(checkpoints_dir):
        return None, 0

    # 1. 如果存在 latest，说明已经跑完所有的 epoch
    latest_path = os.path.join(checkpoints_dir, "resnet_latest.pth")
    if os.path.exists(latest_path):
        return latest_path, total_epochs

    # 2. 否则去寻找最大的 epoch checkpoint
    max_epoch = 0
    best_file = None
    for filename in os.listdir(checkpoints_dir):
        match = re.match(r"resnet_epoch_(\d+)\.pth", filename)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                best_file = filename

    if best_file:
        return os.path.join(checkpoints_dir, best_file), max_epoch

    # 如果什么都没找到
    return None, 0