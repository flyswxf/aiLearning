import os
import sys
import torch
import torch.nn as nn

# Add the project root directory to sys.path so we can import from the 'models' package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.AlexNet import AlexNet
from models.VGG import vgg16
from models.NiN import NiN
from models.GoogLeNet import GoogLeNet



def test_model_layers(model: nn.Module, input_tensor: torch.Tensor):
    """
    通用的逐层打印模型结构与张量形状变化的测试函数（递归版本）。
    完美支持任意深度的嵌套容器 (Sequential, ModuleList, ModuleDict, 或自定义 Module)。
    """
    print(f"\n{'='*60}")
    print(f"Testing Model: {model.__class__.__name__}")
    print(f"Input Shape:   {input_tensor.shape}")
    print(f"{'='*60}")
    
    # 我们需要一个变量来在递归中传递当前的特征图（张量）
    # 在 Python 闭包中，为了能在内部函数中修改外部变量，我们用一个列表包起来
    current_x = [input_tensor]
    
    def _recursive_forward(module: nn.Module, name: str, depth: int = 0):
        # 缩进，让打印结果有层次感
        indent = "  " * depth
        
        # 判断当前模块是否有子模块，以及它是否是一个“顺序容器”
        children = list(module.named_children())
        is_sequential_like = isinstance(module, nn.Sequential) or module is model
        
        if not children or not is_sequential_like:
            # 核心逻辑：如果它是“叶子节点”或者是一个自定义的非顺序结构（如 Inception），
            # 我们就把他当做一个整体执行前向传播，更新特征图，并打印形状
            
            # 注意：像 Dropout 这样的层，前向传播可能会改变内容但不会改变形状，我们依然执行它

            current_x[0] = module(current_x[0])
            print(f"{indent}[-] {name + ': ' if name else ''}{module.__class__.__name__:15s} Output Shape: \t{current_x[0].shape}")
            
        else:
            # 如果它是一个顺序“容器”（比如 Sequential, 或是 模型本身）
            # 我们打印容器的名字，然后递归进入它的子模块
            if name:
                print(f"\n{indent}--- Entering block: {name} ({module.__class__.__name__}) ---")
                
            for child_name, child_module in children:
                _recursive_forward(child_module, child_name, depth + 1)

    # 从模型的根节点开始递归
    _recursive_forward(model, "", depth=0)
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    model = GoogLeNet()
    test_model_layers(model, torch.randn(size=(1, 3, 224, 224), dtype=torch.float32))