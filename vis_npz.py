import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision.utils import make_grid
import torch
from PIL import Image


def visualize_npz(npz_path, output_path=None, num_samples=16, image_size=(3, 64, 64)):
    """
    可视化 .npz 文件中 arr_0 的图像样本。

    参数：
    - npz_path: .npz 文件路径
    - output_path: 可选，保存网格图像的路径
    - num_samples: 显示/保存的样本数量
    - image_size: 每张图像的形状，默认是 (C, H, W)
    """
    data = np.load(npz_path)
    samples = data["arr_0"]

    if len(samples.shape) == 3:  # (N, H, W)
        samples = np.expand_dims(samples, 1)  # 加一个通道维度

    print(f"Loaded {samples.shape[0]} samples with shape {samples.shape[1:]}")

    # 选前 num_samples 个样本
    selected = samples[:num_samples]

    # 转换为 tensor 并做归一化
    tensor = torch.from_numpy(selected)
    if tensor.max() > 1.0:
        tensor = tensor / 255.0

    grid = make_grid(tensor, nrow=int(np.sqrt(num_samples)), normalize=True, padding=2)
    npimg = grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=(8, 8))
    plt.imshow(npimg)
    plt.axis('off')
    plt.title("Samples from npz")
    plt.tight_layout()

    if output_path:
        Image.fromarray((npimg * 255).astype(np.uint8)).save(output_path)
        print(f"Saved sample grid to {output_path}")
    else:
        plt.show()


# 示例用法
if __name__ == "__main__":
    npz_file = "samples/sample_output.npz"
    visualize_npz(npz_file, output_path="sample_grid.png", num_samples=16)
