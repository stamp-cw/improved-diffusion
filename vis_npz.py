import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision.utils import make_grid
import torch
from PIL import Image


def visualize_npz(npz_path, output_path=None, num_samples=16, expected_channels=3):
    """
    可视化 .npz 文件中的图像样本，自动适应不同尺寸的输入。

    参数：
    - npz_path: .npz 文件路径
    - output_path: 可选，保存网格图像的路径
    - num_samples: 显示/保存的样本数量
    - expected_channels: 期望的通道数 (3 for RGB, 1 for grayscale)

    返回：
    - 生成的图像网格 (PIL.Image 对象)
    """
    try:
        # 加载数据
        data = np.load(npz_path)
        if "arr_0" not in data:
            available_keys = list(data.keys())
            if len(available_keys) == 1:  # 如果只有一个key，不管名称是什么都使用
                samples = data[available_keys[0]]
            else:
                raise KeyError(f"No 'arr_0' found in npz file. Available keys: {available_keys}")
        else:
            samples = data["arr_0"]

        print(f"Loaded {samples.shape[0]} samples with shape {samples.shape[1:]}")

        # 自动检测并修正形状
        if len(samples.shape) == 3:  # (N, H, W)
            if expected_channels == 3:
                samples = np.repeat(samples[:, np.newaxis, :, :], 3, axis=1)  # (N, 3, H, W)
            else:
                samples = samples[:, np.newaxis, :, :]  # (N, 1, H, W)
        elif len(samples.shape) == 4:
            if samples.shape[3] in [1, 3]:  # (N, H, W, C)
                samples = np.transpose(samples, (0, 3, 1, 2))  # 转为 (N, C, H, W)

        # 确保通道数正确
        if samples.shape[1] == 1 and expected_channels == 3:
            samples = np.repeat(samples, 3, axis=1)
        elif samples.shape[1] == 3 and expected_channels == 1:
            samples = samples[:, 0:1, :, :]  # 取第一个通道

        # 选择样本并转换
        selected = samples[:num_samples]
        tensor = torch.from_numpy(selected).float()

        # 智能归一化
        if tensor.max() > 1.0 and tensor.dtype != torch.uint8:
            tensor = tensor / 255.0
        elif tensor.max() <= 1.0 and tensor.min() >= 0:
            pass  # 已经是[0,1]范围
        else:
            tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())  # 线性归一化到[0,1]

        # 自动计算最佳网格布局
        nrow = min(8, int(np.sqrt(num_samples)))  # 每行最多8张图

        # 创建图像网格
        grid = make_grid(
            tensor,
            nrow=nrow,
            normalize=False,
            padding=2,
            pad_value=1.0  # 用白色填充
        )

        # 转换为适合显示的格式 (H, W, C)
        npimg = grid.permute(1, 2, 0).numpy()

        # 确保输出在[0,1]范围内
        npimg = np.clip(npimg, 0, 1)

        # 创建图像
        fig = plt.figure(figsize=(12, 12))
        plt.imshow(npimg)
        plt.axis('off')
        plt.title(f"First {len(selected)} samples from {os.path.basename(npz_path)}")
        plt.tight_layout()

        if output_path:
            # 修复路径处理问题
            try:
                output_dir = os.path.dirname(output_path)
                if output_dir:  # 如果指定了目录
                    os.makedirs(output_dir, exist_ok=True)
                Image.fromarray((npimg * 255).astype(np.uint8)).save(output_path)
                print(f"Saved sample grid to {os.path.abspath(output_path)}")
            except Exception as save_error:
                print(f"Warning: Could not save to {output_path}. Error: {save_error}")
                # 尝试保存到当前目录
                fallback_path = os.path.basename(output_path)
                Image.fromarray((npimg * 255).astype(np.uint8)).save(fallback_path)
                print(f"Saved sample grid to {os.path.abspath(fallback_path)} instead")
            plt.close(fig)
            return Image.fromarray((npimg * 255).astype(np.uint8))
        else:
            plt.show()
            return Image.fromarray((npimg * 255).astype(np.uint8))

    except Exception as e:
        print(f"Error processing {npz_path}: {str(e)}")
        raise


if __name__ == "__main__":
    # 示例用法 - 修改为更简单的输出路径
    # npz_file = r"D:\Project\improved-diffusion\samples\samples_64x32x32x3.npz"
    npz_file = r"D:\Project\improved-diffusion\samples\samples_64x32x32x3 (1).npz"
    try:
        result_img = visualize_npz(
            npz_file,
            output_path="sample_grid_output.png",  # 使用当前目录下的简单文件名
            num_samples=64,  # 显示所有64个样本
            expected_channels=3  # 期望RGB图像
        )
    except Exception as e:
        print(f"Visualization failed: {e}")