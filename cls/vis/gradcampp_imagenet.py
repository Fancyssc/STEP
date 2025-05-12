import os
import sys

sys.path.append(os.getcwd())

import argparse

from torchvision import datasets
from timm.data import create_dataset
from timm.data.transforms_factory import create_transform
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import torch
import numpy as np
from omegaconf import OmegaConf


def reshape_transform(tensor):
    """将形状为 (N, tokens, channels) 的 tensor 转换为 (N, channels, H, W)，
    如果存在分类 token，则去除第一个 token。"""
    if tensor.ndim == 3:
        b, n, c = tensor.size()
        # 如果存在分类 token（通常为第一个 token），则检查剩余 tokens 是否能构成正方形
        if (n - 1) > 0 and ((n - 1) ** 0.5) % 1 == 0:
            tensor = tensor[:, 1:, :]
            n = n - 1
        size = int(n**0.5)
        return tensor.transpose(1, 2).reshape(b, c, size, size)
    return tensor


def load_model(model_config):
    if model_config.name == "qkformer":
        from cls.models.static.qkformer_imagenet import qkformer_imagenet

        model = qkformer_imagenet()
        tgt_str = "model.stage3[-1].mlp.id"
    elif model_config.name == "spikformer":
        from cls.models.static.spikformer_imagenet import spikformer_imagenet

        model = spikformer_imagenet()
        tgt_str = None
    else:
        raise ValueError("Invaild model name")

    return model, tgt_str


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = config.ckpt_path

    model, tgt_str = load_model(config.model)
    model.load_state_dict(
        torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"],
        strict=False,
    )
    model = model.to(device)
    model.eval()

    # 选择 MLP 层中添加的 Identity 层，假设在第一个 Block 的 MLP 中
    if tgt_str is None:
        target_layer = model.block[-1].mlp.id
    else:
        target_layer = eval(tgt_str)

    # 加载 Imagenet 数据集中的一张图片（测试集）
    transform = create_transform(
        input_size=224,
        is_training=False,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        crop_pct=1.0,
        interpolation="bicubic",
        color_jitter=0.0,
        auto_augment="rand-m9-n1-mstd0.4-inc1",
    )
    dataset = create_dataset(
        "imagenet", root=config.data_path, is_training=False, transform=transform
    )
    # dataset = datasets.ImageNet(
    #     root=config.data_path, split='train', transform=transform
    # )

    imagenet_classes = config.imagenet_classes
    # 使用 pytorch-grad-cam 库的 GradCAM++ 计算热图
    cam_algorithm = GradCAMPlusPlus(
        model=model,
        target_layers=[target_layer],
        reshape_transform=reshape_transform,
    )
    num_samples = 100
    for i in range(num_samples):
        img_tensor, label = dataset[i]
        input_img = img_tensor.unsqueeze(0).to(device)

        output = model(input_img)
        pred_class = output.argmax(dim=1)

        grayscale_cam = cam_algorithm(
            input_tensor=input_img, targets=[ClassifierOutputTarget(pred_class.item())]
        )[0]

        # 反归一化原图
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_denorm = img_tensor.clone().detach().cpu() * std + mean
        img_np = img_denorm.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)

        plt.figure(figsize=(12, 6))

        # 原图
        plt.subplot(1, 2, 1)
        plt.imshow(img_np)
        plt.title("Original Image")
        plt.axis("off")

        # GradCam++ 热图叠加
        # plt.subplot(1, 2, 2)
        # plt.imshow(img_np)
        # plt.imshow(grayscale_cam, cmap='turbo', alpha=0.4)
        # plt.title(f"GradCam++ Heatmap (Predicted class: {cifar10_classes[pred_class.item()]})")
        # plt.axis('off')

        # GradCam++ 热图叠加（使用官方 show_cam_on_image 方法）
        cam_image = show_cam_on_image(
            img_np, grayscale_cam, use_rgb=True, image_weight=0.6
        )
        plt.subplot(1, 2, 2)
        plt.imshow(cam_image)
        plt.title(
            f"GradCam++ Heatmap (Predicted class: {imagenet_classes[pred_class.item()]})"
        )
        plt.axis("off")

        plt.tight_layout()
        save_path = os.path.join(config.save_path, f"{i}.png")
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Generated {save_path}")
        # plt.show()
