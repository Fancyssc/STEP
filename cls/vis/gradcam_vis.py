from torchvision import transforms, datasets
from timm.data.transforms_factory import create_transform
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import torch
import numpy as np


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


from ..models.static.spikformer_cifar import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt_path = "/home/shensicheng/log/SpikingTransformerBenchmark/cls/Spikformer/spikformer_cifar--cifar10--LIFNode--thres_1.0--tau_2.0--seed_42--epoch_400--20250325-103608/model_best.pth.tar"
model = spikformer_cifar(pretrained=False)
model.load_state_dict(
    torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"],
    strict=False,
)
model = model.to(device)
model.eval()

# 选择 MLP 层中添加的 Identity 层，假设在第一个 Block 的 MLP 中
target_layer = model.block[-1].mlp.id

# 加载 CIFAR10 数据集中的一张图片（测试集）
transform = create_transform(
    input_size=32,
    is_training=False,
    mean=[0.5071, 0.4867, 0.4408],
    std=[0.2675, 0.2565, 0.2761],
    crop_pct=1.0,
    interpolation="bicubic",
    color_jitter=0.0,
    auto_augment="rand-m9-n1-mstd0.4-inc1",
)
dataset = datasets.CIFAR10(
    root="/data/datasets/CIFAR10/", train=False, download=True, transform=transform
)
img_tensor, label = dataset[1]
input_img = img_tensor.unsqueeze(0).to(device)

output = model(input_img)
pred_class = output.argmax(dim=1)

# 使用 pytorch-grad-cam 库的 GradCAM++ 计算热图
cam_algorithm = GradCAMPlusPlus(
    model=model, target_layers=[target_layer], reshape_transform=reshape_transform
)
# grayscale_cam = cam_algorithm(input_tensor=input_img, targets=[ClassifierOutputTarget(pred_class.item())])[0]
grayscale_cam = cam_algorithm(
    input_tensor=input_img, targets=[ClassifierOutputTarget(pred_class.item())]
)
print(grayscale_cam.shape)

# # 反归一化原图
# mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(3, 1, 1)
# std = torch.tensor([0.2675, 0.2565, 0.2761]).view(3, 1, 1)
# img_denorm = img_tensor.clone().detach().cpu() * std + mean
# img_np = img_denorm.permute(1, 2, 0).numpy()
# img_np = np.clip(img_np, 0, 1)

# plt.figure(figsize=(12, 6))

# cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# # 原图
# plt.subplot(1, 2, 1)
# plt.imshow(img_np)
# plt.title("Original Image")
# plt.axis('off')

# # GradCam++ 热图叠加
# # plt.subplot(1, 2, 2)
# # plt.imshow(img_np)
# # plt.imshow(grayscale_cam, cmap='turbo', alpha=0.4)
# # plt.title(f"GradCam++ Heatmap (Predicted class: {cifar10_classes[pred_class.item()]})")
# # plt.axis('off')

# # GradCam++ 热图叠加（使用官方 show_cam_on_image 方法）
# cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True, image_weight=0.6)
# plt.subplot(1, 2, 2)
# plt.imshow(cam_image)
# plt.title(f"GradCam++ Heatmap (Predicted class: {cifar10_classes[pred_class.item()]})")
# plt.axis('off')

# plt.tight_layout()
# plt.savefig("/home/shensicheng/log/SpikingTransformerBenchmark/vis/Spikformer/test1.png", dpi=300, bbox_inches='tight')
# plt.show()
