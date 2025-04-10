import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import os
import sys
from tqdm import tqdm

# 假设你已有训练好的模型或导入模型定义
from braincog.model_zoo.base_module import BaseModule
from timm.models import register_model
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import _cfg
from braincog.base.strategy.surrogate import *
import torch.nn as nn

from ..models.static.spikformer_cifar import *
# 这里添加你的Spikformer模型定义（如果需要）
# 或者导入已有的模型文件

# 加载预训练模型
def load_model(model_path, num_classes=10):
    # 根据你的模型参数进行调整
    model = spikformer_cifar()

    import argparse
    import torch.serialization

    # Add argparse.Namespace to safe globals
    torch.serialization.add_safe_globals([argparse.Namespace])

    try:
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu',
                                weights_only=False)

        # Extract model state dict - looking at your error, the model weights are in 'state_dict'
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)

        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")

    model.eval()
    return model


# 提取attention
class AttentionGrabber:
    def __init__(self, model):
        self.model = model
        self.attention_maps = []
        self.handles = []
        self.setup_hooks() # init时会自动执行 建立hook

    def setup_hooks(self):
        # hook
        for i, block in enumerate(self.model.block): # spikformer中的block 和 attn
            def get_attn(name):
                def hook(module, input, output):

                    # hook是调用了
                    TB, N, C = input[0].shape
                    x_for_qkv = input[0]

                    q = module.q_lif(module.q_bn(module.q_linear(x_for_qkv).transpose(-1, -2)).transpose(-1, -2))
                    q = q.reshape(-1, N, module.num_heads, C // module.num_heads).permute(0, 2, 1, 3)

                    k = module.k_lif(module.k_bn(module.k_linear(x_for_qkv).transpose(-1, -2)).transpose(-1, -2))
                    k = k.reshape(-1, N, module.num_heads, C // module.num_heads).permute(0, 2, 1, 3)

                    attn = (q @ k.transpose(-2, -1)) * module.scale

                    # 提取注意力权重
                    self.attention_maps.append({
                        'layer': name,
                        'map': attn.detach().clone()
                    })

                return hook

            handle = block.attn.register_forward_hook(get_attn(f'block_{i}')) # 为所有的attn创建一个hook
            self.handles.append(handle)

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def clear_maps(self):
        self.attention_maps = []

    def __del__(self):
        self.remove_hooks()


# 加载和预处理图像
def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image)

    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])

    transformed_image = transform(image).unsqueeze(0)
    return transformed_image, original_image


# 提取和处理注意力图
def process_attention_maps(attention_maps, layer_idx=3, head_idx=None, time_step=None):
    """
    处理注意力图
    layer_idx: 要可视化的层索引
    head_idx: 要可视化的注意力头索引，None为平均所有头
    time_step: 要可视化的时间步，None为平均所有时间步
    """
    # 获取指定层的注意力图
    attn_map = attention_maps[layer_idx]['map']  # [TB, heads, N, N]

    # 处理时间步和头
    # 对于Spikformer，TB = T * B，我们需要重新整形以获取各个时间步
    TB, heads, N, N = attn_map.shape
    T = 4  # 时间步数，应该与模型参数中的step一致
    B = TB // T

    attn_map = attn_map.reshape(T, B, heads, N, N)

    # 默认平均所有heads
    if time_step is not None:
        attn_map = attn_map[time_step]
    else:
        attn_map = attn_map.mean(0)  # 平均所有时间步

    # 选择特定的头或平均所有头
    if head_idx is not None:
        attn_map = attn_map[:, head_idx]
    else:
        attn_map = attn_map.mean(1)  # 平均所有头

    # [B, N, N]
    attn_map = attn_map[0]

    patch_size = 4
    num_patches = 32 // patch_size

    # resize
    # 这里假设注意力图的形状已经是 [N, N]，其中N是补丁数量
    if attn_map.shape[0] != num_patches ** 2:
        # 如果形状不匹配，可能需要重新整形或插值
        # 这取决于你的模型是如何处理patch的
        pass

    print(attn_map.shape) # N = 64
    # 上采样到原始图像大小
    attn_map = torch.nn.functional.interpolate(
        attn_map.unsqueeze(0).unsqueeze(0),
        size=(32, 32),
        mode='bilinear'
    ).squeeze()

    # 转换为numpy并标准化
    attn_map = attn_map.cpu().numpy()
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    return attn_map



# 每个类创建一个
def visualize_multiple_images(model, data_dir='./data', output_dir='./attention_maps', num_images=10):

    os.makedirs(output_dir, exist_ok=True)

    # load数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # CIFAR-10 类别
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    grabber = AttentionGrabber(model) # 建立hook 需要等待一次forward才会触发

    # 每个类都选一个image
    for class_idx in range(10):
        for i, (inputs, labels) in enumerate(testloader):
            if labels[0] == class_idx:
                inputs = inputs.to(device)

                # forward一次 触发hook
                with torch.no_grad():
                    _ = model(inputs)

                # 处理注意力图
                attn_map = process_attention_maps(grabber.attention_maps)

                # 可视化
                img = inputs[0].cpu().permute(1, 2, 0).numpy()
                img = img * np.array([0.2470, 0.2435, 0.2616]) + np.array([0.4914, 0.4822, 0.4465])
                img = np.clip(img, 0, 1)

                # 创建可视化
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))

                # 原始
                axs[0].imshow(img)
                axs[0].set_title(f'Input Image: {classes[class_idx]}')
                axs[0].axis('off')

                # attn_map
                im = axs[1].imshow(attn_map, cmap='jet')
                axs[1].set_title('Attention Map')
                axs[1].axis('off')
                plt.colorbar(im, ax=axs[1])

                # overlay
                axs[2].imshow(img)
                im_overlay = axs[2].imshow(attn_map, cmap='jet', alpha=0.4)
                axs[2].set_title('Overlay')
                axs[2].axis('off')
                plt.colorbar(im_overlay, ax=axs[2])

                plt.tight_layout()


                output_path = os.path.join(output_dir, f'attention_class_{classes[class_idx]}.png')
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()

                # 清理hook
                grabber.clear_maps()
                break

    # 清理
    grabber.remove_hooks()


# 主函数示例
def main():
    # 模型路径
    model_path = '/home/shensicheng/log/SpikingTransformerBenchmark/cls/Spikformer/spikformer_cifar--cifar10--LIFNode--thres_1.0--tau_2.0--seed_42--epoch_400--20250325-103608/model_best.pth.tar'  # 替换为你的模型路径

    # 加载模型
    model = load_model(model_path)

    # 单个图像可视化
    # image_path = 'path/to/your/image.jpg'  # 替换为你的图像路径
    # output_path = 'attention_visualization.png'
    # visualize_attention(image_path, model, output_path)

    # 多个CIFAR-10图像的可视化
    visualize_multiple_images(model)

# 可视化注意力图
# def visualize_attention(image_path, model, output_path=None, layer_idx=0, head_idx=None, time_step=None):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#
#     # 加载图像
#     image_tensor, original_image = load_image(image_path)
#     image_tensor = image_tensor.to(device)
#
#     # 提取注意力图
#     grabber = AttentionGrabber(model)
#     with torch.no_grad():
#         _ = model(image_tensor)
#
#     # 处理注意力图
#     attn_map = process_attention_maps(grabber.attention_maps, layer_idx, head_idx, time_step)
#
#     # 调整原始图像大小以匹配处理后的图像
#     original_resized = np.array(Image.fromarray(original_image).resize((32, 32)))
#
#     # 创建可视化
#     fig, axs = plt.subplots(1, 3, figsize=(15, 5))
#
#     # 显示原始图像
#     axs[0].imshow(original_resized)
#     axs[0].set_title('Input Image')
#     axs[0].axis('off')
#
#     # 显示注意力图
#     im = axs[1].imshow(attn_map, cmap='jet')
#     axs[1].set_title('Attention Map')
#     axs[1].axis('off')
#     plt.colorbar(im, ax=axs[1])
#
#     # 显示叠加图
#     axs[2].imshow(original_resized)
#     im_overlay = axs[2].imshow(attn_map, cmap='jet', alpha=0.6)
#     axs[2].set_title('Overlay')
#     axs[2].axis('off')
#     plt.colorbar(im_overlay, ax=axs[2])
#
#     plt.tight_layout()
#
#     if output_path:
#         plt.savefig(output_path, dpi=300, bbox_inches='tight')
#         print(f"Saved visualization to {output_path}")
#
#     plt.show()
#
#     # 清理
#     grabber.remove_hooks()
#     grabber.clear_maps()
#
#     return fig


if __name__ == "__main__":
    main()