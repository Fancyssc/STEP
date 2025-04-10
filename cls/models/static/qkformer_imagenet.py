# from visualizer import get_local
import torch
import torch.nn as nn

# from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode
# from spikingjelly.clock_driven import layer
from braincog.model_zoo.base_module import BaseModule
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from functools import partial
from ..utils.node import *
from braincog.base.strategy.surrogate import *

__all__ = [
    "QKFormer_10_512",
]


def compute_non_zero_rate(x):
    x_shape = torch.tensor(list(x.shape))
    all_neural = torch.prod(x_shape)
    z = torch.nonzero(x)
    print("After attention proj the none zero rate is", z.shape[0] / all_neural)


class MLP(BaseModule):
    def __init__(
        self,
        in_features,
        step=4,
        encode_type="direct",
        mlp_ratio=4.0,
        out_features=None,
        mlp_drop=0.0,
        node=LIFNode,
        tau=2.0,
        act_func=SigmoidGrad,
        threshold=1.0,
        alpha=4.0,
        layer_by_layer=True,
    ):
        super().__init__(step=step, encode_type=encode_type)

        self.in_features = in_features
        self.mlp_ratio = mlp_ratio
        self.out_features = out_features or in_features
        self.hidden_features = int(self.in_features * self.mlp_ratio)

        self.fc_conv1 = nn.Conv2d(
            in_features, self.hidden_features, kernel_size=1, stride=1
        )
        self.fc_bn1 = nn.BatchNorm2d(self.hidden_features)
        self.fc_lif1 = node(
            step=step,
            tau=tau,
            act_func=act_func(alpha=alpha),
            threshold=threshold,
            layer_by_layer=layer_by_layer,
            mem_detach=False,
        )

        self.fc_conv2 = nn.Conv2d(
            self.hidden_features, self.out_features, kernel_size=1, stride=1
        )
        self.fc_bn2 = nn.BatchNorm2d(self.out_features)
        self.fc_lif2 = node(
            step=step,
            tau=tau,
            act_func=act_func(alpha=alpha),
            threshold=threshold,
            layer_by_layer=layer_by_layer,
            mem_detach=False,
        )

    def forward(self, x):
        self.reset()

        TB, C, H, W = x.shape

        x = self.fc_conv1(x)
        x = self.fc_bn1(x)
        x = self.fc_lif1(x)

        x = self.fc_conv2(x)
        x = self.fc_bn2(x)
        x = self.fc_lif2(x)

        return x  # TB, C, H, W


class Token_QK_Attention(BaseModule):
    def __init__(
        self,
        embed_dim,
        step=4,
        encode_type="direct",
        num_heads=12,
        attn_drop=0.0,
        node=LIFNode,
        tau=2.0,
        threshold=1.0,
        act_func=SigmoidGrad,
        alpha=4.0,
        layer_by_layer=True,
    ):
        super().__init__(
            step=step, encode_type=encode_type, layer_by_layer=layer_by_layer
        )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.q_conv = nn.Conv1d(
            embed_dim, embed_dim, kernel_size=1, stride=1, bias=False
        )
        self.q_bn = nn.BatchNorm1d(embed_dim)
        self.q_lif = node(
            step=step,
            tau=tau,
            act_func=act_func(alpha=alpha),
            threshold=threshold,
            layer_by_layer=layer_by_layer,
            mem_detach=False,
        )

        self.k_conv = nn.Conv1d(
            embed_dim, embed_dim, kernel_size=1, stride=1, bias=False
        )
        self.k_bn = nn.BatchNorm1d(embed_dim)
        self.k_lif = node(
            step=step,
            tau=tau,
            act_func=act_func(alpha=alpha),
            threshold=threshold,
            layer_by_layer=layer_by_layer,
            mem_detach=False,
        )

        self.attn_lif = node(
            step=step,
            tau=tau,
            act_func=act_func(alpha=alpha),
            threshold=0.5,
            layer_by_layer=layer_by_layer,
            mem_detach=False,
        )  # special v_thres

        self.proj_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(embed_dim)
        self.proj_lif = node(
            step=step,
            tau=tau,
            act_func=act_func(alpha=alpha),
            threshold=threshold,
            layer_by_layer=layer_by_layer,
            mem_detach=False,
        )

    def forward(self, x):
        self.reset()

        TB, C, H, W = x.shape
        N = H * W
        x = x.flatten(-2, -1)  # TB C N
        x_for_qkv = x

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out)
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.reshape(TB, self.num_heads, C // self.num_heads, N).contiguous()

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out)
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.reshape(TB, self.num_heads, C // self.num_heads, N).contiguous()

        q = torch.sum(q, dim=2, keepdim=True)
        attn = self.attn_lif(q)
        x = torch.mul(attn, k)

        x = x.flatten(1, 2)
        x = self.proj_bn(self.proj_conv(x))
        x = self.proj_lif(x).reshape(TB, C, H, W)

        return x  # TB, C, H, W


class SSA(BaseModule):
    def __init__(
        self,
        embed_dim,
        step=4,
        encode_type="direct",
        num_heads=8,
        scale=0.125,
        attn_drop=0.0,
        node=LIFNode,
        tau=2.0,
        act_func=SigmoidGrad,
        threshold=1.0,
        alpha=4.0,
        layer_by_layer=True,
    ):
        super().__init__(
            step=step, encode_type=encode_type, layer_by_layer=layer_by_layer
        )
        self.num_heads = num_heads
        self.scale = scale
        self.q_conv = nn.Conv1d(
            embed_dim, embed_dim, kernel_size=1, stride=1, bias=False
        )
        self.q_bn = nn.BatchNorm1d(embed_dim)
        self.q_lif = node(
            step=step,
            tau=tau,
            act_func=act_func(alpha=alpha),
            threshold=threshold,
            layer_by_layer=layer_by_layer,
            mem_detach=False,
        )

        self.k_conv = nn.Conv1d(
            embed_dim, embed_dim, kernel_size=1, stride=1, bias=False
        )
        self.k_bn = nn.BatchNorm1d(embed_dim)
        self.k_lif = node(
            step=step,
            tau=tau,
            act_func=act_func(alpha=alpha),
            threshold=threshold,
            layer_by_layer=layer_by_layer,
            mem_detach=False,
        )

        self.v_conv = nn.Conv1d(
            embed_dim, embed_dim, kernel_size=1, stride=1, bias=False
        )
        self.v_bn = nn.BatchNorm1d(embed_dim)
        self.v_lif = node(
            step=step,
            tau=tau,
            act_func=act_func(alpha=alpha),
            threshold=threshold,
            layer_by_layer=layer_by_layer,
            mem_detach=False,
        )

        self.attn_lif = node(
            step=step,
            tau=tau,
            act_func=act_func(alpha=alpha),
            threshold=0.5,
            layer_by_layer=layer_by_layer,
            mem_detach=False,
        )  # special v_thres

        self.proj_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(embed_dim)
        self.proj_lif = node(
            step=step,
            tau=tau,
            act_func=act_func(alpha=alpha),
            threshold=threshold,
            layer_by_layer=layer_by_layer,
            mem_detach=False,
        )

        self.qkv_mp = nn.MaxPool1d(4)

    def forward(self, x):
        self.reset()

        TB, C, H, W = x.shape
        N = H * W
        x = x.flatten(-2, -1)  # TB C N

        x_for_qkv = x

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out)
        q_conv_out = self.q_lif(q_conv_out)
        q = (
            q_conv_out.transpose(-1, -2)
            .reshape(TB, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out)
        k_conv_out = self.k_lif(k_conv_out)  # TB C N
        k = (
            k_conv_out.transpose(-1, -2)
            .reshape(TB, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out)
        v_conv_out = self.v_lif(v_conv_out)
        v = (
            v_conv_out.transpose(-1, -2)
            .reshape(TB, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        x = k.transpose(-2, -1) @ v
        x = (q @ x) * self.scale

        x = x.transpose(2, 3)  # TB H C//H N
        x = self.attn_lif(x).reshape(TB, C, N).contiguous()
        # x = x.flatten(0, 1)
        x = self.proj_lif(self.proj_bn(self.proj_conv(x))).reshape(TB, C, H, W)

        return x  # TB, C, H, W


class TokenSpikingTransformer(nn.Module):
    def __init__(
        self,
        embed_dim,
        step=4,
        encode_type="direct",
        num_heads=12,
        attn_drop=0.0,
        mlp_ratio=4.0,
        node=LIFNode,
        tau=2.0,
        act_func=SigmoidGrad,
        threshold=1.0,
        alpha=4.0,
        layer_by_layer=True,
    ):
        super().__init__()

        self.tssa = Token_QK_Attention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            step=step,
            encode_type=encode_type,
            node=node,
            tau=tau,
            act_func=act_func,
            threshold=threshold,
            alpha=alpha,
            layer_by_layer=layer_by_layer,
        )
        self.mlp = MLP(
            in_features=embed_dim,
            step=step,
            encode_type=encode_type,
            node=node,
            tau=tau,
            act_func=act_func,
            threshold=threshold,
            mlp_ratio=mlp_ratio,
            alpha=alpha,
            layer_by_layer=layer_by_layer,
        )

    def forward(self, x):

        x = x + self.tssa(x)
        x = x + self.mlp(x)

        return x


class SpikingTransformer(nn.Module):
    def __init__(
        self,
        embed_dim,
        step=4,
        encode_type="direct",
        num_heads=12,
        attn_drop=0.0,
        mlp_ratio=4.0,
        scale=0.125,
        node=LIFNode,
        tau=2.0,
        act_func=SigmoidGrad,
        threshold=1.0,
        alpha=4.0,
        layer_by_layer=True,
    ):
        super().__init__()
        self.ssa = SSA(
            embed_dim=embed_dim,
            num_heads=num_heads,
            scale=scale,
            step=step,
            encode_type=encode_type,
            node=node,
            tau=tau,
            act_func=act_func,
            threshold=threshold,
            alpha=alpha,
            layer_by_layer=layer_by_layer,
        )
        self.mlp = MLP(
            in_features=embed_dim,
            step=step,
            encode_type=encode_type,
            node=node,
            tau=tau,
            act_func=act_func,
            threshold=threshold,
            mlp_ratio=mlp_ratio,
            alpha=alpha,
            layer_by_layer=layer_by_layer,
        )

    def forward(self, x):

        x = x + self.ssa(x)
        x = x + self.mlp(x)

        return x


class PatchInit(BaseModule):
    def __init__(
        self,
        step=4,
        encode_type="direct",
        img_h=128,
        img_w=128,
        patch_size=4,
        in_channels=2,
        embed_dim=256,
        node=LIFNode,
        tau=2.0,
        act_func=SigmoidGrad,
        threshold=1.0,
        alpha=4.0,
        layer_by_layer=True,
    ):
        super().__init__(encode_type=encode_type, step=step)

        self.img_h = img_h
        self.img_w = img_w
        self.patch_size = patch_size
        self.patch_nums = self.img_h // self.patch_size * self.img_w // self.patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj_conv = nn.Conv2d(
            in_channels, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj_bn = nn.BatchNorm2d(embed_dim // 2)
        self.proj_lif = node(
            step=step,
            tau=tau,
            act_func=act_func(alpha=alpha),
            threshold=threshold,
            layer_by_layer=layer_by_layer,
            mem_detach=False,
        )

        self.proj1_conv = nn.Conv2d(
            embed_dim // 2,
            embed_dim // 1,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.proj1_bn = nn.BatchNorm2d(embed_dim // 1)
        self.proj1_lif = node(
            step=step,
            tau=tau,
            act_func=act_func(alpha=alpha),
            threshold=threshold,
            layer_by_layer=layer_by_layer,
            mem_detach=False,
        )

        self.proj2_conv = nn.Conv2d(
            embed_dim // 1,
            embed_dim // 1,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.proj2_bn = nn.BatchNorm2d(embed_dim // 1)
        self.proj2_lif = node(
            step=step,
            tau=tau,
            act_func=act_func(alpha=alpha),
            threshold=threshold,
            layer_by_layer=layer_by_layer,
            mem_detach=False,
        )

        self.proj_res_conv = nn.Conv2d(
            embed_dim // 2,
            embed_dim // 1,
            kernel_size=1,
            stride=2,
            padding=0,
            bias=False,
        )
        self.proj_res_bn = nn.BatchNorm2d(embed_dim)
        self.proj_res_lif = node(
            step=step,
            tau=tau,
            act_func=act_func(alpha=alpha),
            threshold=threshold,
            layer_by_layer=layer_by_layer,
            mem_detach=False,
        )

    def forward(self, x):
        self.reset()
        TB, C, H, W = x.shape

        x = self.proj_conv(x)
        x = self.proj_bn(x)
        x = self.proj_lif(x)

        x_feat = x
        x = self.proj1_conv(x)
        x = self.proj1_bn(x)
        x = self.proj1_lif(x)

        x = self.proj2_conv(x)
        x = self.proj2_bn(x)
        x = self.proj2_lif(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat)
        x_feat = self.proj_res_lif(x_feat)

        return x + x_feat  # TB _ _ _


class PatchEmbedding(BaseModule):
    def __init__(
        self,
        step=4,
        encode_type="direct",
        img_h=128,
        img_w=128,
        patch_size=4,
        in_channels=2,
        embed_dim=256,
        node=LIFNode,
        tau=2.0,
        act_func=SigmoidGrad,
        threshold=1.0,
        alpha=4.0,
        layer_by_layer=True,
    ):

        super().__init__(encode_type=encode_type, step=step)

        self.img_h = img_h
        self.img_w = img_w
        self.patch_size = patch_size
        self.patch_nums = self.img_h // self.patch_size * self.img_w // self.patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj3_conv = nn.Conv2d(
            embed_dim // 2, embed_dim, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj3_bn = nn.BatchNorm2d(embed_dim)
        self.proj3_lif = node(
            step=step,
            tau=tau,
            act_func=act_func(alpha=alpha),
            threshold=threshold,
            layer_by_layer=layer_by_layer,
            mem_detach=False,
        )

        self.proj4_conv = nn.Conv2d(
            embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj4_bn = nn.BatchNorm2d(embed_dim)
        self.proj4_maxpool = torch.nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )
        self.proj4_lif = node(
            step=step,
            tau=tau,
            act_func=act_func(alpha=alpha),
            threshold=threshold,
            layer_by_layer=layer_by_layer,
            mem_detach=False,
        )

        self.proj_res_conv = nn.Conv2d(
            embed_dim // 2, embed_dim, kernel_size=1, stride=2, padding=0, bias=False
        )
        self.proj_res_bn = nn.BatchNorm2d(embed_dim)
        self.proj_res_lif = node(
            step=step,
            tau=tau,
            act_func=act_func(alpha=alpha),
            threshold=threshold,
            layer_by_layer=layer_by_layer,
            mem_detach=False,
        )

    def forward(self, x):
        self.reset()
        TB, C, H, W = x.shape
        # Downsampling + Res

        x_feat = x

        x = self.proj3_conv(x)
        x = self.proj3_bn(x)
        x = self.proj3_lif(x)

        x = self.proj4_conv(x)
        x = self.proj4_bn(x)
        x = self.proj4_maxpool(x)
        x = self.proj4_lif(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat)
        x_feat = self.proj_res_lif(x_feat)

        x = x + x_feat  # shortcut

        return x


class hierarchical_spiking_transformer(BaseModule):
    def __init__(
        self,
        step=4,
        img_size_h=128,
        img_size_w=128,
        patch_size=16,
        in_channels=2,
        num_classes=11,
        embed_dims=[64, 128, 256],
        num_heads=[1, 2, 4],
        mlp_ratios=[4, 4, 4],
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=[6, 8, 6],
        sr_ratios=[8, 4, 2],
        node=LIFNode,
        tau=2.0,
        act_func=SigmoidGrad,
        threshold=1.0,
        alpha=4.0,
        layer_by_layer=True,
    ):
        super().__init__(encode_type="direct", step=step, layer_by_layer=layer_by_layer)
        self.num_classes = num_classes
        self.depths = depths
        self.T = step
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depths)
        ]  # stochastic depth decay rule

        patch_embed1 = PatchInit(
            img_h=img_size_h,
            img_w=img_size_w,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dims // 4,
            step=step,
            node=node,
            tau=tau,
            act_func=act_func,
            threshold=threshold,
            alpha=alpha,
            layer_by_layer=layer_by_layer,
        )

        stage1 = nn.ModuleList(
            [
                TokenSpikingTransformer(
                    embed_dim=embed_dims // 4,
                    step=step,
                    num_heads=num_heads,
                    attn_drop=attn_drop_rate,
                    mlp_ratio=mlp_ratios,
                    node=node,
                    tau=tau,
                    act_func=act_func,
                    threshold=threshold,
                    alpha=alpha,
                    layer_by_layer=layer_by_layer,
                )
                for j in range(1)
            ]
        )

        patch_embed2 = PatchEmbedding(
            img_h=img_size_h,
            img_w=img_size_w,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dims // 2,
        )

        stage2 = nn.ModuleList(
            [
                TokenSpikingTransformer(
                    embed_dim=embed_dims // 2,
                    step=step,
                    num_heads=num_heads,
                    attn_drop=attn_drop_rate,
                    mlp_ratio=mlp_ratios,
                    node=node,
                    tau=tau,
                    act_func=act_func,
                    threshold=threshold,
                    alpha=alpha,
                    layer_by_layer=layer_by_layer,
                )
                for j in range(1)
            ]
        )

        patch_embed3 = PatchEmbedding(
            img_h=img_size_h,
            img_w=img_size_w,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dims,
        )

        stage3 = nn.ModuleList(
            [
                SpikingTransformer(
                    embed_dim=embed_dims,
                    step=step,
                    num_heads=num_heads,
                    attn_drop=0.0,
                    mlp_ratio=mlp_ratios,
                    scale=0.125,  # what is scale ??
                    node=node,
                    tau=tau,
                    act_func=act_func,
                    threshold=threshold,
                    alpha=alpha,
                    layer_by_layer=layer_by_layer,
                )
                for j in range(self.depths - 3)
            ]
        )

        setattr(self, f"patch_embed1", patch_embed1)
        setattr(self, f"patch_embed2", patch_embed2)
        setattr(self, f"patch_embed3", patch_embed3)
        setattr(self, f"stage1", stage1)
        setattr(self, f"stage2", stage2)
        setattr(self, f"stage3", stage3)

        # classification head 这里不需要脉冲，因为输入的是在T时长平均发射值
        self.head = (
            nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        )
        self.apply(self._init_weights)

    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed3, H, W):
        if H * W == self.patch_embed3.num_patches:
            return pos_embed
        else:
            return (
                F.interpolate(
                    pos_embed.reshape(1, patch_embed3.H, patch_embed3.W, -1).permute(
                        0, 3, 1, 2
                    ),
                    size=(H, W),
                    mode="bilinear",
                )
                .reshape(1, -1, H * W)
                .permute(0, 2, 1)
            )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):

        stage1 = getattr(self, f"stage1")
        stage2 = getattr(self, f"stage2")
        stage3 = getattr(self, f"stage3")
        patch_embed1 = getattr(self, f"patch_embed1")
        patch_embed2 = getattr(self, f"patch_embed2")
        patch_embed3 = getattr(self, f"patch_embed3")

        x = patch_embed1(x)
        for blk in stage1:
            x = blk(x)

        x = patch_embed2(x)
        for blk in stage2:
            x = blk(x)

        x = patch_embed3(x)
        for blk in stage3:
            x = blk(x)

        _, C, H, W = x.shape
        return x.flatten(-2, -1).mean(-1).reshape(self.step, -1, C).contiguous()

    def forward(self, x):
        # T = self.T
        # x = (x.unsqueeze(0)).repeat(T, 1, 1, 1, 1)
        x = self.encoder(x)
        x = self.forward_features(x)
        x = self.head(x.mean(0))
        return x


@register_model
def qkformer_imagenet(step=1, **kwargs):
    model = hierarchical_spiking_transformer(
        step=kwargs.get("step", 4),
        img_size_h=kwargs.get("img_size", 224),
        img_size_w=kwargs.get("img_size", 224),
        patch_size=kwargs.get("patch_size", 4),
        embed_dims=kwargs.get("embed_dim", 384),
        num_heads=kwargs.get("num_heads", 8),
        mlp_ratios=kwargs.get("mlp_ratio", 4),
        in_channels=kwargs.get("in_channels", 3),
        num_classes=kwargs.get("num_classes", 1000),
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=kwargs.get("depth", 10),
        sr_ratios=1,
        tau=kwargs.get("tau", 2.0),
        threshold=kwargs.get("threshold", 1.0),
        node=kwargs.get("node", LIFNode),
        act_func=kwargs.get("act_func", SigmoidGrad),
        alpha=kwargs.get("alpha", 4.0),
    )
    model.default_cfg = _cfg()
    return model
