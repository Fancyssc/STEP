from braincog.model_zoo.base_module import BaseModule
from timm.models import register_model
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import _cfg
# from utils.node import *
from braincog.base.node import LIFNode
from braincog.base.strategy.surrogate import *

import torch.nn as nn

"""
Spikformer version 1 (ICLR2023)

"""
class SPS(BaseModule):
    """
    :param: node: The neuron model used in the Spiking Transformer. The structure of node should obey BaseNode in Braincog
    :param: step: The number of time steps that the neuron will be simulated for.
    :param: encode_type: The encoding type of the input data. 'direct' for direct encoding
    :param: img_h: The height of the input image.
    :param: img_w: The width of the input image.
    :param: patch_size: The size of the patch.
    :param: in_channels: The number of input channels.
    :param: embed_dims: The dimension of the embedding.
    """
    def __init__(self, step=4, encode_type='direct', img_h=32, img_w=32, patch_size=4, in_channels=3,
                 embed_dims=384, node=LIFNode,tau=2.0,threshold=1.0,act_func=SigmoidGrad, alpha=4.0,layer_by_layer=True):
        super().__init__(step=step, encode_type= encode_type,layer_by_layer=layer_by_layer)

        self.img_h = img_h
        self.img_w = img_w
        self.patch_size = patch_size
        self.patch_nums = self.img_h // self.patch_size * self.img_w // self.patch_size
        self.in_channels = in_channels
        self.embed_dims = embed_dims


        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)
        self.proj_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold, layer_by_layer=layer_by_layer, mem_detach=True)

        self.proj_conv1 = nn.Conv2d(embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims // 4)
        self.proj_lif1 = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,  layer_by_layer=layer_by_layer, mem_detach=True)

        self.proj_conv2 = nn.Conv2d(embed_dims // 4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims // 2)
        self.proj_lif2 = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,  layer_by_layer=layer_by_layer, mem_detach=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv3 = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_lif3 = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,  layer_by_layer=layer_by_layer, mem_detach=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,  layer_by_layer=layer_by_layer, mem_detach=True)


    def forward(self, x):
        self.reset()
        TB, C, H, W = x.shape

        x = self.proj_conv(x)
        x = self.proj_bn(x)
        x = self.proj_lif(x) # TB C H W

        x = self.proj_conv1(x)
        x = self.proj_bn1(x)
        x = self.proj_lif1(x)

        x = self.proj_conv2(x)
        x = self.proj_bn2(x)
        x = self.proj_lif2(x)
        x = self.maxpool2(x)

        x = self.proj_conv3(x)
        x = self.proj_bn3(x)
        x = self.proj_lif3(x)
        x = self.maxpool3(x)

        x_feat = x # TB, -1, H // 4, W // 4
        x = self.rpe_conv(x)
        x = self.rpe_bn(x)
        x = self.rpe_lif(x) # TB, -1, H // 4, W // 4
        x = x + x_feat

        x = x.flatten(-2).transpose(-1, -2)  # TB,N,C

        return x # TB,N,C


class SSA(BaseModule):
    def __init__(self,embed_dim, step=4,encode_type='direct',num_heads=12,attn_scale=0.125,attn_drop=0.,node=LIFNode,tau=2.0,threshold=1.0,act_func=SigmoidGrad, alpha=4.0,layer_by_layer=True):
        super().__init__(step=step, encode_type=encode_type,layer_by_layer=layer_by_layer)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scale = attn_scale
        self.T = step
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.q_bn = nn.BatchNorm1d(embed_dim)
        self.q_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,  layer_by_layer=layer_by_layer, mem_detach=True)

        self.k_linear = nn.Linear(embed_dim,embed_dim)
        self.k_bn = nn.BatchNorm1d(embed_dim)
        self.k_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,  layer_by_layer=layer_by_layer, mem_detach=True)

        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.v_bn = nn.BatchNorm1d(embed_dim)
        self.v_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,  layer_by_layer=layer_by_layer, mem_detach=True)

        self.attn_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=0.5, layer_by_layer=True, mem_detach=True) #special v_thres

        self.proj_linear = nn.Linear(embed_dim, embed_dim)
        self.proj_bn = nn.BatchNorm1d(embed_dim)
        self.proj_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,  layer_by_layer=layer_by_layer, mem_detach=True)

    def forward(self, x):
        self.reset()

        TB, N, C = x.shape
        x_for_qkv = x
        q_linear_out = self.q_linear(x_for_qkv)  # [TB, N, C]
        q_linear_out = self.q_bn(q_linear_out.transpose(-1, -2)).transpose(-1, -2)
        q_linear_out = self.q_lif(q_linear_out)
        q = q_linear_out.reshape(-1, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        k_linear_out = self.k_linear(x_for_qkv)
        k_linear_out = self.k_bn(k_linear_out.transpose(-1, -2)).transpose(-1, -2)
        k_linear_out = self.k_lif(k_linear_out)
        k = k_linear_out.reshape(-1, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        v_linear_out = self.v_linear(x_for_qkv)
        v_linear_out = self.v_bn(v_linear_out.transpose(-1, -2)).transpose(-1, -2)
        v_linear_out = self.v_lif(v_linear_out)
        v = v_linear_out.reshape(-1, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        attn = (q @ k.transpose(-2, -1)) * self.scale
        x = attn @ v
        x = x.transpose(1, 2).reshape(TB, N, C).contiguous()
        x = self.attn_lif(x) # TB N C
        x = self.proj_lif(self.proj_bn(self.proj_linear(x).transpose(-1, -2)).transpose(-1, -2)).reshape(TB, N, C).contiguous()

        return x # TB N C

class MLP(BaseModule):
    def __init__(self, in_features, step=4, encode_type='direct', mlp_ratio = 4.0, out_features=None,mlp_drop=0.,node=LIFNode,tau=2.0,threshold=1.0,act_func=SigmoidGrad, alpha=4.,layer_by_layer=True):
        super().__init__(encode_type=encode_type,step=step,layer_by_layer=layer_by_layer)

        self.out_features = out_features or in_features
        self.hidden_features = int(in_features * mlp_ratio)
        self.mlp_drop = mlp_drop

        self.fc1_linear = nn.Linear(in_features, self.hidden_features)
        self.fc1_bn = nn.BatchNorm1d(self.hidden_features)
        self.fc1_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold, layer_by_layer=layer_by_layer, mem_detach=True)

        self.fc2_linear = nn.Linear(self.hidden_features, out_features)
        self.fc2_bn = nn.BatchNorm1d(self.out_features)
        self.fc2_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold, layer_by_layer=layer_by_layer, mem_detach=True)
    def forward(self, x):
        self.reset()

        TB, N, C = x.shape
        x = self.fc1_linear(x)
        x = self.fc1_bn(x.transpose(-1,-2)).transpose(-1,-2)
        x = self.fc1_lif(x)

        x = self.fc2_linear(x)
        x = self.fc2_bn(x.transpose(-1,-2)).transpose(-1,-2)
        x = self.fc2_lif(x)
        return x # TB N C

# Spikformer block
class Block(nn.Module):
    def __init__(self, embed_dim=384, num_heads=12, step=4,mlp_ratio=4. ,attn_scale=0.125, attn_drop=0.,mlp_drop=0.,node=LIFNode,tau=2.0,threshold=1.0,act_func=SigmoidGrad, alpha=4.,layer_by_layer=True):
        super().__init__()

        self.attn = SSA(embed_dim, step=step, num_heads=num_heads,attn_drop=attn_drop, attn_scale=attn_scale,
                        node=node,tau=tau,act_func=act_func,threshold=threshold,alpha=alpha,layer_by_layer=layer_by_layer)
        # self.layernorm1 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(step=step,in_features=embed_dim,mlp_ratio=mlp_ratio,out_features=embed_dim,mlp_drop=mlp_drop,
                       node=node,tau=tau,act_func=act_func,threshold=threshold,alpha=alpha,layer_by_layer=layer_by_layer)
        # self.layernorm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class Spikformer(BaseModule):
    def __init__(self,
                 step=4, img_size=32, patch_size=4, in_channels=3, num_classes=10,attn_scale=0.125,
                 embed_dim=384, num_heads=12, mlp_ratio=4, mlp_drop=0., attn_drop=0.,
                 depths=4, node=LIFNode,tau=2.0,threshold=1.0,act_func=SigmoidGrad, alpha=4.,layer_by_layer=True
                 ):
        super().__init__(step=step, encode_type='direct',layer_by_layer=layer_by_layer)
        self.T = step  # time step
        self.num_classes = num_classes
        self.depths = depths

        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        patch_embed = SPS(       img_h=img_size,
                                 img_w=img_size,
                                 patch_size=patch_size,
                                 in_channels=in_channels,
                                 embed_dims=embed_dim,
                                 node=node,act_func=act_func,tau=tau,
                                 threshold=threshold,alpha=alpha,
                                 layer_by_layer=layer_by_layer)

        block = nn.ModuleList([Block(
            embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
             attn_drop=attn_drop, attn_scale=attn_scale,layer_by_layer=layer_by_layer,
            node=node,act_func=act_func,tau=tau,threshold=threshold,alpha=alpha)
            for j in range(depths)])

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", block)

        # classification head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):

        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")

        x = patch_embed(x)
        for blk in block:
            x = blk(x) # TB N C
        # dim adjustment
        _, N, C = x.shape
        x = x.reshape(self.step, -1, N, C).contiguous()
        return x.mean(2)

    def forward(self, x):
        self.reset()
        x = self.encoder(x) # TB C H W （lbl auto rerrange）
        x = self.forward_features(x)
        x = self.head(x.mean(0))
        return x

#### models for static datasets
@register_model
def spikformer_cifar(pretrained=False,**kwargs):
    model = Spikformer(
        step=kwargs.get('step', 4),
        img_size=kwargs.get('img_size', 32),
        patch_size=kwargs.get('patch_size', 4),
        in_channels=kwargs.get('in_channels', 3),
        num_classes=kwargs.get('num_classes', 10),
        embed_dim=kwargs.get('embed_dim', 384),
        num_heads=kwargs.get('num_heads', 12),
        mlp_ratio=kwargs.get('mlp_ratio', 4),
        attn_scale=kwargs.get('attn_scale', 0.125),
        mlp_drop=kwargs.get('mlp_drop', 0.0),
        attn_drop=kwargs.get('attn_drop', 0.0),
        depths=kwargs.get('depths', 4),
        tau=kwargs.get('tau', 2.0),
        threshold=kwargs.get('threshold', 1.0),
        node=kwargs.get('node', LIFNode),
        act_func=kwargs.get('act_func', SigmoidGrad),
        alpha=kwargs.get('alpha',4.0)
    )
    model.default_cfg = _cfg()
    return model


