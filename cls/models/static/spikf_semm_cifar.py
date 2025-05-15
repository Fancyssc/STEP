from braincog.model_zoo.base_module import BaseModule
from timm.models import register_model
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import _cfg
from ..utils.node import *
from braincog.base.strategy.surrogate import *
from .spikformer_cifar import BNAndPadLayer

import torch.nn as nn
"""
Spiking Transformer with Experts Mixture (NeurIPS 2024)
"""

class MLP(BaseModule):
    def __init__(self, in_features, step=4, encode_type='direct', mlp_ratio = 4.0, out_features=None,mlp_drop=0.,
                 node=LIFNode,tau=2.0,threshold=1.0,act_func=SigmoidGrad, alpha=4.,layer_by_layer=True):
        super().__init__(encode_type=encode_type,step=step,layer_by_layer=layer_by_layer)

        self.step = step
        self.out_features = out_features or in_features
        self.hidden_features = int(in_features * mlp_ratio)
        self.mlp_drop = mlp_drop

        self.fc1_linear = nn.Linear(in_features, self.hidden_features)
        self.fc1_bn = nn.BatchNorm1d(self.hidden_features)
        self.fc1_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold, layer_by_layer=layer_by_layer, mem_detach=False)

        self.dw_conv = nn.Conv2d(self.hidden_features // 2, self.hidden_features// 2, kernel_size=3, stride=1, padding=1,groups=self.hidden_features // 2, bias=False)
        self.dw_bn = nn.BatchNorm2d(self.hidden_features // 2)
        self.dw_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold, layer_by_layer=layer_by_layer, mem_detach=False)

        self.fc2_linear = nn.Linear(self.hidden_features // 2, out_features)
        self.fc2_bn = nn.BatchNorm1d(self.out_features)
        self.fc2_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold, layer_by_layer=layer_by_layer, mem_detach=False)

    def forward(self, x):
        self.reset()

        TB, N, C = x.shape
        # split for EMSP
        x_for_EMSP = x.reshape(self.step, -1, N, C)
        T, B, _, _ = x_for_EMSP.shape
        H, W = 8, 8

        x = self.fc1_linear(x_for_EMSP.flatten(0, 1))
        x = self.fc1_bn(x.transpose(-1,-2)).transpose(-1,-2).reshape(T, B, N, self.hidden_features).contiguous()
        x = self.fc1_lif(x)

        x1, x2 = torch.chunk(x, 2, dim=3)

        x1 = self.dw_conv(x1.reshape(T * B, H, W, self.hidden_features // 2).permute(0, 3, 1, 2).contiguous())
        x1 = self.dw_bn(x1)
        x1 = self.dw_lif(x1.reshape(T, B, self.hidden_features// 2, H, W).flatten(0, 1)).permute(0, 2, 3, 1).reshape(T, B, N, self.hidden_features // 2)

        x = x1 * x2

        x = self.fc2_linear(x.flatten(0, 1))
        x = self.fc2_bn(x.transpose(-1,-2)).transpose(-1,-2)
        x = self.fc2_lif(x)

        return x # TB N C

class MLP_Unit(BaseModule):
    def __init__(self,in_features, step=4, exp_idx=-1, encode_type='direct', out_features=None, node=LIFNode,tau=2.0,threshold=1.0,act_func=SigmoidGrad, alpha=4.,layer_by_layer=True):
        super().__init__(encode_type=encode_type, step=step, layer_by_layer=layer_by_layer)
        self.out_features = out_features or in_features

        self.unit_linear = nn.Linear(in_features, self.out_features)
        self.unit_bn = nn.BatchNorm1d(out_features)
        self.unit_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold, layer_by_layer=layer_by_layer, mem_detach=False)

        self.out_features = out_features
        self.expert_idx = exp_idx

    def forward(self, x):
        self.reset()
        TB, N, C = x.shape
        x = self.unit_linear(x)
        x = self.unit_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(4, TB//4, N, self.out_features).contiguous()
        x = self.unit_lif(x)
        return x

class SSA(BaseModule):
    def __init__(self,embed_dim, step=4, encode_type='direct',num_heads=12,attn_scale=0.125,attn_drop=0.,
                 node=LIFNode,tau=2.0,threshold=1.0,act_func=SigmoidGrad, alpha=4.0,layer_by_layer=True,
                 expert_mode='base', expert_dim=0, num_expert=4):
        super().__init__(step=step, encode_type=encode_type,layer_by_layer=layer_by_layer,)
        self.num_heads = num_heads
        self.scale = attn_scale
        self.T = step

        if expert_mode == 'small':
            self.dim = expert_dim
        elif expert_mode == 'base':
            self.dim = embed_dim

        # k must be expert dim
        self.k_linear = nn.Linear(embed_dim, expert_dim)
        self.k_bn = nn.BatchNorm1d(expert_dim)
        self.k_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,  layer_by_layer=layer_by_layer, mem_detach=False)

        # v may be expert dim
        self.v_linear = nn.Linear(embed_dim, self.dim)
        self.v_bn = nn.BatchNorm1d(self.dim)
        self.v_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,  layer_by_layer=layer_by_layer, mem_detach=False)

        # self.attn_lif = node(step=step, tau =tau, act_func=act_func(alpha=alpha), threshold=0.5, layer_by_layer=True, mem_detach=False) #special v_thres

        self.proj_linear = nn.Linear(self.dim, embed_dim)
        self.proj_bn = nn.BatchNorm1d(embed_dim)
        self.proj_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,  layer_by_layer=layer_by_layer, mem_detach=False)

        # expert
        self.num_expert = num_expert
        self.expert_dim = expert_dim

        self.router1 = nn.Linear(embed_dim, num_expert)
        self.router2 = nn.BatchNorm1d(num_expert)
        self.router3 = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,  layer_by_layer=layer_by_layer, mem_detach=False)

        self.ff_list = nn.ModuleList([MLP_Unit(in_features=embed_dim, out_features=expert_dim, exp_idx=i) for i in range(num_expert)])

        self.lif_list = nn.ModuleList(
            [node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,  layer_by_layer=layer_by_layer, mem_detach=False)
             for i in range(num_expert)])

    def forward(self, x):
        self.reset()

        TB, N, C = x.shape
        x_for_qkv = x

        k_linear_out = self.k_linear(x_for_qkv)
        k_linear_out = self.k_bn(k_linear_out.transpose(-1, -2)).transpose(-1, -2).reshape(self.step, -1, N, self.expert_dim).contiguous()
        k = self.k_lif(k_linear_out)

        v_linear_out = self.v_linear(x_for_qkv)
        v_linear_out = self.v_bn(v_linear_out.transpose(-1, -2)).transpose(-1, -2).reshape(self.step, -1, N, C) # T B N C
        v = self.v_lif(v_linear_out)

        weights = self.router1(x_for_qkv)
        weights = self.router2(weights.transpose(-1, -2)).reshape(self.step, -1, N, self.num_expert).contiguous()
        weights = self.router3(weights) # weight matrix

        # attention + MoE
        y = 0
        for idx in range(self.num_expert):
            weight_idx = weights[:, :, :,  idx].unsqueeze(dim=-1)
            q = self.ff_list[idx](x_for_qkv)
            attn = q @ k.transpose(-1, -2)
            result = (attn @ v) * self.scale
            result = self.lif_list[idx](result)
            y += weight_idx*result

        y = y.flatten(0, 1) # TB N C
        y = self.proj_lif(self.proj_bn(self.proj_linear(y).transpose(-1, -2)).transpose(-1, -2))

        return y # TB N C

class Block(nn.Module):
    def __init__(self, embed_dim=384, num_heads=12, step=4,mlp_ratio=4. ,attn_scale=0.125, attn_drop=0.,mlp_drop=0.,node=LIFNode,
                 tau=2.0,threshold=1.0,act_func=SigmoidGrad, alpha=4.,layer_by_layer=True, attn_layer='SSA',
                 expert_mode='base', expert_dim=0, num_expert=4):
        super().__init__()

        if attn_layer in globals():
            self.attn  = globals()[attn_layer](
                        embed_dim, step=step, num_heads=num_heads,attn_drop=attn_drop,
                        attn_scale=attn_scale,node=node,tau=tau,act_func=act_func,
                        threshold=threshold,alpha=alpha,layer_by_layer=layer_by_layer,
                        expert_dim=expert_dim, expert_mode=expert_mode, num_expert=num_expert
            )
        # self.layernorm1 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(step=step,in_features=embed_dim,mlp_ratio=mlp_ratio,out_features=embed_dim,mlp_drop=mlp_drop,
                       node=node,tau=tau,act_func=act_func,threshold=threshold,alpha=alpha,layer_by_layer=layer_by_layer)
        # self.layernorm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x

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
                 embed_dims=384, node=LIFNode,tau=2.0,threshold=1.0,act_func=SigmoidGrad, alpha=4.0,layer_by_layer=True,
                 **kwargs):
        super().__init__(step=step, encode_type= encode_type,layer_by_layer=layer_by_layer)

        self.img_h = img_h
        self.img_w = img_w
        self.patch_size = patch_size
        self.patch_nums = self.img_h // self.patch_size * self.img_w // self.patch_size
        self.in_channels = in_channels
        self.embed_dims = embed_dims


        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)
        self.proj_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold, layer_by_layer=layer_by_layer, mem_detach=False)

        self.proj_conv1 = nn.Conv2d(embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims // 4)
        self.proj_lif1 = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,  layer_by_layer=layer_by_layer, mem_detach=False)

        self.proj_conv2 = nn.Conv2d(embed_dims // 4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims // 2)
        self.proj_lif2 = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,  layer_by_layer=layer_by_layer, mem_detach=False)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv3 = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_lif3 = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,  layer_by_layer=layer_by_layer, mem_detach=False)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,  layer_by_layer=layer_by_layer, mem_detach=False)


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

class Spikformer(BaseModule):
    def __init__(self,
                 step=4, img_size=32, patch_size=4, in_channels=3, num_classes=10,attn_scale=0.125,
                 embed_dim=384, num_heads=12, mlp_ratio=4, mlp_drop=0., attn_drop=0.,embed_layer='SPS', attn_layer='SSA',
                 depths=4, node=LIFNode,tau=2.0,threshold=1.0,act_func=SigmoidGrad, alpha=4.,layer_by_layer=True,
                 expert_dim = 96, num_expert = 4, expert_mode='base', **kwargs
                 ):
        super().__init__(step=step, encode_type='direct',layer_by_layer=layer_by_layer)
        self.T = step  # time step
        self.num_classes = num_classes
        self.depths = depths

        # for meta_transformer

        if embed_layer in globals():
            patch_embed = globals()[embed_layer](img_h=img_size,
                                                 img_w=img_size,
                                                 patch_size=patch_size,
                                                 in_channels=in_channels,
                                                 embed_dims=embed_dim,
                                                 node=node, act_func=act_func, tau=tau,
                                                 threshold=threshold, alpha=alpha,
                                                 layer_by_layer=layer_by_layer,
                                                 **kwargs)
        else:
            raise ValueError(f"Unknown embed_layer: {embed_layer}")

        block = nn.ModuleList([Block(
            embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, attn_layer=attn_layer,
             attn_drop=attn_drop, attn_scale=attn_scale,layer_by_layer=layer_by_layer,
            node=node,act_func=act_func,tau=tau,threshold=threshold,alpha=alpha,
            expert_dim = expert_dim, num_expert = num_expert, expert_mode=expert_mode)
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
        if len(x.shape) == 4:
            x = self.encoder(x)

            # sequence datasets
        else:
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1).flatten(0, 1)
        x = self.forward_features(x)
        x = self.head(x.mean(0))
        return x

@register_model
def spikf_semm_cifar(pretrained=False,**kwargs):
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
        alpha=kwargs.get('alpha',4.0),
        ### for meta transformer
        embed_layer=kwargs.get('embed_layer', 'SPS'),
        attn_layer=kwargs.get('attn_layer', 'SSA'),
        ### for sequential datasets
        sequence_length=kwargs.get('sequence_length', 0)
    )
    model.default_cfg = _cfg()
    return model


"""
    classes for meta spiking transformer
"""
class vit_embed(BaseModule):
    def __init__(self, step=4, encode_type='direct', img_h=32, img_w=32, patch_size=4, in_channels=3,
                 embed_dims=384, node=LIFNode, tau=2.0, threshold=1.0, act_func=SigmoidGrad, alpha=4.0,
                 layer_by_layer=True, **kwargs):
        super().__init__(step=step, encode_type=encode_type, layer_by_layer=layer_by_layer)

        self.img_h = img_h
        self.img_w = img_w
        self.patch_size = patch_size
        self.patch_nums = self.img_h // self.patch_size * self.img_w // self.patch_size
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.layer_by_layer = layer_by_layer

        self.proj_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.proj_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,
                             layer_by_layer=layer_by_layer, mem_detach=False)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_nums, self.embed_dims))
        self.output_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,
                               layer_by_layer=layer_by_layer, mem_detach=False)

        # no cls token needed

    def forward(self, x):
        self.reset()

        x = self.proj_conv(x)
        x = self.proj_lif(x)  # TB C H//4 W//4

        # x = x + self.pos_embed
        x = x.flatten(-2, -1).transpose(-2, -1)

        x = x + self.pos_embed  # TB N C
        # x = self.output_lif(x)

        return x

class conv2_embed(BaseModule):
    def __init__(self, step=4, encode_type='direct', img_h=32, img_w=32, patch_size=4, in_channels=3,
                 embed_dims=384, node=LIFNode, tau=2.0, threshold=1.0, act_func=SigmoidGrad, alpha=4.0,
                 layer_by_layer=True,**kwargs):
        super().__init__(step=step, encode_type=encode_type, layer_by_layer=layer_by_layer)

        self.img_h = img_h
        self.img_w = img_w
        self.patch_size = patch_size
        self.patch_nums = self.img_h // self.patch_size * self.img_w // self.patch_size
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.layer_by_layer = layer_by_layer

        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 4)
        self.proj_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,
                             layer_by_layer=layer_by_layer, mem_detach=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv1 = nn.Conv2d(embed_dims // 4, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims)
        self.proj_lif1 = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,
                              layer_by_layer=layer_by_layer, mem_detach=False)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,
                            layer_by_layer=layer_by_layer, mem_detach=False)

    def forward(self, x):
        self.reset()

        TB, C, H, W = x.shape

        x = self.proj_conv(x)
        x = self.proj_bn(x)
        x = self.proj_lif(x)
        x = self.maxpool(x)

        x = self.proj_conv1(x)
        x = self.proj_bn1(x)
        x = self.proj_lif1(x)
        x = self.maxpool1(x)

        x_feat = x  # TB, -1, H // 4, W // 4

        x = self.rpe_conv(x)
        x = self.rpe_bn(x)
        x = self.rpe_lif(x)  # TB, -1, H // 4, W // 4

        x = x + x_feat

        x = x.flatten(-2).transpose(-1, -2)  # TB,N,C

        return x  # TB,N,C

class random_attn(BaseModule):
    def __init__(self,embed_dim, step=4, encode_type='direct',num_heads=12,attn_scale=0.125,attn_drop=0.,
                 node=LIFNode,tau=2.0,threshold=1.0,act_func=SigmoidGrad, alpha=4.0,layer_by_layer=True,
                 expert_mode='base', expert_dim=0, num_expert=4):
        super().__init__(step=step, encode_type=encode_type,layer_by_layer=layer_by_layer,)
        self.num_heads = num_heads
        self.scale = attn_scale
        self.T = step

        if expert_mode == 'small':
            self.dim = expert_dim
        elif expert_mode == 'base':
            self.dim = embed_dim

        # k must be expert dim
        self.k_linear = nn.Linear(embed_dim, expert_dim)
        self.k_linear.weight.requires_grad = False
        self.k_linear.bias.requires_grad = False
        self.k_bn = nn.BatchNorm1d(expert_dim)
        self.k_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,  layer_by_layer=layer_by_layer, mem_detach=False)

        # v may be expert dim
        self.v_linear = nn.Linear(embed_dim, self.dim)
        self.v_linear.weight.requires_grad = False
        self.v_linear.bias.requires_grad = False
        self.v_bn = nn.BatchNorm1d(self.dim)
        self.v_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,  layer_by_layer=layer_by_layer, mem_detach=False)

        # self.attn_lif = node(step=step, tau =tau, act_func=act_func(alpha=alpha), threshold=0.5, layer_by_layer=True, mem_detach=False) #special v_thres

        self.proj_linear = nn.Linear(self.dim, embed_dim)
        self.proj_bn = nn.BatchNorm1d(embed_dim)
        self.proj_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,  layer_by_layer=layer_by_layer, mem_detach=False)

        # expert
        self.num_expert = num_expert
        self.expert_dim = expert_dim

        self.router1 = nn.Linear(embed_dim, num_expert)
        self.router2 = nn.BatchNorm1d(num_expert)
        self.router3 = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,  layer_by_layer=layer_by_layer, mem_detach=False)

        self.ff_list = nn.ModuleList([MLP_Unit(in_features=embed_dim, out_features=expert_dim, exp_idx=i) for i in range(num_expert)])

        self.lif_list = nn.ModuleList(
            [node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,  layer_by_layer=layer_by_layer, mem_detach=False)
             for i in range(num_expert)])

    def forward(self, x):
        self.reset()

        TB, N, C = x.shape
        x_for_qkv = x

        k_linear_out = self.k_linear(x_for_qkv)
        k_linear_out = self.k_bn(k_linear_out.transpose(-1, -2)).transpose(-1, -2).reshape(self.step, -1, N, self.expert_dim).contiguous()
        k = self.k_lif(k_linear_out)

        v_linear_out = self.v_linear(x_for_qkv)
        v_linear_out = self.v_bn(v_linear_out.transpose(-1, -2)).transpose(-1, -2).reshape(self.step, -1, N, C) # T B N C
        v = self.v_lif(v_linear_out)

        weights = self.router1(x_for_qkv)
        weights = self.router2(weights.transpose(-1, -2)).reshape(self.step, -1, N, self.num_expert).contiguous()
        weights = self.router3(weights) # weight matrix

        # attention + MoE
        y = 0
        for idx in range(self.num_expert):
            weight_idx = weights[:, :, :,  idx].unsqueeze(dim=-1)
            q = self.ff_list[idx](x_for_qkv)
            attn = q @ k.transpose(-1, -2)
            result = (attn @ v) * self.scale
            result = self.lif_list[idx](result)
            y += weight_idx*result

        y = y.flatten(0, 1) # TB N C
        y = self.proj_lif(self.proj_bn(self.proj_linear(y).transpose(-1, -2)).transpose(-1, -2))

        return y # TB N C

class sequential_embed(BaseModule):
    def __init__(self, step=4, encode_type='direct', sequence_length=1024, in_channels=3,
                 embed_dims=384, node=LIFNode, tau=2.0, threshold=1.0, act_func=SigmoidGrad, alpha=4.0,
                 layer_by_layer=True, **kwargs):

        super(sequential_embed, self).__init__(step=step, encode_type=encode_type, layer_by_layer=layer_by_layer)

        self.in_channels = in_channels
        self.sequence_length = sequence_length
        self.embed_dims = embed_dims
        self.out_length = 64

        if sequence_length >= 1024:
            downsample_rates = [2, 2, 2, 2]  # 这些是步长，不是指数
        else:
            downsample_rates = [1, 2, 2, 2]
        # 按照要求设置每层的通道数: embed_dim//8, embed_dim//4, embed_dim//2, embed_dim
        channels = [
            in_channels,
            embed_dims // 8,
            embed_dims // 4,
            embed_dims // 2,
            embed_dims
        ]

        self.layers = nn.ModuleList()

        current_length = sequence_length

        for i in range(4):
            stride = downsample_rates[i]
            kernel_size = 5
            target_output_size = current_length // stride
            padding = self._calculate_padding(current_length, kernel_size, stride, target_output_size)

            layer = nn.Sequential(
                nn.Conv1d(channels[i], channels[i + 1], kernel_size=kernel_size,
                          stride=stride, padding=padding),
                nn.BatchNorm1d(channels[i + 1]),
                node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,
                     layer_by_layer=layer_by_layer, mem_detach=False)
            )
            self.layers.append(layer)

            current_length = target_output_size

        # if current_length != self.out_length:
        self.fine_tuning_layer = nn.Sequential(
                nn.AdaptiveAvgPool1d(self.out_length)
            )
        # else:
        #     self.fine_tuning_layer = nn.Identity()

        # 添加可学习的相对位置编码
        self.pe_conv = nn.Conv1d(
            embed_dims,
            embed_dims,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            bias=True
        )
        self.pe_bn = nn.BatchNorm1d(embed_dims)

    def _calculate_padding(self, input_size, kernel_size, stride, target_size):
        padding = ((target_size - 1) * stride + kernel_size - input_size) // 2
        return max(0, int(padding))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = self.fine_tuning_layer(x)

        x_feat = x

        x = self.pe_conv(x)
        x = self.pe_bn(x)

        x = x + x_feat

        return x.permute(0, 2, 1)

class repconv_attn(BaseModule):
    def __init__(self,embed_dim, step=4, encode_type='direct',num_heads=12,attn_scale=0.125,attn_drop=0.,
                 node=LIFNode,tau=2.0,threshold=1.0,act_func=SigmoidGrad, alpha=4.0,layer_by_layer=True,
                 expert_mode='base', expert_dim=0, num_expert=4):
        super().__init__(step=step, encode_type=encode_type,layer_by_layer=layer_by_layer,)
        self.num_heads = num_heads
        self.scale = attn_scale
        self.T = step

        if expert_mode == 'small':
            self.dim = expert_dim
        elif expert_mode == 'base':
            self.dim = embed_dim

        # k must be expert dim

        self.k_linear = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1, padding=0, bias=False),
            BNAndPadLayer(pad_pixels=1, num_features=embed_dim),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 0, groups=embed_dim, bias=False),
            nn.Conv2d(embed_dim, expert_dim, 1, 1, 0, groups=1, bias=False),
        )
        self.k_bn = nn.BatchNorm2d(expert_dim)
        self.k_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,  layer_by_layer=layer_by_layer, mem_detach=False)

        # v may be expert dim
        self.v_linear = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1, padding=0, bias=False),
            BNAndPadLayer(pad_pixels=1, num_features=embed_dim),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 0, groups=embed_dim, bias=False),
            nn.Conv2d(embed_dim, self.dim, 1, 1, 0, groups=1, bias=False),
        )
        self.v_bn = nn.BatchNorm2d(self.dim)
        self.v_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,  layer_by_layer=layer_by_layer, mem_detach=False)

        # self.attn_lif = node(step=step, tau =tau, act_func=act_func(alpha=alpha), threshold=0.5, layer_by_layer=True, mem_detach=False) #special v_thres

        self.proj_linear = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, padding=0, bias=False),
            BNAndPadLayer(pad_pixels=1, num_features=embed_dim),
            nn.Conv2d(self.dim,self.dim, 3, 1, 0, groups=embed_dim, bias=False),
            nn.Conv2d(self.dim, embed_dim, 1, 1, 0, groups=1, bias=False),
        )
        self.proj_bn = nn.BatchNorm2d(embed_dim)
        self.proj_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,  layer_by_layer=layer_by_layer, mem_detach=False)

        # expert
        self.num_expert = num_expert
        self.expert_dim = expert_dim

        self.router1 = nn.Linear(embed_dim, num_expert)
        self.router2 = nn.BatchNorm1d(num_expert)
        self.router3 = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,  layer_by_layer=layer_by_layer, mem_detach=False)

        self.ff_list = nn.ModuleList([MLP_Unit(in_features=embed_dim, out_features=expert_dim, exp_idx=i) for i in range(num_expert)])

        self.lif_list = nn.ModuleList(
            [node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,  layer_by_layer=layer_by_layer, mem_detach=False)
             for i in range(num_expert)])

    def forward(self, x):
        self.reset()

        TB, N, C = x.shape
        H, W = int(N ** 0.5), int(N ** 0.5)
        x = x.permute(0, 2, 1).reshape(TB, C, H, W).contiguous()  # TB C H W
        x_for_qkv = x

        k_linear_out = self.k_linear(x_for_qkv) # TB C H W
        k_linear_out = self.k_bn(k_linear_out).flatten(-2, -1).transpose(-1, -2)
        k_linear_out = k_linear_out.reshape(self.step, -1, N, self.expert_dim).contiguous()
        k = self.k_lif(k_linear_out)

        v_linear_out = self.v_linear(x_for_qkv)
        v_linear_out = self.v_bn(v_linear_out).flatten(-2, -1).transpose(-1, -2)
        v_linear_out = v_linear_out.reshape(self.step, -1, N, C).contiguous() # T B N C
        v = self.v_lif(v_linear_out)

        weights = self.router1(x_for_qkv.flatten(-2, -1).transpose(-1, -2))
        weights = self.router2(weights.transpose(-1, -2)).reshape(self.step, -1, N, self.num_expert).contiguous()
        weights = self.router3(weights) # weight matrix

        # attention + MoE
        y = 0
        for idx in range(self.num_expert):
            weight_idx = weights[:, :, :,  idx].unsqueeze(dim=-1)
            q = self.ff_list[idx](x_for_qkv.flatten(-2, -1).transpose(-2, -1))
            attn = q @ k.transpose(-1, -2)
            result = (attn @ v) * self.scale
            result = self.lif_list[idx](result)
            y += weight_idx*result

        y = y.flatten(0, 1) # TB N C
        y = y.permute(0, 2, 1).reshape(TB, C, H, W).contiguous()
        y = self.proj_lif(self.proj_bn(self.proj_linear(y))).flatten(-2, -1).transpose(-1, -2)

        return y # TB N C