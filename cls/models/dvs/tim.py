import torch
import torch.nn as nn
from sympy import false
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from braincog.model_zoo.base_module import BaseModule
from braincog.base.node.node import *
from braincog.base.strategy.surrogate import *




class MLP(BaseModule):
    def __init__(self,  in_features, step=10, encode_type='direct', mlp_ratio = 4.0, out_features=None,mlp_drop=0.,
                 node=LIFNode,tau=2.0,threshold=1.0,act_func=SigmoidGrad, alpha=4.,layer_by_layer=True):
        super().__init__(step=step, encode_type=encode_type,layer_by_layer=layer_by_layer)
        self.out_features = out_features or in_features
        self.hidden_features = int(in_features * mlp_ratio)
        self.mlp_drop = mlp_drop

        self.fc1_conv = nn.Conv1d(in_features, self.hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(self.hidden_features)
        self.fc1_lif = node(step=step,tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,layer_by_layer=layer_by_layer)

        self.fc2_conv = nn.Conv1d(self.hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = node(step=step,tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,layer_by_layer=layer_by_layer)

        self.c_hidden = self.hidden_features
        self.c_output = out_features

    def forward(self, x):
        self.reset()

        T, B, C, N = x.shape

        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, N).contiguous()  # T B C N
        x = self.fc1_lif(x.flatten(0, 1)).reshape(T, B, self.c_hidden, N).contiguous()

        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, N).contiguous()
        x = self.fc2_lif(x.flatten(0, 1)).reshape(T, B, C, N).contiguous()

        return x


class TIM(BaseModule):
    def __init__(self,  in_channels, encode_type='direct', TIM_alpha=0.5,
                    node=LIFNode,tau=2.0,threshold=1.0,act_func=SigmoidGrad, alpha=4.,):
        super().__init__(step=1, encode_type=encode_type)

        #  channels may depends on the shape of input
        self.interactor = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=5, stride=1,
                                    padding=2, bias=True)

        self.in_lif = node(step=1,tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,layer_by_layer=False)
        self.out_lif =node(step=1,tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,layer_by_layer=False)

        self.tim_alpha = TIM_alpha

    # input [T, B, H, N, C/H]
    def forward(self, x):
        self.reset()

        T, B, H, N, CoH = x.shape

        output = []
        x_tim = torch.empty_like(x[0])

        # temporal interaction

        for i in range(T):
            # 1st step
            if i == 0:
                x_tim = x[i]
                output.append(x_tim)

            # other steps
            else:
                x_tim = self.interactor(x_tim.flatten(0, 1)).reshape(B, H, N, CoH).contiguous()
                x_tim = self.in_lif(x_tim) * self.tim_alpha + x[i] * (1 - self.tim_alpha)
                x_tim = self.out_lif(x_tim)

                output.append(x_tim)

        output = torch.stack(output)  # T B H, N, C/H

        return output  # T B H, N, C/H

class SSA(BaseModule):
    def __init__(self, embed_dim, step=10, encode_type='direct', num_heads=16, TIM_alpha=0.5,attn_scale=0.25,
                 node=LIFNode,tau=2.0,threshold=1.0,act_func=SigmoidGrad, alpha=4.0,layer_by_layer=True, img_size=128, patch_size=16):
        super().__init__(step=step, encode_type=encode_type,layer_by_layer=layer_by_layer)
        assert embed_dim % num_heads == 0, f"dim {embed_dim} should be divided by num_heads {num_heads}."
        self.embed_dim = embed_dim

        self.num_heads = num_heads

        self.in_channels = embed_dim // num_heads

        self.scale = attn_scale

        self.q_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(embed_dim)
        self.q_lif = node(step=step,tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,layer_by_layer=layer_by_layer)

        self.k_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(embed_dim)
        self.k_lif = node(step=step,tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,layer_by_layer=layer_by_layer)

        self.v_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(embed_dim)
        self.v_lif = node(step=step,tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,layer_by_layer=layer_by_layer)

        self.attn_drop = nn.Dropout(0.2)
        self.res_lif = LIFNode(step=step,tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,layer_by_layer=layer_by_layer)
        self.attn_lif = node(step=step,tau=tau, act_func=act_func(alpha=alpha), threshold=0.5,layer_by_layer=layer_by_layer)

        self.proj_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=1, stride=1, bias=False)
        self.proj_bn = nn.BatchNorm1d(embed_dim)
        self.proj_lif = node(step=step,tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,layer_by_layer=layer_by_layer)

        self.TIM = TIM(in_channels= (img_size//patch_size)**2, TIM_alpha=TIM_alpha)

    def forward(self, x):
        self.reset()

        TB, C, N = x.shape

        x_for_qkv = x

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out)
        q_conv_out = self.q_lif(q_conv_out).transpose(-2, -1) #TB N C
        q = q_conv_out.reshape(self.step, -1, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out)
        k_conv_out = self.k_lif(k_conv_out).transpose(-2, -1)
        k = k_conv_out.reshape(self.step, -1, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out)
        v_conv_out = self.v_lif(v_conv_out).transpose(-2, -1)
        v = v_conv_out.reshape(self.step, -1, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        # TIM
        q = self.TIM(q)

        # SSA
        attn = (q @ k.transpose(-2, -1))
        x = (attn @ v) * self.scale

        x = x.transpose(3, 4).reshape(self.step, -1, C, N).contiguous()
        x = self.attn_lif(x.flatten(0, 1))
        x = self.proj_lif(self.proj_bn(self.proj_conv(x))) # TB C N

        return x


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads=16, step=10, TIM_alpha=0.5, mlp_ratio=4., attn_scale = 0.25, img_size=128, patch_size=16,
                  norm_layer=nn.LayerNorm, node=LIFNode,tau=2.0,threshold=1.0,act_func=SigmoidGrad, alpha=4.0,layer_by_layer=True):
        super().__init__()
        self.norm1 = norm_layer(embed_dim)

        self.attn = SSA(embed_dim, step=step, TIM_alpha=TIM_alpha, num_heads=num_heads, attn_scale=attn_scale,img_size=img_size,patch_size=patch_size,
                        node=node,tau=tau,threshold=threshold,act_func=act_func, alpha=alpha,layer_by_layer=layer_by_layer)
        self.norm2 = norm_layer(embed_dim)
        self.mlp = MLP(in_features=embed_dim, step=step,  mlp_ratio=mlp_ratio,
                       node=node,tau=tau,threshold=threshold,act_func=act_func, alpha=alpha,layer_by_layer=layer_by_layer)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SPS(BaseModule):
    def __init__(self, step=10, encode_type='direct', img_size_h=128, img_size_w=128, patch_size=16, in_channels=2,
                 embed_dim=256,  node=LIFNode,tau=2.0,threshold=1.0,act_func=SigmoidGrad, alpha=4.0,layer_by_layer=True):
        super().__init__(step=step, encode_type=encode_type,layer_by_layer=layer_by_layer)
        self.image_size = [img_size_h, img_size_w]

        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W


        self.proj_conv = nn.Conv2d(in_channels, embed_dim // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dim // 8)
        self.proj_lif = node(step=step,tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,layer_by_layer=layer_by_layer)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv1 = nn.Conv2d(embed_dim // 8, embed_dim // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dim // 4)
        self.proj_lif1 = node(step=step,tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,layer_by_layer=layer_by_layer)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv2 = nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dim // 2)
        self.proj_lif2 = node(step=step,tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,layer_by_layer=layer_by_layer)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv3 = nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dim)
        self.proj_lif3 = node(step=step,tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,layer_by_layer=layer_by_layer)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.rpe_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dim)
        self.rpe_lif = node(step=step,tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,layer_by_layer=layer_by_layer)

    def forward(self, x):
        self.reset()

        TB, C, H, W = x.shape

        # # UCF101DVS
        # if self.if_UCF:
        #     x = F.adaptive_avg_pool2d(x.flatten(0, 1), output_size=(64, 64)).reshape(T, B, C, 64, 64)
        #     T, B, C, H, W = x.shape

        x = self.proj_conv(x)# have some fire value
        x = self.proj_bn(x)
        x = self.proj_lif(x)
        x = self.maxpool(x)

        x = self.proj_conv1(x)
        x = self.proj_bn1(x)
        x = self.proj_lif1(x)
        x = self.maxpool1(x)

        x = self.proj_conv2(x)
        x = self.proj_bn2(x)
        x = self.proj_lif2(x)
        x = self.maxpool2(x)

        x = self.proj_conv3(x)
        x = self.proj_bn3(x)
        x = self.proj_lif3(x)
        x = self.maxpool3(x)

        x_rpe = self.rpe_bn(self.rpe_conv(x))
        x_rpe = self.rpe_lif(x_rpe)
        x = x + x_rpe
        x = x.reshape(TB, -1, (H // 16) * (W // 16)).contiguous()

        return x  # TB C N


class Spikformer(nn.Module):
    def __init__(self, step=10, img_size=128, patch_size=16, in_channels=2, num_classes=10,attn_scale=0.25,
                 embed_dim=256, num_heads=16, mlp_ratio=4, depths=4,TIM_alpha=0.5,
                 node=LIFNode,tau=2.0,threshold=1.0,act_func=SigmoidGrad, alpha=4.,layer_by_layer=True
                 ):
        super().__init__()
        self.T = step  # time step
        self.num_classes = num_classes
        self.depths = depths

        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        patch_embed = SPS(step=step,
                          img_size_h=img_size,
                          img_size_w=img_size,
                          patch_size=patch_size,
                          in_channels=in_channels,
                          embed_dim=embed_dim,
                          node=node,tau=tau,threshold=threshold,act_func=act_func, alpha=alpha,layer_by_layer=layer_by_layer)

        block = nn.ModuleList([Block(step=step, TIM_alpha=TIM_alpha, img_size=img_size,patch_size=patch_size,
                                     embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,attn_scale=attn_scale,
                                     node=node,tau=tau,threshold=threshold,act_func=act_func, alpha=alpha,layer_by_layer=layer_by_layer)

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
            x = blk(x)
        return x.mean(3)

    def forward(self, x):
        x = x.permute(1, 0, 2, 3, 4) # T B C H W
        x = x.flatten(0, 1) # TB C H W
        x = self.forward_features(x)
        x = self.head(x.mean(0))
        return x


# Hyperparams could be adjust here

@register_model
def spikformer_dvs(pretrained=False, **kwargs):
    model = Spikformer(
        step=kwargs.get('step', 4),
        img_size=kwargs.get('img_size', 128),
        patch_size=kwargs.get('patch_size', 16),
        in_channels=kwargs.get('in_channels', 2),
        num_classes=kwargs.get('num_classes', 10),
        embed_dim=kwargs.get('embed_dim', 256),
        num_heads=kwargs.get('num_heads', 16),
        mlp_ratio=kwargs.get('mlp_ratio', 4),
        attn_scale=kwargs.get('attn_scale', 0.25),
        depths=kwargs.get('depths', 4),
        tau=kwargs.get('tau', 2.0),
        threshold=kwargs.get('threshold', 1.0),
        node=kwargs.get('node', LIFNode),
        act_func=kwargs.get('act_func', SigmoidGrad),
        alpha=kwargs.get('alpha', 4.0),
        TIM_alpha = kwargs.get('TIM_alpha', 0.5)
        )
    model.default_cfg = _cfg()
    return model