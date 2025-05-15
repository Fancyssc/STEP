from mmengine.model import BaseModule
from braincog.model_zoo.base_module import BaseModule as BR_BaseModule
from timm.models import register_model
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.vision_transformer import _cfg

from braincog.model_zoo.base_module import BaseModule as BR_BaseModule
from braincog.base.node import LIFNode
from braincog.base.strategy.surrogate import *

import torch.nn as nn

from mmseg.registry import MODELS
from mmengine.logging import print_log
from mmengine.runner import CheckpointLoader


__all__ = ["spikformer"]


class MLP(BR_BaseModule):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.0,
        encode_type="direct",
        step=4,
        layer_by_layer=True,
        node=LIFNode,
        tau=2.0,
        act_func=SigmoidGrad,
        alpha=4.0,
        threshold=1.0,
    ):
        super().__init__(
            encode_type=encode_type, step=step, layer_by_layer=layer_by_layer
        )
        out_features = out_features or in_features
        hidden_features = in_features // 3 * 16
        self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm2d(hidden_features)
        self.fc1_lif = node(
            step=step,
            tau=tau,
            act_func=act_func(alpha=alpha),
            threshold=threshold,
            layer_by_layer=layer_by_layer,
            mem_detach=False,
        )

        self.dw_conv = nn.Conv2d(
            hidden_features // 2,
            hidden_features // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features // 2,
        )
        self.dw_bn = nn.BatchNorm2d(hidden_features // 2)
        self.dw_lif = node(
            step=step,
            tau=tau,
            act_func=act_func(alpha=alpha),
            threshold=threshold,
            layer_by_layer=layer_by_layer,
            mem_detach=False,
        )

        self.fc2_conv = nn.Conv2d(
            hidden_features // 2, out_features, kernel_size=1, stride=1
        )
        self.fc2_bn = nn.BatchNorm2d(out_features)
        self.fc2_lif = node(
            step=step,
            tau=tau,
            act_func=act_func(alpha=alpha),
            threshold=threshold,
            layer_by_layer=layer_by_layer,
            mem_detach=False,
        )

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        TB, C, H, W = x.shape
        x = self.fc1_conv(x)
        x = self.fc1_bn(x)
        x = self.fc1_lif(x)

        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.dw_conv(x1)
        x1 = self.dw_bn(x1)
        x1 = self.dw_lif(x1)

        x = x1 * x2

        x = self.fc2_conv(x)
        x = self.fc2_bn(x)
        x = self.fc2_lif(x)
        return x


class Expert_Unit(BR_BaseModule):
    def __init__(
        self,
        in_features,
        out_features,
        drop=0.0,
        encode_type="direct",
        step=4,
        layer_by_layer=True,
        node=LIFNode,
        tau=2.0,
        act_func=SigmoidGrad,
        alpha=4.0,
        threshold=1.0,
    ):
        super().__init__(
            encode_type=encode_type, step=step, layer_by_layer=layer_by_layer
        )
        self.unit_conv = nn.Conv1d(
            in_features, out_features, kernel_size=1, stride=1, bias=False
        )
        self.unit_bn = nn.BatchNorm1d(out_features)
        self.unit_lif = node(
            step=step,
            tau=tau,
            act_func=act_func(alpha=alpha),
            threshold=threshold,
            layer_by_layer=layer_by_layer,
            mem_detach=False,
        )

        self.out_features = out_features

    def forward(self, x, hook=None):
        self.reset()
        TB, C, N = x.shape
        x = self.unit_conv(x)
        x = self.unit_bn(x)
        x = self.unit_lif(x)
        return x


class SSA(BR_BaseModule):
    def __init__(
        self,
        dim,
        expert_dim=96,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
        num_expert=4,
        mode="small",
        step=4,
        encode_type="direct",
        layer_by_layer=True,
        node=LIFNode,
        tau=2.0,
        act_func=SigmoidGrad,
        alpha=4.0,
        threshold=1.0,
    ):

        super().__init__(
            step=step,
            encode_type=encode_type,
            layer_by_layer=layer_by_layer,
        )
        self.dim = dim
        self.expert_dim = expert_dim
        self.num_expert = num_expert
        self.scale = 0.125
        if mode == "base":
            self.d = dim
        elif mode == "small":
            self.d = expert_dim

        self.k_conv = nn.Conv1d(dim, expert_dim, kernel_size=1, stride=1, bias=False)
        #self.k_bn = nn.BatchNorm1d(expert_dim)
        self.k_lif = node(
            step=step,
            tau=tau,
            act_func=act_func(alpha=alpha),
            threshold=threshold,
            layer_by_layer=layer_by_layer,
            mem_detach=False,
        )

        self.v_conv = nn.Conv1d(dim, self.d, kernel_size=1, stride=1, bias=False)
        #self.v_conv_conv = nn.Conv1d(self.d, self.d, kernel_size=1, stride=1, bias=False)
        #self.v_bn = nn.BatchNorm1d(self.d)
        self.v_lif = node(
            step=step,
            tau=tau,
            act_func=act_func(alpha=alpha),
            threshold=threshold,
            layer_by_layer=layer_by_layer,
            mem_detach=False,
        )

        self.router1 = nn.Conv1d(dim, num_expert, kernel_size=1, stride=1)
        self.router2 = nn.BatchNorm1d(num_expert)
        self.router3 = node(
            step=step,
            tau=tau,
            act_func=act_func(alpha=alpha),
            threshold=threshold,
            layer_by_layer=layer_by_layer,
            mem_detach=False,
        )

        self.ff_list = nn.ModuleList(
            [
                Expert_Unit(
                    dim,
                    expert_dim,
                    encode_type=encode_type,
                    step=step,
                    layer_by_layer=layer_by_layer,
                    node=node,
                    tau=tau,
                    act_func=act_func,
                    alpha=alpha,
                    threshold=threshold,
                )
                for i in range(num_expert)
            ]
        )
        self.lif_list = nn.ModuleList(
            [
                node(
                    step=step,
                    tau=tau,
                    act_func=act_func(alpha=alpha),
                    threshold=threshold,
                    layer_by_layer=layer_by_layer,
                    mem_detach=False,
                )
                for i in range(num_expert)
            ]
        )

        self.proj_conv = nn.Conv1d(self.d, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = node(
            step=step,
            tau=tau,
            act_func=act_func(alpha=alpha),
            threshold=threshold,
            layer_by_layer=layer_by_layer,
            mem_detach=False,
        )

        print(self.dim,self.expert_dim,self.k_conv.weight.data.shape)

    def forward(self, x, res_attn):
        
        self.reset()
        TB, C, H, W = x.shape
        x = x.flatten(2)
        TB, C, N = x.shape
        #print(x.shape)
        #print(self.dim)
        #print(self.k_conv)

        if self.expert_dim!=0:

            k_conv_out = self.k_conv(x)
            # k_conv_out = (
            #     self.k_bn(k_conv_out).reshape(T, B, self.expert_dim, N).contiguous()
            # )
            k = self.k_lif(k_conv_out)
    
            v_conv_out = self.v_conv(x)
            # v_conv_out = self.v_bn(v_conv_out).reshape(T, B, -1, N).contiguous()
            v = self.v_lif(v_conv_out)

        # weights = self.router1(x.flatten(0, 1))
        # weights = self.router2(weights).reshape(T, B, self.num_expert, N).contiguous()
        # weights = self.router3(weights)
        weights = self.router1(x)
        weights = self.router2(weights)
        weights = self.router3(weights)

        y = 0
        for idx in range(self.num_expert):
            weight_idx = weights[:, idx, :].unsqueeze(dim=-2)  # TB,1,N
            q = self.ff_list[idx](x).transpose(-1, -2).contiguous()  # TB,N,C
            attn = q @ k  # (TB,N,C) @ (TB,C,N)
            result = attn @ v.transpose(-1, -2).contiguous()  # (TB,N,N) @ (TB,N,C)
            result = self.lif_list[idx](result)
            attn = weight_idx * result.transpose(-1, -2).contiguous()  # => (TB,C,N)
            y += attn

        y = self.proj_lif(self.proj_bn(self.proj_conv(y)).reshape(TB, C, H, W))
        # router_count = torch.sum(weights, dim=-1)
        return y


class Block(nn.Module):
    def __init__(
        self,
        dim,
        expert_dim,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
        num_expert=4,
        encode_type="direct",
        step=4,
        layer_by_layer=True,
        node=LIFNode,
        tau=2.0,
        threshold=1.0,
        alpha=4.0,
        act_func=SigmoidGrad,
    ):
        super().__init__()
        # self.norm1 = norm_layer(dim)
        self.attn = SSA(
            dim,
            expert_dim=expert_dim,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            num_expert=num_expert,
            mode="base",
            encode_type=encode_type,
            step=step,
            layer_by_layer=layer_by_layer,
            node=node,
            tau=tau,
            act_func=act_func,
            alpha=alpha,
            threshold=threshold,
        )
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop,
            encode_type=encode_type,
            step=step,
            layer_by_layer=layer_by_layer,
            node=node,
            tau=tau,
            act_func=act_func,
            alpha=alpha,
            threshold=threshold,
        )

    def forward(self, x, res_attn):
        x_attn = self.attn(x, res_attn)
        x = x + x_attn
        y = self.mlp(x)
        x = x + y
        return x

'''
class SPS(BR_BaseModule):
    def __init__(
        self,
        step=4,
        img_size_h=128,
        img_size_w=128,
        patch_size=4,
        in_channels=2,
        embed_dims=256,
        node=LIFNode,
        tau=2.0,
        threshold=1.0,
        act_func=SigmoidGrad,
        alpha=4.0,
        layer_by_layer=True,
        encode_type='direct',
    ):
        
        super().__init__(
            step=step,
            encode_type=encode_type,
            layer_by_layer=layer_by_layer,
        )
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = (
            self.image_size[0] // patch_size[0],
            self.image_size[1] // patch_size[1],
        )
        self.num_patches = self.H * self.W
        self.proj_conv = nn.Conv2d(
            in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)
        self.proj_lif = node(
            step=step,
            tau=tau,
            act_func=act_func(alpha=alpha),
            threshold=threshold,
            layer_by_layer=layer_by_layer,
            mem_detach=False,
        )
        self.maxpool = torch.nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.proj_conv1 = nn.Conv2d(
            embed_dims // 8,
            embed_dims // 4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.proj_bn1 = nn.BatchNorm2d(embed_dims // 4)
        self.proj_lif1 = node(
            step=step,
            tau=tau,
            act_func=act_func(alpha=alpha),
            threshold=threshold,
            layer_by_layer=layer_by_layer,
            mem_detach=False,
        )
        self.maxpool1 = torch.nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.proj_conv2 = nn.Conv2d(
            embed_dims // 4,
            embed_dims // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.proj_bn2 = nn.BatchNorm2d(embed_dims // 2)
        self.proj_lif2 = node(
            step=step,
            tau=tau,
            act_func=act_func(alpha=alpha),
            threshold=threshold,
            layer_by_layer=layer_by_layer,
            mem_detach=False,
        )

        self.maxpool2 = torch.nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.proj_conv3 = nn.Conv2d(
            embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_lif3 = node(
            step=step,
            tau=tau,
            act_func=act_func(alpha=alpha),
            threshold=threshold,
            layer_by_layer=layer_by_layer,
            mem_detach=False,
        )

        self.maxpool3 = torch.nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.rpe_conv = nn.Conv2d(
            embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif = node(
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
        x = self.proj_conv(x)  # have some fire value
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

        x_feat = x
        x = self.rpe_conv(x)
        x = self.rpe_bn(x)
        x = self.rpe_lif(x)
        x = x + x_feat

        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)  # x.shape: (TB,C,H//16,W//16)
'''

class SPS(BR_BaseModule):
    def __init__(self, img_size_h=512, img_size_w=512, patch_size=32, in_channels=2, step=4, embed_dim=512,):
        super().__init__(encode_type='direct',step=step, embed_dim=embed_dim, layer_by_layer=True)
        # self.image_size = [img_size_h, img_size_w]
        self.patch_size = patch_size
        self.C = in_channels
        self.img_h = img_size_h
        self.img_w = img_size_w
        self.proj_conv = nn.Conv2d(in_channels, embed_dim//8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dim//8)
        self.proj_lif = LIFNode(step=step, tau=2., act_func=SigmoidGrad, threshold=1.,
                               layer_by_layer=True, mem_detach=False)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv1 = nn.Conv2d(embed_dim//8, embed_dim//4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dim//4)
        self.proj_lif1 = LIFNode(step=step, tau=2., act_func=SigmoidGrad, threshold=1.,
                               layer_by_layer=True, mem_detach=False)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv2 = nn.Conv2d(embed_dim//4, embed_dim//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dim//2)
        self.proj_lif2 = LIFNode(step=step, tau=2., act_func=SigmoidGrad, threshold=1.,
                               layer_by_layer=True, mem_detach=False)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv3 = nn.Conv2d(embed_dim//2, embed_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dim)
        self.proj_lif3 = LIFNode(step=step, tau=2., act_func=SigmoidGrad, threshold=1.,
                               layer_by_layer=True, mem_detach=False)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        # 512 x 512 太大 多加一次MP
        self.maxpool4 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.rpe_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dim)
        self.rpe_lif =LIFNode(step=step, tau=2., act_func=SigmoidGrad, threshold=1.,
                               layer_by_layer=True, mem_detach=False)
        self.dim = embed_dim
    def forward(self, x):
        self.reset()

        TB, C, H, W = x.shape
        x = self.proj_conv(x) # have some fire value
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

        x = self.maxpool4(x) # T, B, -1, H//self.patch_size, W//self.patch_size

        x_feat = x
        x = self.rpe_conv(x)
        x = self.rpe_bn(x)
        x = self.rpe_lif(x)
        x = x + x_feat
        if H==512:
            x = x.reshape(TB, self.dim, H // self.patch_size ,-1)
        else:
            x = x.reshape(TB, self.dim, -1 ,W // self.patch_size)

        return x,(x.shape[2],x.shape[3])



@MODELS.register_module()
class SEMM(BaseModule):
    def __init__(
        self,
        T=4,
        img_size_h=128,
        img_size_w=128,
        patch_size=16,
        in_channels=2,
        num_classes=11,
        embed_dim=64,
        expert_dim=96,
        mlp_ratios=[4, 4, 4],
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=[6, 8, 6],
        sr_ratios=[8, 4, 2],
        num_expert=4,
        layer_by_layer=True,
        encode_type="direct",
        node=LIFNode,
        tau=2.0,
        act_func=SigmoidGrad,
        alpha=4.0,
        threshold=1.0,
        num_heads=0,
        decode_mode=None,init_cfg=None,**kwargs
    ):

        super().__init__(init_cfg=init_cfg)
        self.T = T
        self.num_classes = num_classes
        self.depths = depths

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depths)
        ]  # stochastic depth decay rule

        '''
        patch_embed = SPS(
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dim,
            step=T,
            layer_by_layer=layer_by_layer,
            node=node,
            tau=tau,
            act_func=act_func,
            alpha=alpha,
            threshold=threshold,
            encode_type=encode_type,
        )
        '''

        patch_embed = SPS(step=T,img_size_h=img_size_h,img_size_w=img_size_w,
                          patch_size=patch_size,in_channels=in_channels,embed_dim=embed_dim)

        block = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    expert_dim=expert_dim,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,
                    num_expert=num_expert,
                    encode_type=encode_type,
                    step=T,
                    layer_by_layer=layer_by_layer,
                    node=node,
                    tau=tau,
                    act_func=act_func,
                    alpha=alpha,
                    threshold=threshold,
                )
                for j in range(depths)
            ]
        )

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", block)

        # classification head
        """
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )
        """
        self.apply(self._init_weights)

        self.pool=nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))

    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return (
                F.interpolate(
                    pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(
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

    def init_weights(self):
        # logger = MMlogger.get_current_instance()
        if self.init_cfg is None:
            print_log(f'No pre-trained weights for '
                      f'{self.__class__.__name__}, '
                      f'training start from scratch')
            # self.apply(self._init_weights)

            print_log("init_weighting.....")
            self.apply(self._init_weights)
            print_log("Time step: {:}".format(self.T))
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], logger=None, map_location='cpu')
            # state_dict = self.state_dict()
            # import pdb; pdb.set_trace()
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt
            # import pdb; pdb.set_trace()
            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                # 使用mmseg保存的checkpoint中包含backbone, neck, decode_head三个部分
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v
                else:
                    state_dict[k] = v
            # import pdb; pdb.set_trace()
            self.load_state_dict(state_dict, strict=False)
            print_log("--------------Successfully load checkpoint for BACKNONE------------")
            print_log("Time step: {:}".format(self.T))

    def forward_features(self, x):

        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")

        x, (H, W) = patch_embed(x)
        attn = None
        for blk in block:
            x = blk(x, attn)  # TB,C,H,W
        return x 

    def forward(self, x):
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1).flatten(0,1)
        x = self.forward_features(x)
        #x = self.pool(x)
        TB,c,h,w = x.shape
        x = x.reshape(self.T,-1,c,h,w)
        #print(x.shape)
        return x
        # T = 4
        # x = (x.unsqueeze(0)).repeat(T, 1, 1, 1, 1)
        # x = self.forward_features(x)
        # x = self.head(x.mean(0))
        # return x


@register_model
def spikf_semm_imagenet(pretrained=False, **kwargs):
    model = Spikformer(
        img_size_h=kwargs.get("img_size", 224),
        img_size_w=kwargs.get("img_size", 224),
        patch_size=kwargs.get("patch_size", 16),
        in_channels=kwargs.get("in_channels", 3),
        num_classes=kwargs.get("num_classes", 1000),
        embed_dims=kwargs.get("embed_dim", 384),
        expert_dim=kwargs.get("expert_dim", 96),
        mlp_ratios=kwargs.get("mlp_ratio", 4),
        qkv_bias=kwargs.get("qkv_bias", False),
        # qk_scale=None,
        # drop_rate=kwargs.get("drop_rate", 0),
        # attn_drop_rate=0.0,
        # drop_path_rate=kwargs.get("drop_path_rate", 0.1),
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=kwargs.get("depths", 8),
        sr_ratios=kwargs.get("sr_ratios", 1),
        num_expert=kwargs.get("num_expert", 4),
        step=kwargs.get("step", 4),
        tau=kwargs.get("tau", 2.0),
        threshold=kwargs.get("threshold", 1.0),
        node=kwargs.get("node", LIFNode),
        act_func=kwargs.get("act_func", SigmoidGrad),
        alpha=kwargs.get("alpha", 4.0),
    )
    model.default_cfg = _cfg()
    return model
