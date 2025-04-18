import torch
import torch.nn as nn
from braincog.base.strategy.surrogate import SigmoidGrad
# from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from mmengine.model import BaseModule
from mmseg.registry import MODELS
from mmengine.logging import print_log
from mmengine.runner import CheckpointLoader

from collections import OrderedDict

from braincog.model_zoo.base_module import BaseModule as BR_BaseModule
from braincog.base.node import LIFNode



class MLP(BR_BaseModule):
    def __init__(self, in_features, hidden_features=None, out_features=None, step=4, embed_dim=512,drop=0.):
        super().__init__(encode_type='direct',step=step, embed_dim=embed_dim, layer_by_layer=True)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm2d(hidden_features)
        self.fc1_lif = LIFNode(step=step, tau=2., act_func=SigmoidGrad, threshold=1.,
                               layer_by_layer=True, mem_detach=False)

        self.fc2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm2d(out_features)
        self.fc2_lif = LIFNode(step=step, tau=2., act_func=SigmoidGrad, threshold=1.,
                               layer_by_layer=True, mem_detach=False)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        self.reset()

        T, B, C, H, W = x.shape

        x = self.fc1_conv(x.flatten(0,1))
        x = self.fc1_bn(x)
        x = self.fc1_lif(x)

        x = self.fc2_conv(x)
        x = self.fc2_bn(x)
        x = self.fc2_lif(x).reshape(T,B,C,H,W).contiguous()

        return x


class SSA(BR_BaseModule):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0., sr_ratio=1,step=4, embed_dim=512,):
        super().__init__(encode_type='direct',step=step, embed_dim=embed_dim, layer_by_layer=True)
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = LIFNode(step=step, tau=2., act_func=SigmoidGrad, threshold=1.,
                               layer_by_layer=True, mem_detach=False)

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = LIFNode(step=step, tau=2., act_func=SigmoidGrad, threshold=1.,
                               layer_by_layer=True, mem_detach=False)

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = LIFNode(step=step, tau=2., act_func=SigmoidGrad, threshold=1.,
                               layer_by_layer=True, mem_detach=False)

        self.attn_lif = LIFNode(step=step, tau=2., act_func=SigmoidGrad, threshold=1.,
                               layer_by_layer=True, mem_detach=False)

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)

        self.proj_lif = LIFNode(step=step, tau=2., act_func=SigmoidGrad, threshold=1.,
                               layer_by_layer=True, mem_detach=False)

    def forward(self, x, res_attn):
        self.reset()

        T,B,C,H,W = x.shape
        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1) # TB C N
        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out)
        q_conv_out = self.q_lif(q_conv_out).reshape(T,B,C,N).contiguous()
        q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out)
        k_conv_out = self.k_lif(k_conv_out).reshape(T,B,C,N).contiguous()
        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out)
        v_conv_out = self.v_lif(v_conv_out).reshape(T,B,C,N).contiguous()
        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        x = k.transpose(-2,-1) @ v
        x = (q @ x) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = x.flatten(0, 1)
        x = self.attn_lif(x)
        x = self.proj_lif(self.proj_bn(self.proj_conv(x))).reshape(T,B,C,H,W).contiguous()
        return x, v

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1, step=4,):
        super().__init__()

        self.attn = SSA(dim, step=step,num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, step=step)

    def forward(self, x, res_attn):
        x_attn, attn = (self.attn(x, res_attn))
        x = x + x_attn
        x = x + (self.mlp((x)))

        return x, attn


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
    def forward(self, x):
        self.reset()

        T, B, C, H, W = x.shape
        x = self.proj_conv(x.flatten(0, 1)) # have some fire value
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
        x = x.reshape(T, B, -1, self.img_h // self.patch_size , self.img_w // self.patch_size)

        return x

@MODELS.register_module()
class Spikformer(BaseModule):
    def __init__(self,
                 img_size_h=512, img_size_w=512, patch_size=32, in_channels=2, num_classes=11,
                 embed_dim=512, num_heads=8, mlp_ratios=4, qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=8, sr_ratios=1, init_cfg=None, T=4,**kwargs):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.depths = depths
        self.step = T

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        patch_embed = SPS(step=self.step,img_size_h=img_size_h,img_size_w=img_size_w,
                          patch_size=patch_size,in_channels=in_channels,embed_dim=embed_dim)

        block = nn.ModuleList([Block( step=self.step,
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(depths)])

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", block)

    def forward_features(self, x):
        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")

        x = patch_embed(x)
        attn = None
        for blk in block:
            x, attn = blk(x, attn)
        return x # T B C H W


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
            print_log("Time step: {:}".format(self.step))
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
            print_log("Time step: {:}".format(self.step))

    def forward(self, x):
        # expand time dimension
        x = (x.unsqueeze(0)).repeat(self.step, 1, 1, 1, 1)
        # extract features
        feat = self.forward_features(x)
        # average over time dimension and return as list for neck
        return feat
