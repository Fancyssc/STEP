from timm.models import register_model
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import _cfg
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
# from utils.node import *
# from utils.layers import *
import torch.nn as nn

# 本地调用包出错是时可以解注释如下代码
# from cls.utils.layers import *
# from cls.utils.node import *


class SPSv1(SPS):
    """
    params keep the same as model for CIFAR10
    """
    def __init__(self, step=4, encode_type='direct', img_h=32, img_w=32, patch_size=4, in_channels=3,
                 embed_dims=384,node=LIFNode,tau=2.0,act_func=Sigmoid_Grad,threshold=1.0):
        super().__init__(step=step, encode_type= encode_type, img_h=img_h , img_w=img_w , patch_size= patch_size, in_channels=in_channels,
                 embed_dims=embed_dims, node=node, tau= tau, act_func=act_func, threshold=threshold )

        self.maxpool  = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

    def forward(self, x):
        self.reset()
        T, B, C, H, W = x.shape
        assert self.embed_dims % 16 == 0, 'embed_dims must be divisible by 16 for spikformer_imgnet'

        x = self.ConvBnSn(x,self.proj_conv,self.proj_bn,self.proj_lif)
        x = self.maxpool(x.flatten(0, 1)).reshape(T, B, -1, H // 2, W // 2).contiguous()

        x = self.ConvBnSn(x,self.proj_conv1,self.proj_bn1,self.proj_lif1)
        x = self.maxpool1(x.flatten(0, 1)).reshape(T, B, -1, H // 4, W // 4).contiguous()

        x = self.ConvBnSn(x,self.proj_conv2,self.proj_bn2,self.proj_lif2)
        x = self.maxpool2(x.flatten(0,1)).reshape(T, B, -1, H // 8, W // 8).contiguous()

        x = self.ConvBnSn(x,self.proj_conv3,self.proj_bn3,self.proj_lif3)
        x = self.maxpool3(x.flatten(0,1)).reshape(T, B, -1, H // 16, W // 16).contiguous()

        x_feat = x.reshape(T, B, -1, H // 16, W // 16).contiguous()
        x = self.ConvBnSn(x,self.rpe_conv,self.rpe_bn,self.rpe_lif)

        x = x + x_feat
        return x #  T B C H W


class SSAv1(SSA):
    def __init__(self,embed_dim, step=4,encode_type='direct',num_heads=12,scale=0.125,attn_drop=0.,node=LIFNode,tau=2.0,act_func=Sigmoid_Grad,threshold=1.0):
        super().__init__(embed_dim=embed_dim, step=step, encode_type=encode_type, num_heads=num_heads, scale=scale, attn_drop=attn_drop, node=LIFNode, tau=tau, act_func=act_func, threshold=threshold)
        #To keep code simple
        self.q_linear = nn.Conv1d(embed_dim, embed_dim, kernel_size=1, stride=1,bias=False)

        self.k_linear = nn.Conv1d(embed_dim, embed_dim, kernel_size=1, stride=1,bias=False)

        self.v_linear = nn.Conv1d(embed_dim, embed_dim, kernel_size=1, stride=1,bias=False)

        self.proj_linear = nn.Conv1d(embed_dim, embed_dim, kernel_size=1, stride=1,bias=False)


    def forward(self, x):
        T, B, C, H, W = x.shape
        self.reset()
        x = x.flatten(-2,-1).transpose(-2,-1)  # T B N C
        q = self.qkv(x, self.q_linear, self.q_bn, self.q_lif, self.num_heads)
        k = self.qkv(x, self.k_linear, self.k_bn, self.k_lif, self.num_heads)
        v = self.qkv(x, self.v_linear, self.v_bn, self.v_lif, self.num_heads)

        x = self.attn_cal(q, k, v)

        return x.transpose(-2,-1).reshape(T, B, C, H, W) # T B C H W


class MLPv1(MLP):
    def __init__(self, in_features, step=4, encode_type='direct', mlp_ratio = 4.0, out_features=None,mlp_drop=0.,node=LIFNode,tau=2.0,act_func=Sigmoid_Grad,threshold=1.0):
        super().__init__(in_features=in_features, step=step, mlp_ratio=mlp_ratio,mlp_drop=mlp_drop,encode_type=encode_type,out_features=out_features, node=node, tau=tau, act_func=act_func, threshold=threshold)

        self.fc1_linear = nn.Conv2d(in_features, self.hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm2d(self.hidden_features)

        self.fc2_linear = nn.Conv2d(self.hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm2d(self.out_features)

    def forward(self, x):
        self.reset()

        T, B, C, H, W = x.shape

        x = self.fc1_linear(x.flatten(0, 1)) # TB C H W
        x = self.fc1_bn(x).reshape(T, B, -1, H, W).contiguous()
        x = self.fc1_lif(x.flatten(0, 1)).reshape(T, B, -1, H, W).contiguous()

        if self.mlp_drop > 0 :
            x = self.MLP_drop(x)
        x = self.fc2_linear(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, -1, H, W).contiguous()
        x = self.fc2_lif(x.flatten(0, 1)).reshape(T, B, -1, H, W).contiguous()
        return x # T B C H W

# Spikformer block
class Spikf_Block_s(nn.Module):
    def __init__(self, embed_dim=384, num_heads=12, step=4, mlp_ratio=4., scale=0., attn_drop=0.,mlp_drop=0.,node=LIFNode,tau=2.0,act_func=Sigmoid_Grad,threshold=1.0,drop_path=0.):
        super().__init__()
        self.attn = SSAv1(embed_dim, step=step, num_heads=num_heads,attn_drop=attn_drop, scale=scale,node=node,tau=tau,act_func=act_func,threshold=threshold)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp = MLPv1(step=step,in_features=embed_dim,mlp_ratio=mlp_ratio,out_features=embed_dim,mlp_drop=mlp_drop,node=node,tau=tau,act_func=act_func,threshold=threshold)
    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x

class cls_head(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super(cls_head, self).__init__()
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.head(x)
        return x



class SpikformerV1(nn.Module):
    def __init__(self, step=4,img_size=32, patch_size=4, in_channels=3, num_classes=10,
                 embed_dim=384, num_heads=12, mlp_ratio=4, scale=0.125, mlp_drop=0., attn_drop=0.,
                 depths=4, node=LIFNode, tau=2.0, act_func=Sigmoid_Grad, threshold=1.0, drop_path_rate=0.):
        super().__init__()
        self.step = step  # time step
        self.num_classes = num_classes
        self.depths = depths

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        patch_embed = SPSv1(step=step,
                            img_h=img_size,
                            img_w=img_size,
                            patch_size=patch_size,
                            in_channels=in_channels,
                            embed_dims=embed_dim,
                            node=node, tau=tau,
                            act_func=act_func, threshold=threshold)

        block = nn.ModuleList([Spikf_Block_s(step=step, embed_dim=embed_dim,
                                           num_heads=num_heads, mlp_ratio=mlp_ratio,
                                           scale=scale, mlp_drop=mlp_drop, attn_drop=attn_drop,
                                           node=node, tau=2.0, act_func=act_func, threshold=threshold,
                                           drop_path=dpr[j])
                               for j in range(depths)])

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", block)
        # classification head
        self.head = cls_head(embed_dim, num_classes)
        self.apply(self._init_weights)

    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).contiguous().permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).contiguous().permute(0, 2, 1)

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
            x = blk(x) # T B C H W
        return x.flatten(-2, -1).mean(3)

    def forward(self, x):

        x = (x.unsqueeze(0)).repeat(self.step, 1, 1, 1, 1)
        x = self.forward_features(x)
        x = self.head(x.mean(0))
        return x

#### models for static datasets
@register_model
def spikformer_imgnet(pretrained=False,**kwargs):
    model = SpikformerV1(
        step=kwargs.get('step', 4),
        img_size=kwargs.get('img_size', 224),
        patch_size=kwargs.get('patch_size', 16),
        in_channels=kwargs.get('in_channels', 3),
        num_classes=kwargs.get('num_classes', 1000),
        embed_dim=kwargs.get('embed_dim', 512),
        num_heads=kwargs.get('num_heads', 16),
        mlp_ratio=kwargs.get('mlp_ratio', 4),
        scale=kwargs.get('attn_scale', 0.125),
        mlp_drop=kwargs.get('mlp_drop', 0.0),
        attn_drop=kwargs.get('attn_drop', 0.0),
        depths=kwargs.get('depths', 4),
        tau=kwargs.get('tau', 2.0),
        threshold=kwargs.get('threshold', 1.0),
        node=kwargs.get('node', LIFNode),
        act_func=kwargs.get('act_func', Sigmoid_Grad),
    )
    model.default_cfg = _cfg()
    return model

