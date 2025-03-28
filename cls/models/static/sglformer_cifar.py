from braincog.model_zoo.base_module import BaseModule
from einops import rearrange
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from ..utils.node import *
from braincog.base.strategy.surrogate import *

"""
SGLFormer (Frontiers in Neuroscience 2024)
"""

class MLP(BaseModule):
    def __init__(self,  in_features, step=4, encode_type='direct', mlp_ratio = 4.0, out_features=None,mlp_drop=0.,
                 node=LIFNode,tau=2.0,threshold=1.0,act_func=SigmoidGrad, alpha=4.,layer_by_layer=True):
        super().__init__(step=step, encode_type=encode_type,layer_by_layer=layer_by_layer)
        self.out_features = out_features or in_features
        self.hidden_features = int(in_features * mlp_ratio)
        self.mlp_drop = mlp_drop

        self.fc1_conv = nn.Conv2d(in_features, self.hidden_features,1,1,bias=False)
        self.fc1_bn = nn.BatchNorm2d(self.hidden_features)
        self.fc1_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,
                            layer_by_layer=layer_by_layer, mem_detach=False)

        self.fc2_conv = nn.Conv2d(self.hidden_features, self.out_features,1, 1, bias=False)
        self.fc2_bn = nn.BatchNorm2d(self.out_features)
        self.fc2_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,
                            layer_by_layer=layer_by_layer, mem_detach=False)


    def forward(self, x):
        self.reset()

        x = self.fc1_lif(self.fc1_bn(self.fc1_conv(x)))
        x = self.fc2_lif(self.fc2_bn(self.fc2_conv(x)))
        return x  # TB C H W

class GlobalSSA(BaseModule):
    def __init__(self, embed_dim, step=4,encode_type='direct',num_heads=8,attn_scale=0.125,attn_drop=0.,
                 node=LIFNode,tau=2.0,threshold=1.0,act_func=SigmoidGrad, alpha=4.0,layer_by_layer=True):
        super().__init__(step=step, encode_type=encode_type,layer_by_layer=layer_by_layer)
        assert embed_dim % num_heads == 0, f"dim {embed_dim} should be divided by num_heads {num_heads}."
        self.step = step
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scale = attn_scale


        self.q_conv = nn.Conv2d(embed_dim, embed_dim, 1, 1, bias=False)
        self.q_bn = nn.BatchNorm2d(embed_dim)
        self.q_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,
                         layer_by_layer=layer_by_layer, mem_detach=False)

        self.k_conv = nn.Conv2d(embed_dim, embed_dim, 1, 1, bias=False)
        self.k_bn = nn.BatchNorm2d(embed_dim)
        self.k_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,
                          layer_by_layer=layer_by_layer, mem_detach=False)

        self.v_conv = nn.Conv2d(embed_dim, embed_dim, 1, 1, bias=False)
        self.v_bn = nn.BatchNorm2d(embed_dim)
        self.v_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,
                          layer_by_layer=layer_by_layer, mem_detach=False)

        self.attn_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=0.5,
                          layer_by_layer=layer_by_layer, mem_detach=False)

        self.pro_conv = nn.Conv2d(embed_dim, embed_dim, 1, 1, bias=False)
        self.pro_bn = nn.BatchNorm2d(embed_dim)
        self.pro_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,
                          layer_by_layer=layer_by_layer, mem_detach=False)

    def forward(self, x):
        self.reset()
        TB, Co, Ho, Wo = x.shape

        # align for original Partition method
        x_for_ssa = x.reshape(self.step, -1, Co, Ho, Wo)
        T, B, C, H, W = x_for_ssa.shape

        q_conv_out = self.q_conv(x_for_ssa.flatten(0, 1))
        q_conv_out = self.q_bn(q_conv_out)
        q_conv_out = self.q_lif(q_conv_out).reshape(T, B, C, H, W).flatten(3)
        q = q_conv_out.transpose(-1, -2).reshape(T, B, -1, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)

        k_conv_out = self.q_conv(x_for_ssa.flatten(0, 1))
        k_conv_out = self.q_bn(k_conv_out)
        k_conv_out = self.q_lif(k_conv_out).reshape(T, B, C, H, W).flatten(3)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, -1, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)

        v_conv_out = self.q_conv(x_for_ssa.flatten(0, 1))
        v_conv_out = self.q_bn(v_conv_out)
        v_conv_out = self.q_lif(v_conv_out).reshape(T, B, C, H, W).flatten(3)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, -1, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)

        attn = (q @ k.transpose(-2, -1))
        x = (attn @ v) * self.scale
        x = x.flatten(0, 1)
        x = self.attn_lif(x)

        x = x.transpose(-2, -1).reshape(T, B, C, H, W)
        x = self.pro_conv(x.flatten(0, 1))
        x = self.pro_bn(x)
        x = self.pro_lif(x)
        return x



class Partition(nn.Module):
    def __init__(self, num_hw):
        super().__init__()
        self.num_hw = num_hw

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = x.reshape(T, B, C, H // self.num_hw, self.num_hw, W // self.num_hw, self.num_hw)
        x = x.permute(0, 1, 4, 6, 2, 3, 5)
        x = x.reshape(T, -1, C, H // self.num_hw, W // self.num_hw)
        return x


class Integration(nn.Module):
    def __init__(self, num_hw):
        super().__init__()
        self.num_hw = num_hw

    def forward(self, x):
        T, Bnn, C, Hn, Wn = x.shape
        x = x.reshape(T, -1, self.num_hw, self.num_hw, C, Hn, Wn)
        x = x.permute(0, 1, 4, 5, 2, 6, 3)
        x = x.reshape(T, -1, C, int(Hn * self.num_hw), int(Wn * self.num_hw))
        return x


class LocalSSA(BaseModule):
    def __init__(self, embed_dim, step=4,encode_type='direct',num_heads=8,num_hw=2,attn_scale=0.125,attn_drop=0.,
                 node=LIFNode,tau=2.0,threshold=1.0,act_func=SigmoidGrad, alpha=4.0,layer_by_layer=True):
        super().__init__(step=step, encode_type=encode_type,layer_by_layer=layer_by_layer)
        assert embed_dim % num_heads == 0, f"dim {embed_dim} should be divided by num_heads {num_heads}."
        self.step = step
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scale = attn_scale

        self.partition = Partition(num_hw)
        self.integration = Integration(num_hw)

        self.q_conv = nn.Conv2d(embed_dim, embed_dim, 1, 1, bias=False)
        self.q_bn = nn.BatchNorm2d(embed_dim)
        self.q_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,
                         layer_by_layer=layer_by_layer, mem_detach=False)

        self.k_conv = nn.Conv2d(embed_dim, embed_dim, 1, 1, bias=False)
        self.k_bn = nn.BatchNorm2d(embed_dim)
        self.k_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,
                          layer_by_layer=layer_by_layer, mem_detach=False)

        self.v_conv = nn.Conv2d(embed_dim, embed_dim, 1, 1, bias=False)
        self.v_bn = nn.BatchNorm2d(embed_dim)
        self.v_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,
                          layer_by_layer=layer_by_layer, mem_detach=False)

        self.attn_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=0.5,
                          layer_by_layer=layer_by_layer, mem_detach=False)

        self.pro_conv = nn.Conv2d(embed_dim, embed_dim, 1, 1, bias=False)
        self.pro_bn = nn.BatchNorm2d(embed_dim)
        self.pro_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,
                          layer_by_layer=layer_by_layer, mem_detach=False)

    def forward(self, x):
        self.reset()

        TB, Co, Ho, Wo = x.shape

        # align for original Partition method
        x_for_ssa = x.reshape(self.step, -1, Co, Ho, Wo)
        x_for_ssa = self.partition(x_for_ssa)
        T, B, C, H, W = x_for_ssa.shape

        q_conv_out = self.q_conv(x_for_ssa.flatten(0, 1))
        q_conv_out = self.q_bn(q_conv_out)
        q_conv_out = self.q_lif(q_conv_out).reshape(T, B, C, H, W).flatten(3)
        q = q_conv_out.transpose(-1, -2).reshape(T, B, -1, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)

        k_conv_out = self.q_conv(x_for_ssa.flatten(0, 1))
        k_conv_out = self.q_bn(k_conv_out)
        k_conv_out = self.q_lif(k_conv_out).reshape(T, B, C, H, W).flatten(3)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, -1, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)

        v_conv_out = self.q_conv(x_for_ssa.flatten(0, 1))
        v_conv_out = self.q_bn(v_conv_out)
        v_conv_out = self.q_lif(v_conv_out).reshape(T, B, C, H, W).flatten(3)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, -1, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)

        attn = (q @ k.transpose(-2, -1))
        x = (attn @ v) * self.scale
        x = self.attn_lif(x.flatten(0, 1)).reshape(T, B, -1, self.num_heads, C // self.num_heads)

        x = x.transpose(3, 4).reshape(T, B, C, -1).reshape(T, B, C, H, W)
        x = self.pro_conv(x.flatten(0, 1))
        x = self.pro_bn(x)
        x = self.pro_lif(x).reshape(T, B, C, H, W)
        x = self.integration(x).flatten(0, 1)
        return x


class Stem(BaseModule):
    def __init__(self, step=4, encode_type='direct', in_channels=3, embed_dim=384,node=LIFNode,tau=2.0,threshold=1.0,act_func=SigmoidGrad, alpha=4.0,layer_by_layer=True):
        super().__init__(step=step, encode_type=encode_type)

        self.stem_conv = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim//6, kernel_size=3, stride=1, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(embed_dim//6)
        self.stem_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,  layer_by_layer=layer_by_layer, mem_detach=False)

    def forward(self, x):
        self.reset()

        TB, C, H, W = x.shape

        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = self.stem_lif(x)

        return x # TB, C, H, W


class Tokenizer(BaseModule):
    def __init__(self, step=4, encode_type='direct', embed_dim=384,node=LIFNode,tau=2.0,threshold=1.0,act_func=SigmoidGrad, alpha=4.0,layer_by_layer=True):
        super().__init__(step=step, encode_type=encode_type,layer_by_layer=layer_by_layer)

        self.conv1 = nn.Conv2d(embed_dim // 6, embed_dim//4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(embed_dim // 4)
        self.lif1 = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold, layer_by_layer=layer_by_layer, mem_detach=False)

        self.conv2 = nn.Conv2d(embed_dim // 4, embed_dim//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(embed_dim // 2)
        self.lif2 = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold, layer_by_layer=layer_by_layer, mem_detach=False)

        self.conv3 = nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(embed_dim)
        self.lif3 = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold, layer_by_layer=layer_by_layer, mem_detach=False)

        self.conv4 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(embed_dim)
        self.lif4 = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold, layer_by_layer=layer_by_layer, mem_detach=False)

    def forward(self, x):
        self.reset()

        TB, C, H, W = x.shape
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lif1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lif2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.lif3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.lif4(x)

        return x # TB C' H W


class LocalBlock(BaseModule):
    def __init__(self, step=4, encode_type='direct', embed_dim=384,node=LIFNode,tau=2.0,threshold=1.0,act_func=SigmoidGrad, alpha=4.0,layer_by_layer=True):
        super().__init__(step=step, encode_type=encode_type, layer_by_layer=layer_by_layer)
        self.dw_dim = int(4 * embed_dim)

        self.dw_conv1 = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, groups=embed_dim, bias=False)
        self.bn1 = nn.BatchNorm2d(embed_dim)
        self.lif1 = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold, layer_by_layer=layer_by_layer, mem_detach=False)

        self.conv1 = nn.Conv2d(embed_dim, self.dw_dim,1,1,bias=False)
        self.bn2 = nn.BatchNorm2d(self.dw_dim)
        self.lif2 = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold, layer_by_layer=layer_by_layer, mem_detach=False)

        self.dw_conv2 = nn.Conv2d(self.dw_dim,self.dw_dim,3,1,1,groups=self.dw_dim,bias=False)
        self.bn3 = nn.BatchNorm2d(self.dw_dim)
        self.lif3 = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold, layer_by_layer=layer_by_layer, mem_detach=False)

        self.conv2 = nn.Conv2d(self.dw_dim, embed_dim,1,1,bias=False)
        self.bn4 = nn.BatchNorm2d(embed_dim)
        self.lif4 = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold, layer_by_layer=layer_by_layer, mem_detach=False)

    def forward(self, x):
        self.reset()

        TB, C, H, W = x.shape
        x_for_conv = x

        x_for_conv = self.dw_conv1(x_for_conv)
        x_for_conv = self.bn1(x_for_conv)
        x_for_conv = self.lif1(x_for_conv)

        x = x + x_for_conv
        x_res = x

        x = self.lif2(self.bn2(self.conv1(x)))
        x = self.lif3(self.bn3(self.dw_conv2(x)))
        x = self.lif4(self.bn4(self.conv2(x)))
        x = x + x_res
        return x


class SpikingTransformer(nn.Module):
    def __init__(self, embed_dim, step=4, num_heads=8,num_hw=2,attn_scale=0.125,attn_drop=0.,
                 node=LIFNode,tau=2.0,threshold=1.0,act_func=SigmoidGrad, alpha=4.0,layer_by_layer=True,mlp_ratio=4.0):
        super().__init__()
        self.loc = LocalBlock(step=step, embed_dim=embed_dim, node=node, tau=tau,
                              threshold=threshold, act_func=act_func, alpha=alpha, layer_by_layer=layer_by_layer)

        self.ssa1 = LocalSSA(embed_dim, step=step,num_heads=num_heads,num_hw=num_hw,attn_scale=attn_scale,attn_drop=attn_drop,
                 node=node,tau=tau,threshold=threshold,act_func=act_func, alpha=alpha,layer_by_layer=layer_by_layer)
        self.mlp1 = MLP(in_features=embed_dim, step=step, mlp_ratio=mlp_ratio, node=node,
                        tau=tau,threshold=threshold,act_func=act_func, alpha=alpha,layer_by_layer=layer_by_layer)

        self.ssa2 = GlobalSSA(embed_dim, step=step,num_heads=num_heads,attn_scale=attn_scale,attn_drop=attn_drop,
                 node=node,tau=tau,threshold=threshold,act_func=act_func, alpha=alpha,layer_by_layer=layer_by_layer)
        self.mlp2 = MLP(in_features=embed_dim, step=step, mlp_ratio=mlp_ratio, node=node,
                        tau=tau,threshold=threshold,act_func=act_func, alpha=alpha,layer_by_layer=layer_by_layer)
        self.ssa3 = GlobalSSA(embed_dim, step=step,num_heads=num_heads,attn_scale=attn_scale,attn_drop=attn_drop,
                 node=node,tau=tau,threshold=threshold,act_func=act_func, alpha=alpha,layer_by_layer=layer_by_layer)

        self.mlp3 = MLP(in_features=embed_dim, step=step, mlp_ratio=mlp_ratio, node=node,
                        tau=tau,threshold=threshold,act_func=act_func, alpha=alpha,layer_by_layer=layer_by_layer)

    def forward(self, x):
        x = x + self.loc(x)
        x = x + self.ssa1(x)
        x = x + self.mlp1(x)
        x = x + self.ssa2(x)
        x = x + self.mlp2(x)
        x = x + self.ssa3(x)
        x = x + self.mlp3(x)
        return x


class vit_snn(BaseModule):
    def __init__(self,
                 step=4, img_size=32, patch_size=4, in_channels=3, num_classes=10, attn_scale=0.125,
                 embed_dim=384, num_heads=12, mlp_ratio=4, mlp_drop=0., attn_drop=0.,num_hw=2,
                 depths=4, node=LIFNode, tau=2.0, threshold=1.0, act_func=SigmoidGrad, alpha=4., layer_by_layer=True
                 ):
        super().__init__(step=step,encode_type='direct',layer_by_layer=layer_by_layer)
        self.num_classes = num_classes
        self.depths = depths
        self.T = step
        self.num_patches = 64

        self.stem = Stem(step=step, in_channels=in_channels, embed_dim=embed_dim,
                         node=node,tau=tau,threshold=threshold,act_func=act_func, alpha=alpha,layer_by_layer=layer_by_layer)
        self.tokenizer = Tokenizer(step=step, embed_dim=embed_dim,
                         node=node,tau=tau,threshold=threshold,act_func=act_func, alpha=alpha,layer_by_layer=layer_by_layer)
        self.blocks = nn.Sequential(*[SpikingTransformer(embed_dim, step=step,num_heads=num_heads,mlp_ratio=mlp_ratio,
                                                         num_hw=num_hw,attn_scale=attn_scale,attn_drop=attn_drop,node=node,tau=tau,
                                                         threshold=threshold,act_func=act_func, alpha=alpha,
                                                         layer_by_layer=layer_by_layer)
                                      for i in range(depths)])

        # classification head
        self.full_size_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=(self.num_patches, self.T), stride=1,
                                        padding=0, groups=embed_dim)
        self.bn = nn.BatchNorm2d(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        B = B // self.step
        x = self.stem(x)
        x = self.tokenizer(x)
        x = self.blocks(x)
        x = rearrange(x,"(t b) c h w -> t b c h w", t=self.step, b=B)
        x = x.flatten(3).permute(1, 2, 3, 0)
        x = self.bn(self.full_size_conv(x)).squeeze(-1).mean(-1)
        x = x.reshape(B, -1)
        return x

    def forward(self, x):
        self.reset()
        x = self.encoder(x)
        x = self.forward_features(x)
        x = self.head(x)
        return x


@register_model
def sglformer_cifar(pretrained=False, **kwargs):
    model = vit_snn(
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
        depths=kwargs.get('depths', 1),
        tau=kwargs.get('tau', 2.0),
        threshold=kwargs.get('threshold', 1.0),
        node=kwargs.get('node', LIFNode),
        act_func=kwargs.get('act_func', SigmoidGrad),
        alpha=kwargs.get('alpha', 4.0)
    )
    model.default_cfg = _cfg()
    return model

