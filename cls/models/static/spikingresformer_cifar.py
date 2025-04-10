from braincog.model_zoo.base_module import BaseModule
from ..utils.node import *
from typing import Any, List, Mapping
from timm.models.registry import register_model

# class print_module(nn.Module):
#     def __init__(self, test=1):
#         super().__init__()
#         self.test = test
#
#     def forward(self, x):
#         print(x.shape)
#         return x


class GWFFN(BaseModule):
    def __init__(self, step=4, encode_type='direct', in_channels=3,num_conv=1,group_size=64, ratio=4,
                node=LIFNode, tau=2.0, threshold=1.0, act_func=SigmoidGrad, alpha=4.0, layer_by_layer=True):

        super().__init__(step=step, encode_type=encode_type,layer_by_layer=layer_by_layer)
        self.inner_channels = in_channels * ratio

        self.up_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,
                           layer_by_layer=layer_by_layer, mem_detach=False)
        self.up_conv = nn.Conv2d(in_channels, self.inner_channels, kernel_size=1, stride=1,
                                 padding=0,dilation=1,bias=False)
        self.up_bn = nn.BatchNorm2d(self.inner_channels)


        self.conv_module = nn.ModuleList()
        for i in range(num_conv):
            self.conv_module.append(
                nn.Sequential(
                    node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,
                         layer_by_layer=layer_by_layer, mem_detach=False),
                    nn.Conv2d(self.inner_channels, self.inner_channels, kernel_size=3, stride=1,
                              padding=1, groups=self.inner_channels // group_size, bias=False),
                    nn.BatchNorm2d(self.inner_channels),
                )
            )

        self.down_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,
                             layer_by_layer=layer_by_layer, mem_detach=False)
        self.down_conv = nn.Conv2d(self.inner_channels, in_channels, kernel_size=1, stride=1,
                                   padding=0,dilation=1,bias=False)
        self.down_bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        self.reset()

        x_feat_out = x.clone()

        # upsampling
        x = self.up_lif(x)
        x = self.up_conv(x)
        x = self.up_bn(x)

        x_feat_in = x.clone()
        for m in self.conv_module:
            x = m(x)
        x = x + x_feat_in

        # downsampling
        x = self.down_lif(x)
        x = self.down_conv(x)
        x = self.down_bn(x)

        return (x + x_feat_out)

class DSSA(BaseModule):
    """
    :param length is compulsory without default value
    """
    def __init__(self, length, step=4, encode_type='direct', patch_size=4,num_heads=12,
                 embed_dims=384, node=LIFNode,tau=2.0,threshold=1.0,act_func=SigmoidGrad, alpha=4.0,layer_by_layer=True):
        super().__init__(step=step,encode_type=encode_type,layer_by_layer=layer_by_layer)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.length = length

        self.init_firing_rate_x = False
        self.init_firing_rate_attn = False
        # tensors ignored by torch
        self.register_buffer('firing_rate_x', torch.zeros(1, 1, num_heads, 1, 1))
        self.register_buffer('firing_rate_attn', torch.zeros(1, 1, num_heads, 1, 1))

        self.momentum = 0.999
        self.in_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,
                           layer_by_layer=layer_by_layer, mem_detach=False)

        self.W = nn.Conv2d(embed_dims, embed_dims * 2, kernel_size=patch_size, stride=patch_size, padding=0,
                           dilation=1, bias=False)
        self.bn = nn.BatchNorm2d(embed_dims * 2)

        self.attn_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,
                           layer_by_layer=layer_by_layer, mem_detach=False)
        self.attn_out_lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,
                           layer_by_layer=layer_by_layer, mem_detach=False)

        self.proj_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj_norm = nn.BatchNorm2d(embed_dims)

    def forward(self, x):
        self.reset()
        input_shape = x.shape # TB C H W
        B = x.shape[0] // self.step # Batch_size should be int

        x_feat = x.clone()
        x = self.in_lif(x)

        y = self.W(x)
        y = self.bn(y)
        y = y.reshape(self.step, B, self.num_heads, 2 * self.embed_dims // self.num_heads, -1)
        y1, y2 = y[:, :, :, :self.embed_dims // self.num_heads, :], y[:, :, :, self.embed_dims // self.num_heads:, :]
        x = x.reshape(self.step, B, self.num_heads, self.embed_dims  // self.num_heads, -1)

        if self.training:
            firing_rate_x = x.detach().mean((0, 1, 3, 4), keepdim=True)
            if not self.init_firing_rate_x and torch.all(self.firing_rate_x == 0):
                self.firing_rate_x = firing_rate_x
            self.init_firing_rate_x = True
            self.firing_rate_x = self.firing_rate_x * self.momentum + firing_rate_x * (
                    1 - self.momentum)

        scale1 = 1. / torch.sqrt(self.firing_rate_x * (self.embed_dims // self.num_heads))
        attn = torch.matmul(y1.transpose(-1, -2), x)
        attn = attn * scale1

        attn_shape = attn.shape
        attn = self.attn_lif(attn.flatten(0, 1)).reshape(self.step, B, *attn_shape[2:]) # TB , , ,

        if self.training:
            firing_rate_attn = attn.detach().mean((0, 1, 3, 4), keepdim=True)
            if not self.init_firing_rate_attn and torch.all(self.firing_rate_attn == 0):
                self.firing_rate_attn = firing_rate_attn
            self.init_firing_rate_attn = True
            self.firing_rate_attn = self.firing_rate_attn * self.momentum + firing_rate_attn * (
                1 - self.momentum)

        scale2 = 1. / torch.sqrt(self.firing_rate_attn * self.length)
        out = torch.matmul(y2, attn)
        out = out * scale2
        out = out.reshape(*input_shape).contiguous() #TB C H W
        out = self.attn_out_lif(out)

        out = self.proj_conv(out)
        out = self.proj_norm(out)

        return out + x_feat # TB C H W

class DownsampleLayer(BaseModule):
    def __init__(self, in_channels, out_channels, step=4, encode_type='direct', node=LIFNode, tau=2.0, threshold=1.0,
                 act_func=SigmoidGrad, alpha=4.0, layer_by_layer=True):
        super().__init__(step=step, encode_type=encode_type, layer_by_layer=layer_by_layer)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.lif = node(step=step, tau=tau, act_func=act_func(alpha=alpha), threshold=threshold,
                           layer_by_layer=layer_by_layer, mem_detach=False)

    def forward(self, x):
        self.reset()

        x = self.lif(x)
        x = self.conv(x)
        x = self.norm(x)
        return x

class SpikingResformer(BaseModule):
    def __init__(
        self,
        layers: List[List[str]],
        planes: List[int],
        num_heads: List[int],
        patch_sizes: List[int],
        step=4,
        img_size=224,
        in_channels=3,
        num_classes=1000,
        prologue=None,
        group_size=64,
        node=LIFNode,
        tau=2.0,
        threshold=1.0,
        act_func=SigmoidGrad,
        alpha=4.0,
        encode_type='direct',
        layer_by_layer=True,
        **kwargs,
    ):
        super().__init__(step=step,encode_type=encode_type,layer_by_layer=layer_by_layer)
        self.step = step
        self.embed_dim = planes[-1]
        self.skip = ['prologue.0', 'classifier']
        assert len(planes) == len(layers) == len(num_heads) == len(patch_sizes)

        if prologue is None:
            self.prologue = nn.Sequential(
                nn.Conv2d(in_channels, planes[0], 7, 2, 3, bias=False),
                nn.BatchNorm2d(planes[0]),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
            img_size = img_size // 4
        else:
            self.prologue = prologue

        self.layers = nn.Sequential()
        for idx in range(len(planes)):
            sub_layers = nn.Sequential()
            if idx != 0:
                sub_layers.append(
                    DownsampleLayer(planes[idx - 1], planes[idx], node=node,tau=tau, act_func=act_func,
                              encode_type=encode_type,threshold=threshold,layer_by_layer=layer_by_layer,alpha=alpha))
                img_size = img_size // 2
            for name in layers[idx]:
                if name == 'DSSA':
                    sub_layers.append(
                        DSSA(embed_dims=planes[idx], num_heads=num_heads[idx], length=(img_size // patch_sizes[idx])**2,
                             patch_size=patch_sizes[idx],step=self.step, encode_type=encode_type,node=node, act_func=act_func,
                             threshold=threshold,layer_by_layer=layer_by_layer,alpha=alpha))
                elif name == 'GWFFN':
                    sub_layers.append(
                        GWFFN(step=step,in_channels=planes[idx], group_size=group_size, node=node, tau=tau, act_func=act_func,
                              encode_type=encode_type,threshold=threshold,layer_by_layer=layer_by_layer,alpha=alpha))
                else:
                    raise ValueError(name)
            self.layers.append(sub_layers)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1),)
        self.classifier = nn.Linear(planes[-1], num_classes)
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def transfer(self, state_dict: Mapping[str, Any]):
        _state_dict = {k: v for k, v in state_dict.items() if 'classifier' not in k}
        return self.load_state_dict(_state_dict, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # if x.dim() != 5:
        #     x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        #     assert x.dim() == 5
        # else:
        #     #### [B, T, C, H, W] -> [T, B, C, H, W]
        #     x = x.transpose(0, 1)
        x = self.encoder(x) # TB C H W
        x = self.prologue(x)
        x = self.layers(x)
        x = self.avgpool(x)

        #
        x = torch.flatten(x, 2)
        x = torch.squeeze(x, dim=-1).reshape(self.step, -1, self.embed_dim).contiguous() # 去掉最后一个dim
        x = self.classifier(x.mean(0))
        return x

    def no_weight_decay(self):
        ret = set()
        # for name, module in self.named_modules():
        #     if isinstance(module, PLIF):
        #         ret.add(name + '.w')
        return ret

@register_model
def spikingresformer_cifar(**kwargs):
    return SpikingResformer(
        step=kwargs.get('step', 4),
        in_channels=kwargs.get('in_channels', 3),
        num_classes=kwargs.get('num_classes', 10),
        embed_dim=kwargs.get('embed_dim', 384),
        tau=kwargs.get('tau', 2.0),
        threshold=kwargs.get('threshold', 1.0),
        node=kwargs.get('node', LIFNode),
        act_func=kwargs.get('act_func', SigmoidGrad),
        alpha=kwargs.get('alpha', 4.0),
        layers= [
            ['DSSA', 'GWFFN'] * 1,
            ['DSSA', 'GWFFN'] * 2,
            ['DSSA', 'GWFFN'] * 3, ],
        planes=[64, 192, 384],
        num_heads=[1, 3, 6],
        patch_sizes=[4, 2, 1],
        prologue=nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False, ),
            nn.BatchNorm2d(64),
        ),
    )
