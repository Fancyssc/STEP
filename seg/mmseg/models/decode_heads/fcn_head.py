import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead
from braincog.base.node import LIFNode
from braincog.base.strategy.surrogate import SigmoidGrad

from spikingjelly.clock_driven import layer



@MODELS.register_module()
class FCNHead_SNN(BaseDecodeHead):
    """Fully Convolutional Networks for Spiking Neural Networks.

    This head is the SNN adaptation of FCN, designed to work with models
    that have output dimensions of T B C H W.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether to concatenate the input and output of
            convs before classification layer.
        tau (float): Membrane time constant for LIF neurons. Default: 2.0.
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 tau=2.0,
                 **kwargs):
        super(FCNHead_SNN, self).__init__(**kwargs)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.tau = tau

        self.convs = nn.ModuleList()
        for i in range(num_convs):
            self.convs.append(
                nn.Sequential(
                    LIFNode(step=4, tau=2., act_func=SigmoidGrad, threshold=1.,
                            layer_by_layer=True, mem_detach=False),
                    layer.SeqToANNContainer(
                        ConvModule(
                            self.in_channels if i == 0 else self.channels,
                            self.channels,
                            kernel_size=kernel_size,
                            padding=kernel_size // 2,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=None)
                    )
                )
            )

        self.decode_lif = LIFNode(step=4, tau=2., act_func=SigmoidGrad, threshold=1.,
                                  layer_by_layer=True, mem_detach=False)

        if self.concat_input:
            self.conv_cat = layer.SeqToANNContainer(
                ConvModule(
                    self.in_channels + self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=None)
            )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重以确保所有参数都被正确注册"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):

        self.decode_lif.n_reset()
        for node in self.convs:
            if isinstance(node, LIFNode):
                node.n_reset()

        x = inputs  # 直接使用输入，不需要额外处理

        T, B, C, H, W = x.shape

        # 保存输入，用于后续可能的连接操作
        identity = x

        # 通过卷积层处理
        output = x
        conv_outputs = []  # 存储中间输出，确保所有参数都被使用

        for i, conv in enumerate(self.convs):
            output = conv(output)
            conv_outputs.append(output)  # 存储用于梯度流

        if self.concat_input:
            # 沿通道维度连接输入和输出
            T, B, C_out, H, W = output.shape
            _, _, C_in, _, _ = identity.shape

            # 将前两个维度展平用于连接
            output_flat = output.flatten(0, 1)
            identity_flat = identity.flatten(0, 1)

            # 连接并处理
            output_cat = torch.cat([identity_flat, output_flat], dim=1)
            output_cat = self.conv_cat(output_cat)

            # 重新变形回[T, B, C, H, W]
            output = output_cat.reshape(T, B, self.channels, H, W)

        # 应用最终的LIF节点和分类
        T, B, C, H, W = output.shape
        output = self.decode_lif(output.flatten(0, 1)).reshape(T, B, C, H, W)

        # 为分类展平前两个维度
        output_flat = output.flatten(0, 1)
        output_cls = self.cls_seg(output_flat)

        # 重新变形回来并在时间维度上平均
        num_class = output_cls.shape[1]
        result = output_cls.reshape(T, B, num_class, H, W).mean(0)

        # 添加辅助输出以确保所有参数都获得梯度
        if self.training:
            # 计算辅助损失以确保所有参数都获得梯度
            # 这是一个虚拟操作，不会影响实际输出
            aux_outputs = sum([o.mean() for o in conv_outputs]) * 0.0
            result = result + aux_outputs

            # 确保每个参数都被使用
            for name, param in self.named_parameters():
                if param.requires_grad:
                    result = result + param.sum() * 0.0

        return result