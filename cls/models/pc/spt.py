import torch
import math
import globals
import time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from braincog.model_zoo.base_module import BaseModule

from .utils.pointnet_util import  (PointNetSetAbstraction, farthest_point_sample,
                                  index_points, build_spike_node,square_distance)
from timm.models import register_model
from types import SimpleNamespace
from ..utils.node import *

class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels, timestep, spike_mode, use_encoder):
        super().__init__()
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], timestep, spike_mode, use_encoder,
                                         group_all=False, knn=True)

    def forward(self, xyz, points):
        return self.sa(xyz, points)


class TransformerBlock(BaseModule):
    # k: # of neighbor
    def __init__(self, d_points, d_model, k, timestep, spike_mode, use_encoder) -> None:
        super().__init__(step=timestep,encode_type='direct',layer_by_layer=True)
        # spike_mode = "lif"
        self.fc1 = nn.Sequential(
            # build_spike_node(timestep, spike_mode),
            # build_spike_node(timestep, ['lif', 'elif', 'plif', 'if'],
            #                  d_points) if spike_mode is not None else nn.Identity(),  # Mixer Lif/ HD-IF
            LIFNode(step=timestep, threshold=0.5, tau=2., mem_detach=False, layer_by_layer=True),
            nn.Conv1d(d_points, d_model, 1),
            nn.BatchNorm1d(d_model),
            # build_spike_node(timestep, spike_mode) if spike_mode is not None else nn.Identity(),
            LIFNode(step=timestep, threshold=0.5, tau=2., mem_detach=False, layer_by_layer=True),
        )
        self.fc2 = nn.Sequential(
            # build_spike_node(timestep, spike_mode) if spike_mode is not None else nn.Identity(),
            LIFNode(step=timestep, threshold=0.5, tau=2., mem_detach=False, layer_by_layer=True),
            nn.Conv1d(d_model, d_points, 1),
            nn.BatchNorm1d(d_points)
        )
        self.fc_delta = nn.Sequential(
            build_spike_node(timestep, spike_mode),
            nn.Conv2d(3, d_model, 1),
            nn.BatchNorm2d(d_model),
            # build_spike_node(timestep, spike_mode) if spike_mode is not None else nn.ReLU(),
            LIFNode(step=timestep, threshold=0.5, tau=2., mem_detach=False, layer_by_layer=True),
            nn.Conv2d(d_model, d_model, 1),
            nn.BatchNorm2d(d_model),
            # build_spike_node(timestep, spike_mode) if spike_mode is not None else nn.Identity(),
            LIFNode(step=timestep, threshold=0.5, tau=2., mem_detach=False, layer_by_layer=True),
        )
        self.fc_gamma = nn.Sequential(
            # build_spike_node(timestep, spike_mode) if spike_mode is not None else nn.Identity(),
            LIFNode(step=timestep, threshold=0.5, tau=2., mem_detach=False, layer_by_layer=True),
            nn.Conv2d(d_model, d_model, 1),
            nn.BatchNorm2d(d_model),
            # build_spike_node(timestep, spike_mode) if spike_mode is not None else nn.ReLU(),
            LIFNode(step=timestep, threshold=0.5, tau=2., mem_detach=False, layer_by_layer=True),
            nn.Conv2d(d_model, d_model, 1),
            nn.BatchNorm2d(d_model),
        )

        self.w_qs = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1),
            nn.BatchNorm1d(d_model),
            # build_spike_node(timestep, spike_mode) if spike_mode is not None else nn.Identity(),
            LIFNode(step=timestep, threshold=0.5, tau=2., mem_detach=False, layer_by_layer=True),
        )
        self.w_ks = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1),
            nn.BatchNorm1d(d_model),
            # build_spike_node(timestep, spike_mode) if spike_mode is not None else nn.Identity(),
            LIFNode(step=timestep, threshold=0.5, tau=2., mem_detach=False, layer_by_layer=True),
        )
        self.w_vs = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1),
            nn.BatchNorm1d(d_model),
            # build_spike_node(timestep, spike_mode) if spike_mode is not None else nn.Identity(),
            LIFNode(step=timestep, threshold=0.5, tau=2., mem_detach=False, layer_by_layer=True),
        )
        self.k = k
        self.use_encoder = use_encoder

    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyz, features):
        # BrainCog reset
        self.reset()

        T = xyz.shape[0]
        loc = xyz[0] if not self.use_encoder else xyz
        dists = square_distance(loc, loc)  # 每个点对的距离
        knn_idx = dists.argsort()[:, :, :self.k] \
            if not self.use_encoder else \
            dists.argsort()[:, :, :, :self.k]  # 取最近K个点的idx 得到每个点的k近邻索引
        knn_xyz = index_points(loc, knn_idx)
        knn_idx = knn_idx.repeat(T, 1, 1, 1).flatten(0, 1) \
            if not self.use_encoder else \
            knn_idx.flatten(0, 1)

        features = features.flatten(0, 1).permute(0, 2, 1).contiguous()
        pre = features

        x = self.fc1(features)  # d_points -> d_model 上采样

        #  K/V 来自每个点的 k 个邻居，Q 来自当前点本身
        q, k, v = self.w_qs(x), self.w_ks(x), self.w_vs(x)
        k = index_points(k.permute(0, 2, 1), knn_idx).permute(0, 3, 1, 2).contiguous()
        v = index_points(v.permute(0, 2, 1), knn_idx).permute(0, 3, 1, 2).contiguous()

        # 每个点和K近邻的差值
        pos_enc = self.fc_delta((xyz[:, :, :, None] - (knn_xyz.repeat(T, 1, 1, 1, 1) \
                                                           if not self.use_encoder else knn_xyz)).flatten(0, 1).permute(
            0, 3, 1, 2).contiguous())  # T*B, C, N, M

        # broadcast mechanism
        ## Attention机制设计的比较奇怪 q是npoints k是nsamples（knn）因此需要braodcast机制
        attn = self.fc_gamma(q[:, :, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(1)), dim=-1)  # T*B, C, N, M

        res = torch.einsum('bcnm,bcnm->bcn', attn, v + pos_enc)
        res = self.fc2(res) + pre
        res = res.permute(0, 2, 1).reshape(T, xyz.shape[1], xyz.shape[2], -1)
        return res, attn


class Backbone(BaseModule):
    def __init__(self, cfg):
        super().__init__(step=cfg.step, encode_type='direct', layer_by_layer=True)
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.nblocks, cfg.nneighbor, cfg.num_class, cfg.input_dim
        blocks, num_samples = cfg.blocks, cfg.num_samples
        spike_mode, timestep, use_encoder = cfg.spike_mode, cfg.step, cfg.use_encoder
        assert len(blocks) == nblocks + 1, "Block mismatches"

        self.fc1 = nn.Sequential(
            nn.Conv1d(d_points, 32, 1),
            nn.BatchNorm1d(32),
            # build_spike_node(timestep,spike_mode) if spike_mode is not None else nn.ReLU(),
            LIFNode(step=timestep, threshold=0.5, tau=2., mem_detach=False, layer_by_layer=True),
            nn.Conv1d(32, 32, 1),
            nn.BatchNorm1d(32),
        )

        # channel 是点云的channel

        transblock = lambda channel: TransformerBlock(channel, cfg.transformer_dim, nneighbor, timestep,
                                                      spike_mode, use_encoder)
        self.transformer1 = nn.ModuleList(transblock(32) for _ in range(blocks[0]))  # TODO: mutilayer

        # npoints = npoints if not use_encoder else max(num_samples, npoints//timestep)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(nblocks):  # nblocks=4
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(
                TransitionDown(npoints // 4 ** (i + 1), nneighbor, [channel // 2 + 3, channel, channel], timestep,
                               spike_mode, use_encoder))
            for _ in range(blocks[i + 1]):
                self.transformers.append(transblock(channel))

        self.nblocks = nblocks
        self.blocks = blocks

    def forward(self, x):
        # BrainCog reset
        self.reset()

        T, B, N, C = x.shape
        xyz = x[..., :3]
        x = self.fc1(x.flatten(0, 1).permute(0, 2, 1).contiguous())
        x = x.view(T, B, -1, N).permute(0, 1, 3, 2).contiguous()
        points = self.transformer1[0](xyz, x)[0]

        xyz_and_feats = [(xyz, points)]
        id = 0
        for i in range(self.nblocks):
            xyz, points = self.transition_downs[i](xyz, points)  # 逐层下采样
            for _ in range(self.blocks[i + 1]):
                points = self.transformers[id](xyz, points)[0]
                id += 1
            xyz_and_feats.append((xyz, points))
        return points, xyz_and_feats

class SPT(BaseModule):
    def __init__(self, cfg):
        super().__init__(step=cfg.step, encode_type='direct', layer_by_layer=True)
        self.backbone = Backbone(cfg)
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.nblocks, cfg.nneighbor, cfg.num_class, cfg.input_dim
        spike_mode, timestep, use_encoder, num_samples = cfg.spike_mode, cfg.step, cfg.use_encoder, cfg.num_samples
        self.fc2 = nn.Sequential(
            # build_spike_node(timestep, spike_mode) if spike_mode is not None else nn.Identity(),
            LIFNode(step=timestep, threshold=0.5, tau=2., mem_detach=False, layer_by_layer=True),
            nn.Conv1d(32 * 2 ** nblocks, 256, 1),
            nn.BatchNorm1d(256),
            # build_spike_node(timestep, spike_mode) if spike_mode is not None else nn.ReLU(),
            LIFNode(step=timestep, threshold=0.5, tau=2., mem_detach=False, layer_by_layer=True),
            nn.Conv1d(256, 64, 1),
            nn.BatchNorm1d(64),
            # build_spike_node(timestep, spike_mode) if spike_mode is not None else nn.ReLU(),
            LIFNode(step=timestep, threshold=0.5, tau=2., mem_detach=False, layer_by_layer=True),
            nn.Conv1d(64, n_c, 1),
        )
        self.nblocks = nblocks
        self.T = timestep
        self.spike_mode = spike_mode
        self.use_encoder = use_encoder
        self.num_samples = max(npoints // self.T, num_samples)

    def random_SDE(self, x):
        B, N, C = x.shape
        assert N % self.T == 0, "timstep is invalid"
        x = x.view(B, self.T, N // self.T, C).transpose(0, 1)
        return x

    def scan_SDE(self, x):
        B, N, C = x.shape
        _, indices = torch.sort(x[:, :, 0], dim=1)
        x = torch.gather(x, 1, indices.unsqueeze(-1).expand(-1, -1, C))
        x = x.expand(math.ceil(self.num_samples * self.T / N), *x.shape).transpose(0, 1)
        x = x.flatten(1, 2)[:, :self.num_samples * self.T]
        scan = x.view(B, self.T, self.num_samples, C).transpose(0, 1)
        return scan

    def queue_SDE(self, x):
        # x [B N C]
        def queue_mask(loc, fps_idx):
            # dequeue操作 删除已经使用过的点
            mask = torch.ones_like(loc, dtype=torch.bool)
            mask[torch.arange(B).unsqueeze(1), fps_idx] = False
            loc = loc[mask].view(B, -1, 3)
            return loc

        B, N, C = x.shape
        loc = x[..., :3]  # 点云C的前三个channel是坐标
        # npoint -> Ns 每个step需采样的点数
        # res -> 上一个step需要保留的点
        npoint = self.num_samples
        res = (N - npoint) // (self.T - 1) if self.T != 1 else 0

        onion = torch.zeros(self.T, B, npoint, C).to(x.device)
        fps_idx = farthest_point_sample(loc, npoint)  # return [B, npoint]
        # onion -> 当前step根据FPS留下来的点
        onion[0] = index_points(x, fps_idx)  # return [[T], B, S, C]
        loc = queue_mask(loc, fps_idx)  # 删除已经使用过的点

        for i in range(1, self.T):
            if loc.shape[1] == 0:
                onion[i] = onion[i - 1]
            else:
                fps_idx = farthest_point_sample(loc, res)
                onion[i, :, :npoint - res] = onion[i - 1][:, res:]
                onion[i, :, npoint - res:] = index_points(x, fps_idx)
                loc = queue_mask(loc, fps_idx)
        return onion  # 更新并且返回onion [T B S C]

    def forward(self, x):
        # BrainCog reset
        self.reset()
        # Convert to Spike
        assert len(x.shape) < 4, "shape of inputs is invalid"
        st = time.time()
        if self.spike_mode is not None:
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1) \
                if not self.use_encoder else \
                self.queue_SDE(x)  # use_encoder = True 启用编码器
        else:
            x = x.unsqueeze(0)
        end = time.time()
        globals.MID_TIME = end - st

        ### 通过Q-SDE 点云被encode成[T B S C]

        # Backbone
        points, _ = self.backbone(x)

        # Head for cls(including Lif)
        points = points.mean(2) if len(points.shape) == 4 else points.mean(1)
        points = points.unsqueeze(-1)
        res = self.fc2(points.flatten(0, 1))  # cls_head
        res = res.view(self.T, -1, *res.shape[1:]).mean(0)
        return res.squeeze(2)


@register_model
def spt(pretrained=False, **kwargs):
    cfg = SimpleNamespace(**kwargs)
    # print(cfg)
    model= SPT(cfg)
    return model