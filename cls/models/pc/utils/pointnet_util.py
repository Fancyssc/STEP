import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import importlib
from spikingjelly.clock_driven.neuron import *
from .surrogate import *
import farthest_point_sampling_cuda
# from pointnet2_ops import pointnet2_utils

heaviside_sigmoid = HeavisideSigmoid.apply
heaviside_parametric_sigmoid = HeavisideParametricSigmoid.apply

class MoELIFNode(nn.Module):
    def __init__(self, timestep, spike_mode, input_dim, tau=2.0, v_threshold=0.2):
        super(MoELIFNode, self).__init__()
        self.spike_mode = spike_mode
        self.num_experts = len(spike_mode)
        self.tau = tau
        self.v_threshold = v_threshold
        self.k = 4
        self.T = timestep
        self.v_th = 0.2

        self.experts = nn.ModuleList([])
        self.gate = nn.Conv1d(input_dim*self.T, self.num_experts, 1)
        for i in range(self.num_experts):
            if spike_mode[i] == "lif":
                self.experts.append(MultiStepLIFNode( v_threshold=v_threshold, tau=tau, detach_reset=True, backend="cupy"))
            elif spike_mode[i] == "elif":
                self.experts.append(MultiStepEIFNode(v_threshold=v_threshold, tau=tau, detach_reset=True, backend="cupy"))
            elif spike_mode[i] == "plif":
                self.experts.append(MultiStepParametricLIFNode(v_threshold=v_threshold, init_tau=tau, detach_reset=True, backend="cupy"))
            elif spike_mode[i] == "if":
                self.experts.append(MultiStepIFNode( v_threshold=v_threshold, detach_reset=True, backend="cupy"))
    
    def forward(self, x):
        if self.training:
            z = x.view(-1, self.T, *x.shape[1:]).flatten(1, 2)
            gate = F.softmax(self.gate(z), dim=-2).repeat(self.T, 1, 1, 1)
            expert_spikes = torch.stack([expert(x) for expert in self.experts], dim=1)
            expert_outputs = torch.stack([expert.v_seq.flatten(0,1) for expert in self.experts], dim=1).view(self.T, -1, self.num_experts, *x.shape[1:])
            expert_outputs[expert_outputs == 0.0] = self.v_threshold
            output = torch.sum(gate.unsqueeze(3) * expert_outputs, dim=2)
            output = heaviside_parametric_sigmoid(output, self.k, self.v_th)
        else:
            z = x.view(-1, self.T, *x.shape[1:]).flatten(1, 2)
            gate = self.gate(z)
            topk_values, topk_indices = torch.topk(gate, 2, dim=-2) 
            
            gate = torch.zeros_like(gate)
            gate.scatter_(-2, topk_indices, topk_values)
            gate_masked = gate.clone()
            gate_masked[gate == 0] = float('-inf')
            gate = F.softmax(gate_masked, dim=-2).repeat(self.T, 1, 1, 1)
            
            expert_spikes = torch.stack([expert(x) for expert in self.experts], dim=1)
            expert_outputs = torch.stack([expert.v_seq.flatten(0,1) for expert in self.experts], dim=1).view(self.T, -1, self.num_experts, *x.shape[1:])
            expert_outputs[expert_outputs == 0.0] = self.v_threshold
            output = torch.sum(gate.unsqueeze(3) * expert_outputs, dim=2)
            output = heaviside_parametric_sigmoid(output, self.k, self.v_th)
        return output.flatten(0,1)

def build_spike_node(timestep, spike_mode, input_dim=None, tau=2.0, v_threshold=0.5):
    """
    Create a spike neuron node based on the given spike mode.

    Parameters:
    - spike_mode (str): The spike mode to determine the type of neuron node.
                        "lif" creates a MultiStepLIFNode.
                        "elif" creates a MultiStepEIFNode.
    - tau (float, optional): The time constant for neuron activation (default is 2.0).
    - v_threshold (float, optional): The voltage threshold for neuron activation (default is 0.2).

    Returns:
    proj_lif: The created spike neuron node, which can be either a MultiStepLIFNode or MultiStepEIFNode based on the spike_mode.

    Raises:
    ValueError: If an unsupported spike_mode is provided.
    """
    if spike_mode == "lif":
        proj_lif = MultiStepLIFNode(v_threshold=v_threshold, tau=tau, detach_reset=True, backend="cupy")
    elif spike_mode == "elif":
        proj_lif = MultiStepEIFNode(v_threshold=v_threshold, tau=tau, detach_reset=True, backend="cupy")
    elif spike_mode == "plif":
        proj_lif = MultiStepParametricLIFNode(v_threshold=v_threshold, init_tau=tau, detach_reset=True, backend="cupy")
    elif spike_mode == "if":
        proj_lif = MultiStepIFNode(v_threshold=v_threshold, detach_reset=True, backend="cupy")
    elif isinstance(spike_mode, list) and input_dim is not None:
        proj_lif = MoELIFNode(timestep, spike_mode, input_dim, v_threshold=v_threshold)
    else:
        raise ValueError(f"Unsupported spike_mode: {spike_mode}")
    return proj_lif

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm:
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """

    if src.dim() == 4 and src.dim() == 4:
        return torch.sum((src[:, :, :, None] - dst[:, :, None]) ** 2, dim=-1)
    else:
        return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


def index_points(points, idx):
    """
    Input:
        points: input points data, [[T], B, N, C]
        idx: sample index data, [[T], B, S, [K]]
    Return:
        new_points:, indexed points data, [[T], B, S, [K], C]
    """
    idx = idx.to(torch.int64)
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], raw_size[1], -1) if len(raw_size) == 4 else idx.reshape(raw_size[0], -1)
    idx_size = list(idx.shape)
    idx_size.append(points.size(-1))
    res = torch.gather(points, 2 if len(raw_size) == 4 else 1, idx[..., None].expand(*idx_size))
    return res.reshape(*raw_size, -1)

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    centroids = farthest_point_sampling_cuda.farthest_point_sample_cuda(xyz, npoint)[0]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, use_encoder, returnfps=False, knn=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [[T], B, N, 3]
        points: input points data, [[T], B, N, D]
    Return:
        new_xyz: sampled points position data, [[T], B, npoint, nsample, 3]
        new_points: sampled points data, [[T], B, npoint, nsample, 3+D]
    """
    T, B, N, C = xyz.shape
    S = npoint
    loc = xyz[0] if not use_encoder else xyz.flatten(0,1)

    fps_idx = farthest_point_sample(loc, npoint)

    torch.cuda.empty_cache()
    new_xyz = index_points(loc, fps_idx)
    new_xyz = new_xyz.repeat(T, 1, 1, 1) if not use_encoder else new_xyz.view(T, B, -1, C)

    torch.cuda.empty_cache()
    if knn:
        dists = square_distance(new_xyz, xyz)  # B x npoint x N
        idx = dists.argsort()[:, :, :, :nsample]  # B x npoint x K
    else:
        idx = query_ball_point(radius, nsample, xyz, new_xyz)
    torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    torch.cuda.empty_cache()
    grouped_xyz_norm = grouped_xyz - new_xyz.view(T, B, S, 1, C)
    torch.cuda.empty_cache()

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, timestep, spike_mode, use_encoder, group_all, knn=False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.knn = knn
        self.use_encoder = use_encoder
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.mlp_lifs = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            self.mlp_lifs.append(build_spike_node(timestep, spike_mode) if spike_mode is not None else nn.ReLU())
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, N, C]
            points: input points data, [B, N, C]
        Return:
            new_xyz: sampled points position data, [B, S, C]
            new_points_concat: sample points feature data, [B, S, D']
        """
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else: # defualt 是 sample_and_group=False

            # 为原始点云进行FPS采 得到new_xyz -> 为每个中心点查找邻居（KNN）
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, self.use_encoder, knn=self.knn)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        T, B, N, M, _ = new_points.shape
        new_points = new_points.permute(0, 1, 4, 3, 2).flatten(0, 1) # [TB, C+D, nsample, npoint] npoint是采样点数 nsample是每个中心点的领域点数
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  bn(conv(self.mlp_lifs[i](new_points)))
       
        new_points = torch.max(new_points, 2)[0].transpose(1, 2) #对邻域做pooling
        new_points = new_points.reshape(T, B, N, -1)
        return new_xyz, new_points

# NoteL this function swaps N and C
class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp, timestep, spike_mode):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            self.mlp_lifs.append(build_spike_node(timestep, spike_mode))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        T, B, N, C = xyz1.shape
        _, _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, 1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :, :3], idx[:, :, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=3, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(T, B, N, 3, 1), dim=3)

        if points1 is not None:
            # new_points = torch.cat([points1, interpolated_points], dim=-1)
            new_points = points1 + interpolated_points
        else:
            new_points = interpolated_points

        new_points = new_points.flatten(0, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = bn(conv(self.mlp_lifs[i](new_points)))
        return new_points.view(T, B, N, -1)
