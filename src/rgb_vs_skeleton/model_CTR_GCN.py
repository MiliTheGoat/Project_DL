"""CTR-GCN (Chen et al., ICCV 2021) for our 16 classes, following the structure of the authors' code.

Run from the Project_DL folder:
    uv run src/rgb_vs_skeleton/model_CTR_GCN.py      # prints the architecture, parameters, FLOPs and a test forward pass

Input:  (N, C, T, V, 2) float32 = (clips, coordinates, frames, joints, persons); this is what skeleton_dataloader gives.
        skeleton_format "3d25": C = 3 (x y z), V = 25 joints  (original NTU Kinect skeleton)
        skeleton_format "2d13": C = 2 (x y),   V = 13 joints  (shared with HMDB, see skeleton_format.py)
Output: (N, num_classes) raw scores (logits), ready for nn.CrossEntropyLoss
There are no pretrained weights: the model is small (about 1.4M parameters) and is trained from scratch.

Architecture (same as the original):
    data_bn -> 10 blocks (l1-l4: 64 channels, l5-l7: 128, l8-l10: 256; l5 and l8 halve the number of frames)
            -> average over frames, joints and persons -> dropout -> fc
    one block = spatial part (3 CTR-GC units, one per graph subset: self, inward, outward)
              + temporal part (multi-scale: two dilated 5x1 convolutions, a max-pool branch and a 1x1 branch)
              + residual connection
Our changes: the last layer `fc` is 256 -> num_classes (16) instead of 256 -> 60, and the skeleton graph and the
number of input coordinates come from the chosen skeleton format (so the same network also runs on the 2D 13-joint skeleton).
Code base: https://github.com/Uason-Chen/CTR-GCN (model/ctrgcn.py, graph/ntu_rgb_d.py). Cite it and the paper.
"""
import math
import time

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

from skeleton_format import BONES_25, FORMATS

# ------------------------------------------------------------------------------------------------ skeleton graph
NUM_JOINTS = 25          # the original NTU skeleton (kept for older code; new code asks FORMATS for the numbers)
BONES = BONES_25


def edges_to_matrix(edges, num_joints):
    A = np.zeros((num_joints, num_joints), dtype=np.float32)
    for i, j in edges:
        A[j, i] = 1
    return A


def normalize(A):
    """Divide every column by its sum (a joint averages over its neighbours)."""
    degree = A.sum(axis=0)
    degree[degree == 0] = 1
    return A / degree


def build_graph(bones, num_joints):
    """-> (3, V, V): self links, links towards the body centre, links away from the body centre."""
    inward = [(i - 1, j - 1) for i, j in bones]
    outward = [(j, i) for i, j in inward]
    self_links = [(i, i) for i in range(num_joints)]
    return np.stack([edges_to_matrix(self_links, num_joints),
                     normalize(edges_to_matrix(inward, num_joints)),
                     normalize(edges_to_matrix(outward, num_joints))])


# ------------------------------------------------------------------------------------------------ building blocks
def init_conv(conv):
    nn.init.kaiming_normal_(conv.weight, mode="fan_out")
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def init_bn(bn, scale=1.0):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class Zero(nn.Module):
    """Used where a block has no residual connection."""

    def forward(self, x):
        return 0


class TemporalConv(nn.Module):
    """A (k x 1) convolution over time + batch norm."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1), dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        init_conv(self.conv)
        init_bn(self.bn)

    def forward(self, x):
        return self.bn(self.conv(x))


class MultiScaleTemporalConv(nn.Module):
    """Temporal part of a block: 4 parallel branches, each producing a quarter of the output channels."""

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, dilations=(1, 2)):
        super().__init__()
        n_branches = len(dilations) + 2
        assert out_channels % n_branches == 0
        c = out_channels // n_branches

        def reduce():                                      # 1x1 convolution that every branch starts with
            return [nn.Conv2d(in_channels, c, kernel_size=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True)]

        self.branches = nn.ModuleList()
        for dilation in dilations:                         # branches 1-2: dilated temporal convolutions
            self.branches.append(nn.Sequential(*reduce(), TemporalConv(c, c, kernel_size, stride, dilation)))
        self.branches.append(nn.Sequential(                # branch 3: max pooling over time
            *reduce(), nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)), nn.BatchNorm2d(c)))
        self.branches.append(nn.Sequential(                # branch 4: plain 1x1 convolution
            nn.Conv2d(in_channels, c, kernel_size=1, stride=(stride, 1)), nn.BatchNorm2d(c)))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_conv(m)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return torch.cat([branch(x) for branch in self.branches], dim=1)


class CTRGC(nn.Module):
    """Channel-wise Topology Refinement Graph Convolution: the shared graph A is adjusted for every channel,
    using how different two joints' features are (the core idea of the paper)."""

    def __init__(self, in_channels, out_channels, rel_reduction=8):
        super().__init__()
        rel_channels = 8 if in_channels < 16 else in_channels // rel_reduction      # first layer: 2 or 3 coordinates
        self.conv1 = nn.Conv2d(in_channels, rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(rel_channels, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for conv in (self.conv1, self.conv2, self.conv3, self.conv4):
            init_conv(conv)

    def forward(self, x, A, alpha):                        # x: (N, C, T, V)
        x1 = self.conv1(x).mean(dim=2)                     # (N, R, V)  joint features averaged over time
        x2 = self.conv2(x).mean(dim=2)
        x3 = self.conv3(x)                                 # (N, C_out, T, V)
        relation = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))        # (N, R, V, V) pairwise joint differences
        topology = self.conv4(relation) * alpha + A                      # (N, C_out, V, V) refined graph per channel
        return torch.einsum("ncuv,nctv->nctu", topology, x3)


class GraphConv(nn.Module):
    """Spatial part of a block: one CTRGC per graph subset, summed, + batch norm + residual."""

    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        self.convs = nn.ModuleList([CTRGC(in_channels, out_channels) for _ in range(A.shape[0])])
        if in_channels != out_channels:
            self.down = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1), nn.BatchNorm2d(out_channels))
            init_conv(self.down[0])
            init_bn(self.down[1])
        else:
            self.down = nn.Identity()
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))   # the graph itself is learnable
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        init_bn(self.bn, 1e-6)

    def forward(self, x):
        y = sum(conv(x, self.PA[i], self.alpha) for i, conv in enumerate(self.convs))
        return self.relu(self.bn(y) + self.down(x))


class Block(nn.Module):
    """Graph convolution over joints, then multi-scale convolution over time, with a residual connection."""

    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super().__init__()
        self.gcn = GraphConv(in_channels, out_channels, A)
        self.tcn = MultiScaleTemporalConv(out_channels, out_channels, kernel_size=5, stride=stride, dilations=(1, 2))
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = Zero()
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        return self.relu(self.tcn(self.gcn(x)) + self.residual(x))


# ------------------------------------------------------------------------------------------------ the model
class CTRGCN(nn.Module):
    def __init__(self, num_classes=16, skeleton_format="3d25", num_persons=2, dropout=0.0, base=64):
        super().__init__()
        skeleton = FORMATS[skeleton_format]
        self.num_joints, self.in_channels = skeleton["joints"], skeleton["channels"]
        in_channels = self.in_channels
        A = build_graph(skeleton["bones"], self.num_joints)
        self.data_bn = nn.BatchNorm1d(num_persons * in_channels * self.num_joints)
        self.l1 = Block(in_channels, base, A, residual=False)
        self.l2 = Block(base, base, A)
        self.l3 = Block(base, base, A)
        self.l4 = Block(base, base, A)
        self.l5 = Block(base, base * 2, A, stride=2)
        self.l6 = Block(base * 2, base * 2, A)
        self.l7 = Block(base * 2, base * 2, A)
        self.l8 = Block(base * 2, base * 4, A, stride=2)
        self.l9 = Block(base * 4, base * 4, A)
        self.l10 = Block(base * 4, base * 4, A)
        self.drop_out = nn.Dropout(dropout) if dropout else nn.Identity()
        self.fc = nn.Linear(base * 4, num_classes)                      # our change: num_classes outputs (16)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / num_classes))
        init_bn(self.data_bn)

    def blocks(self):
        return [self.l1, self.l2, self.l3, self.l4, self.l5, self.l6, self.l7, self.l8, self.l9, self.l10]

    def prepare(self, x):
        """(N, C, T, V, M) -> normalised (N*M, C, T, V): every person is processed as its own sample."""
        N, C, T, V, M = x.shape
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        return x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

    def forward(self, x):
        N, M = x.shape[0], x.shape[4]
        x = self.prepare(x)
        for block in self.blocks():
            x = block(x)
        x = x.view(N, M, x.shape[1], -1).mean(dim=3).mean(dim=1)        # average over frames+joints, then persons
        return self.fc(self.drop_out(x))


def build_ctr_gcn(num_classes, dropout=0.0, skeleton_format="3d25"):
    return CTRGCN(num_classes=num_classes, skeleton_format=skeleton_format, dropout=dropout)


def count_parameters(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def describe(model, num_frames=64):
    """One line per layer: name, output shape for a test clip, parameters."""
    x = torch.zeros(1, model.in_channels, num_frames, model.num_joints, 2)
    print(f"{'layer':<10}{'output shape':<24}{'params':>12}     (shape = persons, channels, frames, joints)")
    with torch.no_grad():
        x = model.prepare(x)
        print(f"{'data_bn':<10}{str(tuple(x.shape)):<24}{count_parameters(model.data_bn):>12,}")
        for i, block in enumerate(model.blocks(), 1):
            x = block(x)
            print(f"{'l' + str(i):<10}{str(tuple(x.shape)):<24}{count_parameters(block):>12,}")
    print(f"{'fc':<10}{str((1, model.fc.out_features)):<24}{count_parameters(model.fc):>12,}")
    print(f"{'total':<34}{count_parameters(model):>12,}")


def flops(model, x):
    """GFLOPs of one forward pass (torch's own counter, the same one used for the RGB model)."""
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as counter:        # no torch.no_grad() here: the counter fails with it on this model
        model(x)
    return counter.get_total_flops() / 1e9


@hydra.main(version_base=None, config_path="../../configs", config_name="main_config")
def main(cfg: DictConfig) -> None:
    num_classes = len(cfg.classes)
    num_frames = cfg.get("skeleton_num_frames", 64)
    skeleton_format = cfg.get("skeleton_format", "3d25")
    model = build_ctr_gcn(num_classes, dropout=cfg.get("ctrgcn_dropout", 0.0), skeleton_format=skeleton_format).eval()

    print(f"\nCTR-GCN for {num_classes} classes, skeleton format {skeleton_format} "
          f"({model.num_joints} joints, {model.in_channels} coordinates; trained from scratch)\n")
    describe(model, num_frames)
    print(f"\ntrainable parameters: {count_parameters(model, trainable_only=True):,} (all of them)")

    x = torch.randn(2, model.in_channels, num_frames, model.num_joints, 2)
    model.train()                 # untrained batch-norm layers only give sensible numbers in training mode
    t0 = time.perf_counter()
    with torch.no_grad():
        y = model(x)
    dt = time.perf_counter() - t0
    model.eval()
    print(f"\ntest forward pass: input {tuple(x.shape)} -> output {tuple(y.shape)}  (logits, e.g. {y[0, :3].tolist()})")
    print(f"  {flops(model, x[:1]):.2f} GFLOPs per clip (torch counts a multiply-add as 2; papers often report half of this) | "
          f"{dt / 2 * 1000:.0f} ms per clip on CPU (batch of 2)")


if __name__ == "__main__":
    main()