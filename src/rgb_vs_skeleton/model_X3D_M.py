"""X3D-M (Feichtenhofer, CVPR 2020) from PyTorchVideo, ready for our 16 classes.

Run from the Project_DL folder:
    uv run src/rgb_vs_skeleton/Model_X3D_M.py            # prints the architecture, parameters, FLOPs and a test forward pass
Needs:  uv add pytorchvideo torchvision      (plus fvcore, which pytorchvideo pulls in)
The Kinetics-400 weights (~15 MB) are downloaded on first use from dl.fbaipublicfiles.com; needs internet once.

Input:  (N, 3, 16, 224, 224) float32, normalised with mean 0.45 / std 0.225  -> yaml: rgb_num_frames 16, rgb_side 256, rgb_size 224
        X3D-M's pooling layer is built for exactly 16 frames and 224 px, other sizes fail or lose accuracy.
Output: (N, num_classes) raw scores (logits). The original hub model ends with a Softmax; we remove it, because
        nn.CrossEntropyLoss expects logits (training on probabilities would silently train badly).

Architecture (blocks[0..5]): stem -> 4 residual stages -> head (conv 192->432, average pool, conv 432->2048, dropout, linear).
Fine-tuning stages (what `set_trainable` freezes):
    "head"        only the new 2048->16 linear layer is trained (fast warm-up, a few epochs)
    "last_stage"  the last residual stage (blocks[4]) + the whole head
    "all"         everything (use a ~10x smaller learning rate than for the head alone)
Code base: https://github.com/facebookresearch/pytorchvideo  (cite it, together with the X3D paper).
"""
import time

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig

NUM_FRAMES, SIZE = 16, 224                 # fixed by the X3D-M architecture
X3D_MEAN, X3D_STD = (0.45, 0.45, 0.45), (0.225, 0.225, 0.225)
STAGES = ("head", "last_stage", "all")


def build_x3d_m(num_classes, pretrained=True, dropout=0.5):
    """X3D-M with Kinetics-400 weights and a new classifier for `num_classes` classes. Returns logits."""
    from pytorchvideo.models.hub import x3d_m
    try:
        model = x3d_m(pretrained=pretrained)
    except Exception as e:                                    # no internet: still build the model, say so clearly
        print(f"[x3d] could not download the Kinetics weights ({type(e).__name__}) - using RANDOM weights")
        model = x3d_m(pretrained=False)
    head = model.blocks[-1]
    head.proj = nn.Linear(head.proj.in_features, num_classes)   # 2048 -> 400 becomes 2048 -> num_classes
    head.dropout = nn.Dropout(dropout)
    head.activation = nn.Identity()                           # remove the Softmax: training needs logits
    return model


def set_trainable(model, stage):
    """Freeze everything except the chosen part. stage: 'head' | 'last_stage' | 'all'."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    for p in model.parameters():
        p.requires_grad = False
    trainable = {"head": [model.blocks[5].proj],
                 "last_stage": [model.blocks[4], model.blocks[5]],
                 "all": [model]}[stage]
    for module in trainable:
        for p in module.parameters():
            p.requires_grad = True
    return model


def count_parameters(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def describe(model):
    """One line per block: name, output of that block for a test clip, parameters."""
    names = ["stem", "stage 1", "stage 2", "stage 3", "stage 4", "head"]
    x = torch.zeros(1, 3, NUM_FRAMES, SIZE, SIZE)
    print(f"{'block':<12}{'type':<18}{'output shape':<26}{'params':>12}")
    with torch.no_grad():
        for i, block in enumerate(model.blocks):
            x = block(x)
            print(f"{names[i]:<12}{type(block).__name__:<18}{str(tuple(x.shape)):<26}{count_parameters(block):>12,}")
    print(f"{'total':<56}{count_parameters(model):>12,}")


def flops(model, x):
    """GFLOPs of one forward pass (torch's own counter; counts convolutions and linear layers)."""
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as counter, torch.no_grad():
        model(x)
    return counter.get_total_flops() / 1e9


@hydra.main(version_base=None, config_path="../../configs", config_name="main_config")
def main(cfg: DictConfig) -> None:
    num_classes = len(cfg.classes)
    model = build_x3d_m(num_classes, pretrained=cfg.get("x3d_pretrained", True), dropout=cfg.get("x3d_dropout", 0.5)).eval()

    print(f"\nX3D-M for {num_classes} classes\n")
    describe(model)

    print("\ntrainable parameters per fine-tuning stage:")
    for stage in STAGES:
        set_trainable(model, stage)
        print(f"  {stage:<12}{count_parameters(model, trainable_only=True):>12,}")
    set_trainable(model, "all")

    x = torch.randn(2, 3, NUM_FRAMES, SIZE, SIZE)
    t0 = time.perf_counter()
    with torch.no_grad():
        y = model(x)
    dt = time.perf_counter() - t0
    print(f"\ntest forward pass: input {tuple(x.shape)} -> output {tuple(y.shape)}  (logits, e.g. {y[0, :3].tolist()})")
    print(f"  {flops(model, x[:1]):.2f} GFLOPs per clip (torch counts a multiply-add as 2; papers often report half of this) | "
          f"{dt / 2 * 1000:.0f} ms per clip on {'GPU' if x.is_cuda else 'CPU'} (batch of 2)")
    
    if cfg.get("rgb_num_frames", NUM_FRAMES) != NUM_FRAMES or cfg.get("rgb_size", SIZE) != SIZE:
        print(f"\n[WARNING] yaml has rgb_num_frames={cfg.rgb_num_frames}, rgb_size={cfg.rgb_size}; X3D-M needs 16 frames at 224 px "
              f"(rgb_num_frames: 16, rgb_side: 256, rgb_size: 224)")


if __name__ == "__main__":
    main()