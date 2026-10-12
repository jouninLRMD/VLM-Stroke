"""Random 3D affine, intensity and noise augmentation applied to the whole volume before slicing."""
import math
import torch
import torch.nn.functional as F


def augment(x, mask, strength):
    B = x.shape[0]; dev = x.device
    ang = (torch.rand(B, 3, device=dev) * 2 - 1) * math.radians(10 * strength)
    sc = 1 + (torch.rand(B, 1, device=dev) * 2 - 1) * 0.08 * strength
    tr = (torch.rand(B, 3, device=dev) * 2 - 1) * 0.05 * strength
    cx, cy, cz = torch.cos(ang).unbind(1); sx, sy, sz = torch.sin(ang).unbind(1)
    o, z = torch.ones_like(cx), torch.zeros_like(cx)
    Rx = torch.stack([o, z, z, z, cx, -sx, z, sx, cx], 1).view(B, 3, 3)
    Ry = torch.stack([cy, z, sy, z, o, z, -sy, z, cy], 1).view(B, 3, 3)
    Rz = torch.stack([cz, -sz, z, sz, cz, z, z, z, o], 1).view(B, 3, 3)
    A = torch.cat([(Rz @ Ry @ Rx) * sc.view(B, 1, 1), tr.view(B, 3, 1)], 2)
    g = F.affine_grid(A, x.shape, align_corners=False)
    x = F.grid_sample(x, g, mode="bilinear", padding_mode="zeros", align_corners=False)
    mask = (F.grid_sample(mask, g, mode="bilinear", align_corners=False) > 0.5).float()
    x = x * (1 + 0.05 * strength * torch.randn(B, x.shape[1], 1, 1, 1, device=dev)) \
        + 0.05 * strength * torch.randn(B, x.shape[1], 1, 1, 1, device=dev)
    return x + 0.02 * strength * torch.randn_like(x), mask
