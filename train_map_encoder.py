#!/usr/bin/env python3
"""
train_map_encoder.py -- CNN that reads ONLY the blue (obstacle) channel of the maps made by
generate_maps.py and is trained so that every training map is mapped to (almost) the same
feature vector.

Objective: Deep SVDD (one-class). Features z = f(map) are pulled towards a fixed centre c:
        loss = mean_i || f(map_i) - c ||^2
Anti-collapse measures (the loss alone is minimised by a constant network):
    * all conv / linear layers are bias-free, BatchNorm has no affine parameters
      -> the net cannot simply output a constant through a bias term
    * c is fixed (mean of the features of the initial / pre-trained network), never learned
    * optional autoencoder pre-training (--pretrain-epochs) so the starting features carry
      real information about the maps
    * weight decay

Usage:
    python generate_maps.py --out ocean_maps.npz
    python train_map_encoder.py --data ocean_maps.npz --epochs 50 --out map_encoder.pt

Using the trained encoder later:
    ck = torch.load("map_encoder.pt"); enc = Encoder(**ck["cfg"]); enc.load_state_dict(ck["encoder"])
    enc.eval(); z = enc(obstacle_batch)        # obstacle_batch: (B, 1, W, W) float32 in {0, 1}
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------------------------------------------------------- data
def load_split(path, split):
    """Blue channel only -> float tensor (N, 1, W, W), 1 = obstacle."""
    d = np.load(path)
    x = d[split][..., 2].astype(np.float32)
    return torch.from_numpy(x).unsqueeze(1)


# ----------------------------------------------------------------------------- model
class Encoder(nn.Module):
    """4 x (conv3x3 -> BN -> LeakyReLU -> maxpool) -> adaptive pool -> bias-free linear."""

    def __init__(self, feat_dim=64, base=16, pool=4):
        super().__init__()
        ch = [1, base, base * 2, base * 4, base * 8]
        layers = []
        for i, o in zip(ch[:-1], ch[1:]):
            layers += [nn.Conv2d(i, o, 3, padding=1, bias=False),
                       nn.BatchNorm2d(o, affine=False),
                       nn.LeakyReLU(0.1),
                       nn.MaxPool2d(2)]
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(pool)          # works for any map width
        self.fc = nn.Linear(ch[-1] * pool * pool, feat_dim, bias=False)
        self.cfg = dict(feat_dim=feat_dim, base=base, pool=pool)

    def forward(self, x):
        return self.fc(self.pool(self.conv(x)).flatten(1))


class Decoder(nn.Module):
    """Only used for the optional autoencoder pre-training."""

    def __init__(self, feat_dim=64, base=16):
        super().__init__()
        self.base = base
        self.fc = nn.Linear(feat_dim, base * 8 * 4 * 4)
        self.net = nn.Sequential(
            nn.Conv2d(base * 8, base * 4, 3, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(base * 4, base * 2, 3, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(base * 2, base, 3, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(base, 1, 3, padding=1))

    def forward(self, z, size):
        h = self.fc(z).view(z.size(0), self.base * 8, 4, 4)
        for i in range(3):                              # 4 -> 8 -> 16 -> 32, convs in between
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            h = self.net[2 * i:2 * i + 2](h)
        h = F.interpolate(h, size=size, mode="bilinear", align_corners=False)
        return self.net[6](h)                           # logits


# ----------------------------------------------------------------------------- helpers
@torch.no_grad()
def extract(enc, x, device, bs=256):
    enc.eval()
    out = [enc(x[i:i + bs].to(device)).cpu() for i in range(0, len(x), bs)]
    return torch.cat(out)


@torch.no_grad()
def init_center(enc, x, device, eps=0.1):
    c = extract(enc, x, device).mean(0)
    c[(c.abs() < eps) & (c < 0)] = -eps                 # keep c away from 0 (Ruff et al. 2018)
    c[(c.abs() < eps) & (c >= 0)] = eps
    return c.to(device)


def report(name, enc, x, c, device):
    z = extract(enc, x, device)
    d = ((z - c.cpu()) ** 2).sum(1)
    print(f"  {name:5s}: dist^2 to centre  mean {d.mean():.5f}  max {d.max():.5f} | "
          f"feature std across maps (mean over dims) {z.std(0).mean():.5f}")


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="ocean_maps.npz")
    p.add_argument("--out", default="map_encoder.pt")
    p.add_argument("--feat-dim", type=int, default=64)
    p.add_argument("--base", type=int, default=16)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--pretrain-epochs", type=int, default=20, help="autoencoder warm-up (0 = skip)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    dev = a.device
    xtr, xva, xte = (load_split(a.data, s) for s in ("train", "val", "test"))
    size = tuple(xtr.shape[-2:])
    loader = DataLoader(TensorDataset(xtr), batch_size=a.batch_size, shuffle=True, drop_last=len(xtr) > a.batch_size)

    enc = Encoder(a.feat_dim, a.base).to(dev)

    # ---- phase 1: autoencoder pre-training (gives the features real content before collapse pressure)
    if a.pretrain_epochs > 0:
        dec = Decoder(a.feat_dim, a.base).to(dev)
        opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=a.lr, weight_decay=a.weight_decay)
        for ep in range(a.pretrain_epochs):
            enc.train(); dec.train(); tot = 0.0
            for (xb,) in loader:
                xb = xb.to(dev)
                loss = F.binary_cross_entropy_with_logits(dec(enc(xb), size), xb)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(xb)
            print(f"[AE  {ep + 1:3d}/{a.pretrain_epochs}] recon BCE {tot / len(xtr):.4f}")

    # ---- phase 2: Deep SVDD -- pull all maps to one fixed point c
    c = init_center(enc, xtr, dev)
    print("\nBefore SVDD training:")
    report("train", enc, xtr, c, dev)
    opt = torch.optim.Adam(enc.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    for ep in range(a.epochs):
        enc.train(); tot = 0.0
        for (xb,) in loader:
            z = enc(xb.to(dev))
            loss = ((z - c) ** 2).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        sched.step()
        if (ep + 1) % max(1, a.epochs // 10) == 0 or ep == 0:
            print(f"[SVDD {ep + 1:3d}/{a.epochs}] loss {tot / len(xtr):.6f}")

    print("\nAfter SVDD training:")
    for name, x in (("train", xtr), ("val", xva), ("test", xte)):
        report(name, enc, x, c, dev)

    torch.save({"encoder": enc.state_dict(), "center": c.cpu(), "cfg": enc.cfg}, a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
