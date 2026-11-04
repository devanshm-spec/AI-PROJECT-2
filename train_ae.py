#!/usr/bin/env python3
"""
train_ae.py -- convolutional autoencoder on the blue (obstacle) channel of the maps from
generate_maps.py, for out-of-distribution (OOD) detection.

What is trained
    A fully convolutional autoencoder with a SPATIAL bottleneck (keeps layout information,
    unlike a 64-number vector). Loss = per-pixel binary cross-entropy reconstruction.
    Train maps are augmented with random flips / 90-degree rotations (the rectangle process is
    invariant to them), the best epoch is picked on the validation split.

What is saved
    <out>.pt          checkpoint (weights + config)
    <out>_feats.npz   for every split (train / val / test) and every --ood file:
                        <name>_feat   (N, D)  pooled multi-layer features (D = 7 * base)
                        <name>_recon  (N,)    mean reconstruction BCE per map
    These are what you score with: Mahalanobis / kNN on the features, or the recon error.

A quick AUROC (train-fitted Mahalanobis and recon error, test vs each OOD file) is printed at the end.

Usage:
    python train_ae.py --data ocean_maps.npz --out ae --epochs 100 \
        --ood ood_big.npz ood_dense.npz
(each --ood file is a .npz made by generate_maps.py with other parameters; its "test" split is used)
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- data
def load_split(path, split):
    """Blue channel only -> float tensor (N, 1, W, W), 1 = obstacle."""
    d = np.load(path)
    return torch.from_numpy(d[split][..., 2].astype(np.float32)).unsqueeze(1)


def augment(x):
    """Random flip + rotation by a multiple of 90 degrees (one transform per batch)."""
    x = torch.rot90(x, int(torch.randint(4, (1,))), dims=(2, 3))
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=(3,))
    return x


# ----------------------------------------------------------------------------- model
def conv_bn(i, o, stride=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, stride=stride, padding=1, bias=False),
                         nn.BatchNorm2d(o), nn.LeakyReLU(0.1))


class ConvAE(nn.Module):
    """
    100x100 input (any size works):
        e1: (B,  b, 50, 50)   e2: (B, 2b, 25, 25)   e3: (B, 4b, 13, 13)
        bottleneck: (B, bott, 13, 13)   <- spatial code, 16*13*13 = 2704 numbers by default
    Decoder mirrors it with nearest-neighbour upsampling to the encoder's feature-map sizes.
    """

    def __init__(self, base=16, bott=16):
        super().__init__()
        b = base
        self.e1 = nn.Sequential(conv_bn(1, b, 2), conv_bn(b, b))
        self.e2 = nn.Sequential(conv_bn(b, 2 * b, 2), conv_bn(2 * b, 2 * b))
        self.e3 = nn.Sequential(conv_bn(2 * b, 4 * b, 2), conv_bn(4 * b, 4 * b))
        self.to_bott = nn.Conv2d(4 * b, bott, 1)
        self.from_bott = nn.Conv2d(bott, 4 * b, 1)
        self.d3 = nn.Sequential(conv_bn(4 * b, 2 * b), conv_bn(2 * b, 2 * b))
        self.d2 = nn.Sequential(conv_bn(2 * b, b), conv_bn(b, b))
        self.d1 = nn.Sequential(conv_bn(b, b), conv_bn(b, b))
        self.out = nn.Conv2d(b, 1, 3, padding=1)
        self.cfg = dict(base=base, bott=bott)

    def forward(self, x):
        h1 = self.e1(x)
        h2 = self.e2(h1)
        h3 = self.e3(h2)
        z = self.to_bott(h3)
        d = self.from_bott(z)
        d = self.d3(F.interpolate(d, size=h2.shape[-2:], mode="nearest"))
        d = self.d2(F.interpolate(d, size=h1.shape[-2:], mode="nearest"))
        d = self.d1(F.interpolate(d, size=x.shape[-2:], mode="nearest"))
        logits = self.out(d)
        # global-average-pooled features from every encoder stage: b + 2b + 4b + bott = 7b dims
        feat = torch.cat([h.mean((2, 3)) for h in (h1, h2, h3, z)], dim=1)
        return logits, feat


@torch.no_grad()
def run(model, x, device, bs=128):
    """Returns pooled features (N, D) and per-map reconstruction BCE (N,)."""
    model.eval()
    feats, recs = [], []
    for i in range(0, len(x), bs):
        xb = x[i:i + bs].to(device)
        logits, f = model(xb)
        rec = F.binary_cross_entropy_with_logits(logits, xb, reduction="none").mean((1, 2, 3))
        feats.append(f.cpu()); recs.append(rec.cpu())
    return torch.cat(feats).numpy(), torch.cat(recs).numpy()


# ----------------------------------------------------------------------------- scoring
def maha_fit(f_train, ridge=1e-3):
    mu, sd = f_train.mean(0), f_train.std(0) + 1e-6
    z = (f_train - mu) / sd
    cov = np.cov(z, rowvar=False) + ridge * np.eye(z.shape[1])
    return mu, sd, np.linalg.inv(cov)


def maha_score(f, fit):
    mu, sd, prec = fit
    z = (f - mu) / sd
    return np.einsum("nd,de,ne->n", z, prec, z)


def auroc(s_in, s_out):
    """P(score of an OOD map > score of an in-distribution map); 0.5 = useless."""
    try:
        from sklearn.metrics import roc_auc_score
        y = np.r_[np.zeros(len(s_in)), np.ones(len(s_out))]
        return roc_auc_score(y, np.r_[s_in, s_out])
    except ImportError:
        return float((s_out[:, None] > s_in[None, :]).mean())


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="ocean_maps.npz")
    p.add_argument("--out", default="ae", help="prefix for <out>.pt and <out>_feats.npz")
    p.add_argument("--ood", nargs="*", default=[], help="npz files from other distributions (their 'test' split)")
    p.add_argument("--base", type=int, default=16)
    p.add_argument("--bott", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = a.device
    print(f"device: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev.startswith("cuda") else ""))

    x = {s: load_split(a.data, s) for s in ("train", "val", "test")}
    print({k: tuple(v.shape) for k, v in x.items()}, f"| obstacle density train {x['train'].mean():.3f}")
    xtr_dev = x["train"].to(dev)

    model = ConvAE(a.base, a.bott).to(dev)
    n_par = sum(p_.numel() for p_ in model.parameters())
    with torch.no_grad():
        model.eval(); _, f = model(xtr_dev[:2])
    print(f"parameters: {n_par:,} | pooled feature dim: {f.shape[1]} | bottleneck: {a.bott} channels")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    best, best_state = float("inf"), None
    n = len(xtr_dev)
    for ep in range(a.epochs):
        model.train(); perm = torch.randperm(n, device=dev); tot = 0.0
        for i in range(0, n - a.batch_size + 1 if n >= a.batch_size else 1, a.batch_size):
            xb = xtr_dev[perm[i:i + a.batch_size]]
            if not a.no_augment:
                xb = augment(xb)
            logits, _ = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        sched.step()
        _, vrec = run(model, x["val"], dev)
        v = float(vrec.mean())
        if v < best:
            best = v; best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        if ep == 0 or (ep + 1) % max(1, a.epochs // 20) == 0:
            print(f"[{ep + 1:4d}/{a.epochs}] train BCE {tot / n:.4f} | val BCE {v:.4f} | best {best:.4f}")

    model.load_state_dict(best_state)
    torch.save({"model": best_state, "cfg": model.cfg}, f"{a.out}.pt")
    print(f"saved {a.out}.pt (best val BCE {best:.4f})")

    # ---- features + recon error for every split / OOD set
    sets = dict(x)
    for path in a.ood:
        sets[os.path.splitext(os.path.basename(path))[0]] = load_split(path, "test")
    out = {}
    for name, xs in sets.items():
        f, r = run(model, xs, dev)
        out[f"{name}_feat"], out[f"{name}_recon"] = f, r
        print(f"  {name:12s} N={len(xs):4d}  recon BCE mean {r.mean():.4f}  std {r.std():.4f}")
    np.savez_compressed(f"{a.out}_feats.npz", **out)
    print(f"saved {a.out}_feats.npz")

    # ---- quick OOD check (scores fitted on train only)
    fit = maha_fit(out["train_feat"])
    m_test, r_test = maha_score(out["test_feat"], fit), out["test_recon"]
    print("\nAUROC, in-distribution test vs OOD (0.5 = useless, 1.0 = perfect):")
    print(f"  {'OOD set':14s} {'Mahalanobis':>12s} {'recon error':>12s}")
    for name in sets:
        if name in ("train", "val", "test"):
            continue
        print(f"  {name:14s} {auroc(m_test, maha_score(out[f'{name}_feat'], fit)):12.3f} "
              f"{auroc(r_test, out[f'{name}_recon']):12.3f}")
    if not a.ood:
        print("  (no --ood files given; rerun with --ood to get this table)")


if __name__ == "__main__":
    main()
