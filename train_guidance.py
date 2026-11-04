#!/usr/bin/env python3
"""
train_guidance.py -- train the Neural A* guidance encoder (Yonetani et al., ICML 2021).

Pipeline per mini-batch (Fig. 2 of the paper):
  (1) encoder(X, Vs+Vg)                    -> guidance map Phi
  (2) differentiable A*(Phi, X, Vs, Vg)    -> search history C (closed list)
  (3) loss L = || C - P_gt ||_1 / |V|      -> back-propagated through every search step to the encoder
Ground-truth paths P_gt are shortest paths found by an optimal planner (BFS == Dijkstra for unit costs
on the 8-neighbour grid).

Input data (made by generate_maps.py):
    * .npz with arrays  train / val / test   of shape (N,W,W), obstacle = 1, free = 0
    * or a single .npy / .npz array (N,W,W)  -> split with --val-frac / --test-frac
    * or a folder of PNG images (black = obstacle) -> split as above
Use --width to set the map size (maps with another size are resized with nearest neighbour).

Example:
    python train_guidance.py --data ocean_maps.npz --width 100 --epochs 100 --batch-size 8
"""
import argparse
import json
import math
import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from neural_astar import NeuralAstar, standard_astar

NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


# ======================================================================================
# data loading
# ======================================================================================
def _resize_nearest(maps, width):
    if maps.shape[-1] == width and maps.shape[-2] == width:
        return maps
    t = torch.from_numpy(maps).float().unsqueeze(1)
    t = F.interpolate(t, size=(width, width), mode="nearest")
    return t.squeeze(1).numpy()


def _binarize(maps):
    return (np.asarray(maps) > 0.5).astype(np.uint8)       # obstacle = 1, free = 0


def _split(maps, val_frac, test_frac, seed):
    idx = np.random.default_rng(seed).permutation(len(maps))
    n_val, n_test = int(len(maps) * val_frac), int(len(maps) * test_frac)
    return (maps[idx[n_val + n_test:]], maps[idx[:n_val]], maps[idx[n_val:n_val + n_test]])


def load_maps(path, width, val_frac, test_frac, seed):
    """Returns (train, val, test) uint8 arrays (N,width,width) with obstacle = 1, free = 0."""
    if os.path.isdir(path):
        from PIL import Image
        files = sorted(f for f in os.listdir(path) if f.lower().endswith((".png", ".jpg", ".bmp")))
        arr = np.stack([np.asarray(Image.open(os.path.join(path, f)).convert("L")) < 128 for f in files])
        splits = _split(_binarize(arr), val_frac, test_frac, seed)
    else:
        data = np.load(path)
        if hasattr(data, "files"):
            if all(k in data for k in ("train", "val", "test")):
                splits = (_binarize(data["train"]), _binarize(data["val"]), _binarize(data["test"]))
            else:
                splits = _split(_binarize(data[list(data.keys())[0]]), val_frac, test_frac, seed)
        else:
            splits = _split(_binarize(data), val_frac, test_frac, seed)
    return tuple(_resize_nearest(s, width).astype(np.uint8) for s in splits)


# ======================================================================================
# ground-truth generation (optimal planner)
# ======================================================================================
def bfs_from_goal(free_flat, H, W, goal):
    """Unit-cost shortest paths on the 8-neighbour grid (== Dijkstra). parent[] points toward the goal."""
    dist = [-1] * (H * W)
    parent = [-1] * (H * W)
    dist[goal] = 0
    q = deque([goal])
    while q:
        u = q.popleft()
        r, c = divmod(u, W)
        d = dist[u] + 1
        for dr, dc in NEIGHBORS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                v = nr * W + nc
                if free_flat[v] and dist[v] < 0:
                    dist[v] = d
                    parent[v] = u
                    q.append(v)
    return dist, parent


def make_instances_for_map(free_flat, H, W, n_starts, rng, corner_frac, min_dist_frac, tries=8):
    """
    Paper setup: ONE goal per map taken from one of the four corner regions, and n_starts start
    locations sufficiently far from the goal. Returns list of (start, goal, path_indices) or None.
    """
    m = max(1, int(round(corner_frac * min(H, W))))
    corners = [(0, 0), (0, W - m), (H - m, 0), (H - m, W - m)]
    min_d = max(1, int(round(min_dist_frac * W)))
    for ci in rng.permutation(4):
        r0, c0 = corners[ci]
        cand = [r * W + c for r in range(r0, r0 + m) for c in range(c0, c0 + m) if free_flat[r * W + c]]
        if not cand:
            continue
        for _ in range(tries):
            goal = int(rng.choice(cand))
            dist, parent = bfs_from_goal(free_flat, H, W, goal)
            far = [i for i, d in enumerate(dist) if d >= min_d]
            if not far:
                continue
            replace = len(far) < n_starts
            starts = rng.choice(far, size=n_starts, replace=replace)
            out = []
            for s in starts:
                s = int(s)
                path, u = [s], s
                while u != goal:
                    u = parent[u]
                    path.append(u)
                out.append((s, goal, np.asarray(path, dtype=np.int32)))
            return out
    return None


class PlanningDataset(Dataset):
    """Problem instances Q = (X, v_s, v_g) with ground-truth shortest-path maps."""

    def __init__(self, maps_obstacle, n_starts, seed, corner_frac=0.2, min_dist_frac=0.5, name=""):
        self.N, self.H, self.W = maps_obstacle.shape
        self.free = (1 - maps_obstacle).astype(np.float32)          # X: 1 = passable, 0 = obstacle
        rng = np.random.default_rng(seed)
        self.items, skipped = [], 0
        t0 = time.time()
        for i in range(self.N):
            inst = make_instances_for_map(self.free[i].ravel().tolist(), self.H, self.W, n_starts, rng,
                                          corner_frac, min_dist_frac)
            if inst is None:
                skipped += 1
                continue
            self.items += [(i, s, g, p) for (s, g, p) in inst]
        print(f"[{name}] {len(self.items)} instances from {self.N - skipped}/{self.N} maps "
              f"({skipped} maps skipped: no usable goal/start) in {time.time() - t0:.1f}s")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        mi, s, g, path = self.items[k]
        H, W = self.H, self.W
        start = torch.zeros(H * W); start[s] = 1
        goal = torch.zeros(H * W); goal[g] = 1
        gt = torch.zeros(H * W); gt[torch.from_numpy(path.astype(np.int64))] = 1
        return (torch.from_numpy(self.free[mi]).unsqueeze(0), start.view(1, H, W),
                goal.view(1, H, W), gt.view(1, H, W))


# ======================================================================================
# evaluation: Opt, Exp, Hmean (Sec. 4.3)
# ======================================================================================
@torch.no_grad()
def evaluate(model, loader, device, tau, ref_cache=None):
    """
    Opt   : % of instances whose predicted path is a shortest path
    Exp   : mean reduction ratio of node explorations vs. standard A* (same heuristic), in %
    Hmean : harmonic mean of Opt and Exp
    ref_cache: list with the node-expansion count E* of standard A* per instance (filled on first call).
    """
    model.eval()
    first = ref_cache is None
    ref_cache = [] if first else ref_cache
    opts, exps, pos = [], [], 0
    for X, S, G, P in loader:
        X, S, G, P = X.to(device), S.to(device), G.to(device), P.to(device)
        B = X.shape[0]
        if first:
            ref = standard_astar(X, S, G, tau=tau)["histories"].sum(dim=(1, 2, 3))
            ref_cache += ref.cpu().tolist()
        e_star = torch.tensor(ref_cache[pos:pos + B], device=device)
        pos += B
        out = model(X, S, G, return_path=True)
        e = out["histories"].sum(dim=(1, 2, 3))
        path = out["path"]
        is_opt = out["solved"] & ((path * S).sum(dim=(1, 2, 3)) > 0.5) & \
                 (path.sum(dim=(1, 2, 3)) <= P.sum(dim=(1, 2, 3)) + 0.5)
        opts.append(is_opt.float().cpu())
        exps.append(torch.clamp(100.0 * (e_star - e) / e_star, min=0.0).cpu())
    opt = torch.cat(opts).mean().item() * 100
    exp = torch.cat(exps).mean().item()
    hmean = 2 * opt * exp / (opt + exp) if (opt + exp) > 0 else 0.0
    return {"opt": opt, "exp": exp, "hmean": hmean}, ref_cache


# ======================================================================================
# visualisation (optional)
# ======================================================================================
@torch.no_grad()
def save_visualizations(model, dataset, device, out_dir, n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    for k in range(min(n, len(dataset))):
        X, S, G, P = [t.unsqueeze(0).to(device) for t in dataset[k]]
        out = model(X, S, G, return_path=True)
        base = np.stack([X[0, 0].cpu().numpy() * 0.8 + 0.2] * 3, -1)      # obstacles dark, free light gray
        vis = base.copy()
        vis[out["histories"][0, 0].cpu().numpy() > 0.5] = [0.2, 0.8, 0.2]  # explored = green
        vis[out["path"][0, 0].cpu().numpy() > 0.5] = [1, 0, 0]             # path = red
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(base); ax[0].set_title("input (start/goal)")
        for t, c in ((S, "b"), (G, "m")):
            r, cc = divmod(int(t[0, 0].reshape(-1).argmax()), X.shape[-1])
            ax[0].scatter([cc], [r], c=c, s=30)
        ax[1].imshow(vis); ax[1].set_title("Neural A*: explored (green), path (red)")
        ax[2].imshow(out["phi"][0, 0].cpu().numpy(), cmap="Greens_r"); ax[2].set_title("guidance map Phi")
        for a in ax:
            a.axis("off")
        fig.savefig(os.path.join(out_dir, f"result_{k:03d}.png"), dpi=110, bbox_inches="tight")
        plt.close(fig)


# ======================================================================================
# main
# ======================================================================================
def main():
    p = argparse.ArgumentParser(description="Train Neural A* guidance encoder")
    p.add_argument("--data", required=True, help=".npz/.npy file or PNG folder (obstacle = 1, free = 0)")
    p.add_argument("--width", type=int, default=100, help="map width = height (maps are resized if different)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8,
                   help="paper used 100 on 32x32 maps with 16 GB; for 100x100 maps use a small value")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--encoder", default="vgg16_bn", choices=["vgg16_bn", "resnet18", "simple"])
    p.add_argument("--pretrained", action="store_true", help="ImageNet-pretrained backbone (needs internet)")
    p.add_argument("--tau", type=float, default=None, help="temperature of Eq.(3); default sqrt(width)")
    p.add_argument("--train-starts", type=int, default=1, help="start locations per training map")
    p.add_argument("--val-starts", type=int, default=6)
    p.add_argument("--test-starts", type=int, default=15)
    p.add_argument("--corner-frac", type=float, default=0.2, help="goal is drawn from a corner region of this size")
    p.add_argument("--min-dist-frac", type=float, default=0.5,
                   help="start must be at least this fraction of the map width (in path steps) from the goal")
    p.add_argument("--val-frac", type=float, default=0.1, help="only used when --data has no train/val/test keys")
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--max-train-maps", type=int, default=None, help="debug: use only the first N training maps")
    p.add_argument("--max-steps", type=int, default=None, help="cap on A* iterations (default width^2)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--out-dir", default="runs/neural_astar")
    p.add_argument("--vis", type=int, default=0, help="save N result figures of the test split (needs matplotlib)")
    p.add_argument("--eval-only", action="store_true", help="skip training, evaluate --checkpoint on the test split")
    p.add_argument("--checkpoint", default=None)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    tau = args.tau if args.tau is not None else math.sqrt(args.width)
    print(f"device={device}  width={args.width}  tau={tau:.3f}")

    # ---- data ----
    tr, va, te = load_maps(args.data, args.width, args.val_frac, args.test_frac, args.seed)
    if args.max_train_maps:
        tr = tr[:args.max_train_maps]
    print(f"maps: train {len(tr)}, val {len(va)}, test {len(te)}  (obstacle density {tr.mean():.3f})")
    kw = dict(corner_frac=args.corner_frac, min_dist_frac=args.min_dist_frac)
    test_ds = PlanningDataset(te, args.test_starts, args.seed + 3, name="test", **kw)

    model = NeuralAstar(args.encoder, args.pretrained, tau=tau, max_steps=args.max_steps).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"encoder={args.encoder}  parameters={n_params / 1e6:.2f}M")

    if not args.eval_only:
        train_ds = PlanningDataset(tr, args.train_starts, args.seed + 1, name="train", **kw)
        val_ds = PlanningDataset(va, args.val_starts, args.seed + 2, name="val", **kw)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False,
                                  num_workers=args.num_workers)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)

        best, val_ref, log = -1.0, None, []
        for epoch in range(1, args.epochs + 1):
            model.train()
            t0, run_loss, nb = time.time(), 0.0, 0
            for X, S, G, P in train_loader:
                X, S, G, P = X.to(device), S.to(device), G.to(device), P.to(device)
                out = model(X, S, G)                              # (1) encoder  (2) differentiable A*
                loss = F.l1_loss(out["histories"], P)             # (3) Eq.(7): mean |C - P_gt|
                optimizer.zero_grad()
                loss.backward()                                   # back-prop through every search step
                optimizer.step()
                run_loss += loss.item(); nb += 1
            m, val_ref = evaluate(model, val_loader, device, tau, val_ref)
            rec = {"epoch": epoch, "loss": run_loss / max(nb, 1), **m, "time_s": time.time() - t0}
            log.append(rec)
            flag = ""
            if m["hmean"] > best:                                 # keep weights with best validation Hmean
                best = m["hmean"]
                torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch, "val": m},
                           os.path.join(args.out_dir, "best.pt"))
                flag = "  *best*"
            print(f"epoch {epoch:3d}/{args.epochs}  loss {rec['loss']:.5f}  val Opt {m['opt']:.1f}  "
                  f"Exp {m['exp']:.1f}  Hmean {m['hmean']:.1f}  ({rec['time_s']:.0f}s){flag}")
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch},
                       os.path.join(args.out_dir, "last.pt"))
            with open(os.path.join(args.out_dir, "log.json"), "w") as f:
                json.dump(log, f, indent=1)
        ckpt_path = os.path.join(args.out_dir, "best.pt")
    else:
        ckpt_path = args.checkpoint
        assert ckpt_path, "--eval-only needs --checkpoint"

    # ---- final test evaluation with the best checkpoint ----
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    m, _ = evaluate(model, test_loader, device, tau)
    print(f"TEST  Opt {m['opt']:.1f}  Exp {m['exp']:.1f}  Hmean {m['hmean']:.1f}")
    with open(os.path.join(args.out_dir, "test_metrics.json"), "w") as f:
        json.dump(m, f, indent=1)
    if args.vis > 0:
        save_visualizations(model, test_ds, device, os.path.join(args.out_dir, "vis"), args.vis)
        print(f"figures saved to {os.path.join(args.out_dir, 'vis')}")


if __name__ == "__main__":
    main()
