"""
Compute guidance maps with a trained Neural A* guidance encoder and save them for
train_cnn_heuristic.py.

    python make_guidance.py --maps ocean_maps.npz --weights runs/neural_astar/best.pt --out guidance.npz
    python train_cnn_heuristic.py --maps ocean_maps.npz --guidance_npz guidance.npz

Keep this file next to neural_astar.py (the module train_guidance.py imports).

Reads, for each split present (train / val / test):
    <split>          (N,H,W,3) map images, channel 2 (blue) = obstacle
    <split>_start    (N,2) exact (row, col)
    <split>_goal     (N,2) exact (row, col)
Writes:
    <split>_guidance (N,H,W) float32, the encoder output Phi (lower = preferred), 0 on obstacles.

Inputs to the encoder are built exactly as in train_guidance.py: X = 1 for free / 0 for
obstacle, S and G one-hot start / goal maps, all of shape (B,1,H,W). Phi is read from
out["phi"]. Phi does not depend on the A* search, but the model runs it anyway, so
--max_steps 1 (if your NeuralAstar allows it) makes this much faster.
"""
import argparse
import math

import numpy as np
import torch

from neural_astar import NeuralAstar

H = W = 100   # train_cnn_heuristic.py is fixed to 100x100


def load_model(weights, device, max_steps=None):
    ck = torch.load(weights, map_location=device)
    a = ck.get("args", {})
    width = a.get("width", W)
    if width != W:
        print(f"warning: encoder was trained on width {width}, maps here are {W}")
    tau = a.get("tau") or math.sqrt(width)
    ms = max_steps if max_steps is not None else a.get("max_steps")
    model = NeuralAstar(a.get("encoder", "vgg16_bn"), False, tau=tau, max_steps=ms).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"loaded {weights}: encoder={a.get('encoder')} epoch={ck.get('epoch')} val={ck.get('val')}")
    return model


@torch.no_grad()
def guidance_for_split(model, d, split, device, chunk, min_value):
    maps = d[split]
    assert maps.shape[1:3] == (H, W), f"{split}: maps are {maps.shape[1:3]}, expected {(H, W)}"
    N = maps.shape[0]
    obstacle = maps[..., 2] > 0.5
    st, gl = d[f"{split}_start"], d[f"{split}_goal"]
    out = np.zeros((N, H, W), np.float32)
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        B = j - i
        X = torch.from_numpy(~obstacle[i:j]).float().unsqueeze(1)
        S = torch.zeros(B, 1, H, W)
        G = torch.zeros(B, 1, H, W)
        ar = torch.arange(B)
        S[ar, 0, torch.from_numpy(st[i:j, 0]).long(), torch.from_numpy(st[i:j, 1]).long()] = 1
        G[ar, 0, torch.from_numpy(gl[i:j, 0]).long(), torch.from_numpy(gl[i:j, 1]).long()] = 1
        phi = model(X.to(device), S.to(device), G.to(device))["phi"][:, 0].float().cpu().numpy()
        out[i:j] = phi
    free = ~obstacle
    out = np.where(free, np.maximum(out, min_value), 0.0).astype(np.float32)
    v = out[free]
    assert np.isfinite(out).all(), f"{split}: non-finite guidance values"
    assert v.min() >= 0, f"{split}: negative guidance values"
    print(f"[{split}] {N} maps | free-cell guidance min {v.min():.4f} mean {v.mean():.3f} "
          f"max {v.max():.3f} | {100 * (v < 0.01).mean():.1f}% of free cells < 0.01")
    if v.max() > 1.0 + 1e-6:
        print(f"warning: [{split}] guidance max > 1 (expected [0, 1])")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", required=True)
    ap.add_argument("--weights", required=True, help="checkpoint from train_guidance.py (best.pt / last.pt)")
    ap.add_argument("--out", default="guidance.npz")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--min_value", type=float, default=0.0,
                    help="floor for guidance on free cells (e.g. 0.01). Near-zero guidance makes the "
                         "base heuristic ~0; 0.0 leaves the encoder output untouched")
    ap.add_argument("--max_steps", type=int, default=None, help="override A* step cap (phi does not depend on it)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    model = load_model(args.weights, device, args.max_steps)
    d = np.load(args.maps)
    splits = [s for s in ("train", "val", "test") if s in d.files]
    assert splits, f"no train/val/test arrays found in {args.maps} (keys: {d.files})"
    res = {f"{s}_guidance": guidance_for_split(model, d, s, device, args.chunk, args.min_value)
           for s in splits}
    np.savez_compressed(args.out, **res)
    print("saved", args.out, "->", list(res))


if __name__ == "__main__":
    main()
