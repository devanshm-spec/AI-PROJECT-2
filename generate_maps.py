#!/usr/bin/env python3
"""
generate_maps.py -- random "ocean map" generator for Neural A* training.

Each map is a (width x width) binary grid:
    1 = obstacle (black), 0 = free space (white)
filled with many axis-aligned rectangles of varying size placed at random
positions (rectangles may overlap and may be clipped by the map border).

Output (default: ocean_maps.npz) contains three uint8 arrays of shape (N, W, W):
    train, val, test
which is exactly what train_guidance.py reads with --data ocean_maps.npz.

Example:
    python generate_maps.py --width 100 --num-train 800 --num-val 100 --num-test 100 \
        --out ocean_maps.npz --preview preview.png
"""
import argparse
import os
from collections import deque

import numpy as np

NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def largest_component_fraction(grid):
    """Fraction of free cells that belong to the largest 8-connected free component."""
    H, W = grid.shape
    free = (grid == 0).ravel().tolist()
    n_free = sum(free)
    if n_free == 0:
        return 0.0
    seen = [False] * (H * W)
    best = 0
    for s in range(H * W):
        if not free[s] or seen[s]:
            continue
        seen[s] = True
        q = deque([s])
        size = 0
        while q:
            u = q.popleft()
            size += 1
            r, c = divmod(u, W)
            for dr, dc in NEIGHBORS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    v = nr * W + nc
                    if free[v] and not seen[v]:
                        seen[v] = True
                        q.append(v)
        best = max(best, size)
    return best / n_free


def random_rect_map(rng, width, n_rects, min_size, max_size):
    """Place n_rects random rectangles on an empty grid (obstacle = 1)."""
    grid = np.zeros((width, width), dtype=np.uint8)
    for _ in range(n_rects):
        h = int(rng.integers(min_size, max_size + 1))
        w = int(rng.integers(min_size, max_size + 1))
        # allow up to half of the rectangle to hang over the border (it gets clipped),
        # which produces the border-hugging blocks seen in the reference pictures
        r0 = int(rng.integers(-(h // 2), width - h + h // 2 + 1))
        c0 = int(rng.integers(-(w // 2), width - w + w // 2 + 1))
        grid[max(r0, 0):min(r0 + h, width), max(c0, 0):min(c0 + w, width)] = 1
    return grid


def generate_split(rng, count, args, n_min, n_max, s_min, s_max):
    maps = []
    rejected = 0
    while len(maps) < count:
        n_rects = int(rng.integers(n_min, n_max + 1))
        grid = random_rect_map(rng, args.width, n_rects, s_min, s_max)
        density = grid.mean()
        if not (args.min_density <= density <= args.max_density):
            rejected += 1
            continue
        if largest_component_fraction(grid) < args.min_connected:
            rejected += 1
            continue
        maps.append(grid)
    return np.stack(maps).astype(np.uint8), rejected


def save_pngs(maps, folder):
    from PIL import Image
    os.makedirs(folder, exist_ok=True)
    for i, m in enumerate(maps):
        # PNG convention: black (0) = obstacle, white (255) = free
        Image.fromarray(((1 - m) * 255).astype(np.uint8)).save(os.path.join(folder, f"{i:05d}.png"))


def save_preview(maps, path, cols=4, rows=4, scale=3, gap=4):
    from PIL import Image
    n = min(len(maps), cols * rows)
    H, W = maps[0].shape
    canvas = np.full((rows * (H * scale + gap) + gap, cols * (W * scale + gap) + gap), 128, np.uint8)
    for k in range(n):
        r, c = divmod(k, cols)
        tile = np.kron((1 - maps[k]) * 255, np.ones((scale, scale), dtype=np.uint8))
        y, x = gap + r * (H * scale + gap), gap + c * (W * scale + gap)
        canvas[y:y + H * scale, x:x + W * scale] = tile
    Image.fromarray(canvas).save(path)


def main():
    p = argparse.ArgumentParser(description="Random rectangle map generator (obstacle=1, free=0)")
    p.add_argument("--width", type=int, default=100, help="map width = height in cells")
    p.add_argument("--num-train", type=int, default=800)
    p.add_argument("--num-val", type=int, default=100)
    p.add_argument("--num-test", type=int, default=100)
    p.add_argument("--min-rects", type=int, default=None,
                   help="min #rectangles per map (default 25 at width 100, scaled with width^2)")
    p.add_argument("--max-rects", type=int, default=None,
                   help="max #rectangles per map (default 50 at width 100, scaled with width^2)")
    p.add_argument("--min-size", type=int, default=None,
                   help="min rectangle side in cells (default 5 at width 100, scaled with width)")
    p.add_argument("--max-size", type=int, default=None,
                   help="max rectangle side in cells (default 12 at width 100, scaled with width)")
    p.add_argument("--min-density", type=float, default=0.10, help="reject maps with fewer obstacle cells")
    p.add_argument("--max-density", type=float, default=0.35, help="reject maps with more obstacle cells")
    p.add_argument("--min-connected", type=float, default=0.80,
                   help="reject maps whose largest free component holds less than this fraction of free cells")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="ocean_maps.npz")
    p.add_argument("--png-dir", type=str, default=None,
                   help="optional: also write PNGs to <png-dir>/{train,val,test}/ (black = obstacle)")
    p.add_argument("--preview", type=str, default=None, help="optional: write a 4x4 preview PNG")
    args = p.parse_args()

    scale = args.width / 100.0
    n_min = args.min_rects if args.min_rects is not None else max(1, round(25 * scale ** 2))
    n_max = args.max_rects if args.max_rects is not None else max(n_min, round(50 * scale ** 2))
    s_min = args.min_size if args.min_size is not None else max(2, round(5 * scale))
    s_max = args.max_size if args.max_size is not None else max(s_min, round(12 * scale))
    print(f"width={args.width}  rects/map={n_min}-{n_max}  rect side={s_min}-{s_max}")

    rng = np.random.default_rng(args.seed)
    splits = {}
    for name, count in (("train", args.num_train), ("val", args.num_val), ("test", args.num_test)):
        maps, rej = generate_split(rng, count, args, n_min, n_max, s_min, s_max)
        splits[name] = maps
        print(f"{name:5s}: {len(maps)} maps (rejected {rej}), mean obstacle density {maps.mean():.3f}")

    np.savez_compressed(args.out, **splits)
    print(f"saved {args.out}  (keys: train, val, test; shape (N, {args.width}, {args.width}); obstacle=1, free=0)")

    if args.png_dir:
        for name, maps in splits.items():
            save_pngs(maps, os.path.join(args.png_dir, name))
        print(f"PNG files written to {args.png_dir}/")
    if args.preview:
        save_preview(splits["train"], args.preview)
        print(f"preview written to {args.preview}")


if __name__ == "__main__":
    main()
