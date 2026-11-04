#!/usr/bin/env python3
"""
generate_maps.py -- random "ocean map" generator for Neural A* training.

Every sample is a (width x width x 3) RGB-style array (uint8, values 0/1 in the .npz, 0/255 in PNGs):
    channel 0  RED   : goal  (end)   -- exactly one pixel set to 1
    channel 1  GREEN : start         -- exactly one pixel set to 1
    channel 2  BLUE  : obstacle map  -- 1 = obstacle, 0 = free space
The map is filled with many axis-aligned rectangles of varying size placed at random (they may
overlap and may be clipped by the border). Start and goal are always on free cells, are connected
to each other (8-neighbour grid) and are at least --min-dist-frac * width path steps apart.

Output (default: ocean_maps.npz) holds three uint8 arrays of shape (N, W, W, 3): train, val, test.
    * train_guidance.py reads ONLY the blue channel (maps[..., 2]) and samples its own start/goal.
    * a heuristic-learning script can use channels 0 and 1 (goal / start) as well.

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


def bfs_dist(free_flat, H, W, src):
    """8-neighbour BFS distances (in steps) from src; -1 = unreachable."""
    dist = [-1] * (H * W)
    dist[src] = 0
    q = deque([src])
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
                    q.append(v)
    return dist


def place_points(grid, rng, min_dist_frac, goal_corners, corner_frac, tries=10):
    """
    Pick goal (red) and start (green) on free cells of `grid` (obstacle = 1).
    Start is connected to the goal and at least round(min_dist_frac * W) path steps away.
    Returns ((start_r, start_c), (goal_r, goal_c)) or None if no valid pair was found.
    """
    H, W = grid.shape
    free_flat = (grid == 0).ravel().tolist()
    min_d = max(1, int(round(min_dist_frac * W)))
    if goal_corners:                       # goal from one of the four corner regions (paper's setup)
        m = max(1, int(round(corner_frac * W)))
        r0, c0 = [(0, 0), (0, W - m), (H - m, 0), (H - m, W - m)][int(rng.integers(4))]
        cand = [r * W + c for r in range(r0, r0 + m) for c in range(c0, c0 + m) if free_flat[r * W + c]]
    else:
        cand = [i for i, f in enumerate(free_flat) if f]
    if not cand:
        return None
    for _ in range(tries):
        goal = int(cand[int(rng.integers(len(cand)))])
        dist = bfs_dist(free_flat, H, W, goal)
        far = [i for i, d in enumerate(dist) if d >= min_d]
        if far:
            start = int(far[int(rng.integers(len(far)))])
            return divmod(start, W), divmod(goal, W)
    return None


def encode_rgb(grid, start, goal):
    """Stack into (W, W, 3): R = goal, G = start, B = obstacle map."""
    rgb = np.zeros(grid.shape + (3,), dtype=np.uint8)
    rgb[..., 2] = grid
    rgb[goal[0], goal[1], 0] = 1
    rgb[start[0], start[1], 1] = 1
    return rgb


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
        pts = place_points(grid, rng, args.min_dist_frac, args.goal_corners, args.corner_frac)
        if pts is None:
            rejected += 1
            continue
        maps.append(encode_rgb(grid, *pts))
    return np.stack(maps).astype(np.uint8), rejected


def save_pngs(maps, folder):
    from PIL import Image
    os.makedirs(folder, exist_ok=True)
    for i, m in enumerate(maps):
        # RGB PNG: R = goal, G = start, B = obstacles (values 0 / 255)
        Image.fromarray((m * 255).astype(np.uint8), mode="RGB").save(os.path.join(folder, f"{i:05d}.png"))


def save_preview(maps, path, cols=4, rows=4, scale=3, gap=4, marker=2):
    """Human-friendly preview: black = obstacle, white = free, green square = start, red square = goal."""
    from PIL import Image
    n = min(len(maps), cols * rows)
    H, W = maps[0].shape[:2]
    canvas = np.full((rows * (H * scale + gap) + gap, cols * (W * scale + gap) + gap, 3), 128, np.uint8)
    for k in range(n):
        m = maps[k]
        tile = np.kron(((1 - m[..., 2]) * 255).astype(np.uint8), np.ones((scale, scale), np.uint8))
        tile = np.stack([tile] * 3, -1)
        for ch, color in ((0, (255, 0, 0)), (1, (0, 200, 0))):
            r, c = np.argwhere(m[..., ch] > 0)[0]
            tile[max(0, (r - marker) * scale):(r + marker + 1) * scale,
                 max(0, (c - marker) * scale):(c + marker + 1) * scale] = color
        r, c = divmod(k, cols)
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
    p.add_argument("--min-dist-frac", type=float, default=0.5,
                   help="start must be at least this fraction of the width (in path steps) from the goal")
    p.add_argument("--goal-corners", action="store_true",
                   help="draw the goal from one of the four corner regions (default: anywhere on free space)")
    p.add_argument("--corner-frac", type=float, default=0.2, help="corner region size as fraction of width")
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
        print(f"{name:5s}: {len(maps)} maps (rejected {rej}), mean obstacle density {maps[..., 2].mean():.3f}")

    np.savez_compressed(args.out, **splits)
    print(f"saved {args.out}  (keys: train, val, test; shape (N, {args.width}, {args.width}, 3); R=goal, G=start, B=obstacle(1)/free(0))")

    if args.png_dir:
        for name, maps in splits.items():
            save_pngs(maps, os.path.join(args.png_dir, name))
        print(f"PNG files written to {args.png_dir}/")
    if args.preview:
        save_preview(splits["train"], args.preview)
        print(f"preview written to {args.preview}")


if __name__ == "__main__":
    main()
