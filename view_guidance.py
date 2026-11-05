"""
Save pictures of the maps next to their guidance maps.

    python view_guidance.py --maps your_maps.npz --guidance guidance.npz
    python view_guidance.py --maps your_maps.npz --guidance guidance.npz --split val --n 8
    python view_guidance.py --maps your_maps.npz --guidance guidance.npz --idx 0 5 17

Left : map (obstacle = blue, start = green, goal = red)
Right: guidance map (obstacles in blue, free cells coloured, dark = low cost / preferred)
Output: guidance_preview.png (change with --out); add --show to also open a window.
"""
import argparse

import matplotlib
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", required=True, help="the maps npz (with <split>, <split>_start, <split>_goal)")
    ap.add_argument("--guidance", required=True, help="npz written by make_guidance.py")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=6, help="number of random maps to show")
    ap.add_argument("--idx", type=int, nargs="*", default=None, help="specific map indices instead of random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="guidance_preview.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(args.maps)
    g = np.load(args.guidance)[f"{args.split}_guidance"]
    maps, st, gl = d[args.split], d[f"{args.split}_start"], d[f"{args.split}_goal"]
    N = maps.shape[0]
    assert g.shape[0] == N, f"{N} maps but {g.shape[0]} guidance maps"
    idx = args.idx if args.idx else sorted(np.random.default_rng(args.seed)
                                           .choice(N, min(args.n, N), replace=False).tolist())

    fig, ax = plt.subplots(len(idx), 2, figsize=(9, 4.3 * len(idx)), squeeze=False)
    for r, i in enumerate(idx):
        obs = maps[i, ..., 2] > 0.5
        rgb = np.ones(obs.shape + (3,))
        rgb[obs] = (0, 0, 1)
        ax[r, 0].imshow(rgb)
        ax[r, 0].scatter([st[i, 1]], [st[i, 0]], s=60, c="lime", edgecolors="k", label="start")
        ax[r, 0].scatter([gl[i, 1]], [gl[i, 0]], s=60, c="red", edgecolors="k", label="goal")
        ax[r, 0].set_title(f"{args.split} map {i}")
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad((0, 0, 1))
        im = ax[r, 1].imshow(np.ma.masked_where(obs, g[i]), cmap=cmap, vmin=0, vmax=1)
        ax[r, 1].scatter([st[i, 1]], [st[i, 0]], s=40, c="lime", edgecolors="k")
        ax[r, 1].scatter([gl[i, 1]], [gl[i, 0]], s=40, c="red", edgecolors="k")
        free = g[i][~obs]
        ax[r, 1].set_title(f"guidance (min {free.min():.3f}, mean {free.mean():.3f}, max {free.max():.3f})")
        fig.colorbar(im, ax=ax[r, 1], fraction=0.046)
        for a in ax[r]:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(args.out, dpi=100)
    print("saved", args.out)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
