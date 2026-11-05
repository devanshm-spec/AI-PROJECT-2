"""
Train a CNN heuristic for A* on 100x100 grid maps (8-connected).

Input  (B, 4, 100, 100):
    ch0  obstacle map            (0 = free, 1 = obstacle)   [plotted blue]
    ch1  start map (one-hot)                                 [plotted green]
    ch2  goal  map (one-hot)                                 [plotted red]
    ch3  guidance map in [0, 1]  (produced by a guidance CNN)

A* builds its g-values from the guidance map ONLY (there is no separate step cost):
when a node n' is opened from its parent n, the guidance value of n' is added:
    g(n') = g(n) + guidance[n']
So the cost of an edge n -> n' is  guidance[n']  (straight and diagonal moves cost
the same).  The guidance value of the start cell is never added.  Guidance values
must be >= 0 (this is checked); costs of exactly 0 are allowed but give a
heuristic nothing to work with, so keep the guidance strictly positive if you can.
Everything (labels, projection, validation A*) follows from this edge cost.

How admissibility + consistency are GUARANTEED (not just encouraged)
--------------------------------------------------------------------
1. The CNN predicts a raw table  h_raw = base(n) + residual(n),  residual >= 0, with
       base(n) = g_min * chebyshev(n, goal)
   (g_min = smallest guidance value on a free cell; every move enters a cell, so
   it pays at least g_min, and a path needs at least chebyshev-many moves).
2. h_raw is projected onto the set of consistent heuristics:
       h(n) = min_m [ h_raw(m) + d(n -> m) ]      with h(goal) forced to 0,
   where d is the true shortest-path cost on the obstacle graph under the
   guidance-based edge costs.  This is the largest consistent function below
   h_raw, so  h(n) <= c(n, n') + h(n')  for every edge  (consistent), and with
   h(goal) = 0 consistency implies admissibility.
3. Because h_raw >= base and base is itself consistent, the projected h also
   dominates base (it is never worse than the classical heuristic).
4. The projection is computed by parallel Bellman-Ford relaxation on the GPU,
   tracking the arg-min source so gradients flow exactly to the CNN output.

Training objective (expansion-aware)
------------------------------------
A* expands n  iff  g*(n) + h(n) < C*.  So we minimise
    relu(C* - g*(n) - h(n))      (the surrogate for "number of expansions"),
plus a tightness term (h* - h) and a penalty on raw over-estimation.

Usage:
    python train_cnn_heuristic.py --maps ocean_maps.npz --guidance_npz guidance.npz        # real guidance
    python train_cnn_heuristic.py --maps ocean_maps.npz --guidance_mode constant \\
        --guidance_value 1.0                                                              # all-constant map
    python train_cnn_heuristic.py --guidance_mode synthetic                                # fully synthetic
    # at inference:
    #   h = model.heuristic(x)[0]   # (B, H, W) provably consistent table
"""
import argparse
import heapq
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

H = W = 100
S = float(H)                      # normalisation scale for losses / residual
ALLOW_CORNER_CUTTING = True       # must match YOUR A*. True = a diagonal move may pass between two
                                  # diagonally touching obstacles (what generate_maps.py's connectivity
                                  # check assumes). False = diagonals blocked if either side cell is an obstacle.
# ---- WHERE THE GUIDANCE MAP COMES FROM (also settable on the command line) ----
#   "file"      : read precomputed guidance maps from --guidance_npz (needs --maps)
#   "constant"  : every cell gets the same guidance value CONSTANT_GUIDANCE_VALUE (no weights/maps needed)
#   "synthetic" : fake guidance (low values around the optimal path) -- pipeline testing only
GUIDANCE_MODE = "synthetic"
CONSTANT_GUIDANCE_VALUE = 1.0     # used when GUIDANCE_MODE == "constant"; must be > 0 (range [0,1])

SYN_GUIDANCE_MIN = 0.1            # SYNTHETIC guidance maps only: guidance range [MIN, MAX]
SYN_GUIDANCE_MAX = 1.0
SQRT2 = math.sqrt(2.0)
INF = float("inf")

#            NW        N       NE       W       E       SW       S       SE
OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
K_IDX = {o: i for i, o in enumerate(OFFSETS)}


# ----------------------------------------------------------------------------
# Cost model + graph utilities (all batched, all on GPU)
# ----------------------------------------------------------------------------
def neighbour_stack(t, pad):
    """(B,H,W) -> (B,8,H,W) with out[:,k,y,x] = t[:, y+dy_k, x+dx_k]."""
    p = F.pad(t, (1, 1, 1, 1), value=pad)
    return torch.stack(
        [p[:, 1 + dy:1 + dy + H, 1 + dx:1 + dx + W] for dy, dx in OFFSETS], dim=1)


def edge_costs(obstacle, guidance, direction):
    """
    Edge-cost tensor (B,8,H,W), +inf where the move is not allowed.
    Cost of a move = guidance value of the cell that is entered (no step term).
      direction='to_goal'    : C[:,k,y,x] = cost of moving  n -> n+off_k   (enter n+off_k)
      direction='from_start' : C[:,k,y,x] = cost of moving  n+off_k -> n   (enter n)
    Diagonal corner cutting is controlled by ALLOW_CORNER_CUTTING.
    """
    if bool((guidance < 0).any()):
        raise ValueError("guidance values must be >= 0 (edge costs must stay positive)")
    free = ~obstacle
    nb_free = neighbour_stack(free.float(), 0.0) > 0.5
    ok = nb_free & free.unsqueeze(1)
    for k, (dy, dx) in enumerate(OFFSETS):
        if dy != 0 and dx != 0 and not ALLOW_CORNER_CUTTING:
            ok[:, k] &= nb_free[:, K_IDX[(dy, 0)]] & nb_free[:, K_IDX[(0, dx)]]
    if direction == "to_goal":
        c = neighbour_stack(guidance, 0.0)
    else:
        c = guidance.unsqueeze(1).expand(-1, 8, -1, -1)
    return torch.where(ok, c, torch.full_like(c, INF))


def relax(val, C, src=None, max_iter=5000, check_every=8):
    """
    Parallel Bellman-Ford:  val[n] <- min_k ( val[n+off_k] + C[k, n] )  until convergence.
    If `src` (flat index of the cell whose initial value 'won') is given it is tracked.
    """
    for it in range(max_iter):
        best, arg = (neighbour_stack(val, INF) + C).min(1)
        better = best < val
        if src is not None:
            picked = neighbour_stack(src, 0).gather(1, arg.unsqueeze(1)).squeeze(1)
            src = torch.where(better, picked, src)
        val = torch.where(better, best, val)
        if (it + 1) % check_every == 0 and not bool(better.any()):
            break
    return val, src


def dist_field(source_mask, C):
    val = torch.full(source_mask.shape, INF, device=source_mask.device)
    val = val.masked_fill(source_mask, 0.0)
    return relax(val, C)[0]


def grids(device):
    yy = torch.arange(H, device=device).view(1, H, 1).float().expand(1, H, W)
    xx = torch.arange(W, device=device).view(1, 1, W).float().expand(1, H, W)
    return yy, xx


def octile_to_goal(goal):
    """Octile distance (in cells) to the goal + signed offsets. goal: (B,H,W)."""
    g = goal.float()
    yy, xx = grids(g.device)
    s = g.sum((1, 2)).clamp(min=1e-6)
    gy = (g * yy).sum((1, 2)) / s
    gx = (g * xx).sum((1, 2)) / s
    dy = yy - gy.view(-1, 1, 1)
    dx = xx - gx.view(-1, 1, 1)
    a, b = dy.abs(), dx.abs()
    octile = torch.maximum(a, b) + (SQRT2 - 1.0) * torch.minimum(a, b)
    return octile, dy, dx


def base_heuristic(goal, guidance, obstacle):
    """
    g_min * chebyshev: admissible AND consistent for edge cost = guidance[entered cell].
    (chebyshev distance changes by <= 1 per move and every move pays >= g_min.)
    """
    _, dy, dx = octile_to_goal(goal)
    cheb = torch.maximum(dy.abs(), dx.abs())
    g_min = guidance.masked_fill(obstacle, INF).amin((1, 2))
    g_min = torch.where(torch.isfinite(g_min), g_min, torch.zeros_like(g_min)).clamp(min=0.0)
    return g_min.view(-1, 1, 1) * cheb


def project_consistent(h_raw, C_to, goal, obstacle):
    """
    Largest consistent heuristic below h_raw with h(goal)=0 (see module docstring).
    Forward values are exact; gradients flow to the arg-min source cell of h_raw.
    """
    B = h_raw.shape[0]
    init = torch.where(goal, torch.zeros_like(h_raw), h_raw.detach())
    init = init.masked_fill(obstacle, INF)
    src0 = torch.arange(H * W, device=h_raw.device).view(1, H, W).expand(B, H, W).contiguous()
    with torch.no_grad():
        val, src = relax(init, C_to, src0)
    val = torch.where(obstacle, torch.zeros_like(val), val)
    gathered = h_raw.reshape(B, -1).gather(1, src.reshape(B, -1)).reshape(B, H, W)
    h = val + (gathered - gathered.detach())          # value == val, grad -> source cell
    return h


def max_consistency_violation(h, C_to):
    """max over allowed edges of  h(n) - (c(n,n') + h(n'))   (should be <= ~1e-5)."""
    nb = neighbour_stack(h, 0.0)
    viol = h.unsqueeze(1) - (nb + C_to)
    viol = viol.masked_fill(~torch.isfinite(C_to), -INF)
    return float(viol.amax())


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, ch, dilation):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.n1 = nn.GroupNorm(8, ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n2 = nn.GroupNorm(8, ch)

    def forward(self, x):
        y = F.gelu(self.n1(self.c1(x)))
        y = self.n2(self.c2(y))
        return F.gelu(x + y)


class HeuristicNet(nn.Module):
    """4-channel input -> provably admissible & consistent heuristic table."""

    def __init__(self, ch=64, dilations=(1, 2, 4, 8, 16, 32, 64, 1)):
        super().__init__()
        self.stem = nn.Conv2d(4 + 3, ch, 3, padding=1)      # +3 derived goal-relative features
        self.blocks = nn.Sequential(*[ResBlock(ch, d) for d in dilations])
        self.head = nn.Conv2d(ch, 1, 1)
        nn.init.constant_(self.head.bias, -4.0)

    def raw(self, x, obstacle):
        goal = x[:, 2] > 0.5
        octile, dy, dx = octile_to_goal(goal)
        feats = torch.cat([x, (dy / H).unsqueeze(1), (dx / H).unsqueeze(1),
                           (octile / H).unsqueeze(1)], dim=1)
        z = self.blocks(F.gelu(self.stem(feats)))
        residual = S * F.softplus(self.head(z).squeeze(1))               # >= 0
        return base_heuristic(goal, x[:, 3], obstacle) + residual        # >= base

    def heuristic(self, x):
        """returns (h_final, h_raw, C_to).  h_final is admissible + consistent."""
        obstacle = x[:, 0] > 0.5
        goal = x[:, 2] > 0.5
        C_to = edge_costs(obstacle, x[:, 3], "to_goal")
        h_raw = self.raw(x, obstacle)
        h = project_consistent(h_raw, C_to, goal, obstacle)
        return h, h_raw, C_to


# ----------------------------------------------------------------------------
# Synthetic data (swap in your own maps / your guidance CNN here)
# ----------------------------------------------------------------------------
@torch.no_grad()
def ground_truth(obstacle, start, goal, guidance):
    """Exact h* (cost-to-goal), g* (cost-from-start) and C* under edge cost = guidance[entered]."""
    h_star = dist_field(goal, edge_costs(obstacle, guidance, "to_goal"))
    g_star = dist_field(start, edge_costs(obstacle, guidance, "from_start"))
    return h_star, g_star, h_star[start]


@torch.no_grad()
def synthetic_guidance(obstacle, start, goal):
    """
    STAND-IN for a guidance CNN: low values (good) in a blurred band around the optimal path,
    higher values elsewhere, range [SYN_GUIDANCE_MIN, SYN_GUIDANCE_MAX]; 10% of maps get a
    constant (uninformative) guidance.  Use real guidance maps via --guidance_npz instead.
    """
    n, device = obstacle.shape[0], obstacle.device
    C1 = edge_costs(obstacle, torch.ones(n, H, W, device=device), "to_goal")     # 1 per move
    h0, g0 = dist_field(goal, C1), dist_field(start, C1)
    c0 = h0[start]
    excess = (g0 + h0) - c0.view(n, 1, 1)
    delta = torch.empty(n, 1, 1, device=device).uniform_(1.0, 12.0)
    band = (1.0 - excess / delta).clamp(0, 1).nan_to_num(0.0)
    sn = F.avg_pool2d(torch.rand(n, 1, H, W, device=device), 7, 1, 3, count_include_pad=False)[:, 0]
    sn2 = F.avg_pool2d(torch.rand(n, 1, H, W, device=device), 7, 1, 3, count_include_pad=False)[:, 0]
    path_score = (band * (0.5 + sn) + 0.25 * (sn2 > 0.55).float()).clamp(0, 1)
    span = SYN_GUIDANCE_MAX - SYN_GUIDANCE_MIN
    guidance = SYN_GUIDANCE_MIN + span * (1.0 - path_score)
    const = SYN_GUIDANCE_MIN + span * torch.rand(n, 1, 1, device=device)
    guidance = torch.where(torch.rand(n, 1, 1, device=device) < 0.1, const.expand(n, H, W), guidance)
    return guidance.masked_fill(obstacle, 0.0)


@torch.no_grad()
def get_guidance(obstacle, start, goal):
    """Guidance maps for the 'constant' and 'synthetic' modes ('file' is handled in load_dataset)."""
    if GUIDANCE_MODE == "constant":
        return torch.full(obstacle.shape, float(CONSTANT_GUIDANCE_VALUE), device=obstacle.device)
    if GUIDANCE_MODE == "synthetic":
        return synthetic_guidance(obstacle, start, goal)
    raise ValueError(f"GUIDANCE_MODE '{GUIDANCE_MODE}' needs precomputed maps (--maps + --guidance_npz)")


@torch.no_grad()
def _make_chunk(B, device, min_dist=30):
    # --- blobby / salt-and-pepper random obstacle maps, random density ---
    noise = torch.rand(B, 1, H, W, device=device)
    ks = [1, 3, 5, 9]
    stack = torch.stack([noise if k == 1 else
                         F.avg_pool2d(noise, k, 1, k // 2, count_include_pad=False) for k in ks], 0)
    pick = torch.randint(0, len(ks), (B,), device=device)
    blurred = stack[pick, torch.arange(B, device=device)].reshape(B, -1)
    dens = torch.empty(B, 1, device=device).uniform_(0.08, 0.40)
    thr = blurred.sort(1).values.gather(1, ((1 - dens) * (H * W - 1)).long())
    obstacle = (blurred > thr).view(B, H, W)

    # --- start / goal (free cells, at least `min_dist` apart in Chebyshev distance) ---
    free_f = (~obstacle).view(B, -1)
    sidx = torch.rand(B, H * W, device=device).masked_fill(~free_f, -1).argmax(1)
    yy = torch.arange(H, device=device).view(1, H, 1)
    xx = torch.arange(W, device=device).view(1, 1, W)
    cheb = torch.maximum((yy - (sidx // W).view(B, 1, 1)).abs(),
                         (xx - (sidx % W).view(B, 1, 1)).abs()).view(B, -1)
    cand = free_f & (cheb >= min_dist)
    gidx = torch.rand(B, H * W, device=device).masked_fill(~cand, -1).argmax(1)
    has_goal = cand.gather(1, gidx[:, None]).squeeze(1)
    start = F.one_hot(sidx, H * W).bool().view(B, H, W)
    goal = F.one_hot(gidx, H * W).bool().view(B, H, W)

    # --- guidance (synthetic stand-in), ground truth, keep only solvable samples ---
    guidance = get_guidance(obstacle, start, goal)
    h_star, g_star, cstar = ground_truth(obstacle, start, goal, guidance)
    keep = (has_goal & torch.isfinite(cstar)).nonzero().squeeze(1)
    if keep.numel() == 0:
        return None
    x = torch.stack([obstacle.float(), start.float(), goal.float(), guidance], 1)
    return dict(x=x[keep], h_star=h_star[keep], g_star=g_star[keep], cstar=cstar[keep])


@torch.no_grad()
def make_batch(B, device):
    parts, total = [], 0
    while total < B:
        c = _make_chunk(B, device)
        if c is not None:
            parts.append(c)
            total += c["x"].shape[0]
    return {k: torch.cat([p[k] for p in parts])[:B] for k in parts[0]}


# ----------------------------------------------------------------------------
# Maps from generate_maps.py  (R = goal, G = start, B = obstacle; exact coords stored separately)
# ----------------------------------------------------------------------------
@torch.no_grad()
def load_dataset(npz_path, split, guidance_npz, device, chunk=32):
    """
    Builds a training/validation set from the .npz written by generate_maps.py.
    Guidance maps depend on GUIDANCE_MODE: 'file' -> `guidance_npz` (arrays '<split>_guidance',
    shape (N,H,W), floats in [0,1], 0 = best); 'constant' -> one value everywhere;
    'synthetic' -> stand-in maps.  Unsolvable maps are dropped.
    """
    d = np.load(npz_path)
    maps = d[split]
    assert maps.shape[1:3] == (H, W), f"maps are {maps.shape[1:3]}, script is set to {H}x{W}"
    N = maps.shape[0]
    obstacle = torch.from_numpy(maps[..., 2] > 0)
    st, gl = d[f"{split}_start"], d[f"{split}_goal"]          # exact (row, col), ignores marker blobs
    start = torch.zeros(N, H, W, dtype=torch.bool)
    goal = torch.zeros(N, H, W, dtype=torch.bool)
    ar = np.arange(N)
    start[ar, st[:, 0], st[:, 1]] = True
    goal[ar, gl[:, 0], gl[:, 1]] = True
    guid = None
    if GUIDANCE_MODE == "file":
        assert guidance_npz, "GUIDANCE_MODE 'file' needs --guidance_npz"
        guid = torch.from_numpy(np.load(guidance_npz)[f"{split}_guidance"]).float()
        assert guid.shape == (N, H, W), f"guidance shape {tuple(guid.shape)} != {(N, H, W)}"
        assert float(guid.min()) >= 0, "guidance must be >= 0"
        if float(guid.max()) > 1.0 + 1e-6:
            print(f"warning: {split} guidance max is {float(guid.max()):.3f} (expected <= 1)")
    else:
        print(f"[{split}] guidance mode '{GUIDANCE_MODE}'"
              + (f" (value {CONSTANT_GUIDANCE_VALUE})" if GUIDANCE_MODE == "constant" else ""))
    parts = []
    for i in range(0, N, chunk):
        o, s_, g_ = (t[i:i + chunk].to(device) for t in (obstacle, start, goal))
        gd = guid[i:i + chunk].to(device) if guid is not None else get_guidance(o, s_, g_)
        h_star, g_star, cstar = ground_truth(o, s_, g_, gd)
        ok = torch.isfinite(cstar)
        x = torch.stack([o.float(), s_.float(), g_.float(), gd], 1)
        parts.append(dict(x=x[ok], h_star=h_star[ok], g_star=g_star[ok], cstar=cstar[ok]))
    ds = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    print(f"[{split}] {ds['x'].shape[0]}/{N} maps usable "
          f"(mean optimal cost C* = {float(ds['cstar'].mean()):.2f})")
    return ds


def sample_from(ds, B):
    idx = torch.randint(0, ds["x"].shape[0], (B,), device=ds["x"].device)
    return {k: v[idx] for k, v in ds.items()}


# ----------------------------------------------------------------------------
# Reference A* (used for validation: counts expansions, checks optimality)
# ----------------------------------------------------------------------------
def astar(obstacle, guidance, start, goal, h):
    """All inputs are numpy (H,W); start/goal are (y,x). Returns (path_cost, expansions)."""
    obs = obstacle.astype(bool).ravel().tolist()
    gd = guidance.ravel().tolist()
    hh = h.ravel().tolist()
    s, t = start[0] * W + start[1], goal[0] * W + goal[1]
    g = [INF] * (H * W)
    closed = bytearray(H * W)
    g[s] = 0.0
    heap = [(hh[s], 0.0, s)]
    expansions = 0
    while heap:
        _, ng, u = heapq.heappop(heap)             # ties -> larger g first
        if closed[u]:
            continue
        closed[u] = 1
        expansions += 1
        if u == t:
            return g[u], expansions
        uy, ux = divmod(u, W)
        for dy, dx in OFFSETS:
            vy, vx = uy + dy, ux + dx
            if not (0 <= vy < H and 0 <= vx < W):
                continue
            v = vy * W + vx
            if obs[v] or closed[v]:
                continue
            if (dy != 0 and dx != 0 and not ALLOW_CORNER_CUTTING
                    and (obs[(uy + dy) * W + ux] or obs[uy * W + ux + dx])):
                continue
            nv = g[u] + gd[v]               # guidance of v is added to g(v) when v is opened
            if nv < g[v]:
                g[v] = nv
                heapq.heappush(heap, (nv + hh[v], -nv, v))
    return INF, expansions


@torch.no_grad()
def validate(model, val, n_astar=12):
    model.eval()
    x = val["x"]
    h, h_raw, C_to = model.heuristic(x)
    obstacle = x[:, 0] > 0.5
    M = (~obstacle) & torch.isfinite(val["h_star"])
    stats = {
        "consistency_violation": max_consistency_violation(h, C_to),
        "admissibility_violation": float(((h - val["h_star"].nan_to_num(posinf=0.0)) * M).amax()),
        "raw_overshoot_frac": float(((h_raw > val["h_star"] + 1e-4) & M).float().sum() / M.sum()),
        "mean_gap_h*-h": float(((val["h_star"].nan_to_num(posinf=0.0) - h) * M).sum() / M.sum()),
    }
    h_base = base_heuristic(x[:, 2] > 0.5, x[:, 3], obstacle)
    e_learn, e_base, optimal = [], [], True
    for i in range(min(n_astar, x.shape[0])):
        o = x[i, 0].cpu().numpy()
        c = x[i, 3].cpu().numpy()
        st = tuple(int(v) for v in (x[i, 1] > 0.5).nonzero()[0].tolist())
        gl = tuple(int(v) for v in (x[i, 2] > 0.5).nonzero()[0].tolist())
        cl, el = astar(o, c, st, gl, h[i].cpu().numpy())
        cb, eb = astar(o, c, st, gl, h_base[i].cpu().numpy())
        optimal &= abs(cl - float(val["cstar"][i])) < 1e-3
        e_learn.append(el)
        e_base.append(eb)
    stats.update({"A*_expansions_learned": float(np.mean(e_learn)),
                  "A*_expansions_base": float(np.mean(e_base)),
                  "A*_paths_optimal": bool(optimal)})
    model.train()
    return stats


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def compute_loss(model, batch):
    x = batch["x"]
    obstacle = x[:, 0] > 0.5
    h, h_raw, _ = model.heuristic(x)
    h_star, g_star = batch["h_star"], batch["g_star"]
    M = ((~obstacle) & torch.isfinite(h_star) & torch.isfinite(g_star)).float()
    n = M.sum().clamp(min=1.0)
    hs = torch.nan_to_num(h_star, posinf=0.0)
    gs = torch.nan_to_num(g_star, posinf=0.0)

    # cells A* would still expand: g*(n) + h(n) < C*
    l_exp = (F.relu(batch["cstar"].view(-1, 1, 1) - gs - h) * M).sum() / n / S
    # tightness (h <= h* is guaranteed by the projection, so this is >= 0)
    l_gap = ((hs - h) * M).sum() / n / S
    # keep the RAW prediction admissible so the projection has little to repair
    l_over = (F.relu(h_raw - hs) * M).sum() / n / S
    loss = l_exp + 0.2 * l_gap + 5.0 * l_over
    return loss, dict(exp=float(l_exp), gap=float(l_gap), over=float(l_over))


def save_visual(x, h, path="sample.png"):
    """Obstacle = blue, start = green, goal = red, plus guidance and heuristic."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rgb = np.ones((H, W, 3))
    rgb[x[0].cpu().numpy() > 0.5] = (0, 0, 1)
    rgb[x[1].cpu().numpy() > 0.5] = (0, 1, 0)
    rgb[x[2].cpu().numpy() > 0.5] = (1, 0, 0)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(rgb); ax[0].set_title("map")
    ax[1].imshow(x[3].cpu().numpy(), cmap="viridis"); ax[1].set_title("guidance")
    ax[2].imshow(h.cpu().numpy(), cmap="magma"); ax[2].set_title("heuristic h")
    for a in ax:
        a.axis("off")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_every", type=int, default=250)
    ap.add_argument("--ckpt", type=str, default="heuristic_cnn.pt")
    ap.add_argument("--maps", type=str, default=None,
                    help="npz from generate_maps.py (omit to use fully synthetic maps)")
    ap.add_argument("--guidance_npz", type=str, default=None,
                    help="npz with train_guidance / val_guidance (N,H,W) in [0,1] (mode 'file')")
    ap.add_argument("--guidance_mode", choices=["synthetic", "file", "constant"], default=None,
                    help="source of the guidance map (default: 'file' if --guidance_npz is given, "
                         "else the GUIDANCE_MODE constant at the top of the file)")
    ap.add_argument("--guidance_value", type=float, default=None,
                    help="value of the all-constant guidance map (mode 'constant'; > 0, e.g. 1.0)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    global GUIDANCE_MODE, CONSTANT_GUIDANCE_VALUE
    GUIDANCE_MODE = args.guidance_mode or ("file" if args.guidance_npz else GUIDANCE_MODE)
    if args.guidance_value is not None:
        CONSTANT_GUIDANCE_VALUE = args.guidance_value
    if GUIDANCE_MODE == "file" and not (args.maps and args.guidance_npz):
        ap.error("guidance mode 'file' needs both --maps and --guidance_npz")
    if GUIDANCE_MODE == "constant":
        if CONSTANT_GUIDANCE_VALUE <= 0:
            ap.error("constant guidance value must be > 0")
        if CONSTANT_GUIDANCE_VALUE > 1:
            print("warning: constant guidance value > 1 (guidance is expected in [0, 1])")
    print(f"guidance mode: {GUIDANCE_MODE}"
          + (f"  (constant value {CONSTANT_GUIDANCE_VALUE})" if GUIDANCE_MODE == "constant" else ""))

    dev = torch.device(args.device)
    torch.manual_seed(0)
    if args.maps:
        train_ds = load_dataset(args.maps, "train", args.guidance_npz, dev)
        val = load_dataset(args.maps, "val", args.guidance_npz, dev)
        val = {k: v[:64] for k, v in val.items()}
        next_batch = lambda: sample_from(train_ds, args.batch)
    else:
        val = make_batch(32, dev)                   # fixed synthetic validation set
        next_batch = lambda: make_batch(args.batch, dev)
    model = HeuristicNet().to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M  device: {dev}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        batch = next_batch()
        loss, parts = compute_loss(model, batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 25 == 0:
            print(f"[{step:5d}] loss {loss.item():.4f}  exp {parts['exp']:.4f}  "
                  f"gap {parts['gap']:.4f}  over {parts['over']:.4f}  ({time.time() - t0:.0f}s)")
        if step % args.val_every == 0 or step == args.steps:
            print("   VAL:", {k: (round(v, 5) if isinstance(v, float) else v)
                              for k, v in validate(model, val).items()})
            torch.save(model.state_dict(), args.ckpt)

    with torch.no_grad():
        model.eval()
        h = model.heuristic(val["x"][:1])[0][0]
    try:
        save_visual(val["x"][0], h)
    except Exception as e:                           # matplotlib optional
        print("skipped visualisation:", e)


if __name__ == "__main__":
    main()
