#!/usr/bin/env python3
"""
stochastic_search.py -- adaptive-K SeeA* (uniform sampling) with CNN/Mahalanobis OOD detection
and a Neural-A*-style guidance map for g.

Pipeline (once per map): AE features -> Mahalanobis s -> empirical CDF u -> t -> K -> guidance map
-> SeeA*(uniform, K) with g(v')=g(v*)+Phi(v'), h = Chebyshev.

Needs train_ae.py in the same folder (or pass --ae-module).
Quick check without checkpoints:  python stochastic_search.py --self-test
"""
import argparse, csv, importlib.util, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ reuse train_ae.py
def import_train_ae(path=None):
    if path:
        spec = importlib.util.spec_from_file_location("train_ae", path)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
    import train_ae
    return train_ae


# ------------------------------------------------------------------ OOD -> u -> t -> K
class OODCalibrator:
    """Mahalanobis fit on TRAIN features; empirical CDF built from ID VALIDATION scores only."""
    def __init__(self, ae, f_train, f_val, ridge):
        self.ae = ae
        self.fit = ae.maha_fit(f_train, ridge)
        self.s_val = np.sort(ae.maha_score(f_val, self.fit))

    def score(self, feat):
        return float(self.ae.maha_score(feat, self.fit)[0])

    def cdf(self, s):  # F(s) = (1/N) * #{s_i <= s}
        return float(np.searchsorted(self.s_val, s, side="right") / len(self.s_val))


def compute_t(u, u0, u1):
    return float(min(1.0, max(0.0, (u - u0) / (u1 - u0))))


def compute_K(t, k_min, k_max, gamma):
    """K = Kmax^(1-t^gamma) * Kmin^(t^gamma); t=0 -> Kmax, t=1 -> Kmin."""
    w = t ** gamma
    k_float = (k_max ** (1.0 - w)) * (k_min ** w)
    k_int = int(min(k_max, max(k_min, round(k_float))))
    return k_float, k_int


# ------------------------------------------------------------------ guidance model
def build_guidance(ckpt_path, cfg_path, device):
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = json.load(open(cfg_path)) if cfg_path else {}
    mode = cfg.get("encoder_input", "m+")          # "m+" = [map, start+goal]; "m" = [map]
    if isinstance(obj, nn.Module):
        return obj.to(device).eval(), mode
    sd = obj
    for k in ("state_dict", "model", "model_state_dict"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]; break
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        raise SystemExit("Guidance checkpoint is a state_dict: need `pip install segmentation-models-pytorch` "
                         "(Neural A* encoder), or supply a full pickled nn.Module.")
    net = smp.Unet(encoder_name=cfg.get("encoder_name", "vgg16_bn"), encoder_weights=None,
                   classes=1, in_channels=cfg.get("in_channels", 2 if mode == "m+" else 1),
                   encoder_depth=cfg.get("encoder_depth", 4),
                   decoder_channels=cfg.get("decoder_channels", [256, 128, 64, 32]))
    last = None
    for strip in ("", "model.", "encoder.", "net.", "module."):   # tolerate Lightning/DataParallel prefixes
        cand = {(k[len(strip):] if strip and k.startswith(strip) else k): v for k, v in sd.items()}
        try:
            net.load_state_dict(cand, strict=True); last = None; break
        except RuntimeError as e:
            last = e
    if last is not None:
        raise SystemExit("Guidance checkpoint does not match the assumed architecture. Provide the true "
                         "configuration via --guidance-model-config (encoder_name, in_channels, encoder_depth, "
                         f"decoder_channels, encoder_input) or a pickled module.\nDetails: {str(last)[:1500]}")
    return net.to(device).eval(), mode


@torch.no_grad()
def guidance_map(net, mode, passable, start, goal, device, pad_to=32):
    H, W = passable.shape
    m = torch.from_numpy(passable.astype(np.float32))[None, None]   # guidance model: 1 = passable
    sg = torch.zeros(1, 1, H, W); sg[0, 0, start[0], start[1]] = 1; sg[0, 0, goal[0], goal[1]] = 1
    x = torch.cat([m, sg], 1) if mode == "m+" else m
    ph, pw = (-H) % pad_to, (-W) % pad_to
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), value=0.0)
    out = net(x.to(device))
    if out.min() < 0 or out.max() > 1:
        out = torch.sigmoid(out)                    # paper: sigmoid-activated final layer
    phi = out[0, 0, :H, :W].float().cpu().numpy()
    return np.clip(np.nan_to_num(phi, nan=1.0), 0.0, None)   # nonnegative, same size as map


# ------------------------------------------------------------------ SeeA* (uniform sampling)
_NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]   # 8-neighbour (paper)


def seea_star(passable, phi, start, goal, K, seed=0, max_exp=10**9):
    """K=None (or >=|OPEN|) -> whole OPEN, i.e. standard A*. Exactly one node expanded per iteration.
    g(v') = g(v*) + Phi(v'); f = g + Chebyshev h; f is computed from the CURRENT g (no stale keys)."""
    H, W = passable.shape
    assert passable.shape == phi.shape and (phi >= 0).all()
    assert passable[start] and passable[goal], "start/goal must be passable"
    rng = np.random.default_rng(seed)
    N = H * W; s = start[0] * W + start[1]; gl = goal[0] * W + goal[1]
    rr, cc = np.divmod(np.arange(N), W)
    h = np.maximum(np.abs(rr - goal[0]), np.abs(cc - goal[1])).astype(np.float64)   # Chebyshev
    g = np.full(N, np.inf); g[s] = 0.0
    parent = np.full(N, -1, dtype=np.int64)
    state = np.zeros(N, dtype=np.int8)                  # 0 unseen, 1 OPEN, 2 CLOSED
    phi_f = phi.reshape(-1); pas = passable.reshape(-1)
    open_list = [s]; pos = {s: 0}; state[s] = 1
    generated, expanded, iters, found = 1, 0, 0, False
    t0 = time.perf_counter()
    while open_list and expanded < max_exp:
        n = len(open_list)
        iters += 1
        if K is None or K >= n:
            cand_ids = np.asarray(open_list)
        else:
            idx = rng.choice(n, size=int(K), replace=False)           # uniform, without replacement
            cand_ids = np.asarray([open_list[i] for i in idx])
        f = g[cand_ids] + h[cand_ids]
        v = int(cand_ids[int(np.argmin(f))])                          # min-f inside the subset
        i = pos.pop(v); last = open_list.pop()                        # O(1) removal from OPEN
        if i < len(open_list):
            open_list[i] = last; pos[last] = i
        state[v] = 2; expanded += 1
        if v == gl:
            found = True; break
        vr, vc = divmod(v, W)
        for dr, dc in _NB:
            r, c = vr + dr, vc + dc
            if not (0 <= r < H and 0 <= c < W): continue
            u = r * W + c
            if not pas[u] or state[u] == 2: continue
            gn = g[v] + phi_f[u]                                      # Neural A* Eq. 5/6
            if state[u] == 0:
                state[u] = 1; g[u] = gn; parent[u] = v
                pos[u] = len(open_list); open_list.append(u); generated += 1
            elif gn < g[u]:
                g[u] = gn; parent[u] = v
    runtime = time.perf_counter() - t0
    path = []
    if found:
        v = gl
        while v != -1:
            path.append((v // W, v % W)); v = int(parent[v])
        path.reverse()
    return dict(found=found, path=path, path_len=len(path), cost=float(g[gl]) if found else float("nan"),
                expanded=expanded, generated=generated, open_end=len(open_list),
                runtime=runtime, iters=iters)


# ------------------------------------------------------------------ data helpers
def load_test_maps(path, key, index):
    d = np.load(path)
    arr = d[key] if hasattr(d, "files") else d
    if arr.ndim == 2: arr = arr[None]
    elif arr.ndim == 3 and arr.shape[-1] == 3: arr = arr[None]
    if arr.ndim == 4: arr = arr[..., 2]                  # blue channel, like train_ae.load_split
    arr = arr.astype(np.float32)
    sel = range(len(arr)) if index == "all" else [int(i) for i in index.split(",")]
    return [(i, arr[i]) for i in sel]


def parse_rc(s):
    r, c = s.split(","); return int(r), int(c)


def random_start_goal(passable, rng):
    free = np.argwhere(passable); mind = 0.5 * min(passable.shape)
    for _ in range(1000):
        a, b = free[rng.integers(len(free))], free[rng.integers(len(free))]
        if max(abs(a[0] - b[0]), abs(a[1] - b[1])) >= mind: return tuple(int(x) for x in a), tuple(int(x) for x in b)
    raise SystemExit("could not sample start/goal")


# ------------------------------------------------------------------ main
def report(tag, r):
    print(f"  [{tag}] found={r['found']} path_len={r['path_len']} cost={r['cost']:.4f} expanded={r['expanded']} "
          f"generated={r['generated']} open_end={r['open_end']} time={r['runtime']:.4f}s iters={r['iters']}")


def self_test():
    assert compute_K(0, 4, 64, 1)[1] == 64 and compute_K(.5, 4, 64, 1)[1] == 16 and compute_K(1, 4, 64, 1)[1] == 4
    assert compute_t(0.5, .95, .995) == 0 and compute_t(1.0, .95, .995) == 1
    rng = np.random.default_rng(0)
    pas = rng.random((20, 20)) > 0.2; pas[0, 0] = pas[19, 19] = True
    phi = np.full((20, 20), 0.5)
    for K in (None, 1, 4, 64):
        r = seea_star(pas, phi, (0, 0), (19, 19), K, seed=1)
        print("self-test K=", K, {k: r[k] for k in ("found", "path_len", "cost", "expanded", "iters")})
        if r["found"]:
            assert abs(r["cost"] - 0.5 * (r["path_len"] - 1)) < 1e-9     # g = sum of Phi along path
    print("self-test OK (K formula, thresholds, search bookkeeping)")


def main():
    p = argparse.ArgumentParser()
    g = p.add_argument_group("OOD")
    g.add_argument("--ae-checkpoint"); g.add_argument("--data", help="ID npz with train/val splits")
    g.add_argument("--test-map", help=".npy/.npz map(s): (H,W), (H,W,3), (N,H,W,3)")
    g.add_argument("--test-key", default="test"); g.add_argument("--test-index", default="0",
                   help="comma list of indices or 'all'")
    g.add_argument("--ridge", type=float, default=1e-3); g.add_argument("--batch-size", type=int, default=128)
    g.add_argument("--ae-module", default=None, help="path to train_ae.py if not importable")
    a = p.add_argument_group("adaptive K")
    a.add_argument("--k-min", type=int, default=4); a.add_argument("--k-max", type=int, default=64)
    a.add_argument("--u0", type=float, default=0.95); a.add_argument("--u1", type=float, default=0.995)
    a.add_argument("--gamma", type=float, default=1.0)
    m = p.add_argument_group("guidance"); m.add_argument("--guidance-checkpoint")
    m.add_argument("--no-guidance", action="store_true",
                   help="FALLBACK for testing only: Phi=1 everywhere (unit cost), no guidance model loaded")
    m.add_argument("--guidance-model-config", default=None)
    s = p.add_argument_group("search")
    s.add_argument("--start"); s.add_argument("--goal"); s.add_argument("--random-start-goal", action="store_true")
    s.add_argument("--seed", type=int, default=0); s.add_argument("--max-expansions", type=int, default=10**9)
    s.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    c = p.add_argument_group("comparison")
    c.add_argument("--compare", action="store_true", help="also run A* and fixed-K SeeA*")
    c.add_argument("--compare-k", default="1,4,16,64"); c.add_argument("--repeats", type=int, default=1)
    p.add_argument("--csv", default="results.csv"); p.add_argument("--self-test", action="store_true")
    A = p.parse_args()

    if A.self_test: return self_test()
    for need in ("ae_checkpoint", "data", "test_map") + (() if A.no_guidance else ("guidance_checkpoint",)):
        if getattr(A, need) is None: p.error(f"--{need.replace('_', '-')} is required")
    if not A.u1 > A.u0: p.error("need u1 > u0")
    if A.k_max < A.k_min or A.k_min < 1: p.error("need k_max >= k_min >= 1")
    if A.gamma <= 0: p.error("need gamma > 0")
    if not (A.start and A.goal) and not A.random_start_goal: p.error("give --start/--goal or --random-start-goal")

    dev = A.device if (not A.device.startswith("cuda") or torch.cuda.is_available()) else "cpu"
    ae = import_train_ae(A.ae_module)
    ck = torch.load(A.ae_checkpoint, map_location=dev)
    ae_net = ae.ConvAE(**ck["cfg"]).to(dev); ae_net.load_state_dict(ck["model"]); ae_net.eval()

    f_tr, _ = ae.run(ae_net, ae.load_split(A.data, "train"), dev, A.batch_size)
    f_va, _ = ae.run(ae_net, ae.load_split(A.data, "val"), dev, A.batch_size)
    cal = OODCalibrator(ae, f_tr, f_va, A.ridge)
    print(f"fitted Mahalanobis on {len(f_tr)} train maps; CDF from {len(f_va)} ID validation maps")

    if A.no_guidance:
        gnet, gmode = None, None
        print("WARNING: --no-guidance -> Phi=1 (plain unit-cost A*/SeeA*), NOT Neural A* guidance")
    else:
        gnet, gmode = build_guidance(A.guidance_checkpoint, A.guidance_model_config, dev)
    rng = np.random.default_rng(A.seed)
    new_csv = not os.path.exists(A.csv); rows = []

    for idx, obst in load_test_maps(A.test_map, A.test_key, A.test_index):
        passable = obst < 0.5
        x = torch.from_numpy(obst)[None, None]
        feat, _ = ae.run(ae_net, x, dev, 1)
        sc = cal.score(feat); u = cal.cdf(sc); t = compute_t(u, A.u0, A.u1)
        k_float, K = compute_K(t, A.k_min, A.k_max, A.gamma)       # once per map, fixed for the search
        if A.random_start_goal: st, gl = random_start_goal(passable, rng)
        else: st, gl = parse_rc(A.start), parse_rc(A.goal)
        phi = np.ones(passable.shape) if A.no_guidance else guidance_map(gnet, gmode, passable, st, gl, dev)

        print(f"\n=== map {idx} | start={st} goal={gl} ===")
        print(f"OOD: mahalanobis={sc:.4f}  u={u:.4f}  u0={A.u0} u1={A.u1}  t={t:.4f}")
        print(f"K:   k_min={A.k_min} k_max={A.k_max} gamma={A.gamma}  K_float={k_float:.4f}  K={K}")
        base = dict(map=idx, start=st, goal=gl, mahalanobis=sc, u=u, t=t, K_float=k_float, K_adaptive=K, seed=A.seed)

        methods = [("adaptive", K)]
        if A.compare:
            methods = [("A*", None)] + [(f"fixedK={k}", int(k)) for k in A.compare_k.split(",")] + methods
        for name, k in methods:
            rs = [seea_star(passable, phi, st, gl, k, A.seed + j, A.max_expansions) for j in range(A.repeats)]
            r = rs[0]
            if A.repeats > 1:
                for key in ("path_len", "cost", "expanded", "generated", "open_end", "runtime", "iters"):
                    r[key] = float(np.nanmean([q[key] for q in rs]))
                r["found"] = all(q["found"] for q in rs)
            report(name, r)
            if name == "adaptive" and r["found"] and A.repeats == 1:
                print("  path:", r["path"])
            rows.append({**base, "method": name, "K_used": "all" if k is None else k,
                         **{k_: r[k_] for k_ in ("found", "path_len", "cost", "expanded", "generated",
                                                  "open_end", "runtime", "iters")}})
    with open(A.csv, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if new_csv: w.writeheader()
        w.writerows(rows)
    print(f"\nsaved {len(rows)} rows to {A.csv}")


if __name__ == "__main__":
    main()
