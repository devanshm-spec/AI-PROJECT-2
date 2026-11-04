"""
neural_astar.py -- Neural A* (Yonetani et al., ICML 2021, "Path Planning using Neural A* Search")

Contents
    chebyshev_heuristic   H matrix (Chebyshev distance + 0.001 * Euclidean tie-breaker)
    DifferentiableAstar   Sec. 3.1 / Algorithm 2 -- A* written with matrix operations
    GuidanceEncoder       U-Net (VGG-16 backbone by default) producing the guidance map Phi in [0,1]
    NeuralAstar           encoder + differentiable A*  (the full planner)

Conventions (same as the paper)
    X          (B,1,H,W) binary map, 1 = passable, 0 = obstacle   (NOTE: the *dataset* stores obstacle = 1;
                the training script converts it with X = 1 - map)
    start_map  (B,1,H,W) one-hot (V_s)
    goal_map   (B,1,H,W) one-hot (V_g)
    Phi        (B,1,H,W) guidance cost, > 0
    histories  (B,1,H,W) closed list C (all nodes selected by the search)
"""
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def make_neighbor_kernel(dtype=torch.float32, device=None):
    """Fixed 3x3 kernel K of the paper: ones everywhere except the centre (8-neighbourhood)."""
    k = torch.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=dtype, device=device)
    return k.view(1, 1, 3, 3)


def chebyshev_heuristic(goal_map, tie_break=1e-3):
    """
    H(v) = Chebyshev distance(v, goal) + tie_break * Euclidean distance(v, goal)   (Sec. 4.2).
    goal_map: (B,1,H,W) one-hot.  Returns (B,1,H,W).
    """
    B, _, H, W = goal_map.shape
    idx = goal_map.reshape(B, -1).argmax(dim=1)
    gy = (idx // W).to(goal_map.dtype).view(B, 1, 1, 1)
    gx = (idx % W).to(goal_map.dtype).view(B, 1, 1, 1)
    ys = torch.arange(H, device=goal_map.device, dtype=goal_map.dtype).view(1, 1, H, 1)
    xs = torch.arange(W, device=goal_map.device, dtype=goal_map.dtype).view(1, 1, 1, W)
    dy = (ys - gy).abs()
    dx = (xs - gx).abs()
    cheb = torch.maximum(dy, dx)
    euc = torch.sqrt(dy ** 2 + dx ** 2)
    return cheb + tie_break * euc


@torch.no_grad()
def backtrack(parent, start_idx, goal_idx, H, W):
    """
    Algorithm 1 'Backtrack': follow Parent pointers from the goal to the start.
    parent: (B, H*W) long (-1 = never reached).  Returns (B,1,H,W) binary path map.
    """
    B, HW = parent.shape
    ar = torch.arange(B, device=parent.device)
    path = torch.zeros(B, HW, device=parent.device)
    cur = goal_idx.clone()
    valid = parent[ar, cur] >= 0              # goal was reached
    path[ar, cur] = 1.0
    for _ in range(HW):
        go = valid & (cur != start_idx)
        if not bool(go.any()):
            break
        nxt = parent[ar, cur]
        cur = torch.where(go, nxt, cur)
        path[ar[go], cur[go]] = 1.0
    return path.view(B, 1, H, W)


# --------------------------------------------------------------------------------------
# differentiable A*
# --------------------------------------------------------------------------------------
class DifferentiableAstar(nn.Module):
    """
    Differentiable A* module (Sec. 3.1, Algorithm 2).

    All lists are binary matrices of the map size:
        O  open list      C  closed list (search history)      G  accumulated guidance cost
    One iteration (for every sample of the mini-batch in parallel):
        Eq.(3)  V* = Imax( exp(-(G+H)/tau) * O / <exp(-(G+H)/tau), O> )    (straight-through)
        Eq.(8)  O <- O - eta V*,  C <- C + eta V*                          (freeze once goal found)
        Eq.(4)  Vnbr  = (V* conv K) * X * (1-O) * (1-C)     (newly opened neighbours)
                Vnbr' = (V* conv K) * X * O * (1-C)         (neighbours already in the open list)
        Eq.(5,6) G <- G' * Vnbr + min(G', G) * Vnbr' + G * (1 - Vnbr - Vnbr'),   G' = <G,V*> + Phi
    Gradients are disabled (detach) for O, Vnbr, Vnbr' and for G inside Eq.(6), exactly as in the paper;
    this keeps the recurrent back-propagation chain small enough to train.

    Args:
        tau        temperature of Eq.(3). None -> sqrt(map width), the paper's choice.
        tie_break  weight of the Euclidean tie-breaking term added to the Chebyshev heuristic.
        max_steps  safety cap on the number of search iterations (None -> H*W).
    """

    def __init__(self, tau=None, tie_break=1e-3, max_steps=None):
        super().__init__()
        self.tau = tau
        self.tie_break = tie_break
        self.max_steps = max_steps
        self.register_buffer("kernel", make_neighbor_kernel(), persistent=False)

    def forward(self, phi, X, start_map, goal_map, return_path=False):
        B, _, H, W = X.shape
        dev, dt = X.device, X.dtype
        tau = self.tau if self.tau is not None else math.sqrt(W)
        K = self.kernel.to(device=dev, dtype=dt)
        Hmat = chebyshev_heuristic(goal_map, self.tie_break)

        # ---- initialisation (Algorithm 2, lines 1-6) ----
        O = start_map.detach().clone()                    # open list  = {v_s}
        C = torch.zeros_like(X)                           # closed list = {}
        G = torch.zeros_like(X)                           # g(v_s) = 0
        done = torch.zeros(B, 1, 1, 1, device=dev, dtype=dt)   # 1 once the goal has been selected
        HW = H * W
        if return_path:
            start_idx = start_map.reshape(B, -1).argmax(1)
            goal_idx = goal_map.reshape(B, -1).argmax(1)
            ar = torch.arange(B, device=dev)
            parent = torch.full((B, HW), -1, dtype=torch.long, device=dev)
            parent[ar, start_idx] = start_idx

        max_steps = self.max_steps if self.max_steps is not None else HW
        steps = 0
        for _ in range(max_steps):
            if bool((done > 0.5).all()):
                break
            steps += 1

            # ---- node selection, Eq.(3) ----
            # exp(-(G+H)/tau) * O / <exp(.), O>  ==  softmax over the open list (numerically stable form)
            logits = (-(G + Hmat) / tau).masked_fill(O < 0.5, -1e9)
            soft = F.softmax(logits.reshape(B, -1), dim=1).view(B, 1, H, W)
            hard = F.one_hot(soft.detach().reshape(B, -1).argmax(1), HW).to(dt).view(B, 1, H, W)
            Vstar = hard + soft - soft.detach()          # Imax: one-hot forward, identity backward

            empty = (O.sum(dim=(1, 2, 3), keepdim=True) < 0.5).to(dt)     # open list exhausted
            searching = (1.0 - done) * (1.0 - empty)                       # samples still running
            reached = (hard * goal_map).sum(dim=(1, 2, 3), keepdim=True)   # 1 if goal selected now

            # ---- Eq.(8): update open / closed lists (frozen for finished samples) ----
            # NOTE: the goal itself is added to C at the step it is selected (the ground-truth path
            # contains the goal, so this is the sensible reading of Eq.(8); the difference is 1 pixel).
            C = C + searching * Vstar
            O = O - searching * hard

            # ---- node expansion, Eq.(4) (nothing is expanded once the goal is selected) ----
            expand = searching * (1.0 - reached)
            nbr = F.conv2d(hard, K, padding=1)                              # V* conv K   (no grad)
            closed = C.detach()
            Vnbr = nbr * X * (1.0 - O) * (1.0 - closed) * expand
            Vnbr_bar = nbr * X * O * (1.0 - closed) * expand

            # ---- cost update, Eq.(5)-(6) ----
            g_sel = (G.detach() * Vstar).sum(dim=(1, 2, 3), keepdim=True)   # <G, V*>  (G detached)
            G_new = g_sel + phi                                              # G'
            if return_path:
                with torch.no_grad():
                    improved = (Vnbr_bar > 0.5) & (G_new.detach() < G.detach())
                    upd = ((Vnbr > 0.5) | improved).reshape(B, -1)
                    sel_idx = hard.reshape(B, -1).argmax(1)
                    parent = torch.where(upd, sel_idx[:, None].expand(B, HW), parent)
            G = G_new * Vnbr + torch.minimum(G_new, G) * Vnbr_bar + G * (1.0 - Vnbr - Vnbr_bar)
            O = O + Vnbr

            done = ((done + searching * reached + empty) > 0.5).to(dt)

        out = {"histories": C, "steps": steps, "path": None, "solved": None}
        if return_path:
            out["path"] = backtrack(parent, start_idx, goal_idx, H, W)
            out["solved"] = parent[ar, goal_idx] >= 0
        return out


# --------------------------------------------------------------------------------------
# encoder
# --------------------------------------------------------------------------------------
def _conv_block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class SimpleUNet(nn.Module):
    """Small built-in U-Net (3 down-samplings). Used when --encoder simple or when smp is missing."""

    def __init__(self, in_ch=2, base=32):
        super().__init__()
        b = base
        self.e1, self.e2, self.e3 = _conv_block(in_ch, b), _conv_block(b, 2 * b), _conv_block(2 * b, 4 * b)
        self.bott = _conv_block(4 * b, 8 * b)
        self.up3, self.d3 = nn.ConvTranspose2d(8 * b, 4 * b, 2, stride=2), _conv_block(8 * b, 4 * b)
        self.up2, self.d2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, stride=2), _conv_block(4 * b, 2 * b)
        self.up1, self.d1 = nn.ConvTranspose2d(2 * b, b, 2, stride=2), _conv_block(2 * b, b)
        self.head = nn.Conv2d(b, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        b = self.bott(F.max_pool2d(e3, 2))
        d3 = self.d3(torch.cat([self.up3(b), e3], 1))
        d2 = self.d2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)


class GuidanceEncoder(nn.Module):
    """
    Fully-convolutional encoder (Sec. 3.2 'Encoder design').
    Input  : concat(X, V_s + V_g)  -> 2 channels
    Output : guidance map Phi in [0,1] (sigmoid), same size as the map.
    arch   : 'vgg16_bn' (default, U-Net + VGG-16 backbone via segmentation_models_pytorch),
             'resnet18' (paper's ablation), or 'simple' (built-in small U-Net, no extra dependency).
    """

    def __init__(self, arch="vgg16_bn", pretrained=False):
        super().__init__()
        self.net = None
        if arch != "simple":
            try:
                import segmentation_models_pytorch as smp
                self.net = smp.Unet(encoder_name=arch,
                                    encoder_weights="imagenet" if pretrained else None,
                                    in_channels=2, classes=1, activation=None)
            except ImportError:
                warnings.warn("segmentation_models_pytorch not installed "
                              "(pip install segmentation-models-pytorch) -> falling back to the built-in U-Net.")
        if self.net is None:
            self.net = SimpleUNet(in_ch=2)

    def forward(self, X, start_map, goal_map):
        x = torch.cat([X, start_map + goal_map], dim=1)
        H, W = x.shape[-2:]
        ph, pw = (-H) % 32, (-W) % 32            # U-Net with 5 stages needs sizes divisible by 32 (100 -> 128)
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), value=0.0)
        return torch.sigmoid(self.net(x))[..., :H, :W]


# --------------------------------------------------------------------------------------
# full planner
# --------------------------------------------------------------------------------------
class NeuralAstar(nn.Module):
    """Neural A* = GuidanceEncoder + DifferentiableAstar."""

    def __init__(self, encoder="vgg16_bn", pretrained=False, tau=None, tie_break=1e-3, max_steps=None):
        super().__init__()
        self.encoder = GuidanceEncoder(encoder, pretrained)
        self.astar = DifferentiableAstar(tau=tau, tie_break=tie_break, max_steps=max_steps)

    def forward(self, X, start_map, goal_map, return_path=False):
        phi = self.encoder(X, start_map, goal_map)                       # (1) guidance map
        out = self.astar(phi, X, start_map, goal_map, return_path)       # (2) differentiable A*
        out["phi"] = phi
        return out


@torch.no_grad()
def standard_astar(X, start_map, goal_map, tau=None, tie_break=1e-3, return_path=False):
    """Vanilla A* with the same heuristic: the differentiable module with unit guidance cost Phi = 1."""
    astar = DifferentiableAstar(tau=tau, tie_break=tie_break).to(X.device)
    return astar(torch.ones_like(X), X, start_map, goal_map, return_path)
