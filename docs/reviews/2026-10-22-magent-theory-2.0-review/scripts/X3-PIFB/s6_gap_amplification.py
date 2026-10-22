r"""X3-PIFB check 6 (exploratory, numerical observation only): does the ch. 12 limiting field amplify a
small initial pairing asymmetry? Four agents u = (-x3, -x1, x1, x3) (reflection-symmetric), with gaps
g_out = x3 - x1 (outer pairs) and g_in = 2 x1 (middle). Under the no-interaction control du/dt = -u the
ratio g_out/g_in is constant. Under the full limit field it can drift; drift toward 0 means the outer
pairs {1,2},{3,4} are being selected, drift up means the middle pair {2,3} is being selected.
"""
from __future__ import annotations

import numpy as np


def rows(u: np.ndarray) -> np.ndarray:
    diff = u[:, None] - u[None, :]
    lg = -0.5 * diff ** 2
    lg -= lg.max(axis=1, keepdims=True)
    w = np.exp(lg)
    return w / w.sum(axis=1, keepdims=True)


def field(u: np.ndarray) -> np.ndarray:
    w = rows(u)
    s = w + w.T
    return -u + (s * (u[None, :] - u[:, None])).sum(axis=1)


def ratio_path(u0: np.ndarray, tmax: float = 6.0, dt: float = 1e-3) -> list[tuple[float, float]]:
    u = u0.copy()
    out = []
    t = 0.0
    nxt = 0.0
    while t <= tmax:
        if t >= nxt - 1e-12:
            g_out = 0.5 * ((u[1] - u[0]) + (u[3] - u[2]))
            g_in = u[2] - u[1]
            out.append((round(t, 2), g_out / g_in))
            nxt += 1.0
        k1 = field(u)
        k2 = field(u + 0.5 * dt * k1)
        k3 = field(u + 0.5 * dt * k2)
        k4 = field(u + dt * k3)
        u = u + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t += dt
    return out


for scale in (1.0, 2.0, 3.0):
    for eps in (-0.05, 0.0, 0.05):
        x1 = scale * (1.0 + eps)
        x3 = scale * 3.0
        u0 = np.array([-x3, -x1, x1, x3])
        path = ratio_path(u0)
        print(f"scale={scale}, eps={eps:+.2f}: ratio g_out/g_in over t=0..6 ->",
              [f"{r:.3f}" for _, r in path])


# Threshold events (rule open:selector, eps=1/10, delta=1/20) for the same four-agent data,
# full limit field versus the no-peer-force control u0 e^{-t}.
EPS, DEL = 0.1, 0.05


def block_row(w: np.ndarray, I: list[int], J: list[int]) -> float:
    return w[np.ix_(I, J)].sum() / len(I)


def qualifies(w: np.ndarray, blocks: list[list[int]], I: list[int], J: list[int]) -> bool:
    ext = [K for K in blocks if K is not I and K is not J]
    for X, Y in ((I, J), (J, I)):
        bxy = block_row(w, X, Y)
        if not (bxy > EPS and bxy - sum(block_row(w, X, K) for K in ext) > DEL):
            return False
    return True


def events(u0: np.ndarray, mode: str, tmax: float = 8.0, dt: float = 1e-3) -> list[tuple[float, str]]:
    u = u0.copy()
    blocks: list[list[int]] = [[i] for i in range(len(u0))]
    ev: list[tuple[float, str]] = []
    t = 0.0
    while t < tmax:
        w = rows(u)
        merged = True
        while merged:
            merged = False
            for a_ in range(len(blocks)):
                for b_ in range(a_ + 1, len(blocks)):
                    I, J = blocks[a_], blocks[b_]
                    if qualifies(w, blocks, I, J):
                        ev.append((round(t, 3), f"{I}+{J}"))
                        blocks = [K for K in blocks if K is not I and K is not J] + [I + J]
                        merged = True
                        break
                if merged:
                    break
        if mode == "full":
            k1 = field(u); k2 = field(u + 0.5 * dt * k1); k3 = field(u + 0.5 * dt * k2); k4 = field(u + dt * k3)
            u = u + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        else:
            u = u0 * np.exp(-(t + dt))
        t += dt
    return ev


print("threshold events (full vs control):")
for scale in (2.0, 3.0, 4.0):
    for eps in (-0.05, 0.0):
        x1 = scale * (1.0 + eps); x3 = scale * 3.0
        u0 = np.array([-x3, -x1, x1, x3])
        print(f"  scale={scale}, eps={eps:+.2f}: full {events(u0, 'full')} | control {events(u0, 'control')}")
