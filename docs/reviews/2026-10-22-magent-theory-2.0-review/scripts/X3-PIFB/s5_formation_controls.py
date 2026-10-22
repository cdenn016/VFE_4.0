r"""X3-PIFB check 5: do the 2.0 formation events require the peer interaction? (no-interaction controls)

(a) Conditional formation (ch. 11): belief shapes obey d alpha_i/dt = -2(alpha_i - alpha_0) for every
    kappa >= 0 (the shape coefficient of the force (form:density-force) has no kappa), so the
    threshold events of thm form:selection are identical at kappa = 0 (no population coupling).
    Checked here by recomputing the event times from the stated formulas for kappa-independent shapes.
(b) Open-law routing (ch. 12): limiting field (open:limit)
        du_i/dt = -u_i + sum_j (w_ij + w_ji)(u_j - u_i),  w_ij = softmax_j(-(u_i-u_j)^2/2) incl. self,
    on the configuration used in the proof of prop open:routing, versus the control in which the
    peer forces are deleted (du/dt = -u) and the rows are only read out. The preferential-pair rule
    (eps = 1/10, delta = 1/20, both directions) is evaluated along both trajectories.
"""
from __future__ import annotations

import math

import numpy as np

# (a) ch. 11 event times are functions of initial shape gaps only
d1, d2, d3, delta = 0.2, 0.6, 0.3, 0.1
tL, tR = 0.5 * math.log(d1 / delta), 0.5 * math.log(d3 / delta)
t2 = 0.5 * math.log((d2 + (d1 + d3) / 2) / delta)
print(f"(a) ch.11 events from shapes alone (kappa-free): t_L={tL:.4f}, t_R={tR:.4f}, t_2={t2:.4f}")

# (b) ch. 12 limiting field
A, b0, R = 4096.0, 2.0, 1.0e7
u0 = np.array([-A - b0, -A + b0, A - b0, A + b0, R - 2.0, R + 2.0])
EPS, DEL = 0.1, 0.05


def rows(u: np.ndarray) -> np.ndarray:
    diff = u[:, None] - u[None, :]
    logits = -0.5 * diff ** 2
    logits -= logits.max(axis=1, keepdims=True)
    w = np.exp(logits)
    return w / w.sum(axis=1, keepdims=True)


def field(u: np.ndarray) -> np.ndarray:
    w = rows(u)
    s = w + w.T
    return -u + (s * (u[None, :] - u[:, None])).sum(axis=1)


def block_row(w: np.ndarray, I: list[int], J: list[int]) -> float:
    return w[np.ix_(I, J)].sum() / len(I)


def qualifies(w: np.ndarray, blocks: list[list[int]], I: list[int], J: list[int]) -> bool:
    ext = [K for K in blocks if K is not I and K is not J]
    for X, Y in ((I, J), (J, I)):
        bxy = block_row(w, X, Y)
        if not (bxy > EPS and bxy - sum(block_row(w, X, K) for K in ext) > DEL):
            return False
    return True


def run(mode: str, tmax: float = 9.0, dt: float = 1e-3) -> list[tuple[float, str]]:
    u = u0.copy()
    blocks: list[list[int]] = [[i] for i in range(6)]
    events: list[tuple[float, str]] = []
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
                        events.append((round(t, 3), f"{I}+{J}"))
                        blocks = [K for K in blocks if K is not I and K is not J] + [I + J]
                        merged = True
                        break
                if merged:
                    break
        if mode == "full":
            k1 = field(u)
            k2 = field(u + 0.5 * dt * k1)
            k3 = field(u + 0.5 * dt * k2)
            k4 = field(u + dt * k3)
            u = u + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        else:
            u = u0 * math.exp(-(t + dt))
        t += dt
    return events


ev_full = run("full")
ev_ctrl = run("control")
print("(b) events, full interacting limit field :", ev_full)
print("    events, no-peer-force control (u0 e^-t):", ev_ctrl)
