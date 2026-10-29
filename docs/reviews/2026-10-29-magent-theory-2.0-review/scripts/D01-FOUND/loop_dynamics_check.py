r"""Does the reciprocal Fisher dynamics respond to a loop transport in the stabilizer of the law?

Lens D01-FOUND.  Claim under test (chapters/01_foundations.tex:246): "Every interaction cost of
ch:interactions, ch:dynamics compares laws, so the dynamics defined there cannot respond to a loop
transport that lies in the stabilizer of the transported law."

Finite tables on Z_4, permutation channels, potential
    E = sum_a KL(nu_a || r_a) + beta * sum_{(a <- b)} KL(nu_a || (Theta_ab)_# nu_b),
rows fixed, unit Fisher flow  d/dt log nu_a = -(phi_a - E_{nu_a} phi_a)  with phi_a the first
variation (receiver term log(nu_a / Theta_# nu_b), source term -(nu_a / Theta_# nu_b) o Theta).
Gauge-invariant comparison quantities: edge costs, local terms (references transform with the
frame), and the loop comparison KL(nu_j || W_# nu_j).  RK4, 4000 steps on [0, 1].
Numerical agreement or disagreement at the tested instance is evidence, not a proof.
"""
from __future__ import annotations

import math

import numpy as np

N = 4
beta = 1.0


def perm_push(p: np.ndarray, perm: list[int]) -> np.ndarray:
    out = np.zeros_like(p)
    for y, fy in enumerate(perm):
        out[fy] += p[y]
    return out


def kl(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(p * np.log(p / q)))


def velocities(laws: list[np.ndarray], refs: list[np.ndarray], edges: list[tuple[int, int, list[int]]]) -> list[np.ndarray]:
    phis = [np.log(laws[a] / refs[a]) for a in range(len(laws))]
    for (a, b, perm) in edges:
        comp = perm_push(laws[b], perm)
        ratio = laws[a] / comp
        phis[a] = phis[a] + beta * np.log(ratio)                       # receiver role
        phis[b] = phis[b] - beta * np.array([ratio[perm[y]] for y in range(N)])  # source role
    return [-(phis[a] - laws[a] @ phis[a]) for a in range(len(laws))]


def run(laws0: list[np.ndarray], refs: list[np.ndarray], edges: list[tuple[int, int, list[int]]],
        T: float = 1.0, steps: int = 4000) -> list[np.ndarray]:
    ls = [np.log(p) for p in laws0]
    dt = T / steps

    def norm(l_: np.ndarray) -> np.ndarray:
        e = np.exp(l_ - l_.max()); return e / e.sum()

    def f_(ls_: list[np.ndarray]) -> list[np.ndarray]:
        return velocities([norm(l_) for l_ in ls_], refs, edges)

    for _ in range(steps):
        k1 = f_(ls)
        k2 = f_([l + dt / 2 * k for l, k in zip(ls, k1)])
        k3 = f_([l + dt / 2 * k for l, k in zip(ls, k2)])
        k4 = f_([l + dt * k for l, k in zip(ls, k3)])
        ls = [l + dt / 6 * (a + 2 * b + 2 * c + d) for l, a, b, c, d in zip(ls, k1, k2, k3, k4)]
    return [norm(l_) for l_ in ls]


ident = [0, 1, 2, 3]
shift = [1, 2, 3, 0]
unif = np.full(N, 0.25)

print("== 1  single edge i <- j, source uniform: passive equivalence check")
r_i = np.array([0.4, 0.3, 0.2, 0.1])
out_id = run([r_i.copy(), unif.copy()], [unif, unif], [(0, 1, ident)])
out_sh = run([r_i.copy(), unif.copy()], [unif, unif], [(0, 1, shift)])
inv_shift = [3, 0, 1, 2]
print("  receivers differ by:", float(np.max(np.abs(out_id[0] - out_sh[0]))))
print("  source(shift) - shift^{-1}# source(id):", float(np.max(np.abs(out_sh[1] - perm_push(out_id[1], inv_shift)))))
print("  (passive change by the shift at j maps one configuration to the other; no physical response)")

print("== 2  self-loop at one agent, Theta_ii = id versus shift, nu(0) uniform (in both stabilizers)")
r = np.array([0.5, 0.25, 0.15, 0.10])
v_id = velocities([unif.copy()], [r], [(0, 0, ident)])[0]
v_sh = velocities([unif.copy()], [r], [(0, 0, shift)])[0]
print("  t=0 log-velocity difference:", float(np.max(np.abs(v_id - v_sh))))
p_id = run([unif.copy()], [r], [(0, 0, ident)])[0]
p_sh = run([unif.copy()], [r], [(0, 0, shift)])[0]
print("  t=1 laws: id", np.round(p_id, 6), " shift", np.round(p_sh, 6))
print("  gauge-invariant KL(nu(1)||r): id", kl(p_id, r), " shift", kl(p_sh, r))
H = lambda p: float(-np.sum(p * np.log(p)))
print("  gauge-invariant entropy H(nu(1)): id", H(p_id), " shift", H(p_sh))
print("  loop comparison KL(nu(1)||Theta#nu(1)): id", kl(p_id, perm_push(p_id, ident)), " shift", kl(p_sh, perm_push(p_sh, shift)))

print("== 3  two-cycle i <- j <- i; B: (Theta_ij, Theta_ji) = (id, id); A: (shift, id); nu_j(0) uniform")
r_i = np.array([0.4, 0.3, 0.2, 0.1]); nu_i0 = np.array([0.1, 0.2, 0.3, 0.4])
refs = [r_i, unif]
edgesB = [(0, 1, ident), (1, 0, ident)]
edgesA = [(0, 1, shift), (1, 0, ident)]
lawsA0 = [nu_i0.copy(), unif.copy()]
for name, edges in [("B", edgesB), ("A", edgesA)]:
    D = [kl(lawsA0[a], perm_push(lawsA0[b], pm)) for a, b, pm in edges]
    print(f"  t=0 {name}: edge costs", np.round(D, 12))
vB = velocities([p.copy() for p in lawsA0], refs, edgesB)
vA = velocities([p.copy() for p in lawsA0], refs, edgesA)
print("  t=0 source (j) log-velocity B:", np.round(vB[1], 6), " A:", np.round(vA[1], 6))
outB = run([p.copy() for p in lawsA0], refs, edgesB)
outA = run([p.copy() for p in lawsA0], refs, edgesA)
for name, edges, out in [("B", edgesB, outB), ("A", edgesA, outA)]:
    D = [kl(out[a], perm_push(out[b], pm)) for a, b, pm in edges]
    W = [edges[1][2][edges[0][2][y]] for y in range(N)]  # loop based at j: first Theta_ij then Theta_ji
    loop = kl(out[1], perm_push(out[1], W))
    loc = [kl(out[a], refs[a]) for a in range(2)]
    print(f"  t=1 {name}: edge costs {np.round(D, 6)}, local terms {np.round(loc, 6)}, loop comparison at j {loop:.6f}")

print("== 4  second-order mechanism for the self-loop: KL(nu(t)||H#nu(t)) = (t^2/2) sum nu0 (v - v o H^-1)^2 + O(t^3)")
v0 = velocities([unif.copy()], [r], [(0, 0, shift)])[0]      # = record-free velocity, loop force vanishes at nu0
Hinv = [3, 0, 1, 2]
coef = 0.5 * float(np.sum(unif * (v0 - v0[Hinv]) ** 2))
for t_ in [0.02, 0.01, 0.005]:
    p_t = run([unif.copy()], [r], [(0, 0, shift)], T=t_, steps=400)[0]
    val = kl(p_t, perm_push(p_t, shift))
    print(f"  t={t_}: KL(nu(t)||shift#nu(t)) / t^2 = {val / t_**2:.6f}   predicted coefficient = {coef:.6f}")

print("== 5  first-order separation in the two-cycle: d/dt of the gauge-invariant edge costs at t=0")
for name, edges in [("B", edgesB), ("A", edgesA)]:
    rates = []
    for (a, b, pm) in edges:
        D0 = kl(lawsA0[a], perm_push(lawsA0[b], pm))
        hs = []
        for T_ in [1e-3, 5e-4]:
            out_ = run([p.copy() for p in lawsA0], refs, edges, T=T_, steps=50)
            hs.append((kl(out_[a], perm_push(out_[b], pm)) - D0) / T_)
        rates.append(2 * hs[1] - hs[0])   # Richardson extrapolation of the forward difference
    print(f"  {name}: dD_ij/dt(0) = {rates[0]:.6f}, dD_ji/dt(0) = {rates[1]:.6f}")
