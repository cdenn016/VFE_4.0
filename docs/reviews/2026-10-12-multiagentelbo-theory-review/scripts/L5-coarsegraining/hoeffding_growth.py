r"""L5 checks: Hoeffding coordinates (thm:rg-hoeffding-action-isomorphism), the exact
interaction step and its derivative cocycle (lem:rg-interaction-step,
prop:rg-interaction-native-growth-bound), the two-agent remark, and the
intra-block invariance proposition, on binary networks with N <= 6 enumerated exhaustively.

Conventions follow c11: A(psi)(z) = -log sum_y Cdag(z,y) exp(-psi(y)),
Cdag(z,y) = mbar(y) C(y,z) / mbarc(z); hd = Hoeffding extraction relative to a product
reference nu; G-norm = sum_B ||g_B||_inf; quotient norm = oscillation norm (sup-inf)/2.
"""
from __future__ import annotations

import itertools
import numpy as np

rng = np.random.default_rng(20261012)


# ---------------------------------------------------------------- general product spaces
def coord_expectation(f: np.ndarray, nus: list[np.ndarray], keep: frozenset[int]) -> np.ndarray:
    r"""(E_B f)(y_B) = \int f(y_B, y'_{B^c}) nu_{B^c}(dy'), returned as a function on the full grid."""
    out = f.copy()
    for i, nu in enumerate(nus):
        if i in keep:
            continue
        m = np.tensordot(out, nu, axes=([i], [0]))  # integrate axis i
        out = np.expand_dims(m, axis=i) * np.ones_like(out)
    return out


def hoeffding_projector(f, nus, B):
    n = len(nus)
    out = np.zeros_like(f)
    for r in range(len(B) + 1):
        for Bp in itertools.combinations(sorted(B), r):
            out += (-1) ** (len(B) - r) * coord_expectation(f, nus, frozenset(Bp))
    return out


def check_hoeffding_identities() -> dict:
    sizes = (3, 2, 4)
    nus = [rng.dirichlet(np.ones(s)) for s in sizes]
    n = len(sizes)
    subsets = [frozenset(c) for r in range(n + 1) for c in itertools.combinations(range(n), r)]
    f = rng.normal(size=sizes)
    res = {}
    # E_B E_B' = E_{B cap B'}
    err = 0.0
    for B in subsets:
        for Bp in subsets:
            lhs = coord_expectation(coord_expectation(f, nus, B), nus, Bp)
            rhs = coord_expectation(f, nus, B & Bp)
            err = max(err, np.abs(lhs - rhs).max())
    res["E_B E_B' - E_{B cap B'}"] = err
    P = {B: hoeffding_projector(f, nus, B) for B in subsets}
    res["sum_B P_B f - f"] = np.abs(sum(P.values()) - f).max()
    err = 0.0
    for B in subsets:
        for Bp in subsets:
            lhs = hoeffding_projector(P[B], nus, Bp)
            rhs = P[B] if B == Bp else 0 * f
            err = max(err, np.abs(lhs - rhs).max())
    res["P_B' P_B - 1{B=B'} P_B"] = err
    # product formula prod_{i in B} (I - E_{V\{i}}) E_B
    err = 0.0
    full = frozenset(range(n))
    for B in subsets:
        g = coord_expectation(f, nus, B)
        for i in B:
            g = g - coord_expectation(g, nus, full - {i})
        err = max(err, np.abs(g - P[B]).max())
    res["product formula - Mobius sum"] = err
    # component depends only on y_B, zero nu-mean in each coordinate of B
    err_dep, err_mean = 0.0, 0.0
    for B in subsets:
        err_dep = max(err_dep, np.abs(coord_expectation(P[B], nus, B) - P[B]).max())
        for i in B:
            err_mean = max(err_mean, np.abs(coord_expectation(P[B], nus, full - {i})).max())
    res["P_B f depends only on y_B"] = err_dep
    res["P_B f has zero mean in each i in B"] = err_mean
    # hd norm bound: ||hd[f]||_G <= (3^n - 1) ||[f]||_osc on random f
    ratios = []
    for _ in range(2000):
        f2 = rng.normal(size=sizes) * rng.exponential(size=sizes)
        G = sum(np.abs(hoeffding_projector(f2, nus, B)).max() for B in subsets if B)
        osc = 0.5 * (f2.max() - f2.min())
        ratios.append(G / osc)
    res["max ||hd f||_G / ||[f]||_osc (bound 26)"] = max(ratios)
    return res


def check_sharpness() -> dict:
    out = {}
    for n in range(1, 7):
        p = 0.999
        mu = 2 * p - 1
        nus = [np.array([1 - p, p])] * n  # index 0 -> y=-1, 1 -> y=+1
        grid = np.array(list(itertools.product([-1, 1], repeat=n))).reshape((2,) * n + (n,))
        f = np.prod(grid, axis=-1).astype(float)
        G = 0.0
        for r in range(1, n + 1):
            for B in itertools.combinations(range(n), r):
                G += np.abs(hoeffding_projector(f, nus, frozenset(B))).max()
        formula = (4 * p - 1) ** n - (2 * p - 1) ** n
        out[n] = (G, formula, 3 ** n - 1)
    return out


# ---------------------------------------------------------------- binary networks (flat arrays)
class BinarySpace:
    """{-1,1}^n with a product reference; Walsh-type basis phi_B = prod_{i in B}(y_i - mu_i)."""

    def __init__(self, n: int, p: np.ndarray):
        self.n = n
        self.p = np.asarray(p, float)  # P(y_i = +1)
        self.mu = 2 * self.p - 1
        self.Y = np.array(list(itertools.product([-1, 1], repeat=n)), float)  # (2^n, n)
        self.subsets = [B for r in range(n + 1) for B in itertools.combinations(range(n), r)]
        cols = []
        for B in self.subsets:
            cols.append(np.prod(self.Y[:, list(B)] - self.mu[list(B)], axis=1) if B else np.ones(2 ** n))
        self.Phi = np.array(cols).T  # (2^n, 2^n)
        self.Phi_inv = np.linalg.inv(self.Phi)
        self.supnorm = np.array([np.prod(1 + np.abs(self.mu[list(B)])) if B else 1.0 for B in self.subsets])

    def hd(self, f: np.ndarray) -> np.ndarray:
        """Coefficients c_B (B nonempty) of the Hoeffding components of f."""
        return (self.Phi_inv @ f)[1:]

    def asm(self, c: np.ndarray) -> np.ndarray:
        return self.Phi[:, 1:] @ c

    def Gnorm(self, c: np.ndarray) -> float:
        return float(np.sum(np.abs(c) * self.supnorm[1:]))


def reverse_kernel(mbar: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    joint = mbar[:, None] * C  # (y, z)
    mc = joint.sum(0)
    return (joint / mc[None, :]).T, mc  # Cdag (z, y)


def action(Cdag: np.ndarray, psi: np.ndarray) -> np.ndarray:
    return -np.log(Cdag @ np.exp(-psi))


def tilted(Cdag: np.ndarray, psi: np.ndarray) -> np.ndarray:
    W = Cdag * np.exp(-psi)[None, :]
    return W / W.sum(1, keepdims=True)


def dobrushin(K: np.ndarray) -> float:
    d = 0.0
    for a in range(K.shape[0]):
        for b in range(K.shape[0]):
            d = max(d, 0.5 * np.abs(K[a] - K[b]).sum())
    return d


def op_norm(D: np.ndarray, S_in: BinarySpace, S_out: BinarySpace) -> float:
    """Exact norm G_in -> G_out: each H_B is one-dimensional for binary agents, so the
    l1-weighted operator norm is the max weighted column sum."""
    cols = np.abs(D) * S_out.supnorm[1:, None]
    return float(np.max(cols.sum(0) / S_in.supnorm[1:]))


def interaction_step(S0: BinarySpace, S1: BinarySpace, Cdag: np.ndarray, g: np.ndarray) -> np.ndarray:
    psi = S0.asm(g)
    return S1.hd(action(Cdag, psi))


def derivative(S0, S1, Cdag, g):
    psi = S0.asm(g)
    U = tilted(Cdag, psi)  # (z, y)
    return S1.hd(U @ S0.Phi[:, 1:]), U  # columns: images of basis directions


def random_full_support_law(n: int, conc: float = 1.0) -> np.ndarray:
    return rng.dirichlet(conc * np.ones(2 ** n))


def block_channel(S0: BinarySpace, S1: BinarySpace, blocks: list[list[int]], kernels: list[np.ndarray]) -> np.ndarray:
    """Block-product channel; kernels[I] is (2^{|I|}, 2) row stochastic, index order of itertools.product."""
    C = np.ones((2 ** S0.n, 2 ** S1.n))
    for a, y in enumerate(S0.Y):
        for b, z in enumerate(S1.Y):
            val = 1.0
            for I, (blk, K) in enumerate(zip(blocks, kernels)):
                idx = int("".join("1" if y[i] > 0 else "0" for i in blk), 2)
                val *= K[idx, 1 if z[I] > 0 else 0]
            C[a, b] = val
    return C


def det_kernel(fun, size: int) -> np.ndarray:
    K = np.zeros((2 ** size, 2))
    for idx, y in enumerate(itertools.product([-1, 1], repeat=size)):
        K[idx, 1 if fun(np.array(y)) > 0 else 0] = 1.0
    return K


def run_orbit(n_list, channels_builder, n_trials=200, scale=1.0, label=""):
    worst = {"ratio_native": 0.0, "telescope_err": 0.0, "fd_err": 0.0, "dobrushin_bound_violation": -np.inf,
             "min_slack_native": np.inf}
    for _ in range(n_trials):
        spaces = [BinarySpace(n, rng.uniform(0.15, 0.85, size=n)) for n in n_list]
        mbar = random_full_support_law(n_list[0], conc=0.7)
        Cs = channels_builder(spaces)
        laws, Cdags = [mbar], []
        for C in Cs:
            Cd, mc = reverse_kernel(laws[-1], C)
            Cdags.append(Cd)
            laws.append(mc)
        g = [rng.normal(size=2 ** n_list[0] - 1) * scale / np.sqrt(2 ** n_list[0])]
        Ds, Us = [], []
        for k in range(len(Cs)):
            D, U = derivative(spaces[k], spaces[k + 1], Cdags[k], g[k])
            Ds.append(D)
            Us.append(U)
            g.append(interaction_step(spaces[k], spaces[k + 1], Cdags[k], g[k]))
        Dprod = Ds[0]
        for D in Ds[1:]:
            Dprod = D @ Dprod
        # telescoped form hd_n U_{n-1} ... U_0 asm_0
        Uprod = Us[0]
        for U in Us[1:]:
            Uprod = U @ Uprod
        tele = spaces[-1].hd(Uprod @ spaces[0].Phi[:, 1:])
        worst["telescope_err"] = max(worst["telescope_err"], np.abs(tele - Dprod).max())
        # finite-difference derivative of the composite map
        h = 1e-6
        dirv = rng.normal(size=g[0].shape)
        def comp(gg):
            for k in range(len(Cs)):
                gg = interaction_step(spaces[k], spaces[k + 1], Cdags[k], gg)
            return gg
        fd = (comp(g[0] + h * dirv) - comp(g[0] - h * dirv)) / (2 * h)
        worst["fd_err"] = max(worst["fd_err"], np.abs(fd - Dprod @ dirv).max() / (1 + np.abs(fd).max()))
        nrm = op_norm(Dprod, spaces[0], spaces[-1])
        bound = 3 ** n_list[-1] - 1
        worst["ratio_native"] = max(worst["ratio_native"], nrm / bound)
        worst["min_slack_native"] = min(worst["min_slack_native"], bound - nrm)
        # sharper: bound * prod of tilted Dobrushin coefficients
        dprod = np.prod([dobrushin(U) for U in Us])
        worst["dobrushin_bound_violation"] = max(worst["dobrushin_bound_violation"], nrm - bound * dprod)
    return worst


def check_growth() -> dict:
    out = {}
    blocks = [[0, 1], [2, 3], [4, 5]]

    def det_parity(spaces):
        K = det_kernel(lambda y: y[0] * y[1], 2)
        C0 = block_channel(spaces[0], spaces[1], blocks, [K] * 3)
        Kmaj = det_kernel(lambda y: np.sign(y.sum()), 3)
        C1 = block_channel(spaces[1], spaces[2], [[0, 1, 2]], [Kmaj])
        return [C0, C1]

    def stoch_block(spaces):
        C0 = block_channel(spaces[0], spaces[1], blocks, [rng.dirichlet(np.ones(2), size=4) for _ in range(3)])
        C1 = block_channel(spaces[1], spaces[2], [[0, 1, 2]], [rng.dirichlet(np.ones(2), size=8)])
        return [C0, C1]

    def generic(spaces):
        C0 = rng.dirichlet(0.5 * np.ones(2 ** spaces[1].n), size=2 ** spaces[0].n)
        C1 = rng.dirichlet(0.5 * np.ones(2 ** spaces[2].n), size=2 ** spaces[1].n)
        return [C0, C1]

    def det_keep_three(spaces):
        Kfirst = det_kernel(lambda y: y[0], 2)
        C0 = block_channel(spaces[0], spaces[1], blocks, [Kfirst] * 3)
        Kpair = det_kernel(lambda y: y[0] * y[1], 2)
        Kid = det_kernel(lambda y: y[0], 1)
        C1 = block_channel(spaces[1], spaces[2], [[0, 1], [2]], [Kpair, Kid])
        return [C0, C1]

    for scale in (0.5, 3.0):
        out[f"6->3->1 deterministic parity/majority, scale {scale}"] = run_orbit([6, 3, 1], det_parity, scale=scale)
        out[f"6->3->1 stochastic block-product, scale {scale}"] = run_orbit([6, 3, 1], stoch_block, scale=scale)
        out[f"6->3->1 generic random channels, scale {scale}"] = run_orbit([6, 3, 1], generic, scale=scale)
        out[f"6->3->2 deterministic keep/parity, scale {scale}"] = run_orbit([6, 3, 2], det_keep_three, scale=scale)
    return out


def adversarial_search() -> dict:
    """Try to approach the bound 3^{|V_n|}-1 for a one-step 2->1 and 4->2 deterministic projection
    with a strongly biased target reference: ratio is hd_n norm limited."""
    best = {}
    for (n0, n1) in [(2, 1), (4, 2), (6, 3)]:
        top = 0.0
        for _ in range(400):
            p0 = rng.uniform(0.02, 0.98, size=n0)
            S0 = BinarySpace(n0, p0)
            # law equal to the product reference so that the pushed law is a product
            mbar = np.prod(np.where(S0.Y > 0, p0, 1 - p0), axis=1)
            S1 = BinarySpace(n1, p0[:n1])  # target reference = pushed marginals (keep first n1 agents)
            K = det_kernel(lambda y: y[0], 1)
            C = block_channel(S0, S1, [[i] for i in range(n1)], [K] * n1)
            Cd, _ = reverse_kernel(mbar, C)
            g = rng.normal(size=2 ** n0 - 1) * rng.uniform(0, 4)
            D, _ = derivative(S0, S1, Cd, g)
            top = max(top, op_norm(D, S0, S1))
        best[f"{n0}->{n1} keep-first, max ||D|| (bound {3 ** n1 - 1})"] = top
    return best


def check_two_agent() -> dict:
    S0 = BinarySpace(2, [0.5, 0.5])
    S1 = BinarySpace(1, [0.5])
    mbar = np.full(4, 0.25)
    C = np.zeros((4, 2))
    for a, y in enumerate(S0.Y):
        C[a, 1 if y[0] > 0 else 0] = 1.0
    Cd, _ = reverse_kernel(mbar, C)
    out = {}
    err = 0.0
    for _ in range(200):
        x1, x2, x12 = rng.normal(size=3)
        psi = -x1 * S0.Y[:, 0] - x2 * S0.Y[:, 1] - x12 * S0.Y[:, 0] * S0.Y[:, 1]
        c = S1.hd(action(Cd, psi))  # coefficient of phi_{I} = z
        xi1p = x1 + np.arctanh(np.tanh(x2) * np.tanh(x12))
        err = max(err, abs(c[0] - (-xi1p)))
    out["two-agent xi1' formula error"] = err
    # derivative of pair direction at psi = -xi2 y2
    x2 = 0.7
    g = S0.hd(-x2 * S0.Y[:, 1])
    D, _ = derivative(S0, S1, Cd, g)
    # basis order of subsets: (0,), (1,), (0,1)
    out["D(pair direction phi_{12}=y1y2) coefficient vs tanh(xi2)"] = (D[0, 2], np.tanh(x2))
    out["||D_0|| at this point (bound of prop: 3^1-1=2; remark quotes 3^2-1=8)"] = op_norm(D, S0, S1)
    return out


def check_intra_block() -> dict:
    """Product base law + block-product channel: components of T(g) on |B|>=2 vanish for g in the
    intra-block sector, for an arbitrary product reference at the target."""
    worst = 0.0
    worst_generic = 0.0
    for _ in range(100):
        p0 = rng.uniform(0.1, 0.9, size=6)
        S0 = BinarySpace(6, p0)
        mbar = np.prod(np.where(S0.Y > 0, p0, 1 - p0), axis=1)
        S1 = BinarySpace(3, rng.uniform(0.1, 0.9, size=3))
        blocks = [[0, 1], [2, 3], [4, 5]]
        C = block_channel(S0, S1, blocks, [rng.dirichlet(np.ones(2), size=4) for _ in range(3)])
        Cd, _ = reverse_kernel(mbar, C)
        g = np.zeros(63)
        for k, B in enumerate(S0.subsets[1:]):
            if any(set(B) <= set(b) for b in blocks):
                g[k] = rng.normal() * 2
        out = interaction_step(S0, S1, Cd, g)
        multi = [k for k, B in enumerate(S1.subsets[1:]) if len(B) >= 2]
        worst = max(worst, np.abs(out[multi]).max())
        # control: a non-product base law generates coarse pair components
        mbar2 = random_full_support_law(6)
        Cd2, _ = reverse_kernel(mbar2, C)
        out2 = interaction_step(S0, S1, Cd2, g)
        worst_generic = max(worst_generic, np.abs(out2[multi]).max())
    return {"max |multi-agent component| (product base)": worst,
            "control: max |multi-agent component| (non-product base)": worst_generic}


def gaussian_scope() -> dict:
    """The native bound is an L^inf statement. For Gaussian agents with a correlated base law
    the L^2(mbar) analogue of ||hd|| is unbounded: f = y1 + y2 has L^2(mbar) norm sqrt(2+2rho)
    while its two singleton components have L^2(mbar) norm 1 each."""
    out = {}
    for rho in (0.0, -0.9, -0.99, -0.9999):
        out[rho] = 2.0 / np.sqrt(2 + 2 * rho)
    return out


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    print("== Hoeffding identities (alphabets 3x2x4, random product reference)")
    for k, v in check_hoeffding_identities().items():
        print(f"  {k}: {v:.3e}")
    print("== Sharpness (4p-1)^n-(2p-1)^n at p=0.999")
    for n, (G, fo, b) in check_sharpness().items():
        print(f"  n={n}: G-norm {G:.6f}, formula {fo:.6f}, |diff| {abs(G - fo):.2e}, bound {b}")
    print("== Two-agent remark")
    for k, v in check_two_agent().items():
        print(f"  {k}: {v}")
    print("== Intra-block invariance")
    for k, v in check_intra_block().items():
        print(f"  {k}: {v:.3e}")
    print("== Exact derivative cocycle on binary networks (exhaustive state enumeration)")
    for k, v in check_growth().items():
        print(f"  {k}:")
        for kk, vv in v.items():
            print(f"     {kk}: {vv:.3e}")
    print("== Adversarial one-step search toward the bound")
    for k, v in adversarial_search().items():
        print(f"  {k}: {v:.4f}")
    print("== Gaussian scope: L2(mbar) extraction ratio 2/sqrt(2+2 rho)")
    for k, v in gaussian_scope().items():
        print(f"  rho={k}: {v:.3f}")
