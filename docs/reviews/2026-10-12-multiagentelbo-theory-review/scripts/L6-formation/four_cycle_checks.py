r"""L6-formation: exact checks of the four-cycle witness (c13 prop:rg-formation-endogenous-cycle,
c14 prop:phil-seeded-contrast-formation, ca_selection_proofs sheet determinants and saddle spectrum,
ca_temperature_bounds prop:phil-temperature-tracking).

Coordinates x = (kbar_0..3, mbar_0..3, omega_0..3); edge e_i : i -> i+1,
r^b_i = kbar_{i+1} - kbar_i - omega_i,  r^m_i = mbar_{i+1} - mbar_i - omega_i.
Optimizer precision (times T): A_S = H_S + T (I + R_ev), R_ev = 2 D^T D, D x = kbar - mbar.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm, null_space

np.set_printoptions(precision=6, suppress=True)


def cov_b(i: int) -> np.ndarray:
    a = np.zeros(12)
    a[(i + 1) % 4] += 1.0
    a[i] -= 1.0
    a[8 + i] -= 1.0
    return a


def cov_m(i: int) -> np.ndarray:
    a = np.zeros(12)
    a[4 + (i + 1) % 4] += 1.0
    a[4 + i] -= 1.0
    a[8 + i] -= 1.0
    return a


def edge_of(i: int, j: int) -> int:
    """Edge index joining receiver i and source j (j = i +- 1)."""
    return i if j == (i + 1) % 4 else (i - 1) % 4


Dm = np.zeros((4, 12))
for i in range(4):
    Dm[i, i] = 1.0
    Dm[i, 4 + i] = -1.0
R_ev = 2.0 * Dm.T @ Dm


def H_of(S: tuple[int, ...]) -> np.ndarray:
    H = np.zeros((12, 12))
    for i, j in enumerate(S):
        e = edge_of(i, j)
        H += np.outer(cov_b(e), cov_b(e)) + np.outer(cov_m(e), cov_m(e))
    return H


ASSIGN = list(itertools.product(*[((i - 1) % 4, (i + 1) % 4) for i in range(4)]))
S0 = (1, 0, 3, 2)
S1 = (3, 2, 1, 0)


def main() -> None:
    T = 1e-3
    out = {}
    # Gram and eigenstructure of H_M
    G = np.array([[cov_b(0) @ cov_b(0), cov_b(0) @ cov_m(0)], [cov_m(0) @ cov_b(0), cov_m(0) @ cov_m(0)]])
    out["gram"] = G
    HM = H_of(S0)
    ev = np.linalg.eigvalsh(HM)
    out["H_M eigenvalues"] = np.round(ev, 12)
    e, f = 0, 2
    npl = {g: (cov_b(g) + cov_m(g)) / (2 * math.sqrt(2)) for g in (e, f)}
    nmi = {g: (cov_b(g) - cov_m(g)) / 2 for g in (e, f)}
    res = []
    for g in (e, f):
        res.append(np.linalg.norm(HM @ npl[g] - 8 * npl[g]))
        res.append(np.linalg.norm(HM @ nmi[g] - 4 * nmi[g]))
        res.append(np.linalg.norm(R_ev @ npl[g]))
        res.append(np.linalg.norm(R_ev @ nmi[g] - 4 * nmi[g]))
    out["eigvec residuals (H_M n+, n-, R_ev n+, R_ev n-)"] = max(res)
    KM = HM + T * (np.eye(12) + R_ev)
    # held path
    xo = np.concatenate([np.zeros(8), np.ones(4)])
    p = sum(cov_b(g) + cov_m(g) for g in (e, f))
    errs = []
    for t in (0.1, 0.25, 1.0, 3.0):
        xt = expm(-KM * t) @ xo
        xb = math.exp(-T * t) * (xo + (1 - math.exp(-8 * t)) / 4 * p)
        errs.append(np.linalg.norm(xt - xb))
        rint = cov_b(0) @ xt + math.exp(-(8 + T) * t)
        rext = cov_b(1) @ xt - math.exp(-T * t) * (-1.5 + math.exp(-8 * t) / 2)
        errs.append(abs(rint))
        errs.append(abs(rext))
    out["held path / r_int / r_ext max residual"] = max(errs)
    # Z_M, W_M, loop, block diagonality
    ZM = np.stack([(npl[e] - npl[f]) / math.sqrt(2), nmi[e], nmi[f]], axis=1)
    WM = null_space(ZM.T)
    loop = np.concatenate([np.zeros(8), np.ones(4)])
    out["dim W_M"] = WM.shape[1]
    out["|loop . Z_M|"] = np.abs(loop @ ZM).max()
    zplus = (npl[e] + npl[f]) / math.sqrt(2)
    out["|P_Z zplus| (fast common normal retained)"] = np.linalg.norm(ZM.T @ zplus)
    kerH = null_space(HM)
    out["|Z_M^T ker H_M| (kernel retained)"] = np.abs(ZM.T @ kerH).max()
    out["|W_M^T K_M Z_M| (block diagonal)"] = np.abs(WM.T @ KM @ ZM).max()
    out["K_Z diag"] = np.diag(ZM.T @ KM @ ZM)
    out["K_Z expected"] = np.array([8 + T, 4 + 5 * T, 4 + 5 * T])
    Kphys = HM + T * np.eye(12)
    out["physical Z diag"] = np.diag(ZM.T @ Kphys @ ZM)
    # exterior common residual annihilates Z_M
    out["|ext common residual . Z_M|"] = max(
        np.abs(((cov_b(g) + cov_m(g)) / 2) @ ZM).max() for g in (1, 3)
    )
    # D_e in hidden coordinates
    rng = np.random.default_rng(0)
    errD = 0.0
    for _ in range(20):
        x = rng.normal(size=12)
        zp, z1, z2, z3 = zplus @ x, ZM[:, 0] @ x, ZM[:, 1] @ x, ZM[:, 2] @ x
        De = 0.5 * ((cov_b(e) @ x) ** 2 + (cov_m(e) @ x) ** 2)
        Df = 0.5 * ((cov_b(f) @ x) ** 2 + (cov_m(f) @ x) ** 2)
        errD = max(errD, abs(De - ((zp + z1) ** 2 + z2 ** 2)), abs(Df - ((zp - z1) ** 2 + z3 ** 2)))
    out["D_e, D_f hidden-coordinate identity residual"] = errD
    # physical comparison identity: KL(q||Pi)+E|D z|^2 = KL(q||nu)+log((4+5T)/(4+T)), Gaussian q on Z_M
    def kl_gauss(m, S, P):  # KL(N(m,S) || N(0,P^{-1}))
        k = len(m)
        return 0.5 * (np.trace(P @ S) + m @ P @ m - k - math.log(np.linalg.det(P @ S)))

    Pnu = np.diag([8 + T, 4 + 5 * T, 4 + 5 * T]) / T
    PPi = np.diag([8 + T, 4 + T, 4 + T]) / T
    DZ = Dm @ ZM
    errKL = 0.0
    for _ in range(5):
        m = rng.normal(size=3) * 0.01
        L = rng.normal(size=(3, 3)) * 0.01
        S = L @ L.T + 1e-4 * np.eye(3)
        Edz2 = np.trace(DZ.T @ DZ @ S) + m @ DZ.T @ DZ @ m
        lhs = kl_gauss(m, S, PPi) + Edz2 / (2 * 0.5)
        rhs = kl_gauss(m, S, Pnu) + math.log((4 + 5 * T) / (4 + T))
        errKL = max(errKL, abs(lhs - rhs))
    out["physical comparison KL identity residual"] = errKL
    # one-coordinate OU relative entropy formula
    lam, z0, t = 4.3, 0.007, 0.9
    a = math.exp(-2 * lam * t)
    m_, v_ = math.exp(-lam * t) * z0, T * (1 - a) / lam
    kl_true = 0.5 * (v_ / (T / lam) + m_ ** 2 / (T / lam) - 1 - math.log(v_ / (T / lam)))
    out["OU KL formula residual"] = abs(kl_true - 0.5 * (-a - math.log(1 - a) + lam * a * z0 ** 2 / T))
    # Gaussian MGF identity E exp((a+Z)^2/T)
    Tq = 0.3
    for s2 in (Tq / (8 + Tq), Tq / (4 + 5 * Tq)):
        aa = 0.2
        val = quad(lambda z: math.exp((aa + z) ** 2 / Tq - z * z / (2 * s2)) / math.sqrt(2 * math.pi * s2), -5, 5)[0]
        form = (1 - 2 * s2 / Tq) ** -0.5 * math.exp(aa ** 2 / (Tq * (1 - 2 * s2 / Tq)))
        out[f"MGF identity rel. residual s2={s2:.4f}"] = abs(val / form - 1)
    out["MGF factors (<= sqrt(4/3), sqrt(2)) at T->0"] = ((1 - 2 / 8) ** -0.5, (1 - 2 / 4) ** -0.5)
    # constants
    eps = 0.01
    Te = eps ** 8
    ts = 2 * math.log(1 / eps) + 1
    pe = 73728 * eps ** 4 * ts + 4 * ts * math.exp(-1 / Te)
    out["p_eps at eps=1/100"] = pe
    grid = np.linspace(1e-4, 0.01, 2000)
    out["eps^4 t* increasing on grid"] = bool(np.all(np.diff(grid ** 4 * (2 * np.log(1 / grid) + 1)) > 0))
    p0 = 3 * math.exp(-0.25) / 512
    out["p0, 4 e^{-1/(4T)} at T=1e-16"] = (p0, 4 * math.exp(-1 / (4 * 1e-16)))
    out["1/46080 check"] = ((2 * math.pi) ** -6 * math.pi ** 6 / math.factorial(6), 1 / 46080)
    out["16*46080"] = 16 * 46080
    out["row ratio 0.5511/0.4489"] = 0.5511 / 0.4489
    out["eps t* at 1/100 and (1+4 log 10)/100"] = (eps * ts, (1 + 4 * math.log(10)) / 100)
    out["selected-residual bound at t_eps, eps=1/100 (<= eps^2)"] = (
        eps ** 16 + math.sqrt(3) * 0.01 * eps ** 8 + eps ** 2 / 8, eps ** 2
    )
    out["internal RE at t=0 lower (1-sqrt3/100)^2/2"] = (1 - math.sqrt(3) / 100) ** 2 / 2
    out["exterior margin at 1/4: e^{-T t*}(3/2-e^{-2}/2)-sqrt3(rho+delta)"] = (
        math.exp(-Te * ts) * (1.5 - math.exp(-2) / 2) - math.sqrt(3) * (0.01 + eps ** 2 / (8 * math.sqrt(3)))
    )
    out["selected at 1/4: e^{-2}+sqrt3(rho+delta)"] = math.exp(-2) + math.sqrt(3) * (0.01 + eps ** 2 / (8 * math.sqrt(3)))
    for k, v in out.items():
        print(f"{k}: {v}")

    # ---------- sheet determinants and classes (selection proof, temperature tracking)
    print("\n# sheet determinants")
    def mult(S):
        m = [0] * 4
        for i, j in enumerate(S):
            m[edge_of(i, j)] += 1
        return tuple(m)

    def dets(T):
        return {S: np.linalg.det(H_of(S) + T * (np.eye(12) + R_ev)) for S in ASSIGN}

    def d_formula(T):
        return {
            "M": 25 * T ** 8 * (T + 8) ** 2 * (5 * T + 4) ** 2,
            "E": 5 * T ** 6 * (T ** 3 + 16 * T ** 2 + 77 * T + 112) * (125 * T ** 3 + 200 * T ** 2 + 85 * T + 8),
            "N": 5 * T ** 6 * (T + 4) * (5 * T + 2) * (T ** 2 + 12 * T + 28) * (25 * T ** 2 + 30 * T + 4),
            "C": 5 * T ** 5 * (T + 2) * (T + 4) ** 2 * (T + 6) * (5 * T + 2) ** 2 * (5 * T + 4),
        }

    def cls(S):
        m = sorted(mult(S))
        mm = mult(S)
        if m == [0, 0, 2, 2]:
            return "M"
        if m == [1, 1, 1, 1]:
            return "C"
        # path of 3 edges: 2 at an end vs 2 in the middle
        k2 = mm.index(2)
        k0 = mm.index(0)
        return "E" if (k2 - k0) % 4 in (1, 3) else "N"

    for T in (0.5, 0.05, 0.01):
        D = dets(T)
        F = d_formula(T)
        worst = max(abs(D[S] / F[cls(S)] - 1) for S in ASSIGN)
        counts = {c: sum(1 for S in ASSIGN if cls(S) == c) for c in "MENC"}
        print(f"T={T}: max rel. residual det vs class formula = {worst:.2e}; class counts {counts}")
    # flip graph classes
    pairs = set()
    for S in ASSIGN:
        for i in range(4):
            S2 = list(S)
            S2[i] = (i - 1) % 4 if S[i] == (i + 1) % 4 else (i + 1) % 4
            pairs.add(tuple(sorted((cls(S), cls(tuple(S2))))))
    print("one-receiver flip class pairs:", sorted(pairs))
    # active bounds (temperature tracking): d_class / (5 T^5)
    T = 0.01
    F = d_formula(T)
    print("active dets at T=1/100 / T-power:",
          {"M/T^3": F["M"] / (5 * T ** 5) / T ** 3, "E/T": F["E"] / (5 * T ** 5) / T,
           "N/T": F["N"] / (5 * T ** 5) / T, "C": F["C"] / (5 * T ** 5)})
    # passive determinant: ker of all H_S
    Kcommon = null_space(np.vstack([H_of(S) for S in ASSIGN]))
    Om = np.eye(12) + R_ev
    print("dim common kernel:", Kcommon.shape[1], " eig of Omega on it:",
          np.round(np.linalg.eigvalsh(Kcommon.T @ Om @ Kcommon), 10),
          " max ||H_S||:", max(np.linalg.norm(H_of(S), 2) for S in ASSIGN))
    print("K0 = (2pi)^{7/2}/((pi^3/840) sqrt5120) =", (2 * math.pi) ** 3.5 / ((math.pi ** 3 / 840) * math.sqrt(5120)),
          " 210 sqrt(2pi/5) =", 210 * math.sqrt(2 * math.pi / 5),
          " vol B_7(1/2) =", math.pi ** 3.5 / math.gamma(4.5) / 2 ** 7, " pi^3/840 =", math.pi ** 3 / 840)

    # ---------- reduced field, saddle spectrum, adiabatic escape time
    print("\n# reduced field")
    for T in (0.05, 0.01):
        D = dets(T)
        w = {S: (D[S] / T ** 12) ** -0.5 for S in ASSIGN}
        def sig(S):
            return np.array([1 if S[i] == (i + 1) % 4 else -1 for i in range(4)])
        sigs = {S: sig(S) for S in ASSIGN}
        def field(p):
            Z = 0.0
            num = np.zeros(4)
            for S in ASSIGN:
                s = sigs[S]
                pi = np.prod(np.where(s > 0, p, 1 - p))
                Z += pi * w[S]
                num += pi * w[S] * (s > 0)
            return num / Z - p
        h = 1e-6
        J = np.zeros((4, 4))
        for j in range(4):
            dp = np.zeros(4); dp[j] = h
            J[:, j] = (field(0.5 + dp) - field(0.5 - dp)) / (2 * h)
        wM = w[S0]
        rE = min(w[S] for S in ASSIGN if cls(S) == "E") / wM
        rN = min(w[S] for S in ASSIGN if cls(S) == "N") / wM
        rC = min(w[S] for S in ASSIGN if cls(S) == "C") / wM
        D0 = 1 + 4 * rE + 2 * rN + rC
        pred = sorted([(3 - 2 * rN - rC) / D0, (-1 - 2 * rN + 3 * rC) / D0, (-1 + 2 * rN - rC) / D0, (-1 + 2 * rN - rC) / D0])
        print(f"T={T}: r_E={rE:.4e} (sqrt(40/7)T={math.sqrt(40/7)*T:.4e}), r_N={rN:.4e}, r_C={rC:.4e} "
              f"(sqrt(5/3)T^1.5={math.sqrt(5/3)*T**1.5:.4e})")
        print("  Jacobian eigenvalues:", np.round(np.sort(np.linalg.eigvals(J).real), 6), " predicted:", np.round(pred, 6))
        alpha = (3 - 2 * rN - rC) / D0
        # adiabatic ODE p' = eps V(p), from p = 1/2 + eps*I
        vdir = np.array([1, -1, 1, -1]) / 2.0
        Ivec = 0.7 * vdir + np.array([0.3, 0.1, -0.2, 0.05])
        for eps in (1e-2, 1e-3, 1e-4):
            p = 0.5 + eps * Ivec
            s, ds = 0.0, 0.01
            while True:
                k1 = field(p); k2 = field(p + ds / 2 * k1); k3 = field(p + ds / 2 * k2); k4 = field(p + ds * k3)
                p = p + ds / 6 * (k1 + 2 * k2 + 2 * k3 + k4); s += ds
                pM = np.array([1, 0, 1, 0.0])
                if np.max(np.abs(p - pM)) < 1e-2 or np.max(np.abs(p - (1 - pM))) < 1e-2 or s > 200:
                    break
            print(f"  eps={eps:g}: slow time to reach vertex nbhd = {s:.3f}; log(1/eps)/alpha = {math.log(1/eps)/alpha:.3f}; "
                  f"difference = {s - math.log(1/eps)/alpha:.3f}; vertex = {np.round(p, 3)}")


if __name__ == "__main__":
    main()
