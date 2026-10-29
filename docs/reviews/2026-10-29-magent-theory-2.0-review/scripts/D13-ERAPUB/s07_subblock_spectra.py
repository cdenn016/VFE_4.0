r"""D13-ERAPUB s07: can the collective phase of thm:public-phase be refined to a proper sub-block
near the common reference? (chapters/09_public_compression.tex, sec:public-recursive-requirement)

For N = 4 and block I = {0, 1}, the block-synchronized set S_I = {z_0 = z_1} is the fixed-point
subspace of the transposition (0 1), hence invariant. A strong-stable (asymptotic-phase)
projection onto S_I governed by a rate condition needs every normal decay rate (intra-block
differences z_0 - z_1) to exceed every tangential decay rate along S_I. This script computes
both spectra of DV(0) on a finite alphabet, using the chart field of s04_phase_rates.py.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('s04', os.path.join(HERE, 's04_phase_rates.py'))
s04 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s04)


def jacobian(N, J, Tq, Ts):
    K = s04.K
    PA, PB = s04.basis_mean_zero(s04.A), s04.basis_mean_zero(s04.B)
    Pfull = np.block([[PA, np.zeros((K, K - 1))], [np.zeros((K, K - 1)), PB]])
    dim = 2 * K - 2
    n = N * dim
    Jac = np.zeros((n, n))
    h = 1e-6
    for k in range(n):
        e = np.zeros(n)
        e[k] = h
        Zp = np.array([Pfull @ e[i * dim:(i + 1) * dim] for i in range(N)])
        Fp, Fm = s04.field(Zp, N, J, Tq, Ts), s04.field(-Zp, N, J, Tq, Ts)
        Jac[:, k] = np.concatenate([np.linalg.lstsq(Pfull, (Fp[i] - Fm[i]) / (2 * h), rcond=None)[0] for i in range(N)])
    return Jac, dim


def main():
    N = 4
    for J, Tq, Ts in ((0.5, 0.8, 1.5), (0.0, 1.0, 1.0)):
        Jac, dim = jacobian(N, J, Tq, Ts)
        I = np.eye(dim)
        Z = np.zeros((dim, dim))
        # tangent space of S_I: (w, w, z2, z3)
        Tb = np.vstack([np.hstack([I, Z, Z]), np.hstack([I, Z, Z]), np.hstack([Z, I, Z]), np.hstack([Z, Z, I])])
        # normal space: (d, -d, 0, 0)
        Nb = np.vstack([I, -I, Z, Z])
        tan = np.linalg.lstsq(Tb, Jac @ Tb, rcond=None)[0]
        inv_res = np.abs(Tb @ tan - Jac @ Tb).max()
        nor = np.linalg.lstsq(Nb, Jac @ Nb, rcond=None)[0]
        inv_res_n = np.abs(Nb @ nor - Jac @ Nb).max()
        rt = np.sort(-np.linalg.eigvals(tan).real)
        rn = np.sort(-np.linalg.eigvals(nor).real)
        kap = 2 * N / (N - 1)
        print(f'J={J}, Tq={Tq}, Ts={Ts}, N={N}, kappa_N={kap:.4f}; S_I invariant to {inv_res:.1e}, normal space invariant to {inv_res_n:.1e}')
        print(f'   tangential decay rates on S_I: {np.round(rt, 6)}')
        print(f'   normal (intra-block) decay rates: {np.round(rn, 6)}')
        print(f'   slowest normal rate {rn.min():.6f} vs fastest tangential rate {rt.max():.6f}: rate condition holds: {rn.min() > rt.max()}')


if __name__ == '__main__':
    main()
