r"""D13-ERAPUB s06: prop:public-gibbs-identifies, cor:public-gibbs-channel and the
deterministic-channel force statement (chapters/09_public_compression.tex, sec 9.3).

(a) identity channel, lambda uniform on 5 points, three children with weights (0.5, 0.3, 0.2):
    along one line r = 1 + t h with generic mean-zero h, the entropy-corrected normalizer
    sum_j w_j exp((iota_j + t v_j)/T) is fitted; weights are recovered by a Vandermonde solve.
(b) quantifier scope of the remark after the proof: for a FIXED line, two different collections
    with equal slopes v_j and equal tilted weights w_j e^{iota_j/T} give identical responses on
    that line, while a different line separates them.
(c) fixed Markov kernel K (3 -> 2): two collections with equal channelled laws give equal
    normalizers for random receivers; a collection with a different channelled law does not.
(d) incoming Fisher force: d/dt log q_j = beta (K(r/nu_j) - 1); for deterministic K the image
    velocity is beta (r - nu_j) and depends only on nu_j; for a nondeterministic K two sources
    with equal images have different image velocities.
(e) witness of eq:public-restricted-force: equal intercepts, intercept velocities 1/9 and
    (5 sqrt 5 - 10)/9.
"""
from __future__ import annotations

import numpy as np


def kl(a, b, lam):
    return float(np.sum(lam * a * np.log(a / b)))


def Z(r, F, w, T, lam):
    return sum(wj * np.exp(-kl(r, f, lam) / T) for f, wj in zip(F, w))


def part_a(rng):
    n, T = 5, 0.7
    lam = np.ones(n) / n
    F = [rng.dirichlet(np.ones(n)) / lam for _ in range(3)]  # densities against lam
    w = np.array([0.5, 0.3, 0.2])
    h = rng.normal(size=n)
    h -= lam @ h
    ts = np.linspace(-0.05, 0.05, 9)
    y = np.array([np.exp(np.sum(lam * (1 + t * h) * np.log(1 + t * h)) / T) * Z(1 + t * h, F, w, T, lam) for t in ts])
    v = np.array([np.sum(lam * h * np.log(f)) for f in F])
    iota = np.array([np.sum(lam * np.log(f)) for f in F])
    M = np.exp(np.outer(ts, v) / T)
    coef = np.linalg.lstsq(M, y, rcond=None)[0]
    print(f'(a) recovered tilted weights / exp(iota/T): {np.round(coef / np.exp(iota / T), 12)} (true {w})')


def part_b(rng):
    n, T = 4, 1.0
    lam = np.ones(n) / n
    h = np.array([1.0, -1.0, 0.0, 0.0])  # fixed probe line
    f1 = np.array([1.0, 1.0, 1.0, 1.0])
    # second density with the same slope v = <h, log f> = 0 but a different intercept
    f2 = np.array([1.0, 1.0, 1.6, 0.4])
    f2 = f2 / (lam @ f2)
    i1, i2 = np.sum(lam * np.log(f1)), np.sum(lam * np.log(f2))
    v1, v2 = np.sum(lam * h * np.log(f1)), np.sum(lam * h * np.log(f2))
    w1, w2 = 1.0, np.exp((i1 - i2) / T)
    diffs = [abs(Z(1 + t * h, [f1], [w1], T, lam) - Z(1 + t * h, [f2], [w2], T, lam)) for t in np.linspace(-0.3, 0.3, 13)]
    h2 = np.array([0.0, 0.0, 1.0, -1.0])
    d2 = abs(Z(1 + 0.2 * h2, [f1], [w1], T, lam) - Z(1 + 0.2 * h2, [f2], [w2], T, lam))
    print(f'(b) slopes {v1:.3e}, {v2:.3e}; max response difference on the fixed line {max(diffs):.2e}; on another line {d2:.3e}')


def part_c(rng):
    lam = np.ones(2) / 2
    Kmat = np.array([[0.9, 0.1], [0.2, 0.8], [0.55, 0.45]])
    q1 = np.array([0.2, 0.3, 0.5])
    ker = np.linalg.svd(np.vstack([Kmat.T, np.ones(3)]))[2][-1]
    q2 = q1 + 0.1 * ker / np.abs(ker).max()
    q3 = np.array([0.6, 0.3, 0.1])
    nu = [q @ Kmat for q in (q1, q2, q3)]
    print(f'(c) channelled laws: {np.round(nu[0], 6)}, {np.round(nu[1], 6)}, {np.round(nu[2], 6)}; q1 != q2: {not np.allclose(q1, q2)}')
    worst12, min13 = 0.0, np.inf
    for _ in range(200):
        r = rng.dirichlet(np.ones(2)) / lam
        z1 = Z(r, [nu[0] / lam], [1.0], 0.5, lam)
        z2 = Z(r, [nu[1] / lam], [1.0], 0.5, lam)
        z3 = Z(r, [nu[2] / lam], [1.0], 0.5, lam)
        worst12 = max(worst12, abs(z1 - z2))
        min13 = min(min13, abs(z1 - z3))
    print(f'(c) equal channelled laws: max |Z1-Z2| = {worst12:.1e}; different channelled law: min |Z1-Z3| = {min13:.2e} over 200 receivers')


def part_d():
    r = np.array([0.35, 0.65])
    for name, Kmat in (('deterministic', np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])),
                       ('nondeterministic', np.array([[0.9, 0.1], [0.2, 0.8], [0.55, 0.45]]))):
        qa = np.array([0.15, 0.25, 0.6])
        if name == 'deterministic':
            qb = np.array([0.32, 0.08, 0.6])
        else:
            ker = np.linalg.svd(np.vstack([Kmat.T, np.ones(3)]))[2][-1]
            qb = qa + 0.08 * ker / np.abs(ker).max()
        vels = []
        for q in (qa, qb):
            nu = q @ Kmat
            dlogq = Kmat @ (r / nu) - 1.0  # beta = 1
            vels.append((q * dlogq) @ Kmat)
        print(f'(d) {name}: images {np.round(qa @ Kmat, 6)} / {np.round(qb @ Kmat, 6)}; image velocities '
              f'{np.round(vels[0], 6)} / {np.round(vels[1], 6)}; r - nu = {np.round(r - qa @ Kmat, 6)}')


def part_e():
    lam = np.ones(3) / 3
    c = (1 + np.sqrt(5)) / 8
    for q in (np.array([0.25, 0.25, 0.5]), np.array([c, c, 1 - 2 * c])):
        f = q / lam
        print(f'(e) q = {np.round(q, 6)}: intercept {lam @ np.log(f):.12f}; velocity {lam @ (1 / f) - 1:.9f}')
    print(f'(e) 1/9 = {1/9:.9f}; (5 sqrt5 - 10)/9 = {(5*np.sqrt(5)-10)/9:.9f}')


if __name__ == '__main__':
    rng = np.random.default_rng(5)
    part_a(rng); part_b(rng); part_c(rng); part_d(); part_e()
