r"""R-ERA script 2: chapter 9 checks.

(a) Binary two-agent phase coefficient for J in [1/4, 1): forward-backward truncation
    Pi_T = phi_{-T}(log-density average of Phi_T x); compare (p-m)/(m a^2) with
    c = 3(2-J)/(2(5-J)).  Also compare with a mean-coordinate average (chart independence).
(b) Lipschitz bound ||Df|| <= 1+|J| on random finite-alphabet points (max norm).
(c) Restricted-probe witness: equal intercepts, different intercept velocities.
(d) Fixed channel K: incoming force on the image closes for a deterministic K.
"""
import numpy as np
from scipy.integrate import solve_ivp


def fine(t, x, J):
    x1, x2 = x
    f1 = -(1 - x1**2) * (2 * np.arctanh(x1) - np.arctanh(x2) - J * x1) - (x1 - x2)
    f2 = -(1 - x2**2) * (2 * np.arctanh(x2) - np.arctanh(x1) - J * x2) - (x2 - x1)
    return [f1, f2]


def back(t, th, J):
    return th - J * np.tanh(th)  # reversed diagonal flow in log-density coordinate


def phase(m, a, J, T, chart='log'):
    s = solve_ivp(fine, (0, T), [m + a, m - a], args=(J,), rtol=1e-13, atol=1e-16, method='DOP853')
    xT = s.y[:, -1]
    th = np.arctanh(xT).mean() if chart == 'log' else np.arctanh(xT.mean())
    b = solve_ivp(back, (0, T), [th], args=(J,), rtol=1e-13, atol=1e-16, method='DOP853')
    return np.tanh(b.y[0, -1])


def part_a():
    for J in (0.3, 0.5, 0.8, 0.95):
        c = 3 * (2 - J) / (2 * (5 - J))
        out = []
        for s in (0.04, 0.02, 0.01):
            p = phase(s, s, J, 12.0)
            pm = phase(s, s, J, 12.0, chart='mean')
            out.append(((p - s) / s**3, abs(p - pm)))
        print(f'(a) J={J}: c={c:.6f}; ratios', ['%.6f' % r for r, _ in out], 'log vs mean chart |dp|', ['%.1e' % d for _, d in out])
    # truncation convergence at J=0.8
    J = 0.8
    vals = [phase(0.05, 0.05, J, T) for T in (4, 6, 8, 10, 12)]
    print('(a) J=0.8 Pi_T for T=4..12:', ['%.15f' % v for v in vals])


def part_b(rng):
    worst = 0.0
    for _ in range(2000):
        n = 6
        A = rng.dirichlet(np.ones(n)); B = rng.dirichlet(np.ones(n))
        phi = np.array([1, -1] * 3, float); psi = np.array([1, 1, -1, -1, 1, -1], float)
        J = rng.uniform(-0.999, 0.999)
        u = rng.normal(0, 2, n); v = rng.normal(0, 2, n)
        q = A * np.exp(u); q /= q.sum(); s = B * np.exp(v); s /= s.sum()
        h = rng.uniform(-1, 1, n); k = rng.uniform(-1, 1, n)
        nh = max(abs(h).max(), abs(k).max())
        cov_s = s @ (psi * k) - (s @ psi) * (s @ k)
        cov_q = q @ (phi * h) - (q @ phi) * (q @ h)
        d1 = -h + J * phi * cov_s; d2 = -k + J * psi * cov_q
        worst = max(worst, max(abs(d1).max(), abs(d2).max()) / (nh * (1 + abs(J))))
    print(f'(b) max ||Df[h,k]|| / ((1+|J|)||(h,k)||) over 2000 samples: {worst:.6f} (<= 1 expected)')


def part_c():
    lam = np.ones(3) / 3
    c = (1 + np.sqrt(5)) / 8
    for q in (np.array([0.25, 0.25, 0.5]), np.array([c, c, 1 - 2 * c])):
        f = q / lam
        b = lam @ np.log(f)
        bdot = lam @ (1 / f) - 1  # r_0 = r_theta = 1, alpha = 1
        print(f'(c) q={q.round(6)}: intercept {b:.12f}, intercept velocity {bdot:.6f}')
    print('(c) (5 sqrt5 - 10)/9 =', (5 * np.sqrt(5) - 10) / 9)


def part_d(rng):
    # deterministic merge {0,1,2}->{0,1}; nondeterministic K
    Kdet = np.array([[1, 0], [1, 0], [0, 1.0]])
    Kran = np.array([[0.9, 0.1], [0.2, 0.8], [0.55, 0.45]])
    r = np.array([0.3, 0.7])
    for name, K in (('det', Kdet), ('nondet', Kran)):
        q1 = np.array([0.1, 0.3, 0.6]); nu = q1 @ K
        # second input with the same image
        if name == 'det':
            q2 = np.array([0.35, 0.05, 0.6])
        else:
            # move along the kernel of K^T within the simplex
            ker = np.linalg.svd(np.vstack([K.T, np.ones(3)]))[2][-1]
            q2 = q1 + 0.05 * ker / np.abs(ker).max()
        assert np.allclose(q2 @ K, nu)
        vel = []
        for q in (q1, q2):
            nuq = q @ K
            force = K @ (r / nuq) - 1          # d/dt log q (beta = 1)
            qdot = q * force
            vel.append(qdot @ K)
        print(f'(d) {name}: image velocities {vel[0].round(6)} vs {vel[1].round(6)}; r - nu = {(r - nu).round(6)}')


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    part_a(); part_b(rng); part_c(); part_d(rng)
