r"""L06-ERA script 2: integrate the four-agent erasure system of chapters/08_smooth_erasure.tex.

State: 24 Bernoulli means theta = (x_i, z_i, h_i, y_i, w_i, hh_i), i = 0..3.
Flow: unit Fisher gradient flow of the Gibbs-reduced potential,
    \dot\theta = -(1-\theta^2) \partial_\theta E_red,
with the envelope gradient d E_red = d E_0 + sum beta^* dD^q + sum gamma^* dD^s.
The analytic gradient is checked against a complex-step derivative of E_red.

Reports: gate-closure times, box invariance, preferred sources before and
after closure, and the within-box public separation over a long horizon.
Single-threaded, short.
"""
import numpy as np
from scipy.integrate import solve_ivp

J = np.log(3.0)
HSTAR = -0.5
H0 = 0.25
DELTA = 0.04
RHO = 0.01
EPS = 0.015
ZETA = 0.5
N = 4


def chi(r):
    r = np.real(r) if not np.iscomplexobj(r) else r
    rr = np.real(r)
    if rr <= 0:
        return 0.0 * r
    if rr >= 1:
        return 1.0 + 0.0 * r
    e1 = np.exp(-1 / r)
    e2 = np.exp(-1 / (1 - r))
    return e1 / (e1 + e2)


def dchi(r):
    if r <= 0 or r >= 1:
        return 0.0
    e1 = np.exp(-1 / r)
    e2 = np.exp(-1 / (1 - r))
    s = e1 + e2
    de1 = e1 / r**2
    de2 = e2 / (1 - r)**2
    return (de1 * s - e1 * (de1 - de2)) / s**2


def alpha(H):
    return chi((H - H0) / DELTA)


def dalpha(H):
    return dchi((H - H0) / DELTA) / DELTA


def d(x, z):
    return 0.5 * (1 + x) * np.log((1 + x) / (1 + z)) + 0.5 * (1 - x) * np.log((1 - x) / (1 - z))


def Deps(x, v):
    return d(x, (1 - EPS) * x + EPS * v)


def unpack(th):
    th = th.reshape(6, N)
    return th  # rows: x, z, h, y, w, hh


def energy(th, T):
    x, z, h, y, w, hh = unpack(th)
    psi = lambda t: d(t, 0.0 * t)
    E0 = (psi(x) + psi(y) - J * x * y + np.log(np.cosh(J)) + psi(z) + psi(w)
          + d(h, HSTAR + 0 * h) + d(hh, HSTAR + 0 * hh)).sum()
    aq = alpha(h.mean())
    as_ = alpha(hh.mean())
    red = 0.0
    for i in range(N):
        cq = [Deps(x[i], x[j]) + d(z[i], aq * RHO * z[j]) + d(h[i], HSTAR + 0 * h[i]) for j in range(N) if j != i]
        cs = [Deps(y[i], y[j]) + d(w[i], as_ * RHO * w[j]) + d(hh[i], HSTAR + 0 * hh[i]) for j in range(N) if j != i]
        red += -T * np.log(sum(np.exp(-c / T) for c in cq) / (N - 1))
        red += -T * np.log(sum(np.exp(-c / T) for c in cs) / (N - 1))
    return E0 + red


def rows(pub, priv, gate, T):
    """Cost matrix and Gibbs rows for one sector."""
    a = alpha(gate.mean())
    C = np.full((N, N), np.inf)
    for i in range(N):
        for j in range(N):
            if i != j:
                C[i, j] = Deps(pub[i], pub[j]) + d(priv[i], a * RHO * priv[j]) + d(gate[i], HSTAR)
    W = np.exp(-(C - np.nanmin(np.where(np.isinf(C), np.nan, C), axis=1, keepdims=True)) / T)
    W[np.isinf(C)] = 0.0
    W /= W.sum(axis=1, keepdims=True)
    return C, W


def grad_sector(pub, priv, gate, other_pub, T):
    """Gradient of E_red with respect to (pub, priv, gate) of one sector."""
    a = alpha(gate.mean())
    da = dalpha(gate.mean())
    _, B = rows(pub, priv, gate, T)
    gp = np.zeros(N)
    gz = np.zeros(N)
    gh = np.zeros(N)
    for i in range(N):
        gp[i] = np.arctanh(pub[i]) - J * other_pub[i]
        gz[i] = np.arctanh(priv[i])
        gh[i] = 2 * (np.arctanh(gate[i]) - np.arctanh(HSTAR))
        for j in range(N):
            if j == i:
                continue
            cpub = (1 - EPS) * pub[i] + EPS * pub[j]
            gp[i] += B[i, j] * (np.arctanh(pub[i]) - np.arctanh(cpub) + (1 - EPS) * (cpub - pub[i]) / (1 - cpub**2))
            gz[i] += B[i, j] * (np.arctanh(priv[i]) - np.arctanh(a * RHO * priv[j]))
            # incoming: row j uses i as source
            cin = (1 - EPS) * pub[j] + EPS * pub[i]
            gp[i] += B[j, i] * EPS * (cin - pub[j]) / (1 - cin**2)
            cz = a * RHO * priv[i]
            gz[i] += B[j, i] * (cz - priv[j]) / (1 - cz**2) * a * RHO
    corr = 0.0
    for aa in range(N):
        for j in range(N):
            if j == aa:
                continue
            cz = a * RHO * priv[j]
            corr += B[aa, j] * RHO * priv[j] * (cz - priv[aa]) / (1 - cz**2)
    gh += da / N * corr
    return gp, gz, gh


def grad(th, T):
    x, z, h, y, w, hh = unpack(th)
    gx, gz, gh = grad_sector(x, z, h, y, T)
    gy, gw, ghh = grad_sector(y, w, hh, x, T)
    return np.concatenate([gx, gz, gh, gy, gw, ghh])


def rhs(t, th, T):
    return -(1 - th**2) * grad(th, T)


def complex_step_grad(th, T, hstep=1e-30):
    g = np.zeros_like(th)
    for k in range(th.size):
        e = np.zeros(th.size, complex)
        e[k] = 1j * hstep
        g[k] = np.imag(energy(th.astype(complex) + e, T)) / hstep
    return g


def initial():
    # agents 0..3: public signs (+,-,+,-); private signs (+,+,-,-) -> crossed pairs (0,1),(2,3)
    x = np.array([0.4952, -0.4961, 0.5048, -0.5068])
    y = np.array([0.4968, -0.4943, 0.5031, -0.5052])
    z = ZETA * np.array([1, 1, -1, -1.0])
    w = ZETA * np.array([1, 1, -1, -1.0])
    h = np.array([0.2712, 0.2745, 0.2768, 0.2790])
    hh = np.array([0.2721, 0.2733, 0.2776, 0.2752])
    return np.concatenate([x, z, h, y, w, hh])


def main():
    for T in (1.0, 1e-4):
        th0 = initial()
        # gradient check at the initial point (gate open) and a perturbed point
        rng = np.random.default_rng(1)
        thp = th0 + 0.01 * rng.standard_normal(th0.size)
        thp[8:12] = th0[8:12]  # keep gates in transition band
        if T == 1.0:
            for pt in (th0, thp):
                ga, gc = grad(pt, T), complex_step_grad(pt, T)
                print(f'T={T}: analytic vs complex-step gradient max |diff| = {np.abs(ga-gc).max():.2e}')
        x, z, h, y, w, hh = unpack(th0)
        _, Bq = rows(x, z, h, T)
        _, Bs = rows(y, w, hh, T)
        print(f'T={T}: initial alpha_q={alpha(h.mean()):.4f}, alpha_s={alpha(hh.mean()):.4f}')
        print(f'T={T}: initial preferred sources q: {Bq.argmax(1)}, s: {Bs.argmax(1)}; max row weight {Bq.max():.4f}')
        ts = np.concatenate([np.linspace(0, 0.2, 201), np.linspace(0.25, 120, 400)])
        sol = solve_ivp(rhs, (0, 120), th0, t_eval=ts, args=(T,), method='LSODA', rtol=1e-10, atol=1e-13)
        Y = sol.y
        aq = np.array([alpha(Y[8:12, k].mean()) for k in range(Y.shape[1])])
        as_ = np.array([alpha(Y[20:24, k].mean()) for k in range(Y.shape[1])])
        tq = sol.t[np.argmax(aq == 0)]
        tsec = sol.t[np.argmax(as_ == 0)]
        print(f'T={T}: first sampled time with alpha_q==0: {tq:.4f}; alpha_s==0: {tsec:.4f}; bound 3delta/(2m)={3*DELTA/(2*1.328834):.4f}')
        print(f'T={T}: alpha stays 0 afterwards: q {bool(np.all(aq[sol.t>=tq]==0))}, s {bool(np.all(as_[sol.t>=tsec]==0))}')
        pubx, puby = Y[0:4], Y[12:16]
        sgn = np.array([1, -1, 1, -1])
        inbox = np.all((sgn[:, None] * pubx > 0.49) & (sgn[:, None] * pubx < 0.51)
                       & (sgn[:, None] * puby > 0.49) & (sgn[:, None] * puby < 0.51))
        print(f'T={T}: all public states stayed in their boxes on sampled times: {inbox}')
        print(f'T={T}: private max |z|,|w| over run: {np.abs(Y[4:8]).max():.4f}, {np.abs(Y[16:20]).max():.4f}')
        k_after = np.argmax(sol.t > max(tq, tsec) + 0.01)
        thk = Y[:, k_after]
        x, z, h, y, w, hh = unpack(thk)
        _, Bq = rows(x, z, h, T)
        _, Bs = rows(y, w, hh, T)
        print(f'T={T}: preferred sources just after closure (t={sol.t[k_after]:.3f}) q: {Bq.argmax(1)}, s: {Bs.argmax(1)}; '
              f'same-box weight row0 {Bq[0,2]:.4f}')
        sep_plus = np.hypot(pubx[0] - pubx[2], puby[0] - puby[2])
        sep_minus = np.hypot(pubx[1] - pubx[3], puby[1] - puby[3])
        for tt in (0, 1, 5, 10, 20, 40, 80, 120):
            kk = np.argmin(np.abs(sol.t - tt))
            print(f'T={T}: t={sol.t[kk]:6.2f}  in-box separations +: {sep_plus[kk]:.3e}  -: {sep_minus[kk]:.3e}')
        # decay rate estimate between t=40 and 120
        k1, k2 = np.argmin(np.abs(sol.t - 40)), np.argmin(np.abs(sol.t - 120))
        rate = -np.log(sep_plus[k2] / sep_plus[k1]) / (sol.t[k2] - sol.t[k1])
        print(f'T={T}: fitted in-box separation decay rate (40..120): {rate:.4f}')
        print(f'T={T}: final public x: {pubx[:, -1]}, y: {puby[:, -1]}')
    # local antisymmetric-mode prediction at x=y=1/2 without peers
    xs = 0.5
    Jac = (1 - xs**2) * np.array([[1 / (1 - xs**2), -J], [-J, 1 / (1 - xs**2)]])
    print('local Fisher-Hessian eigenvalues at (1/2,1/2):', np.linalg.eigvalsh(Jac))


if __name__ == '__main__':
    main()
