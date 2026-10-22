r"""R-ERA script 1: null-flow control for the erasure construction (chapter 8).

Full flow: unit Fisher gradient flow of the Gibbs-reduced potential (L06-ERA s02 code).
Null flow: every peer cost D_ij (j != i) deleted, all else unchanged, so
    \dot\theta = -(1-\theta^2) \partial_\theta E_0.
Selector: each agent's preferred source = argmin_j of the original row cost evaluated
at the current state (belief sector).  We record the pairing history and switch times
along both flows, and the late in-box separation along the full flow.
"""
import importlib.util
import sys
import numpy as np
from scipy.integrate import solve_ivp

P = r"C:/Users/chris and christine/Desktop/VFE_4.0/docs/reviews/2026-10-22-magent-theory-2.0-review/scripts/L06-ERA/s02_erasure_simulation.py"
spec = importlib.util.spec_from_file_location("s02", P)
s02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s02)
N, J, HSTAR = s02.N, s02.J, s02.HSTAR


def null_rhs(t: float, th: np.ndarray) -> np.ndarray:
    x, z, h, y, w, hh = s02.unpack(th)
    g = np.concatenate([np.arctanh(x) - J * y, np.arctanh(z), np.arctanh(h) - np.arctanh(HSTAR),
                        np.arctanh(y) - J * x, np.arctanh(w), np.arctanh(hh) - np.arctanh(HSTAR)])
    return -(1 - th**2) * g


def history(sol) -> list:
    out, last = [], None
    for k in range(sol.y.shape[1]):
        x, z, h, y, w, hh = s02.unpack(sol.y[:, k])
        C, _ = s02.rows(x, z, h, 1.0)
        pref = tuple(int(np.argmin(np.where(np.isinf(C[i]), np.inf, C[i]))) for i in range(N))
        if pref != last:
            out.append((round(float(sol.t[k]), 4), pref))
            last = pref
    return out


def main() -> None:
    th0 = s02.initial()
    ts = np.linspace(0, 0.3, 3001)
    full = solve_ivp(s02.rhs, (0, 0.3), th0, t_eval=ts, args=(1.0,), method='LSODA', rtol=1e-10, atol=1e-13)
    null = solve_ivp(null_rhs, (0, 0.3), th0, t_eval=ts, method='LSODA', rtol=1e-10, atol=1e-13)
    print('full-flow preferred-source history (time, argmin per agent):', history(full))
    print('null-flow preferred-source history (time, argmin per agent):', history(null))
    aq_f = np.array([s02.alpha(full.y[8:12, k].mean()) for k in range(len(ts))])
    aq_n = np.array([s02.alpha(null.y[8:12, k].mean()) for k in range(len(ts))])
    print('gate closure time full %.4f null %.4f' % (ts[np.argmax(aq_f == 0)], ts[np.argmax(aq_n == 0)]))
    # null flow: public signs preserved
    print('null-flow public signs at t=0.3:', np.sign(null.y[0:4, -1]), np.sign(null.y[12:16, -1]))


if __name__ == '__main__':
    sys.exit(main())
