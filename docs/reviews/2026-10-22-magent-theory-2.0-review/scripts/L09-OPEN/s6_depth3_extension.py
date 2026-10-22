r"""L09-OPEN extension probe: three nested levels in the limiting field open:limit.

Ten agents: a dyadic core u = s3 A3 + s2 A2 + s1 * 2 (s_k = +-1), plus an exterior pair at R +- 2.
Uses the fixed selector (eps=1/10, delta=1/20). Also prints the self-similarity identity:
for two collapsed clusters of m coincident agents at -a, +a, the inherited cross mass is
B = v/(1+v), v = e^{-2a^2}, and a' = -a(1+4B), independent of m.
Numerical exploration only; it proves nothing.
"""
from __future__ import annotations

import itertools
import numpy as np
from scipy.integrate import solve_ivp

import importlib.util, pathlib
_spec = importlib.util.spec_from_file_location("s2", pathlib.Path(__file__).with_name("s2_limit_field_events.py"))
s2 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(s2)


def self_similarity() -> None:
    for m in (1, 2, 4):
        a = 0.9
        u = np.array([-a] * m + [a] * m) + 1e-9 * np.arange(2 * m)  # (nearly) collapsed clusters
        beta = s2.weights(u)
        B = beta[:m, m:].sum(axis=1).mean()
        v = np.exp(-2 * a * a)
        ad = -(s2.field(0, u)[:m].mean()) * (-1)  # d/dt of left center is -a', left center is -a
        print(f"  m={m}: B={B:.10f} (v/(1+v)={v/(1+v):.10f}); a'={-s2.field(0,u)[:m].mean():.8f} vs -a(1+4B)={-a*(1+4*v/(1+v)):.8f}")


def main() -> None:
    print("[self-similarity of collapsed clusters]")
    self_similarity()
    A2, A3 = 50.0, 2500.0
    core = [s3 * A3 + s2_ * A2 + s1 * 2.0 for s3 in (-1, 1) for s2_ in (-1, 1) for s1 in (-1, 1)]
    L = np.log(A3) + 2.0
    R = (A3 + 20.0) * np.exp(L) * 0.01 + 30 * np.exp(L)
    u0 = np.array(core + [R - 2, R + 2])
    T_end = L + 12
    sol = solve_ivp(s2.field, (0, T_end), u0, rtol=1e-10, atol=1e-10, dense_output=True, method="LSODA")
    ts = np.linspace(0, T_end, 30001)
    events, P = s2.run_selector(sol, ts, len(u0), "depth3")
    print(f"[depth-3 probe] A2={A2}, A3={A3}, R={R:.3e}")
    for e in events:
        print(f"  merge t={e[0]:.3f} block {e[1]} margin {e[2]:.1e}")


if __name__ == "__main__":
    main()
