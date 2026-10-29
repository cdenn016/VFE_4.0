r"""D13-ERAPUB s02c: integrator-independence check of s02b. Recomputes eta_c for the full and
null flows with DOP853 (rtol 1e-12, atol 1e-14) instead of LSODA (rtol 1e-11)."""
from __future__ import annotations

import importlib.util
import os

from scipy.integrate import solve_ivp
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('s02', os.path.join(HERE, 's02_erasure_switching.py'))
s02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s02)


def integrate(th0, rhs, t_end):
    return solve_ivp(rhs, (0, t_end), th0, method='DOP853', rtol=1e-12, atol=1e-14, dense_output=True)


s02.integrate = integrate
spec2 = importlib.util.spec_from_file_location('s02b', os.path.join(HERE, 's02b_switch_order_threshold.py'))
s02b = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(s02b)
s02b.s02 = s02


def main():
    t_end = 0.03
    out = {}
    for name, rhs in (('full', s02.rhs_full), ('null', s02.rhs_null)):
        out[name] = brentq(lambda e: s02b.gap(e, rhs, t_end), -0.0045, -0.0030, xtol=1e-12)
        print(f'DOP853 {name}: eta_c = {out[name]:.10f}')
    print(f'DOP853 eta_c(full) - eta_c(null) = {out["full"] - out["null"]:+.3e}')


if __name__ == '__main__':
    main()
