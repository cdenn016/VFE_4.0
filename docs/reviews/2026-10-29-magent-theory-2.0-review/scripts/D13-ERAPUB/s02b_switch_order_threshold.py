r"""D13-ERAPUB s02b: locate, separately for the reciprocal flow and for the null flow, the
threshold initial shift eta_c of agent 0's public state at which agents 0 and 1 switch their
preferred source at the same instant (chapters/08_smooth_erasure.tex, sec:erase-status).

If eta_c(full) != eta_c(null), then every preparation with eta strictly between the two values
has a different per-agent switching order along the two flows, and hence a different
mutual-pair partition sequence; the set of such preparations is open in the 24-dimensional
admitted family because the switching times depend continuously on the initial data and the
inequalities are strict. Uses the model of s02_erasure_switching.py (same directory).
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('s02', os.path.join(HERE, 's02_erasure_switching.py'))
s02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s02)


def switch_time_of(sol, agent, t_end):
    ev, _ = s02.switch_times(sol, t_end, ngrid=1200)
    ts = [e[0] for e in ev if e[1] == agent]
    return ts[0]


def gap(eta, rhs, t_end):
    sol = s02.integrate(s02.prep(eta), rhs, t_end)
    return switch_time_of(sol, 0, t_end) - switch_time_of(sol, 1, t_end)


def main():
    t_end = 0.06
    res = {}
    for name, rhs in (('full', s02.rhs_full), ('null', s02.rhs_null)):
        g_lo, g_hi = gap(-0.0045, rhs, t_end), gap(-0.0030, rhs, t_end)
        eta_c = brentq(lambda e: gap(e, rhs, t_end), -0.0045, -0.0030, xtol=1e-12)
        res[name] = eta_c
        print(f'{name}: gap(t_0 - t_1) at eta=-0.0045: {g_lo:+.3e}, at eta=-0.0030: {g_hi:+.3e}; eta_c = {eta_c:.10f}')
    lo, hi = sorted(res.values())
    mid = 0.5 * (lo + hi)
    print(f'eta_c(full) - eta_c(null) = {res["full"] - res["null"]:+.3e}; midpoint eta = {mid:.10f}')
    for name, rhs in (('full', s02.rhs_full), ('null', s02.rhs_null)):
        th = s02.prep(mid)
        ev, _ = s02.switch_times(s02.integrate(th, rhs, t_end), t_end, ngrid=1200)
        seq = s02.history_from_events(th, ev, s02.mutual_partition)
        comp = s02.history_from_events(th, ev, s02.component_partition)
        print(f'  at midpoint, {name}: order {[e[1] for e in ev]}, times {[f"{e[0]:.9f}" for e in ev]}')
        print(f'      mutual-pair sequence {seq}')
        print(f'      component sequence   {comp}')


if __name__ == '__main__':
    main()
