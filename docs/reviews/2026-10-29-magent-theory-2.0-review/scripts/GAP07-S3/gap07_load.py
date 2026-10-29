r"""GAP07-S3 helper: load the definition blocks of item7_checks/checks.py without running its scans.

Executes lines 1-231 (checks (1)-(5), the SharedPop class, records(), the instance constants and the fixed draw y0)
and lines 413-525 (F_batch, block_stats, all_rgs, interval_minimizer, DerivedPop, dpop), with stdout suppressed.
Also reproduces the random draws of checks (11), (12), (14) (generator rng) and (22), (24) (generator rngd) by
replaying exactly the generator calls that checks.py makes before them, so the reruns use the same states.
Read-only use of the manuscript directory; nothing is written there.
"""
import os, io, contextlib
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

SRC = r"C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/item7_checks/checks.py"


def load():
    with open(SRC, encoding="utf-8") as fh:
        L = fh.readlines()
    assert "y0 = o + 0.005" in L[230], L[230]
    assert L[412].startswith("# Derived selector"), L[412]
    assert L[523].startswith("    return DerivedPop"), L[523]
    ns = {"__name__": "item7_defs"}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile("".join(L[:231]) + "".join(L[412:525]), "checks_defs", "exec"), ns)
    return ns


def draws(ns):
    r"""Replay the generator calls of checks.py after line 231.

    rng (seed 20261023): check (11) draws one state per record gap 1.0, 1.1, 1.2, 1.3; check (12) twelve states;
    check (14) one state per gap 1.1, 1.2, 1.3 (checks (6)-(10), (13) make no rng calls).
    rngd (seed 20261024): check (18) 5 x standard_normal(2); check (19) 2000 x random(); check (20)
    5 x integers(0, 5, 12); check (22) 200 x random(12) (null box); check (24) 12 x random(12).
    The rng calls of checks (18)-(19) go to rng after check (14) and do not affect the draws used here.
    """
    rng = ns["rng"]; o = ns["o"]; records = ns["records"]
    out = {}
    out["c11"] = {}
    for gp in (1.0, 1.1, 1.2, 1.3):
        oo = records(gp); out["c11"][gp] = (oo, oo + 0.005 * (2 * rng.random(len(oo)) - 1))
    out["c12"] = [o + 0.005 * (2 * rng.random(len(o)) - 1) for _ in range(12)]
    out["c14"] = {}
    for gp in (1.1, 1.2, 1.3):
        oo = records(gp); out["c14"][gp] = (oo, oo + 0.005 * (2 * rng.random(len(oo)) - 1))
    rngd = np.random.default_rng(20261024)
    for _ in range(5):
        rngd.standard_normal(2)
    for _ in range(2000):
        rngd.random()
    for _ in range(5):
        rngd.integers(0, 5, 12)
    out["c22"] = [o + 0.005 * (2 * rngd.random(12) - 1) for _ in range(200)]
    out["c24"] = [o + 0.005 * (2 * rngd.random(12) - 1) for _ in range(12)]
    return out
