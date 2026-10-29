r"""GAP07-S3: remaining numerical statements in scope.

(a) it7:null proof constants, rigorous (mpmath iv), for every record gap s in {1.0, 1.1, 1.2, 1.3} of checks (11), (14):
    with r_min = smallest |o_i| - 0.005 and r_max = largest |o_i| + 0.005, the evaluator term |y| beta0(y) on
    [r_min, r_max] is below the record term rho * 0.005 = 2.5e-5 at the box boundary, so the box is forward invariant, and
    adjacent agents stay at distance >= 0.14 (KL >= 0.0588 > delta_c). |y| beta0(y) is decreasing for |y| > 1/sqrt(6).
(b) (F2) paragraph (line 252): the record-determined limits of the differences of member deviations,
    rho/(1+rho) |o_i - o_j|, at most 0.0015 (first stage) and 0.007 (unions).
(c) it7:depth: local exponents of the certified middle roots d_u(n), n = 3, 4, 6 (claim: "roughly as n^{-1/2}").
(d) Check (15) (attention-source example, no records) rerun at a = 0, 0.3, 0.6 with reduced horizons 45, 105, 250, with the
    authors' attn_run (checks.py lines 335-373), and its null control: deleting peer costs leaves each agent's row with
    its own evaluator target eta_i only, and y(0) = eta is then an equilibrium, so the null never selects.
"""
import os, sys, time, io, contextlib
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
from mpmath import iv, mp, mpf, nstr, log

T0 = time.time()
iv.dps = 30; mp.dps = 30
PS = iv.mpf("0.05")
print("(a) it7:null constants:")
for s in ("1.0", "1.1", "1.2", "1.3"):
    gap = mpf(s); centres = [mpf("2.5") - gap / 2, mpf("2.5") + gap / 2]
    absrec = sorted(c + o_ for c in centres for o_ in (mpf("-0.15"), mpf(0), mpf("0.15")))
    rmin, rmax = absrec[0] - mpf("0.005"), absrec[-1] + mpf("0.005")
    Y = iv.mpf([rmin, rmax])
    # beta0(y) = ps e^{-3y^2} / (ps e^{-3y^2} + 1 - ps) <= ps/(1-ps) e^{-3 y^2}; |y| beta0 decreasing for |y| > 1/sqrt 6
    b0 = PS / (1 - PS) * iv.exp(-3 * iv.mpf(rmin) ** 2)
    term = iv.mpf(rmin) * b0
    print(f"    s = {s}: |y| in [{nstr(rmin, 5)}, {nstr(rmax, 5)}]; beta0 <= {nstr(mpf(b0._mpi_[1]), 4)};"
          f" evaluator term <= {nstr(mpf(term._mpi_[1]), 4)} < record term 2.5e-5: {mpf(term._mpi_[1]) < mpf('2.5e-5')};"
          f" adjacent KL >= 3 (0.14)^2 = {nstr(3 * mpf('0.14') ** 2, 4)} > 0.05")
print("(b) deviation limits: first stage", round(0.005 / 1.005 * 0.3, 6), "; unions",
      round(0.005 / 1.005 * 1.4, 6), "(text: at most 0.0015 and 0.007)")
du = {3: mpf("0.963036186873"), 4: mpf("0.770747038692"), 6: mpf("0.592923517107")}
print("(c) local exponents of d_u(n): 3->4", nstr(log(du[3] / du[4]) / log(mpf(4) / 3), 4), "; 4->6",
      nstr(log(du[4] / du[6]) / log(mpf(6) / 4), 4), "; 3->6", nstr(log(du[3] / du[6]) / log(mpf(2)), 4),
      "(text: roughly 1/2)")

SRC = r"C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/item7_checks/checks.py"
L = open(SRC, encoding="utf-8").readlines()
assert L[334].startswith("def attn_run") and L[372].startswith("    return hist")
ns = {"np": np}
exec(compile("".join(L[133:145]) + "".join(L[334:374]), "attn", "exec"), ns)       # comps (134-144) and attn_run
eta = np.array([c_ + o_ for c_ in (-3.2, -1.8, 1.8, 3.2) for o_ in (-0.3, 0.0, 0.3)])
stored = {0.0: ([3.2] * 4 + [39.98] * 2), 0.3: ([3.2] * 4 + [100.18] * 2), 0.6: ([3.2] * 4 + [245.28] * 2)}
print("(d) check (15), reduced horizons:")
for a, Tend in ((0.0, 45.0), (0.3, 105.0), (0.6, 250.0)):
    t1 = time.time(); h_ = ns["attn_run"](eta, a, T_end=Tend)
    print(f"    a = {a}: blocks {sorted((l_, b_) for _, l_, b_ in h_)}; times {[x_[0] for x_ in h_]};"
          f" identical to stored times: {[x_[0] for x_ in h_] == stored[a]}; {time.time() - t1:.1f} s")
    if time.time() - T0 > 20: break
lam_, ps_ = 6.0, 0.05
D = lam_ / 2 * (eta - eta) ** 2
b0 = ps_ * np.exp(-D) / (ps_ * np.exp(-D) + 1 - ps_)
v0 = b0 * (eta - eta)
print("    null control at y(0) = eta: velocity", np.abs(v0).max(), "; smallest adjacent KL",
      round(float(lam_ / 2 * np.min(np.diff(np.sort(eta))) ** 2), 4), "> delta_c = 0.05, so the null never selects")
print("elapsed", round(time.time() - T0, 1), "s")
