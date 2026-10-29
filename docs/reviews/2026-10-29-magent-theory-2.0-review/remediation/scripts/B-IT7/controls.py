r"""B-IT7: the three controls of the repaired chapter-05 standard for the Item 7 instance.

(N1) label-preserving null: peer costs deleted, self weight a e^{-D_ii}/(a e^{-D_ii} + 1 - a) (the 'null' mode of
     checks.py); after a handoff the control remains the fine flow.
(N2) renormalized null: peer labels removed, self (evaluator) weight one: y' = (g - y) + rho (o - y).
(N3) frozen-environment control: nothing deleted, every other agent's law frozen at t = 0 in every cost and row;
     after a handoff the control remains the fine (pre-handoff, individual-row) flow.
The selector is evaluated on block statistics computed from the fine states (SharedPop.select / DerivedPop.select).
Usage: python controls.py declared | derived <Tp>
"""
import os, sys, io, contextlib
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

SRC = r"C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/item7_checks/checks.py"
lines = open(SRC, encoding="utf-8").read().splitlines()
ns = {}
with contextlib.redirect_stdout(io.StringIO()):
    exec("\n".join(lines[:231]), ns)          # definitions, checks (1)-(5), fixed draw y0 (line 231)
    exec("\n".join(lines[412:525]), ns)       # derived-selector definitions
SharedPop, DerivedPop = ns["SharedPop"], ns["DerivedPop"]
o, y0 = ns["o"], ns["y0"]
lam, rho, ps, g, delta, Delta = (ns[k] for k in ("lam", "rho", "ps", "g", "delta", "Delta"))


def ctl_vel(s, y):
    if s.mode == "lp":
        Dii = lam / 2 * (y - g) ** 2
        b0 = ps * np.exp(-Dii) / (ps * np.exp(-Dii) + 1 - ps)
        return b0 * (g - y) + rho * (s.o - y)
    if s.mode == "rn":
        return (g - y) + rho * (s.o - y)
    if s.mode == "fr":
        Y = np.tile(s.y_init, (s.N, 1)); np.fill_diagonal(Y, g)          # sources: frozen peers, evaluator as self
        D = lam / 2 * (y[:, None] - Y) ** 2
        pri = np.full((s.N, s.N), s.pp); np.fill_diagonal(pri, ps)
        W = pri * np.exp(-(D - D.min(1, keepdims=True))); W /= W.sum(1, keepdims=True)
        return (W * Y).sum(1) - y + rho * (s.o - y)
    return SharedPop.vel(s, y)


class SCtl(SharedPop):
    vel = ctl_vel


class DCtl(DerivedPop):
    vel = ctl_vel


def run(cls, yy, Tp, mode, Tend, dt=0.05):
    P = cls(o, lam, rho, ps, g, delta, Delta, Tp, mode=mode); P.y_init = yy.copy()
    yf = P.run(yy, Tend, dt=dt)
    return [(h[0], h[1], h[2]) for h in P.hist], yf


def seq(h):
    return tuple(sorted((lev, b) for _, lev, b in h))   # modulo exchange of disjoint merges


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "declared":
        rs = np.random.default_rng(1029)
        draws = [y0] + [o + 0.005 * (2 * rs.random(len(o)) - 1) for _ in range(12)]
        for mode, Tend in (("lp", 60.0), ("rn", 60.0), ("fr", 400.0), ("full", 30.0)):
            out = [run(SCtl, d, 2.0, mode, Tend) for d in draws]
            h0, yf0 = out[0]
            print(f"[declared, Tp=2] {mode}: fixed draw events {[(t, lev, len(b)) for t, lev, b in h0]}")
            if mode == "fr":
                L = yf0[:6]; print("    fr fixed draw: final left pools", np.round([L[:3].mean(), L[3:].mean()], 4),
                                   "gap", round(float(L[3:].mean() - L[:3].mean()), 4))
            seqs = {seq(h) for h, _ in out}
            print(f"    distinct partition sequences over {len(draws)} states: {len(seqs)};",
                  "event-time range", (min(min(t for t, _, _ in h) for h, _ in out if h) if any(h for h, _ in out) else None,
                                       max(max(t for t, _, _ in h) for h, _ in out if h) if any(h for h, _ in out) else None))
    else:
        Tp = float(sys.argv[2])
        for mode, Tend in (("lp", 40.0), ("rn", 40.0), ("fr", 120.0)):
            h, yf = run(DCtl, y0, Tp, mode, Tend)
            print(f"[derived, Tp={Tp}] {mode}: fixed draw events {[(t, lev, len(b)) for t, lev, b in h]}")
