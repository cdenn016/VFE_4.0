r"""D27-ITEM7: independent hybrid simulator of the twelve-agent instance (it7:ch-instance) with the declared
selector it7:selector in continuous-time semantics.

Written from the text: per-agent Gibbs flow with individual rows before a handoff, shared-label rows after it
(it7:exact (i)-(ii)), routing of outside receivers to parents, candidate = connected component of the coherence
graph of units that is walled from every other unit, handoff when a candidate has persisted for tau_p.
Candidate onsets are located by bisection inside the RK4 step and handoffs are executed at onset + tau_p exactly.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

LAM, RHO, PS, G, T = 6.0, 0.005, 0.05, 0.0, 1.0
DC, DW, TP = 0.05, 0.5, 2.0


class Hybrid:
    def __init__(self, o, y0, tp=TP, mode="full"):
        self.o = np.asarray(o, float); self.N = len(o); self.pp = (1 - PS) / (self.N - 1)
        self.y = np.array(y0, float); self.t = 0.0; self.tp = tp; self.mode = mode
        self.units = [(i,) for i in range(self.N)]
        self.cand = {}; self.hist = []; self.level = {(i,): 0 for i in range(self.N)}
        self._index()

    def _index(self):
        K = len(self.units)
        self.lab = np.empty(self.N, int)
        self.A = np.zeros((K, self.N))
        for j, u in enumerate(self.units):
            self.lab[list(u)] = j; self.A[j, list(u)] = 1.0 / len(u)
        self.n = np.array([len(u) for u in self.units], float)

    def vel(self, y):
        if self.mode == "null":                        # peer costs deleted, labels and priors kept
            D = LAM / 2 * (y - G) ** 2
            b0 = PS * np.exp(-D / T) / (PS * np.exp(-D / T) + 1 - PS)
            return b0 * (G - y) + RHO * (self.o - y)
        m = self.A @ y; n = self.n
        C = (n[:, None] * LAM / 2) * (m[:, None] - m[None, :]) ** 2   # (K, K) receiver unit x source unit
        Cg = n * LAM / 2 * (m - G) ** 2
        Pr = np.tile(n * self.pp, (len(n), 1)); np.fill_diagonal(Pr, (n - 1) * self.pp)
        big = np.where(Pr > 0, C, np.inf)
        cmin = np.minimum(big.min(1), Cg)
        W = Pr * np.exp(-(C - cmin[:, None]) / T); Wg = PS * np.exp(-(Cg - cmin) / T)
        target = (W @ m + Wg * G) / (W.sum(1) + Wg)
        return target[self.lab] - y + RHO * (self.o - y)

    def rk4(self, y, h):
        k1 = self.vel(y); k2 = self.vel(y + h / 2 * k1); k3 = self.vel(y + h / 2 * k2); k4 = self.vel(y + h * k3)
        return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def margin(self, y, C):
        """g_C > 0 iff the set C of current units (indices) is a candidate."""
        m = self.A @ y
        inside = np.sort(m[list(C)])
        coh = DC - LAM / 2 * np.max(np.diff(inside)) ** 2
        out = [j for j in range(len(m)) if j not in C]
        wall = min(LAM / 2 * (m[i] - m[j]) ** 2 for i in C for j in out) - DW if out else np.inf
        return min(coh, wall)

    def candidates(self, y):
        m = self.A @ y; K = len(m); order = np.argsort(m)
        found = []
        start = 0
        for k in range(1, K + 1):                      # 1-D: components are runs of the sorted means
            if k == K or LAM / 2 * (m[order[k]] - m[order[k - 1]]) ** 2 >= DC:
                comp = tuple(sorted(order[start:k].tolist())); start = k
                if len(comp) >= 2 and self.margin(y, comp) > 0:
                    found.append(comp)
        return found

    def key(self, comp):
        return frozenset(a for j in comp for a in self.units[j])

    def run(self, Tend, h=0.01, record=None):
        """Integrate to Tend; record(t, y) is called after every step."""
        while self.t < Tend - 1e-13:
            tnext = min([on + self.tp for on in self.cand.values()], default=np.inf)
            step = min(h, Tend - self.t, max(tnext - self.t, 0.0))
            y_prev, t_prev = self.y.copy(), self.t
            if step > 0:
                self.y = self.rk4(self.y, step); self.t += step
            # handoff due now?
            due = [(on + self.tp, kk) for kk, on in self.cand.items() if on + self.tp <= self.t + 1e-12]
            if due:
                _, kk = min(due, key=lambda z: z[0])
                comp = [j for j, u in enumerate(self.units) if set(u) <= kk]
                if self.margin(self.y, comp) > 0:
                    lev = 1 + max(self.level[self.units[j]] for j in comp)
                    newu = tuple(sorted(kk))
                    self.units = [u for j, u in enumerate(self.units) if j not in comp] + [newu]
                    self.level[newu] = lev; self.hist.append((self.t, lev, newu)); self._index()
                del self.cand[kk]
                y_prev, t_prev = self.y.copy(), self.t      # structural change: onsets measured from here
            # candidate bookkeeping
            now = {self.key(c): c for c in self.candidates(self.y)}
            for kk in list(self.cand):
                if kk not in now: del self.cand[kk]
            for kk, c in now.items():
                if kk in self.cand: continue
                comp = [j for j, u in enumerate(self.units) if set(u) <= kk]
                if t_prev < self.t and self.margin(y_prev, comp) <= 0:
                    lo, hi = 0.0, self.t - t_prev           # bisection for the onset inside the last step
                    for _ in range(40):
                        mid = (lo + hi) / 2
                        if self.margin(self.rk4(y_prev, mid), comp) > 0: hi = mid
                        else: lo = mid
                    self.cand[kk] = t_prev + hi
                else:
                    self.cand[kk] = self.t
            if record is not None: record(self.t, self.y)
        return self.y
