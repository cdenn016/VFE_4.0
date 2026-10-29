r"""GAP06-S2: exact event-driven integrator for the fusion inclusion of chapters/06_fusion.tex.

The flow is W mu' in -dF(mu), with
    F(mu) = sum_i n_i U(mu_i - x_i) + lam sum_{i<j} n_i n_j (|mu_i^q - mu_j^q| + |mu_i^s - mu_j^s|),
    U(p, s) = (a/2)(p^2 + s^2) - c p s,
and x_i = 0 in the chapter (fus:potential); nonzero x_i gives the heterogeneous-reference variant
of 2026-10-29 D11-FUS-08.  The right derivative is the minimal W-norm element of -W^{-1} dF
(Brezis 1973).  In each sector the agents are grouped into clusters of exactly equal values.
Between events every cluster moves with the multiplicity-weighted mean of its members' free
forces plus the constant external sign forces, so the state solves an affine linear ODE, which is
integrated exactly with a matrix exponential.  Events are (i) closure of the gap between adjacent
clusters of a sector and (ii) loss of feasibility of the internal subgradients of a cluster
(cut inequalities of the weighted complete-graph base polytope).  Event times are located on a
time grid and refined with brentq on the exact affine solution.  At a closure the clusters merge;
feasibility of the merged cluster is checked by a box-constrained least-squares problem for the
minimal W-norm velocity.  This integrator is independent of the proximal scheme used by D11-FUS.
"""
from itertools import combinations
import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq, lsq_linear


class FusionFlow:
    def __init__(self, n, a, c, lam, xref=None, tol=1e-12):
        self.n = np.asarray(n, dtype=float)
        self.r = len(self.n)
        self.a, self.c, self.lam = float(a), float(c), float(lam)
        self.x = np.zeros((self.r, 2)) if xref is None else np.asarray(xref, dtype=float)
        self.tol = tol

    # index of coordinate (agent i, sector al) in the flat state
    @staticmethod
    def idx(i, al):
        return 2 * i + al

    def free_force_affine(self):
        r"""F0_i^al = -a (mu_i^al - x_i^al) + c (mu_i^al' - x_i^al') as (Mf, bf) with F0 = Mf mu + bf."""
        r = self.r
        Mf = np.zeros((2 * r, 2 * r)); bf = np.zeros(2 * r)
        for i in range(r):
            for al in (0, 1):
                be = 1 - al
                k = self.idx(i, al)
                Mf[k, self.idx(i, al)] = -self.a
                Mf[k, self.idx(i, be)] = self.c
                bf[k] = self.a * self.x[i, al] - self.c * self.x[i, be]
        return Mf, bf

    def build_phase(self, clusters):
        r"""Affine system mu' = M mu + b for fixed sorted clusters (clusters[al] = list of lists)."""
        r = self.r
        Mf, bf = self.free_force_affine()
        M = np.zeros((2 * r, 2 * r)); b = np.zeros(2 * r)
        for al in (0, 1):
            cl = clusters[al]
            nC = [self.n[C].sum() for C in cl]
            for p, C in enumerate(cl):
                # constant external sign force: clusters below push down (sign +1), above up
                below = sum(nC[:p]); above = sum(nC[p + 1:])
                ext = -self.lam * (below - above)
                rowM = np.zeros(2 * r); rowb = ext
                for j in C:
                    w = self.n[j] / nC[p]
                    rowM += w * Mf[self.idx(j, al)]
                    rowb += w * bf[self.idx(j, al)]
                for i in C:
                    M[self.idx(i, al)] = rowM
                    b[self.idx(i, al)] = rowb
        return M, b

    def event_functions(self, clusters):
        r"""Linear event functions g = G [mu; 1] that must stay positive within a phase.

        Gaps between adjacent clusters, and cut margins lam n_S n_{C\S} - sum_{j in S} y_j with
        y_j = n_j (F0_j - v_C) for every proper nonempty subset S of every cluster C."""
        r = self.r
        Mf, bf = self.free_force_affine()
        rows, labels = [], []
        for al in (0, 1):
            cl = clusters[al]
            for p in range(len(cl) - 1):
                g = np.zeros(2 * r + 1)
                g[self.idx(cl[p + 1][0], al)] += 1.0
                g[self.idx(cl[p][0], al)] -= 1.0
                rows.append(g); labels.append(('gap', al, p))
            for p, C in enumerate(cl):
                if len(C) < 2:
                    continue
                nC = self.n[C].sum()
                # y_j = n_j (F0_j - sum_k (n_k/nC) F0_k), affine in mu
                F0rows = {j: np.concatenate([Mf[self.idx(j, al)], [bf[self.idx(j, al)]]]) for j in C}
                vC = sum((self.n[k] / nC) * F0rows[k] for k in C)
                for size in range(1, len(C)):
                    for S in combinations(C, size):
                        nS = self.n[list(S)].sum()
                        g = np.zeros(2 * r + 1)
                        g[-1] += self.lam * nS * (nC - nS)
                        for j in S:
                            g -= self.n[j] * (F0rows[j] - vC)
                        rows.append(g); labels.append(('cut', al, p, S))
        if not rows:
            return np.zeros((0, 2 * r + 1)), labels
        return np.array(rows), labels

    def min_norm_velocity_cluster(self, mu, clusters, al, p):
        r"""Minimal W-norm velocities of the members of cluster p in sector al (box LSQ in eps).

        The external sign force uses the cluster order, so clusters that share a position
        right after a split are treated as separating (subgradient +-1 between them)."""
        Mf, bf = self.free_force_affine()
        F0 = Mf @ mu.ravel() + bf
        cl = clusters[al]
        C = cl[p]
        below = sum(self.n[D].sum() for D in cl[:p])
        above = sum(self.n[D].sum() for D in cl[p + 1:])
        ext = -self.lam * (below - above)
        f = np.array([F0[self.idx(j, al)] + ext for j in C])
        pairs = list(combinations(range(len(C)), 2))
        G = np.zeros((len(C), len(pairs)))
        for q, (u, v) in enumerate(pairs):
            G[u, q] = -self.lam * self.n[C[v]]
            G[v, q] = +self.lam * self.n[C[u]]
        sw = np.sqrt(self.n[C])
        res = lsq_linear(sw[:, None] * G, -sw * f, bounds=(-1, 1), method='bvls', tol=1e-14)
        vel = f + G @ res.x
        return vel

    def split_cluster(self, mu, clusters, al, p):
        C = clusters[al][p]
        vel = self.min_norm_velocity_cluster(mu, clusters, al, p)
        order = np.argsort(vel, kind='mergesort')
        groups, cur = [], [C[order[0]]]
        scale = 1.0 + np.abs(vel).max()
        for a_, b_ in zip(order[:-1], order[1:]):
            if vel[b_] - vel[a_] > 1e-9 * scale:
                groups.append(cur); cur = [C[b_]]
            else:
                cur.append(C[b_])
        groups.append(cur)
        return groups  # ascending velocity, all at the same position

    def initial_clusters(self, mu):
        clusters = []
        for al in (0, 1):
            order = np.argsort(mu[:, al], kind='mergesort')
            cl = [[int(order[0])]]
            for i in order[1:]:
                if abs(mu[i, al] - mu[cl[-1][0], al]) <= self.tol * (1 + abs(mu[i, al])):
                    cl[-1].append(int(i))
                else:
                    cl.append([int(i)])
            clusters.append(cl)
        return clusters

    def merge_touching(self, mu, clusters):
        r"""Merge adjacent clusters that touch (gap <= tol) and approach (or are at rest relative
        to each other).  Needed when two events of different sectors occur at the same instant."""
        changed = True
        any_change = False
        while changed:
            changed = False
            M, b = self.build_phase(clusters)
            v = (M @ mu.ravel() + b).reshape(self.r, 2)
            for al in (0, 1):
                cl = clusters[al]
                for p in range(len(cl) - 1):
                    lo, hi = cl[p], cl[p + 1]
                    gap = mu[hi[0], al] - mu[lo[0], al]
                    scale = 1.0 + abs(mu[hi[0], al])
                    if gap <= 1e-10 * scale and v[hi[0], al] - v[lo[0], al] <= 1e-10 * (1 + abs(v).max()):
                        clusters[al] = cl[:p] + [lo + hi] + cl[p + 2:]
                        changed = True; any_change = True
                        break
                if changed:
                    break
        return clusters, any_change

    def normalize(self, mu, clusters):
        r"""Set members of each cluster exactly equal (weighted mean); resolve infeasible clusters."""
        mu = mu.copy()
        changed = True
        while changed:
            changed = False
            clusters, ch = self.merge_touching(mu, clusters)
            changed = changed or ch
            for al in (0, 1):
                for C in clusters[al]:
                    w = self.n[C] / self.n[C].sum()
                    mu[C, al] = float(w @ mu[C, al])
                newcl = []
                for p, C in enumerate(clusters[al]):
                    if len(C) >= 2:
                        groups = self.split_cluster(mu, clusters, al, p)
                        if len(groups) > 1:
                            changed = True
                        newcl.extend(groups)
                    else:
                        newcl.append(C)
                clusters[al] = newcl
        return mu, clusters

    def sync_partition(self, clusters):
        lab = {}
        for al in (0, 1):
            for p, C in enumerate(clusters[al]):
                for i in C:
                    lab.setdefault(i, []).append(p)
        groups = {}
        for i in range(self.r):
            groups.setdefault(tuple(lab[i]), []).append(i)
        return tuple(sorted(tuple(sorted(g)) for g in groups.values()))

    def run(self, mu0, T, dt=2e-3, max_events=200, record_sector=False):
        mu = np.asarray(mu0, dtype=float).copy()
        clusters = self.initial_clusters(mu)
        hist = [(0.0, self.sync_partition(clusters))]   # raw initial coincidence partition
        mu, clusters = self.normalize(mu, clusters)
        t = 0.0
        sp0 = self.sync_partition(clusters)
        if sp0 != hist[-1][1]:
            hist.append((0.0, sp0))                      # immediate split at t = 0+
        sector_hist = []
        for _ in range(max_events):
            M, b = self.build_phase(clusters)
            r2 = 2 * self.r
            Maug = np.zeros((r2 + 1, r2 + 1)); Maug[:r2, :r2] = M; Maug[:r2, -1] = b
            G, labels = self.event_functions(clusters)
            y0 = np.concatenate([mu.ravel(), [1.0]])
            if G.shape[0] == 0:
                break
            E = expm(Maug * dt)
            y = y0.copy(); gprev = G @ y; tt = 0.0
            hit = None
            while t + tt < T:
                y_next = E @ y
                gnext = G @ y_next
                cross = np.where((gprev > 0) & (gnext <= 0))[0]
                if cross.size:
                    best = None
                    for k in cross:
                        fk = lambda s, k=k: G[k] @ (expm(Maug * s) @ y0)
                        tk = brentq(fk, tt, tt + dt, xtol=1e-15, rtol=1e-15, maxiter=200)
                        if best is None or tk < best[0]:
                            best = (tk, k)
                    hit = best
                    break
                y, gprev = y_next, gnext; tt += dt
            if hit is None:
                mu = (expm(Maug * (T - t)) @ y0)[:r2].reshape(self.r, 2)
                t = T
                break
            te, k = hit
            yE = expm(Maug * te) @ y0
            mu = yE[:r2].reshape(self.r, 2)
            t += te
            lab = labels[k]
            if lab[0] == 'gap':
                al, p = lab[1], lab[2]
                merged = clusters[al][p] + clusters[al][p + 1]
                clusters[al] = clusters[al][:p] + [merged] + clusters[al][p + 2:]
                if record_sector:
                    sector_hist.append((t, 'merge', al, tuple(sorted(merged))))
            else:
                al, p, S = lab[1], lab[2], lab[3]
                C = clusters[al][p]
                rest = [j for j in C if j not in S]
                clusters[al] = clusters[al][:p] + [rest, list(S)] + clusters[al][p + 1:]
                if record_sector:
                    sector_hist.append((t, 'split', al, tuple(sorted(S)), tuple(sorted(rest))))
            before = [list(map(tuple, cl)) for cl in clusters]
            mu, clusters = self.normalize(mu, clusters)
            after = [list(map(tuple, cl)) for cl in clusters]
            if record_sector and before != after:
                sector_hist.append((t, 'regroup', before, after))
            sp = self.sync_partition(clusters)
            if sp != hist[-1][1]:
                hist.append((t, sp))
        return hist, mu, sector_hist
