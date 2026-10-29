"""Shared construction of the tangent communication model (it2:model) for D24-LDYN checks.
Edges are (sender j, receiver i). Dictionaries omega_e in gl(K) are row-major vectors of
length K^2 with the Frobenius inner product; kappa_i(omega) = omega c_i."""
import numpy as np
from itertools import permutations


def kappa(c):
    K = len(c)
    return np.kron(np.eye(K), c[None, :])          # K x K^2, kappa @ vec(w) = w @ c


def code_kron(T):
    K = T.shape[1]
    return np.kron(T, np.eye(K))                    # vec(T w) = (T x I) vec(w)


def null_basis(M, tol=1e-10):
    u, s, vt = np.linalg.svd(M)
    r = int((s > tol).sum())
    return vt[r:].T, vt[:r].T                       # (kernel basis, row-space basis)


def simple_cycles(N, edges):
    """All simple directed cycles as lists of edge indices (brute force, small graphs)."""
    E = {(j, i): k for k, (j, i) in enumerate(edges)}
    cyc = set()
    for L in range(2, N + 1):
        for verts in permutations(range(N), L):
            if verts[0] > min(verts):
                continue
            ks = []
            ok = True
            for a in range(L):
                j, i = verts[a], verts[(a + 1) % L]   # edge j -> i
                if (j, i) not in E:
                    ok = False
                    break
                ks.append(E[(j, i)])
            if ok:
                cyc.add(tuple(sorted(ks)))
    return [list(c) for c in sorted(cyc)]


def spd(n, rng, lo, hi):
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    return Q @ np.diag(rng.uniform(lo, hi, size=n)) @ Q.T


class Model:
    def __init__(self, N, K, edges, rng, sigma2=1.0, lam=0.7, gscale=1.0, c_zero=False,
                 P=None, Tlist=None, Rlist=None, cvec=None, grounded=True, ranks=None):
        self.N, self.K, self.edges = N, K, edges
        self.E = len(edges)
        self.sigma2, self.lam = sigma2, lam
        d = K * K
        self.d = d
        self.P = P if P is not None else [spd(K, rng, 0.5, 2.0) for _ in range(N)]
        self.m = [rng.normal(size=K) for _ in range(N)]
        if cvec is not None:
            self.c = cvec
        else:
            self.c = [np.zeros(K) if c_zero else rng.normal(size=K) for _ in range(N)]
        self.T, self.R, self.o, self.what, self.Gbar = [], [], [], [], []
        for k in range(self.E):
            if Tlist is not None:
                T = Tlist[k]
            else:
                n = ranks[k] if ranks is not None else int(rng.integers(1, K + 1))
                T = rng.normal(size=(n, K))
            n = T.shape[0]
            self.T.append(T)
            self.R.append(Rlist[k] if Rlist is not None else spd(n, rng, 0.5, 2.0))
            self.o.append(rng.normal(size=n))
            self.what.append(rng.normal(size=d))
            TK = code_kron(T)
            M = spd(TK.shape[0], rng, 0.5, 2.0)
            self.Gbar.append((TK.T @ M @ TK) if grounded else np.zeros((d, d)))
        self.gscale = gscale
        self.loops = simple_cycles(N, edges)
        self.W = [spd(d, rng, 0.5, 2.0) for _ in self.loops]
        self.iy = [np.arange(i * K, (i + 1) * K) for i in range(N)]
        off = N * K
        self.iw = [np.arange(off + k * d, off + (k + 1) * d) for k in range(self.E)]
        self.n = off + self.E * d
        self.inn = [[k for k, (j, i) in enumerate(edges) if i == a] for a in range(N)]
        self.out = [[k for k, (j, i) in enumerate(edges) if j == a] for a in range(N)]
        self.blocks = [np.concatenate([self.iy[a]] + [self.iw[k] for k in self.inn[a]]) for a in range(N)]
        self.Lmat = []
        for k, (j, i) in enumerate(edges):
            L = np.zeros((self.T[k].shape[0], self.n))
            L[:, self.iy[i]] += self.T[k]
            L[:, self.iy[j]] -= self.T[k]
            L[:, self.iw[k]] -= self.T[k] @ kappa(self.c[i])
            self.Lmat.append(L)
        self.Bg = []
        for cyc in self.loops:
            B = np.zeros((d, self.n))
            for k in cyc:
                B[:, self.iw[k]] += np.eye(d)
            self.Bg.append(B)

    def G(self, k):
        return self.gscale * self.Gbar[k]

    def global_J_eta(self, Rrep=None, ozero=False):
        """Global energy it2:global-energy; Rrep[k] replaces R_e (used for E^S)."""
        J = np.zeros((self.n, self.n))
        eta = np.zeros(self.n)
        for a in range(self.N):
            J[np.ix_(self.iy[a], self.iy[a])] += self.P[a]
            eta[self.iy[a]] += self.P[a] @ self.m[a]
        for k in range(self.E):
            w = self.iw[k]
            J[np.ix_(w, w)] += np.eye(self.d) / self.sigma2 + self.G(k)
            eta[w] += self.G(k) @ self.what[k]
            Rk = self.R[k] if Rrep is None else Rrep[k]
            Ri = np.linalg.inv(Rk)
            L = self.Lmat[k]
            J += L.T @ Ri @ L
            if not ozero:
                eta += L.T @ Ri @ self.o[k]
        for g, B in enumerate(self.Bg):
            J += self.lam * B.T @ self.W[g] @ B
        return J, eta

    def per_agent_Q(self, a, mode='plug', Sig=None, Vd=None, ozero=False):
        """Full-vector quadratic of agent a's energy it2:agent-energy: E_a = x^T Q x/2 - q^T x."""
        Q = np.zeros((self.n, self.n))
        q = np.zeros(self.n)
        Q[np.ix_(self.iy[a], self.iy[a])] += self.P[a]
        q[self.iy[a]] += self.P[a] @ self.m[a]
        for k in self.inn[a]:
            j, i = self.edges[k]
            w = self.iw[k]
            Q[np.ix_(w, w)] += np.eye(self.d) / self.sigma2 + self.G(k)
            q[w] += self.G(k) @ self.what[k]
            if mode == 'plug':
                Rk = self.R[k]
            else:
                Rk = self.R[k] + self.T[k] @ Sig[j] @ self.T[k].T
            Ri = np.linalg.inv(Rk)
            L = self.Lmat[k]
            Q += L.T @ Ri @ L
            if not ozero:
                q += L.T @ Ri @ self.o[k]
        if self.lam > 0:
            for g, cyc in enumerate(self.loops):
                if not any(k in self.inn[a] for k in cyc):
                    continue
                if mode == 'plug':
                    Om = self.lam * self.W[g]
                else:
                    Vsum = sum(Vd[f] for f in cyc if f not in self.inn[a])
                    Om = np.linalg.inv(np.linalg.inv(self.lam * self.W[g]) + Vsum)
                B = self.Bg[g]
                Q += B.T @ Om @ B
        return Q, q

    def per_agent_H_b(self, mode='plug', Sig=None, Vd=None, ozero=False):
        H = np.zeros((self.n, self.n))
        b = np.zeros(self.n)
        A = np.zeros((self.n, self.n))
        for a in range(self.N):
            Q, q = self.per_agent_Q(a, mode, Sig, Vd, ozero)
            bl = self.blocks[a]
            H[bl, :] = Q[bl, :]
            b[bl] = q[bl]
            A[np.ix_(bl, bl)] = Q[np.ix_(bl, bl)]
        return H, b, A

    def law_level_cov_map(self, Sig, Vd):
        """One synchronous per-agent covariance update S_a = A_a(peers)^{-1}; returns marginals."""
        newS, newV = [None] * self.N, [None] * self.E
        for a in range(self.N):
            Q, _ = self.per_agent_Q(a, 'law', Sig, Vd)
            bl = self.blocks[a]
            S = np.linalg.inv(Q[np.ix_(bl, bl)])
            pos = {g: t for t, g in enumerate(bl)}
            iy = [pos[g] for g in self.iy[a]]
            newS[a] = S[np.ix_(iy, iy)]
            for k in self.inn[a]:
                iw = [pos[g] for g in self.iw[k]]
                newV[k] = S[np.ix_(iw, iw)]
        return newS, newV

    def law_level_fixed_point(self, start=0.0, tol=1e-13, maxit=20000):
        Sig = [start * np.eye(self.K) for _ in range(self.N)]
        Vd = [start * np.eye(self.d) for _ in range(self.E)]
        for it in range(maxit):
            S2, V2 = self.law_level_cov_map(Sig, Vd)
            err = max(max(np.abs(S2[a] - Sig[a]).max() for a in range(self.N)),
                      max(np.abs(V2[k] - Vd[k]).max() for k in range(self.E)))
            Sig, Vd = S2, V2
            if err < tol:
                return Sig, Vd, it + 1
        return Sig, Vd, maxit
