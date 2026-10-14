"""E7: (1) native free partition Z_L = 4^{-L^2} sum_z r_L(z) of def:rg-pressure-native equals
sum_S t^{|S|} b^{|d_2 S|} and the plaquette Ising form with K = (1/2) log(1/b),
H_p = (1/2) log t + (c_p/2) log b, at L = 2, 3; (2) the finite raw pressure p_L = N^{-1} log Z_L
has d^2 p_L / dj^2 at j = 0 equal to [4(L-1)(L-2)(b^6-b^8) - (L-1)^2 b^8]/L^2, whose limit is
4 b^6 - 5 b^8 (negative for b^2 > 4/5)."""
import itertools, math
import numpy as np

def geometry(L):
    edges = [((x1, x2), a) for x1 in range(L) for x2 in range(L) for a in (1, 2)]
    idx = {e: k for k, e in enumerate(edges)}
    plaq = []
    for x1 in range(L - 1):
        for x2 in range(L - 1):
            plaq.append([idx[((x1, x2), 1)], idx[((x1 + 1, x2), 2)], idx[((x1, x2 + 1), 1)], idx[((x1, x2), 2)]])
    return len(edges), plaq

def Z_native(L, b, t):
    nE, plaq = geometry(L)
    tot = 0.0
    for z in itertools.product((-1, 1), repeat=nE):
        r = 1.0
        for e in range(nE):
            r *= 1 + b * z[e]
        for p in plaq:
            r *= 1 + t * z[p[0]] * z[p[1]] * z[p[2]] * z[p[3]]
        tot += r
    return tot / 4 ** (L * L)

def Z_dual(L, b, t):
    nE, plaq = geometry(L)
    tot = 0.0
    for S in itertools.product((0, 1), repeat=len(plaq)):
        cnt = np.zeros(nE, int)
        for s, p in zip(S, plaq):
            if s:
                cnt[p] += 1
        tot += t ** sum(S) * b ** int((cnt % 2).sum())
    return tot

def Z_ising(L, b, t):
    nE, plaq = geometry(L)
    K = 0.5 * math.log(1 / b)
    inc = np.zeros(nE, int)
    for p in plaq:
        inc[p] += 1
    c = [sum(1 for e in p if inc[e] == 1) for p in plaq]
    pairs = [(i, k) for i in range(len(plaq)) for k in range(i + 1, len(plaq)) if set(plaq[i]) & set(plaq[k])]
    H = [0.5 * math.log(t) + 0.5 * cp * math.log(b) for cp in c]
    tot = 0.0
    for s in itertools.product((-1, 1), repeat=len(plaq)):
        E = sum(K * (s[i] * s[k] - 1) for i, k in pairs) + sum(H[i] * (s[i] + 1) for i in range(len(plaq)))
        tot += math.exp(E)
    return tot

for L in (2, 3):
    for b, t in ((0.4, 0.07), (0.9, 0.3)):
        zn, zd, zi = Z_native(L, b, t), Z_dual(L, b, t), Z_ising(L, b, t)
        print(f"L={L} b={b} t={t}: native {zn:.15e} dual rel {abs(zd-zn)/zn:.1e} ising rel {abs(zi-zn)/zn:.1e}")

# (2) second j-derivative of p_L at j = 0 by the dual sum (t = tanh j), central difference
for L in (3, 4):
    for bval in (0.5, 0.95):
        h = math.atanh(bval); d = 1e-3
        f = lambda j: math.log(Z_dual(L, bval, math.tanh(j))) / L ** 2
        num = (f(d) - 2 * f(0.0) + f(-d)) / d ** 2
        exact = (4 * (L - 1) * (L - 2) * (bval ** 6 - bval ** 8) - (L - 1) ** 2 * bval ** 8) / L ** 2
        print(f"L={L} b={bval}: p_L,jj(h,0) FD {num:.6f} formula {exact:.6f}; bulk 4b^6-5b^8 = {4*bval**6-5*bval**8:.6f}")
