r"""L9 check S2: the native free model of def:rg-thermo-native-families / eq:rg-pressure-native-factor
is exactly a 2D Ising lattice gas in a field on the (L-1)x(L-1) plaquette grid.

Z_L(b,t) = 4^{-N} sum_z prod_e (1+b z_e) prod_p (1+t P_p)
         = sum_{S subset plaquettes} t^{|S|} b^{|boundary_2(S)|},
where boundary_2(S) is the set of edges lying in an odd number of plaquettes of S.
(high-temperature expansion: expand both products; a monomial survives the counting average iff
its edge set is the mod-2 boundary of its plaquette set.)  In spins sigma_p in {0,1} the weight is
t^{sum sigma} b^{#disagreeing nearest-neighbour plaquette pairs + sum_p n_bd(p) sigma_p}, a
ferromagnetic (b<1) Ising lattice gas with field log t and a boundary field.
Checks: brute force L=2,3 against the dual sum; dual row-transfer pressure for L up to 13 and the
Hessian identity N^{-1} I = Hess A_L at L=6 via the dual transfer matrix vs exhaustive enumeration at L=3.
"""
import itertools
import numpy as np

def edges_and_plaquettes(L):
    E = [(x, y, a) for x in range(L) for y in range(L) for a in (1, 2)]
    idx = {e: k for k, e in enumerate(E)}
    P = []
    for x in range(L - 1):
        for y in range(L - 1):
            P.append([idx[(x, y, 1)], idx[(x + 1, y, 2)], idx[(x, y + 1, 1)], idx[(x, y, 2)]])
    return E, P

def Z_brute(L, b, t):
    E, P = edges_and_plaquettes(L)
    nE = len(E)
    tot = 0.0
    for z in itertools.product([1, -1], repeat=nE):
        z = np.array(z)
        w = np.prod(1 + b * z)
        for p in P:
            w *= 1 + t * np.prod(z[p])
        tot += w
    return tot / 2**nE

def Z_dual(L, b, t):
    E, P = edges_and_plaquettes(L)
    tot = 0.0
    for S in itertools.product([0, 1], repeat=len(P)):
        cnt = np.zeros(len(E), dtype=int)
        for s, p in zip(S, P):
            if s:
                cnt[p] += 1
        tot += t**sum(S) * b**int((cnt % 2).sum())
    return tot

rng = np.random.default_rng(0)
for L in (2, 3):
    for _ in range(2):
        b, t = rng.uniform(0.05, 0.95), rng.uniform(0.01, 0.9)
        zb, zd = Z_brute(L, b, t), Z_dual(L, b, t)
        print(f"L={L} b={b:.3f} t={t:.3f}: brute={zb:.15f} dual={zd:.15f} rel.diff={abs(zb-zd)/zb:.2e}")

# dual row transfer matrix on the (L-1)x(L-1) plaquette grid
def logZ_transfer(L, b, t):
    n = L - 1
    if n == 0:
        return 0.0
    states = np.array(list(itertools.product([0, 1], repeat=n)))
    # weight of one row: field t, horizontal disagreements b, left/right boundary edges b
    def row_w(s):
        w = t**s.sum() * b**int(np.sum(s[1:] != s[:-1]))
        w *= b**int(s[0]) * b**int(s[-1])
        return w
    rw = np.array([row_w(s) for s in states])
    # vertical disagreements between consecutive rows
    D = (states[:, None, :] != states[None, :, :]).sum(-1)
    T = b**D
    bottom = b**states.sum(1)
    top = b**states.sum(1)
    v = bottom * rw
    logscale = 0.0
    for _ in range(n - 1):
        v = (v @ T) * rw
        m = v.max(); v /= m; logscale += np.log(m)
    return np.log((v * top).sum()) + logscale

for L in (2, 3):
    b, t = 0.4, 0.07
    print(f"transfer vs dual sum L={L}:", logZ_transfer(L, b, t), np.log(Z_dual(L, b, t)))

def A_L(L, h, j):
    b, t = np.tanh(h), np.tanh(j)
    N = L * L
    p = logZ_transfer(L, b, t) / N
    return p + 2 * np.log(np.cosh(h)) + (L - 1)**2 / N * np.log(np.cosh(j))

h0, j0 = 0.6, 0.06   # t = tanh j ~ 0.06 < 1/12
print("pressure sequence p_L (raw) at h=0.6, j=0.06:")
for L in (4, 6, 8, 10, 12, 13):
    print("  L =", L, " p_L =", logZ_transfer(L, np.tanh(h0), np.tanh(j0)) / L**2)

# Hessian identity: N^{-1} Fisher = Hess A_L, at L=3 by exhaustive enumeration in natural coordinates
L = 3
E, P = edges_and_plaquettes(L)
confs = np.array(list(itertools.product([1, -1], repeat=len(E))), dtype=float)
Th = confs.sum(1)
Tj = np.stack([np.prod(confs[:, p], 1) for p in P], 1).sum(1)
w = np.exp(h0 * Th + j0 * Tj); w /= w.sum()
T = np.stack([Th, Tj], 1); m = w @ T
I = ((T - m).T * w) @ (T - m) / L**2
eps = 1e-4
f = lambda h, j: A_L(L, h, j)
H = np.array([[(f(h0+eps,j0)-2*f(h0,j0)+f(h0-eps,j0))/eps**2,
               (f(h0+eps,j0+eps)-f(h0+eps,j0-eps)-f(h0-eps,j0+eps)+f(h0-eps,j0-eps))/(4*eps**2)],
              [0, (f(h0,j0+eps)-2*f(h0,j0)+f(h0,j0-eps))/eps**2]]); H[1,0]=H[0,1]
print("L=3: N^-1 I (enumeration) =", I.ravel(), "\n     Hess A_L (dual transfer, FD) =", H.ravel(), " max diff", np.abs(I-H).max())
