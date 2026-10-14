r"""L9 check S1: 1D Ising-type chain analog of the native lattice pressure/Fisher statements.

Chain of N signs z_k, raw factor prod(1+b z_k) prod(1+t z_k z_{k+1}), b=tanh h, t=tanh j,
normalized counting reference 2^{-N}.  Checks, in the 1D analog:
 (a) raw/natural pressure conversion  p_N = A_N - log cosh h - (N-1)/N log cosh j   [eq:rg-thermo-native-pressure-conversion]
 (b) free-law Fisher / N = Hess A_N exactly                                           [prop:rg-thermo-native-ctwo-metric]
 (c) infinite-law Fisher rate (stationary Markov chain, row Fisher with past fixed)
     equals Hess A of the transfer-matrix pressure                                    [eq:rg-thermo-native-identity-density, I_1 = H]
 (d) pushed (block-parity) Fisher rate: free-law push and infinite-law push agree      [thm:rg-thermo-native-free-push-ctwo]
 (e) area contraction I_{aB} <= a^d I_B with d=1                                       [prop:rg-spatial-fisher-contraction]
 (f) convexity: Hess A positive definite.
"""
import itertools
import numpy as np
import sympy as sp

# ---------- (a),(b): free chain ----------
def logZ_raw(h, j, N):
    b, t = np.tanh(h), np.tanh(j)
    s = np.array([1.0, -1.0])
    v = 0.5 * (1 + b * s)
    T = 0.5 * (1 + t * np.outer(s, s)) * (1 + b * s)[None, :]
    for _ in range(N - 1):
        v = v @ T
    return np.log(v.sum())

def logZ_nat(h, j, N):
    s = np.array([1.0, -1.0])
    v = 0.5 * np.exp(h * s)
    T = 0.5 * np.exp(j * np.outer(s, s)) * np.exp(h * s)[None, :]
    for _ in range(N - 1):
        v = v @ T
    return np.log(v.sum())

h0, j0, N = 0.37, 0.21, 12
pN = logZ_raw(h0, j0, N) / N
AN = logZ_nat(h0, j0, N) / N
conv = AN - np.log(np.cosh(h0)) - (N - 1) / N * np.log(np.cosh(j0))
print("(a) conversion residual =", abs(pN - conv), " scale |p_N| =", abs(pN))

N2 = 10
configs = np.array(list(itertools.product([1, -1], repeat=N2)), dtype=float)
Th = configs.sum(1)
Tj = (configs[:, :-1] * configs[:, 1:]).sum(1)
w = np.exp(h0 * Th + j0 * Tj); w /= w.sum()
T = np.stack([Th, Tj], 1)
mean = w @ T
I_enum = ((T - mean).T * w) @ (T - mean)
eps = 1e-4
def F(h, j): return logZ_nat(h, j, N2)
H = np.zeros((2, 2))
H[0, 0] = (F(h0 + eps, j0) - 2 * F(h0, j0) + F(h0 - eps, j0)) / eps**2
H[1, 1] = (F(h0, j0 + eps) - 2 * F(h0, j0) + F(h0, j0 - eps)) / eps**2
H[0, 1] = H[1, 0] = (F(h0 + eps, j0 + eps) - F(h0 + eps, j0 - eps) - F(h0 - eps, j0 + eps) + F(h0 - eps, j0 - eps)) / (4 * eps**2)
print("(b) max|I_enum - N Hess A_N| =", np.abs(I_enum - H).max(), " scale", np.abs(I_enum).max(), "(central-difference truncation level)")

# ---------- (c) ----------
hs, js = sp.symbols('h j', real=True)
lam = sp.exp(js) * sp.cosh(hs) + sp.sqrt(sp.exp(2 * js) * sp.sinh(hs)**2 + sp.exp(-2 * js))
A = sp.log(lam / 2)
HessA = sp.hessian(A, (hs, js))
HA = np.array(HessA.subs({hs: h0, js: j0}).evalf(30), dtype=float)

def chain_params(h, j):
    s = np.array([1.0, -1.0], dtype=complex)
    K = 0.5 * np.exp(j * np.outer(s, s) + 0.5 * h * (s[:, None] + s[None, :]))
    a, bb, c = K[0, 0], K[0, 1], K[1, 1]
    lmax = 0.5 * (a + c) + np.sqrt(0.25 * (a - c)**2 + bb * bb)
    v = np.array([bb, lmax - a])
    P = K * v[None, :] / (lmax * v[:, None])
    pi = v * v / (v * v).sum()
    return P, pi

eps_c = 1e-20
ghP = np.imag(np.log(chain_params(h0 + 1j * eps_c, j0)[0])) / eps_c
gjP = np.imag(np.log(chain_params(h0, j0 + 1j * eps_c)[0])) / eps_c
P, pi = chain_params(h0, j0); P = P.real; pi = pi.real
Irate = np.zeros((2, 2))
for s1 in range(2):
    for s2 in range(2):
        g = np.array([ghP[s1, s2], gjP[s1, s2]])
        Irate += pi[s1] * P[s1, s2] * np.outer(g, g)
print("(c) Hess A (transfer matrix, exact) =", HA.ravel())
print("    infinite-law row-Fisher rate     =", Irate.ravel())
print("    max residual =", np.abs(HA - Irate).max(), " scale", np.abs(HA).max())
print("(f) eigenvalues of Hess A:", np.linalg.eigvalsh(HA))

# ---------- (d),(e) ----------
sgn = np.array([1.0, -1.0])

def pushed_logprob_infinite(h, j, B, ys):
    P, pi = chain_params(h, j)
    M = {1: np.zeros((2, 2), dtype=complex), -1: np.zeros((2, 2), dtype=complex)}
    F0 = {1: np.zeros(2, dtype=complex), -1: np.zeros(2, dtype=complex)}
    for xs in itertools.product([0, 1], repeat=B):
        par = int(np.prod(sgn[list(xs)]))
        tail = 1.0 + 0j
        for k in range(1, B):
            tail = tail * P[xs[k - 1], xs[k]]
        for a in range(2):
            M[par][a, xs[-1]] += P[a, xs[0]] * tail
        F0[par][xs[-1]] += pi[xs[0]] * tail
    out = []
    for y in ys:
        v = F0[int(y[0])]
        for yk in y[1:]:
            v = v @ M[int(yk)]
        out.append(np.log(v.sum()))
    return np.array(out)

def pushed_logprob_free(h, j, B, ys):
    b, t = np.tanh(h), np.tanh(j)
    L = len(ys[0])
    def logw(y):
        v = None
        for blk in range(L):
            for k in range(B):
                if blk == 0 and k == 0:
                    v = np.zeros((2, 2), dtype=complex)
                    for a in range(2):
                        v[a, a] = 0.5 * (1 + b * sgn[a])
                    continue
                nv = np.zeros((2, 2), dtype=complex)
                for a in range(2):
                    for par in range(2):
                        for c in range(2):
                            npar = c if k == 0 else (par ^ c)
                            nv[c, npar] += v[a, par] * 0.5 * (1 + t * sgn[a] * sgn[c]) * (1 + b * sgn[c])
                v = nv
            want = 0 if y[blk] > 0 else 1
            keep = np.zeros_like(v)
            keep[:, want] = v[:, want]
            v = keep
        return np.log(v.sum())
    lz = logZ_raw(h, j, B * L)
    return np.array([logw(y) for y in ys]) - lz

def fisher(fun, h, j, B, L):
    ys = np.array(list(itertools.product([1, -1], repeat=L)), dtype=float)
    lp = fun(h, j, B, ys).real
    p = np.exp(lp)
    gh = np.imag(fun(h + 1j * eps_c, j, B, ys)) / eps_c
    gj = np.imag(fun(h, j + 1j * eps_c, B, ys)) / eps_c
    G = np.stack([gh, gj], 1)
    return (G.T * p) @ G, p.sum()

print("(d) pushed Fisher per coarse cell")
rates = {}
for B in (1, 2, 3, 6):
    Lmax = {1: 12, 2: 11, 3: 9, 6: 7}[B]
    I_prev = None
    for L in range(2, Lmax + 1):
        I_inf, tot = fisher(pushed_logprob_infinite, h0, j0, B, L)
        if I_prev is not None:
            inc = I_inf - I_prev
        I_prev = I_inf
    I_free, tot2 = fisher(pushed_logprob_free, h0, j0, B, Lmax)
    I_free_m1, _ = fisher(pushed_logprob_free, h0, j0, B, Lmax - 1)
    rates[B] = inc
    print(f"  B={B}: rate (increment at L={Lmax}) =", inc.ravel(), " sum p =", round(float(tot.real), 12), round(float(tot2.real), 12))
    print(f"        free-law increment          =", (I_free - I_free_m1).ravel(), " |diff| =", np.abs((I_free - I_free_m1) - inc).max())
    print(f"        I_inf/L - I_free/L (O(1/L)) =", ((I_inf - I_free) / Lmax).ravel())
print("    B=1 rate vs Hess A residual:", np.abs(rates[1] - HA).max())
print("(e) eigenvalues of a*I_B - I_aB (must be >= 0):")
for (a, B) in ((2, 1), (3, 1), (6, 1), (3, 2), (2, 3)):
    print(f"   a={a}, B={B}:", np.linalg.eigvalsh(a * rates[B] - rates[a * B]))
