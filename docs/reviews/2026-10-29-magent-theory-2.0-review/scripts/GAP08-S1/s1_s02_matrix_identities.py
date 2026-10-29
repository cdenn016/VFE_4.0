r"""GAP08-S1 script 1: supplements/s02_geometry.tex, lem:sgeom-lowrank, prop:sgeom-rankone,
lem:sgeom-frechet, the stacked-interface remark and eq:sgeom-moment-derivative.

Independent routes (not the series argument of the manuscript):
  * phi_1(Z) is computed from the augmented-matrix identity
        exp([[Z, I],[0, 0]]) = [[e^Z, phi_1(Z)],[0, I]]
    and, separately, by direct series summation; both at 50 digits (mpmath).
  * D exp_L[H] is computed from the block-triangular identity
        exp([[L, H],[0, L]]) = [[e^L, D exp_L[H]],[0, e^L]]
    and compared with the integral of eq:sgeom-frechet (mpmath quadrature, 2x2)
    and with a central difference at 50 digits.
  * the rank-one determinant uses the matrix determinant lemma symbolically.
Numerical agreement supports an identity at the tested instances; the proofs are in the report.
"""
import mpmath as mp
import sympy as sp
import random

mp.mp.dps = 50
random.seed(20261029)


def rmat(m, n, scale=1):
    return mp.matrix([[mp.mpf(random.randint(-9, 9)) / 7 * scale for _ in range(n)] for _ in range(m)])


def eye(n):
    return mp.eye(n)


def maxabs(A):
    return max(abs(A[i, j]) for i in range(A.rows) for j in range(A.cols))


def phi1_aug(Z):
    r = Z.rows
    M = mp.zeros(2 * r, 2 * r)
    for i in range(r):
        for j in range(r):
            M[i, j] = Z[i, j]
        M[i, r + i] = 1
    E = mp.expm(M)
    P = mp.zeros(r, r)
    for i in range(r):
        for j in range(r):
            P[i, j] = E[i, r + j]
    return P


def phi1_series(Z, nterms=200):
    r = Z.rows
    S = mp.zeros(r, r)
    term = eye(r)  # Z^{n-1}/n! for n = 1
    fact = mp.mpf(1)
    S = S + term
    P = eye(r)
    for n in range(2, nterms):
        P = P * Z
        fact = fact * n
        S = S + P / fact
    return S


print("== lem:sgeom-lowrank: e^{OT} = I + O phi_1(TO) T ==")
cases = [(4, 1), (5, 2), (3, 5), (6, 6), (7, 2)]
for d, r in cases:
    O = rmat(d, r)
    T = rmat(r, d)
    lhs = mp.expm(O * T)
    P_aug = phi1_aug(T * O)
    P_ser = phi1_series(T * O)
    rhs = eye(d) + O * P_aug * T
    print(f"(d,r)=({d},{r}): |e^OT - (I+O phi1 T)| = {mp.nstr(maxabs(lhs - rhs), 5)}; "
          f"|phi1_aug - phi1_series| = {mp.nstr(maxabs(P_aug - P_ser), 5)}")

# singular TO: choose O with columns in ker T direction so TO is singular but nonzero
d, r = 4, 3
T = rmat(r, d)
O = rmat(d, r)
# force T*O to have rank 2: replace third column of O by a vector in ker T
ker = mp.matrix(d, 1)
# kernel of T (3x4) via solving; use numpy-like approach with mpmath: pick null vector by SVD
U_, S_, V_ = mp.svd_r(T, full_matrices=True)
nullv = V_[V_.rows - 1, :].T  # last right-singular vector
for i in range(d):
    O[i, 2] = nullv[i]
TO = T * O
print("singular TO: det(TO) =", mp.nstr(mp.det(TO), 5),
      "; |e^OT - (I+O phi1 T)| =", mp.nstr(maxabs(mp.expm(O * T) - (eye(d) + O * phi1_aug(TO) * T)), 5))

# nilpotent TO exactly with sympy: O = [e1, e2], T rows chosen so TO = [[0,1],[0,0]]
Os = sp.Matrix([[1, 0], [0, 1], [0, 0]])
Ts = sp.Matrix([[0, 1, 5], [0, 0, 3]])
TOs = Ts * Os
assert TOs == sp.Matrix([[0, 1], [0, 0]])
phi_nil = sp.eye(2) + TOs / 2  # phi_1 series terminates: I + Z/2!
lhs = (Os * Ts).exp()
rhs = sp.eye(3) + Os * phi_nil * Ts
print("nilpotent TO (sympy exact): e^OT - (I + O(I+Z/2)T) =", sp.simplify(lhs - rhs).tolist())

# kernel and image statements of prop:int-rank (same data as lem:sgeom-lowrank)
d, r = 5, 2
O = rmat(d, r)
T = rmat(r, d)
E = mp.expm(O * T)
U_, S_, V_ = mp.svd_r(T, full_matrices=True)
worst_ker = mp.mpf(0)
for k in range(r, d):
    v = V_[k, :].T
    worst_ker = max(worst_ker, maxabs(E * v - v))
# image test: residual of projecting columns of (E - I) onto im O
Q, Rr = mp.qr(O)
Qr = mp.matrix(d, r)
for i in range(d):
    for j in range(r):
        Qr[i, j] = Q[i, j]
D = E - eye(d)
proj_res = maxabs(D - Qr * (Qr.T * D))
print(f"prop:int-rank (d,r)=({d},{r}): max |e^OT v - v| over ker T basis = {mp.nstr(worst_ker, 5)}; "
      f"|(e^OT - I) - P_imO (e^OT - I)| = {mp.nstr(proj_res, 5)}; det e^OT = {mp.nstr(mp.det(E), 8)} "
      f"vs e^tr(OT) = {mp.nstr(mp.e ** sum((O*T)[i, i] for i in range(d)), 8)}")

print("\n== stacked interfaces (s02:26) ==")
d = 4
O1, T1 = rmat(d, 1), rmat(1, d)
O2, T2 = rmat(d, 2), rmat(2, d)
O = mp.matrix(d, 3)
T = mp.matrix(3, d)
for i in range(d):
    O[i, 0] = O1[i, 0]
    O[i, 1] = O2[i, 0]
    O[i, 2] = O2[i, 1]
    T[0, i] = T1[0, i]
    T[1, i] = T2[0, i]
    T[2, i] = T2[1, i]
full = mp.expm(O1 * T1 + O2 * T2)
stack = eye(d) + O * phi1_aug(T * O) * T
diag_only = eye(d) + O1 * phi1_aug(T1 * O1) * T1 + O2 * phi1_aug(T2 * O2) * T2
print("full stacked formula residual:", mp.nstr(maxabs(full - stack), 5))
print("diagonal-block replacement error (generic):", mp.nstr(maxabs(full - diag_only), 5))
# make both cross products vanish: T2 O1 = 0 and T1 O2 = 0
# choose O1 in ker T2 and O2 columns in ker T1
U_, S_, V_ = mp.svd_r(T2, full_matrices=True)
o1 = V_[V_.rows - 1, :].T  # in ker T2 (T2 is 2x4, kernel dim 2)
U1, S1, V1 = mp.svd_r(T1, full_matrices=True)
O1b = o1
O2b = mp.matrix(d, 2)
for j in range(2):
    vj = V1[V1.rows - 1 - j, :].T  # in ker T1
    for i in range(d):
        O2b[i, j] = vj[i]
print("cross products |T2 O1|, |T1 O2| =", mp.nstr(maxabs(T2 * O1b), 3), mp.nstr(maxabs(T1 * O2b), 3))
full_b = mp.expm(O1b * T1 + O2b * T2)
diag_b = eye(d) + O1b * phi1_aug(T1 * O1b) * T1 + O2b * phi1_aug(T2 * O2b) * T2
print("diagonal-block formula with both cross products zero:", mp.nstr(maxabs(full_b - diag_b), 5))
# one-sided vanishing: T2 O1 = 0 but T1 O2 != 0
diag_c = eye(d) + O1b * phi1_aug(T1 * O1b) * T1 + O2 * phi1_aug(T2 * O2) * T2
full_c = mp.expm(O1b * T1 + O2 * T2)
print("one-sided vanishing (T2 O1 = 0, T1 O2 != 0): |T1 O2| =", mp.nstr(maxabs(T1 * O2), 4),
      "; diagonal-block error =", mp.nstr(maxabs(full_c - diag_c), 5))

print("\n== prop:sgeom-rankone (sympy, symbolic u, v in R^3) ==")
u = sp.Matrix(sp.symbols('u1:4'))
v = sp.Matrix(sp.symbols('v1:4'))
t = (v.T * u)[0, 0]
L = u * v.T
for n in range(2, 7):
    assert sp.expand(L ** n - t ** (n - 1) * L) == sp.zeros(3, 3)
print("L^n = t^(n-1) L for n = 2..6: exact")
c = sp.Symbol('c')
detlemma = sp.expand((sp.eye(3) + c * L).det() - (1 + c * t))
print("det(I + c u v^T) - (1 + c t) =", detlemma)
phi1t = (sp.exp(t) - 1) / t
print("so det e^L = 1 + phi_1(t) t =", sp.simplify(1 + phi1t * t), "(= e^t)")
for tv in [mp.mpf('-0.874'), mp.mpf(0), mp.mpf('1.188'), mp.mpf('3.5')]:
    # numeric instance in d = 3 with v^T u = tv
    uu = mp.matrix([1, 2, -1])
    vv = mp.matrix([tv, 0, 0])
    Ln = uu * vv.T
    ph = (mp.e ** tv - 1) / tv if tv != 0 else mp.mpf(1)
    res = maxabs(mp.expm(Ln) - (eye(3) + ph * Ln))
    print(f"t = {mp.nstr(tv, 4)}: |e^L - (I + phi1(t) L)| = {mp.nstr(res, 5)}, "
          f"det e^L - e^t = {mp.nstr(mp.det(mp.expm(Ln)) - mp.e ** tv, 5)}")

print("\n== lem:sgeom-frechet ==")


def frechet_block(Lm, Hm):
    n = Lm.rows
    M = mp.zeros(2 * n, 2 * n)
    for i in range(n):
        for j in range(n):
            M[i, j] = Lm[i, j]
            M[n + i, n + j] = Lm[i, j]
            M[i, n + j] = Hm[i, j]
    E = mp.expm(M)
    P = mp.zeros(n, n)
    for i in range(n):
        for j in range(n):
            P[i, j] = E[i, n + j]
    return P


for n in [2, 3, 4]:
    Lm = rmat(n, n)
    Hm = rmat(n, n)
    Db = frechet_block(Lm, Hm)
    h = mp.mpf('1e-18')
    Dc = (mp.expm(Lm + h * Hm) - mp.expm(Lm - h * Hm)) / (2 * h)
    print(f"n={n}: |block - central difference(h=1e-18)| = {mp.nstr(maxabs(Db - Dc), 5)}; "
          f"|[L,H]| = {mp.nstr(maxabs(Lm*Hm - Hm*Lm), 4)}; |D - e^L H| = {mp.nstr(maxabs(Db - mp.expm(Lm)*Hm), 4)}")
    if n == 2:
        mp.mp.dps = 30
        Dint = mp.zeros(2, 2)
        for i in range(2):
            for j in range(2):
                f = lambda s, i=i, j=j: (mp.expm((1 - s) * Lm) * Hm * mp.expm(s * Lm))[i, j]
                Dint[i, j] = mp.quad(f, [0, 1])
        print(f"n=2: |integral (eq:sgeom-frechet) - block| = {mp.nstr(maxabs(Dint - Db), 5)}")
        mp.mp.dps = 50

# commuting case
Lm = rmat(3, 3)
Hm = 2 * Lm * Lm - Lm + eye(3) / 3
print("commuting H = p(L): |D exp_L[H] - e^L H| =", mp.nstr(maxabs(frechet_block(Lm, Hm) - mp.expm(Lm) * Hm), 5))

# Daleckii-Krein form for diagonal L, symbolic
l1, l2, s = sp.symbols('l1 l2 s', real=True)
h12 = sp.Symbol('h12')
integrand = sp.exp((1 - s) * l1) * h12 * sp.exp(s * l2)
I12 = sp.integrate(integrand, (s, 0, 1), conds='none')
print("diagonal L, (1,2) entry of the integral minus h12 (e^l1 - e^l2)/(l1 - l2):",
      sp.simplify(I12 - h12 * (sp.exp(l1) - sp.exp(l2)) / (l1 - l2)))

print("\n== eq:sgeom-moment-derivative (sympy, d = 2) ==")
eps, lam = sp.symbols('epsilon lambda')
mi = sp.Matrix(sp.symbols('mi1:3'))
mj = sp.Matrix(sp.symbols('mj1:3'))
dmi = sp.Matrix(sp.symbols('dmi1:3'))
dmj = sp.Matrix(sp.symbols('dmj1:3'))
s11, s12, s22, d11, d12, d22 = sp.symbols('s11 s12 s22 d11 d12 d22')
Sig = sp.Matrix([[s11, s12], [s12, s22]])
dSig = sp.Matrix([[d11, d12], [d12, d22]])
Leps = lam * (mi + eps * dmi) * (mj + eps * dmj).T * (Sig + eps * dSig).inv()
dL = sp.diff(Leps, eps).subs(eps, 0)
Si = Sig.inv()
formula = lam * (dmi * mj.T * Si + mi * dmj.T * Si - mi * mj.T * Si * dSig * Si)
print("dL - formula =", sp.simplify(dL - formula).tolist())
