"""R-INH checks for the chapter 5 additions (single-threaded, small)."""
from fractions import Fraction as Fr
import itertools
import numpy as np
import sympy as sp

# (A) Rogers-Pitman intertwining example
T = [[Fr(3, 10), Fr(1, 2), Fr(1, 5)], [Fr(2, 5), Fr(1, 5), Fr(2, 5)], [Fr(1, 4), Fr(1, 4), Fr(1, 2)]]
R = [0, 0, 1]
Lam = [[Fr(1, 2), Fr(1, 2), Fr(0)], [Fr(0), Fr(0), Fr(1)]]
def mm(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]
Phi = [[Fr(1) if R[x] == z else Fr(0) for z in range(2)] for x in range(3)]
Tc = mm(mm(Lam, T), Phi)
print("(A) Tc =", Tc, " LamT == TcLam:", mm(Lam, T) == mm(Tc, Lam))
print("    strong lumpability fails:", mm(T, Phi)[0] != mm(T, Phi)[1])
def path_law(p0, n):
    out = {}
    for xs in itertools.product(range(3), repeat=n):
        pr = p0[xs[0]]
        for a, b in zip(xs, xs[1:]):
            pr *= T[a][b]
        key = tuple(R[x] for x in xs)
        out[key] = out.get(key, 0) + pr
    return out
def parent_law(nu, n):
    out = {}
    for zs in itertools.product(range(2), repeat=n):
        pr = nu[zs[0]]
        for a, b in zip(zs, zs[1:]):
            pr *= Tc[a][b]
        out[zs] = pr
    return out
nu = [Fr(1, 3), Fr(2, 3)]
p0 = [sum(nu[z] * Lam[z][x] for z in range(2)) for x in range(3)]
print("    path laws agree for nu Lam (n=5):", path_law(p0, 5) == parent_law(nu, 5))
d = path_law([Fr(1), 0, 0], 5); e = parent_law([Fr(1), 0], 5)
print("    TV-type L1/2 difference from delta_1:", sum(abs(d[k] - e[k]) for k in e) / 2)

# (B) coarsest strongly lumpable refinement
rng = np.random.default_rng(0)
P = rng.random((6, 6)); P[1] = P[0]; P[3] = P[2]; P /= P.sum(1, keepdims=True)
P[1] = P[0]; P[3] = P[2]
Q = [0, 0, 0, 0, 1, 1]
def refine(blocks):
    lab = {}
    for x in range(6):
        sig = (blocks[x],) + tuple(round(P[x, [y for y in range(6) if blocks[y] == b]].sum(), 12) for b in sorted(set(blocks)))
        lab.setdefault(sig, len(lab));
    return [lab[(blocks[x],) + tuple(round(P[x, [y for y in range(6) if blocks[y] == b]].sum(), 12) for b in sorted(set(blocks)))] for x in range(6)]
b = Q
while True:
    nb = refine(b)
    if len(set(nb)) == len(set(b)):
        break
    b = nb
print("(B) refinement fixed point:", b)

# (C) joint response example
for (u, v) in [(0, 0), (0, 1)]:
    law = {}
    for xi in (0, 1):
        k = (xi ^ v, u ^ xi); law[k] = law.get(k, 0) + 0.5
    print("(C) x=", (u, v), "law(O,R(X'))=", law)

# (D) sharp gradient criterion: center of mass
x1, x2, s = sp.symbols('x1 x2 s'); W = sp.Function('W')
E = x1**2 / 2 + x2**2 / 2 + W(x1 - x2)
grad = sp.Matrix([sp.diff(E, x1), sp.diff(E, x2)])
Rz = (x1 + x2) / sp.sqrt(2)
pushed = (sp.Matrix([[1, 1]]) / sp.sqrt(2) * (-grad))[0]
print("(D) DR(-grad E) + R =", sp.simplify(pushed + Rz))

# (E) mean-field identity on finite tables: 3 agents, alphabet 3, constant-output moment channel
K = 3; N = 3; Tstat = np.array([0.0, 1.0, 2.0]); G0 = np.array([0.2, 0.3, 0.5]); Tq = 0.7
pi = np.full((N, N), 0.5); np.fill_diagonal(pi, 0.0)
def C(M):
    w = np.exp(np.array([0.3, -0.2, 0.1]) * M / N + np.array([0.0, 0.5, -0.3])); return w / w.sum()
def Ered(q):
    M = sum(q[i] @ Tstat for i in range(N)); c = C(M)
    e = sum(np.sum(q[i] * np.log(q[i] / G0)) for i in range(N))
    for i in range(N):
        D = np.sum(q[i] * np.log(q[i] / c))
        Z = sum(pi[i, j] * np.exp(-D / Tq) for j in range(N) if j != i)
        e -= Tq * np.log(Z)
    return e
q = rng.random((N, K)); q /= q.sum(1, keepdims=True)
h = np.zeros_like(q); eps = 1e-6
for i in range(N):
    for k in range(K):
        qp = q.copy(); qp[i, k] += eps; qm = q.copy(); qm[i, k] -= eps
        h[i, k] = (Ered(qp) - Ered(qm)) / (2 * eps)
dlog = -(h - (q * h).sum(1, keepdims=True))
for (i, j) in [(0, 1), (1, 2)]:
    r = dlog[i] - dlog[j] + 2 * np.log(q[i] / q[j])
    print("(E) pair", (i, j), "residual spread over k:", np.ptp(r))
# null flow rate
