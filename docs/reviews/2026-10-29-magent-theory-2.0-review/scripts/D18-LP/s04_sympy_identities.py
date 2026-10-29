r"""D18-LP check 4: exact symbolic identities.

(a) lp:sum-example / lp:sum-failure: with B1=B3=b e1, B2=b e2, A=C=0, l1=l3=0, l2=v, build the
    rows from the general row formula (lp:weights) and the B field (lp:field) symbolically,
    then Omega = det(S, V_S)/|S|^2 and dOmega/du (u = sech(v/2)).
(b) lp:causal-law-jets: second contextual derivative of R_theta(c) Bbar(c) in two context
    coordinates, generic functions.
(c) lp:retention identity U(R_a S) R_a = U(S).
(d) lp:initial-phase-score-defect sign: psi_parent = psi_fine - alpha0(c).
"""
import sympy as sp

b, v, s, k, E = sp.symbols("b v s k E", positive=True)
I1I0 = sp.Symbol("rho", positive=True)  # I1(b)/I0(b)
u = sp.sech(v / 2)
L = [0, v, 0]
m = [sp.tanh(Li / 2) for Li in L]
Bv = [sp.Matrix([b, 0]), sp.Matrix([0, b]), sp.Matrix([b, 0])]
Mi = [I1I0 * Bi / b for Bi in Bv]
# s = 1/(5 I0(J)) (t_i = 0 because A=C=0), k = 2/(5 I0(b)), E = exp(b I1/I0)
w = sp.zeros(3, 3)
for i in range(3):
    for j in range(3):
        if i == j:
            w[i, j] = s
        else:
            expo = L[j] * m[i] / 2 + (Bv[j].T * Mi[i])[0]
            w[i, j] = k * sp.exp(expo) / sp.cosh(L[j] / 2)
w = w.subs(sp.exp(b * I1I0), E)
beta = sp.Matrix(3, 3, lambda i, j: w[i, j] / sum(w[i, r] for r in range(3)))
VB = [-Bv[i] + sum((beta[i, j] * Bv[j] for j in range(3) if j != i), sp.zeros(2, 1)) for i in range(3)]
S = Bv[0] + Bv[1] + Bv[2]
VS = VB[0] + VB[1] + VB[2]
Omega = sp.simplify((S[0] * VS[1] - S[1] * VS[0]) / (S[0] ** 2 + S[1] ** 2))
uu = sp.Symbol("u", positive=True)
Om_u = sp.simplify(Omega.rewrite(sp.exp).subs(sp.exp(v / 2), sp.Symbol("q")))
D = s + k * E + k * uu
claimed = sp.Rational(1, 5) * ((4 * k * uu - 2 * k * E) / D - 2 * k / (s + 2 * k))
# substitute u = sech(v/2) numerically-exactly: compare Omega(v) with claimed(u(v))
diff = sp.simplify(Omega - claimed.subs(uu, u))
print("(a) Omega - claimed (after u = sech(v/2)):", sp.simplify(diff.rewrite(sp.exp)))
dOm = sp.simplify(sp.diff(claimed, uu) - (4 * k * s + 6 * k ** 2 * E) / (5 * D ** 2))
print("(a) dOmega/du - (4ks+6k^2E)/(5D^2):", dOm)
print("    S =", list(S))

# (b) causal law jets
c1, c2 = sp.symbols("c1 c2")
th = sp.Function("theta")(c1, c2)
Bb = sp.Matrix([sp.Function("bx")(c1, c2), sp.Function("by")(c1, c2)])
R = lambda a: sp.Matrix([[sp.cos(a), -sp.sin(a)], [sp.sin(a), sp.cos(a)]])
J2 = sp.Matrix([[0, -1], [1, 0]])
Bt = R(th) * Bb
lhs = sp.diff(Bt, c1, c2)
ta, tb, tab = sp.diff(th, c1), sp.diff(th, c2), sp.diff(th, c1, c2)
rhs = R(th) * (sp.diff(Bb, c1, c2) + ta * J2 * sp.diff(Bb, c2) + tb * J2 * sp.diff(Bb, c1)
               + tab * J2 * Bb - ta * tb * Bb)
print("(b) lp:causal-law-jets second-order residual:", sp.simplify(lhs - rhs).T)
lhs1 = sp.diff(Bt, c1)
rhs1 = R(th) * (sp.diff(Bb, c1) + ta * J2 * Bb)
print("(b) first-order residual:", sp.simplify(lhs1 - rhs1).T)

# (c) U(R_a S) R_a = U(S)
Sx, Sy, a = sp.symbols("S_x S_y a", real=True)
Uf = lambda SS: sp.Matrix([[SS[0], SS[1]], [-SS[1], SS[0]]]) / sp.sqrt(SS[0] ** 2 + SS[1] ** 2)
Svec = sp.Matrix([Sx, Sy])
print("(c) U(R_a S) R_a - U(S):", sp.simplify(Uf(R(a) * Svec) * R(a) - Uf(Svec)))

# (d) score defect: parent density p~(psi) = p(psi + alpha0(c)); score at corresponding samples
cc, ps_ = sp.symbols("c psi", real=True)
al = sp.Function("alpha0")(cc)
Bx_, By_ = sp.Function("Bx")(cc), sp.Function("By")(cc)
logq = lambda P: Bx_ * sp.cos(P) + By_ * sp.sin(P)  # c-dependent normalizer is rotation invariant
fine_score = sp.diff(logq(ps_), cc)
parent_logq = logq(ps_ + al)  # parent sample psi_par = psi_fine - alpha0
parent_score_at_corr = sp.diff(parent_logq, cc).subs(ps_, ps_ - al)  # evaluated at psi_par = psi - alpha0
lphys = -sp.diff(logq(ps_), ps_)
print("(d) parent score - (fine score - alpha0' * l_phys):",
      sp.simplify(parent_score_at_corr - (fine_score - sp.diff(al, cc) * lphys)))
