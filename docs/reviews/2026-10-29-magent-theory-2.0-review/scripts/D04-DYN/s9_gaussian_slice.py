r"""D04-DYN s9: on the unit-variance Gaussian slice the reciprocal potential is a mean-field global free energy.

Three agents with q_i = N(m_i, 1), trivial model sector, evaluators g_i = N(c_i, w_i), identity channels, fixed rows
beta_ij on directed edges (rows held fixed, so the row penalties are constants).  Normalized global model:
y_i ~ N(c_i, w_i) independently, one record o_ij | y ~ N(y_i - y_j, 1/beta_ij) per directed edge, all registered at 0.
Check symbolically that E(m) - F(Q(m); o=0) is independent of the means, with Q(m) = prod_i N(m_i, 1);
and that off the slice (free variances) the difference depends on the variances.
"""
import sympy as sp

m = sp.symbols("m0:3", real=True)
v = sp.symbols("v0:3", positive=True)
c = sp.symbols("c0:3", real=True)
w = sp.symbols("w0:3", positive=True)
edges = [(0, 1), (1, 2), (2, 0), (0, 2)]
b = {e: sp.Symbol(f"b{e[0]}{e[1]}", positive=True) for e in edges}


def KLg(ma, va, mb, vb):
    return sp.Rational(1, 2) * (va / vb + (ma - mb) ** 2 / vb - 1 + sp.log(vb / va))


def potential(var):
    E = sum(KLg(m[i], var[i], c[i], w[i]) for i in range(3))
    E += sum(b[(i, j)] * KLg(m[i], var[i], m[j], var[j]) for (i, j) in edges)
    return E


def free_energy(var):
    F = sum(KLg(m[i], var[i], c[i], w[i]) for i in range(3))
    for (i, j) in edges:
        # E_Q[-log N(0; y_i - y_j, 1/beta)] = (1/2) log(2 pi / beta) + (beta/2) E[(y_i - y_j)^2]
        F += sp.Rational(1, 2) * sp.log(2 * sp.pi / b[(i, j)]) + b[(i, j)] / 2 * ((m[i] - m[j]) ** 2 + var[i] + var[j])
    return F


diff_slice = sp.simplify(potential([1, 1, 1]) - free_energy([1, 1, 1]))
print("on the slice v_i = 1: E - F =", diff_slice)
print("  depends on the means:", any(diff_slice.has(mi) for mi in m))
diff_off = sp.simplify(potential(list(v)) - free_energy(list(v)))
print("off the slice: d(E - F)/d v_0 =", sp.simplify(sp.diff(diff_off, v[0])))
print("  depends on the means off the slice:", any(sp.diff(diff_off, mi) != 0 for mi in m))
print("  d(E - F)/d m_0 off the slice:", sp.simplify(sp.diff(diff_off, m[0])))
