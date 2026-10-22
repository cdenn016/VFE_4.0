r"""L05-FUS: is the fixed-variance lognormal family invariant if variances are allowed to move?

Free log variances v_q, v_s for an agent of multiplicity n (recognition N(mu, diag(v_q, v_s)) in log
coordinates).  Local KL against N(0,(nA)^{-1}); peer term sqrt(2 KL(N(mu_i, s_i) || N(m_ij, s_j)))/(1-chi)
with restored message variances s = v + 1 - 1/n and transported source mean m_ij = mu_j + chi (mu_i - mu_j).
Report the variance derivatives at v = 1/n (where the chapter fixes the variances).
"""
import sympy as sp

a, c, n, chi = sp.symbols('a c n chi', positive=True)
vq, vs, mq, ms = sp.symbols('v_q v_s mu_q mu_s', real=True)
A = sp.Matrix([[a, -c], [-c, a]])
P = n * A
S0 = sp.diag(vq, vs)
m = sp.Matrix([mq, ms])
KL = sp.Rational(1, 2) * ((P * S0).trace() - 2 + (m.T * P * m)[0] - sp.log(P.det()) - sp.log(S0.det()))
dKL = sp.simplify(sp.diff(KL, vq).subs(vq, 1 / n))
print("d(local KL)/d v_q at v_q = 1/n:", dKL, "  (zero iff a = 1)")

mi, mj, si, sj = sp.symbols('m_i m_j s_i s_j', positive=True)
mt = mj + chi * (mi - mj)
kl = sp.Rational(1, 2) * (si / sj - 1 - sp.log(si / sj) + (mi - mt)**2 / sj)
d = sp.sqrt(2 * kl) / (1 - chi)
d_si = sp.simplify(sp.diff(d, si).subs({si: 1, sj: 1}))
d_sj = sp.simplify(sp.diff(d, sj).subs({si: 1, sj: 1}))
print("d(peer)/d s_receiver at s=1:", d_si)
print("d(peer)/d s_source   at s=1:", d_sj, " (nonzero whenever m_i != m_j)")
num = d_sj.subs({mi: 1.0, mj: 0.0, chi: 0.3})
print("   numeric example m_i=1, m_j=0, chi=0.3:", float(num))
