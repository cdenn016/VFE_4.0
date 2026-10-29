r"""GAP08-S1 script 7: strengthening of prop:sup-row-equivalence (supplements/s03_variation.tex:48-64).

Claim checked: with Gaussian sources N(b_j, W) sharing one held covariance W, R = T W, zero record and
zero visible cross covariance, the receiver law p need not be Gaussian.  For any receiver law with mean
a, covariance V and finite differential entropy H(p),
    C_j - KL(p || N(b_j, W)) / T = c_{TW} + d/(2T) + H(p)/T - c_W/T,
which does not depend on j, so the likelihood rows and the divergence rows coincide.  For an affine
receiver family X = a + M eps with a fixed standardized eps, H(p) = H(eps) + (1/2) log det V, so
eq:sup-covariance-difference holds verbatim.

Test receiver: eps uniform on [-sqrt3, sqrt3]^2 (mean 0, covariance I, entropy log 12), X = a + M eps.
C_j and the cross-entropy term of KL_j are computed by 2-D Gauss-Legendre quadrature over eps
(exact for these quadratic integrands); H(p) = log 12 + log|det M| is exact for a uniform law.
"""
import mpmath as mp

mp.mp.dps = 30
d = 2
T = mp.mpf('0.7')
a = mp.matrix([mp.mpf('0.3'), mp.mpf('-0.4')])
M = mp.matrix([[mp.mpf('1.1'), mp.mpf('0.2')], [mp.mpf('-0.3'), mp.mpf('0.8')]])
W = mp.matrix([[mp.mpf('1.5'), mp.mpf('0.4')], [mp.mpf('0.4'), mp.mpf('0.9')]])
bs = [mp.matrix([1, 0]), mp.matrix([mp.mpf('-0.5'), 2]), mp.matrix([mp.mpf('0.25'), mp.mpf('-1.5')])]
pis = [mp.mpf('0.2'), mp.mpf('0.5'), mp.mpf('0.3')]
s3 = mp.sqrt(3)


def c_of(Sig):
    return mp.log(mp.det(2 * mp.pi * Sig)) / 2


def E_eps(f):
    """Expectation over eps uniform on [-sqrt3, sqrt3]^2."""
    return mp.quad(lambda e1, e2: f(e1, e2), [-s3, s3], [-s3, s3], method='gauss-legendre') / 12


def scores(Mm):
    R = T * W
    Ri = R ** -1
    Wi = W ** -1
    V = Mm * Mm.T
    H = mp.log(12) + mp.log(abs(mp.det(Mm)))
    out = []
    for b in bs:
        def cj(e1, e2):
            x = a + Mm * mp.matrix([e1, e2])
            r = x - b
            return c_of(R) + ((r.T * Ri * r)[0, 0] + (Ri * W)[0, 0] + (Ri * W)[1, 1]) / 2
        def ce(e1, e2):
            x = a + Mm * mp.matrix([e1, e2])
            r = x - b
            return c_of(W) + (r.T * Wi * r)[0, 0] / 2
        C = E_eps(cj)
        KL = -H + E_eps(ce)
        out.append((C, KL))
    return out, V, H


sc, V, H = scores(M)
diffs = [C - KL / T for C, KL in sc]
pred = c_of(T * W) + mp.mpf(d) / (2 * T) + H / T - c_of(W) / T
print("C_j - KL_j/T for the three sources:", [mp.nstr(x, 20) for x in diffs])
print("predicted constant c_TW + d/(2T) + H(p)/T - c_W/T:", mp.nstr(pred, 20))
print("max deviation:", mp.nstr(max(abs(x - pred) for x in diffs), 5))
rowC = [p * mp.e ** (-C) for p, (C, KL) in zip(pis, sc)]
rowD = [p * mp.e ** (-KL / T) for p, (C, KL) in zip(pis, sc)]
rowC = [x / sum(rowC) for x in rowC]
rowD = [x / sum(rowD) for x in rowD]
print("rows from C:", [mp.nstr(x, 12) for x in rowC])
print("rows from KL/T:", [mp.nstr(x, 12) for x in rowD])
# Gaussian formula of eq:sup-row-constant would give a different constant for this receiver
gauss_const = c_of(T * W) + mp.mpf(d) / T - mp.log(mp.det(W) / mp.det(V)) / (2 * T)
print("Gaussian-receiver constant of eq:sup-row-constant at the same V:", mp.nstr(gauss_const, 20),
      "(differs by (H(p) - H(N(a,V)))/T =", mp.nstr((H - mp.log(mp.det(2 * mp.pi * mp.e * V)) / 2) / T, 10), ")")
# covariance derivative along the affine family M -> M + h dM
dM = mp.matrix([[mp.mpf('0.3'), mp.mpf('-0.1')], [mp.mpf('0.2'), mp.mpf('0.5')]])
h = mp.mpf('1e-10')
sp_, _, _ = scores(M + h * dM)
sm_, _, _ = scores(M - h * dM)
dV = dM * M.T + M * dM.T
target = ((V ** -1) * dV)
target = (target[0, 0] + target[1, 1]) / (2 * T)
for j in range(3):
    fd = ((sp_[j][0] - sp_[j][1] / T) - (sm_[j][0] - sm_[j][1] / T)) / (2 * h)
    print(f"source {j}: d(C_j - KL_j/T) along dM = {mp.nstr(fd, 15)}; Tr(V^-1 dV)/(2T) = {mp.nstr(target, 15)}")
