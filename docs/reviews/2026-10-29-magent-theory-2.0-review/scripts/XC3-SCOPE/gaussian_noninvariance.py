"""XC3-SCOPE check of prop:dyn-gaussian-noninvariance and of the sentence after
def:dyn-restricted-flow ("the restricted flow is the Fisher-orthogonal projection of the
full velocity").  One-dimensional source q_i=N(0,1) with Gaussian evaluator g=N(0,1),
one receiver q_k=N(m,s2) attending to i through the identity channel with beta_ki=1.
First variation in q_i (centering removes constants):
   phi = log q_i - log g - q_k/q_i ;  full Fisher velocity v = -(phi - E_{q_i} phi).
(a) Fraction of the Fisher norm of v outside the Gaussian tangent space span{1,x,x^2}.
(b) The Fisher norm E_{q_i}[(q_k/q_i)^2] is finite iff s2 < 2 (closed form), so for wider
    receivers the full velocity is not a Fisher tangent vector and the restricted flow is a
    projection only in the weak sense (pairing with the family tangent vectors).
Single-threaded, < 2 s."""
import numpy as np
from numpy.polynomial.hermite_e import hermegauss

def frac_outside(m, s2, deg=200):
    x, w = hermegauss(deg); w = w/np.sqrt(2*np.pi)          # E under N(0,1)
    qi = np.exp(-x**2/2)/np.sqrt(2*np.pi); qk = np.exp(-(x-m)**2/(2*s2))/np.sqrt(2*np.pi*s2)
    phi = -(qk/qi)                                         # log qi - log g = 0 here
    v = -(phi - np.sum(w*phi))
    B = np.vstack([np.ones_like(x), x, x**2]).T
    G = B.T @ (w[:, None]*B); c = B.T @ (w*v); coef = np.linalg.solve(G, c)
    res = v - B @ coef
    return float(np.sum(w*res**2)/np.sum(w*v**2)), coef

def fisher_norm_source(m, s2):
    a = 1/s2 - 0.5
    if a <= 0: return np.inf
    b = 2*m/s2; c = m**2/s2
    return float((2*np.pi)**0.5/(2*np.pi*s2)*np.sqrt(np.pi/a)*np.exp(b**2/(4*a) - c))

if __name__ == "__main__":
    for (m, s2) in [(0.5, 0.5), (1.0, 0.8), (1.0, 1.2), (0.0, 1.5)]:
        f, coef = frac_outside(m, s2)
        print("m=%.2f s2=%.2f: Fisher-norm fraction outside Gaussian tangent = %.4f ; E[(qk/qi)^2] = %.4f" % (m, s2, f, fisher_norm_source(m, s2)))
    for s2 in (1.9, 1.99, 1.999, 2.0, 3.0):
        print("s2=%.3f: E_{q_i}[(q_k/q_i)^2] = %s" % (s2, fisher_norm_source(0.0, s2)))
    # weak projection: pairings E_{q_i}[v x^n] = -E_{q_k}[x^n] + const, finite for every s2
    print("pairings with tangent vectors stay finite: E_qk[x^2] = m^2+s2, e.g. s2=3 ->", 0.0**2+3.0)
