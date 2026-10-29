"""B-LINKS small verification checks (single-threaded, a few seconds).

(a) straight-line quartic of the chapter 18 critical mode, exact rationals;
(b) near-cancellation of the two dominant quartic terms;
(c) two-agent plug-in per-agent instability example of D24-LDYN-01;
(d) law-level covariance fixed point: monotone iteration from 0 and from the upper bound agree (random instance).
"""
from fractions import Fraction as Fr
import numpy as np

A4 = Fr(236914859731152, 655026269697361)
D3 = Fr(-31633266640, 129477420379)
K3 = Fr(21485600170644766720, 2074468196131542287)
Tc = Fr(3269, 10118)
c = Fr(-657534393590529472, 6223404588394626861)
assert A4 + 4*D3 - K3 + 32*Tc == 6*c, "quartic identity"
line = A4 + 4*D3 + 32*Tc
print("(a) straight-line fourth derivative", line, float(line), " c_line =", float(line/6), " c =", float(c))
print("(b) 3k^TH^{-1}k =", float(K3), " 32 T_c =", float(32*Tc), " relative gap", float((K3-32*Tc)/K3))

# (c) plug-in two-agent example: codes T(1->0)=(1,0), T(0->1)=(0,1), R=1/q, c_i=0, no dictionaries active
P0 = np.array([[10., 3.], [3., 1.]]); P1 = np.array([[1., 3.], [3., 10.]])
e1 = np.array([1., 0.]); e2 = np.array([0., 1.])
for q in (0.25, 0.5, 0.75, 1.0, 4.0):
    A0 = P0 + q*np.outer(e1, e1); A1 = P1 + q*np.outer(e2, e2)
    # agent 0 receives from 1 through e1: record e1^T(y0 - y1); agent 1 receives from 0 through e2
    A = np.block([[A0, np.zeros((2, 2))], [np.zeros((2, 2)), A1]])
    C = np.block([[np.zeros((2, 2)), q*np.outer(e1, e1)], [q*np.outer(e2, e2), np.zeros((2, 2))]])
    Jac = -(np.eye(4) - np.linalg.solve(A, C))
    ev = np.linalg.eigvals(Jac)
    print(f"(c) q={q}: max Re eig = {ev.real.max():+.6f}, predicted (2q-1)/(q+1) = {(2*q-1)/(q+1):+.6f}")

# (d) law-level covariance map on a random complete exchange of three agents, K=2, lambda=0 (records only)
rng = np.random.default_rng(3)
K = 2; nodes = range(3); edges = [(j, i) for i in nodes for j in nodes if i != j]
T = {e: rng.normal(size=(1, K)) for e in edges}; R = {e: np.array([[0.3]]) for e in edges}
c_i = {i: rng.normal(size=K) for i in nodes}
P = {i: (lambda a: a@a.T + 0.5*np.eye(K))(rng.normal(size=(K, K))) for i in nodes}
s2inv = 1.0
def kappa(i):  # omega (K x K, vec row-major) -> omega c_i
    return np.kron(np.eye(K), c_i[i][None, :])
def Phi(Sig):
    out = {}
    for i in nodes:
        ins = [e for e in edges if e[1] == i]
        d = K + K*K*len(ins); A = np.zeros((d, d)); A[:K, :K] += P[i]
        for k, e in enumerate(ins):
            sl = slice(K + K*K*k, K + K*K*(k+1)); A[sl, sl] += s2inv*np.eye(K*K)
            j = e[0]; Rt = R[e] + T[e]@Sig[j]@T[e].T; L = np.zeros((1, d))
            L[:, :K] = T[e]; L[:, sl] = -T[e]@kappa(i)
            A += L.T@np.linalg.solve(Rt, L)
        out[i] = np.linalg.inv(A)[:K, :K]
    return out
lo = {i: np.zeros((K, K)) for i in nodes}; hi = {i: np.linalg.inv(P[i]) for i in nodes}
for _ in range(200):
    lo, hi = Phi(lo), Phi(hi)
print("(d) |iterate from 0 - iterate from upper bound| =", max(np.abs(lo[i]-hi[i]).max() for i in nodes))
