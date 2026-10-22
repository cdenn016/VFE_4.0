r"""X4-PRIOR check of the 2.0 open-formation construction (chapter 12) in its scaled limit.

Limit field (eq. open:limit):  u_i' = -u_i + sum_j (w_ij + w_ji)(u_j - u_i),
w_ij = exp(-(u_i-u_j)^2/2) / sum_k exp(-(u_i-u_k)^2/2)   (self included).
Selector (def. open:selector) with eps=1/10, delta=1/20, one sector.

Question: does the second-stage (parent) event time track the supplied initial scale A,
i.e. t_parent - log(A) ~ const (a supplied-scale sequence in the sense of the earlier SPEC R14)?
Also: identity-self versus selective-self channel factor exp(H/T) in the native row.
"""
import numpy as np
from scipy.integrate import solve_ivp

EPS, DEL = 0.1, 0.05


def weights(u):
    d = u[:, None] - u[None, :]
    L = -0.5 * d**2
    L -= L.max(axis=1, keepdims=True)
    W = np.exp(L)
    return W / W.sum(axis=1, keepdims=True)


def field(t, u):
    W = weights(u)
    S = W + W.T
    return -u + (S * (u[None, :] - u[:, None])).sum(axis=1)


def block_row(W, I, J):
    return W[np.ix_(I, J)].sum() / len(I)


def qualifies(W, blocks, I, J):
    for (X, Y) in ((I, J), (J, I)):
        b = block_row(W, X, Y)
        ext = sum(block_row(W, X, K) for K in blocks if K is not X and K is not Y)
        if not (b > EPS and b - ext > DEL):
            return False
    return True


def run(A, R=None, tmax=None):
    core = [-A - 2, -A + 2, A - 2, A + 2]
    u0 = np.array(core + ([R - 2, R + 2] if R is not None else []), dtype=float)
    if tmax is None:
        tmax = np.log(max(A, 4)) + 6
    ts = np.linspace(0, tmax, 4001)
    sol = solve_ivp(field, (0, tmax), u0, t_eval=ts, rtol=1e-10, atol=1e-12, method='LSODA')
    n = len(u0)
    singles = [[i] for i in range(n)]
    pair_t, par_t = None, None
    for k, t in enumerate(sol.t):
        W = weights(sol.y[:, k])
        if pair_t is None:
            if qualifies(W, singles, singles[0], singles[1]) and qualifies(W, singles, singles[2], singles[3]):
                pair_t = t
        else:
            blocks = [[0, 1], [2, 3]] + ([[4], [5]] if n == 6 else [])
            if n == 6 and qualifies(W, blocks, blocks[2], blocks[3]):
                blocks = [[0, 1], [2, 3], [4, 5]]
            if qualifies(W, blocks, blocks[0], blocks[1]):
                par_t = t
                break
    return pair_t, par_t


print("A, t_pair, t_parent, t_parent - log(A)   [four-agent core]")
for A in (64, 256, 1024, 4096, 16384):
    tp, tq = run(A)
    print(A, round(tp, 4), round(tq, 4), round(tq - np.log(A), 4))

print("six agents, third pair at R = 1000 A:")
for A in (256, 4096):
    tp, tq = run(A, R=1000.0 * A)
    print(A, round(tp, 4), round(tq, 4), round(tq - np.log(A), 4))

# Identity-self versus selective-self channel: the receiver angular entropy H cancels from
# the Gibbs normalization only when the self channel is the selective channel K.
# With an identity self channel D_ii = 0 while D_ij = H + d_ij, so every peer weight is
# multiplied by exp(-H/T) relative to self.
for kappa in (0.5, 1.0, 2.0):
    from scipy.special import i0, i1
    H = kappa * i1(kappa) / i0(kappa) - np.log(i0(kappa))  # KL(vonMises_kappa || Haar)
    for h in (0.3, 0.1):
        T = h**2
        print("kappa=%.1f H=%.4f T=%.3f  exp(-H/T)=%.3e" % (kappa, H, T, np.exp(-H / T)))
