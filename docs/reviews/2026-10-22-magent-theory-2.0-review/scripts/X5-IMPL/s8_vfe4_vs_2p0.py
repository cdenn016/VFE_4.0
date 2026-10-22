"""VFE_4.0 objective/route versus the MAgent 2.0 law potential.

(a) eq:dyn-jensen for the VFE_4.0 Gaussian communication record (A=B=I, o=0, independent laws).
(b) VFE_4.0 rows softmax(log pi - C_j) versus 2.0 Gibbs rows softmax(log pi - D_j/T).
(c) Time parametrization: VFE_4.0 damped CAVI (moment interpolation), VFE_4.0 fisher_natural
    (natural interpolation) and the continuous Fisher flow of one Gaussian block.
(d) ch.05 inh:smooth-mixture-failure and inh:smooth-boundary-failure numbers.
"""
import numpy as np

def kl(m1, S1, m2, S2):
    S2i = np.linalg.inv(S2); d = m2 - m1
    return 0.5 * (np.trace(S2i @ S1) + d @ S2i @ d - len(m1) + np.linalg.slogdet(S2)[1] - np.linalg.slogdet(S1)[1])
def ent(S): return 0.5 * (len(S) * (1 + np.log(2 * np.pi)) + np.linalg.slogdet(S)[1])
rng = np.random.default_rng(2)
K = 2
def spd(scale=1.0):
    B = rng.standard_normal((K, K)); return scale * (B @ B.T + 0.3 * np.eye(K))
a, V = rng.standard_normal(K), spd()
R = 0.5 * np.eye(K)
def C(b, W):   # VFE_4.0 bank_costs for A=B=I, o=0, independent receiver/sender
    d = a - b; Ri = np.linalg.inv(R)
    return 0.5 * np.linalg.slogdet(2 * np.pi * R)[1] + 0.5 * (d @ Ri @ d + np.trace(Ri @ (V + W)))
b, W = rng.standard_normal(K), spd()
# Jensen term E_q KL(p || p^w) with p^w the source posterior given receiver value w
Pw = np.linalg.inv(np.linalg.inv(W) + np.linalg.inv(R))
Ri = np.linalg.inv(R); Pwi = np.linalg.inv(Pw)
# closed form: m(w)-b = Pw R^{-1}(w-b), so E_q (m-b)^T Pw^{-1} (m-b) = (a-b)^T R^{-1} Pw R^{-1} (a-b) + tr(R^{-1} Pw R^{-1} V)
quad = (a - b) @ Ri @ Pw @ Ri @ (a - b) + np.trace(Ri @ Pw @ Ri @ V)
gap = 0.5 * (np.trace(Pwi @ W) + quad - K + np.linalg.slogdet(Pw)[1] - np.linalg.slogdet(W)[1])
lhs = C(b, W); rhs = ent(V) + kl(a, V, b, W + R) + gap
print(f'(a) C={lhs:.12f}; H(q)+KL(q||pK)+E_q KL(p||p^w) = {rhs:.12f}; residual={lhs - rhs:.2e}; Jensen gap={gap:.4f}')
# (b) rows
best_flip = None
for trial in range(2000):
    srcs = [(rng.standard_normal(K) * 0.7, spd(rng.uniform(0.05, 3))) for _ in range(3)]
    Cs = np.array([C(bb, WW) for bb, WW in srcs])
    Ds = np.array([kl(a, V, bb, WW) for bb, WW in srcs])            # identity channel, T=1
    DK = np.array([kl(a, V, bb, WW + R) for bb, WW in srcs])       # channel N(.,R), T=1
    r1 = np.exp(-(Cs - Cs.min())); r1 /= r1.sum()
    r2 = np.exp(-(Ds - Ds.min())); r2 /= r2.sum()
    r3 = np.exp(-(DK - DK.min())); r3 /= r3.sum()
    if np.argmax(r1) != np.argmax(r2) and np.argmax(r1) != np.argmax(r3):
        best_flip = (r1, r2, r3); break
print('(b) example rows: VFE_4.0', np.round(best_flip[0], 4), ' 2.0 identity-channel', np.round(best_flip[1], 4),
      ' 2.0 noisy-channel', np.round(best_flip[2], 4), f'(found at trial {trial})')
# (c) time parametrization of one scalar Gaussian block relaxing to N(m*,1/P*)
mstar, Pstar, m0, P0 = 1.0, 10.0, 0.0, 0.1
def first_time(update, w, thr=0.5):
    m, P = m0, P0
    for k in range(1, 200000):
        m, P = update(m, P, w)
        if m >= thr: return k * w
def cavi_moment(m, P, w):
    S, Ss = 1 / P, 1 / Pstar; d = m - mstar
    m2 = (1 - w) * m + w * mstar; S2 = (1 - w) * S + w * Ss + w * (1 - w) * d * d
    return m2, 1 / S2
def natural(m, P, w):
    P2 = (1 - w) * P + w * Pstar; return ((1 - w) * P * m + w * Pstar * mstar) / P2, P2
# continuous Fisher flow: P(t)=P*+(P0-P*)e^{-t}, eta(t)=P* m* + (P0 m0 - P* m*) e^{-t}
ts = np.linspace(0, 5, 500001); Pt = Pstar + (P0 - Pstar) * np.exp(-ts); et = Pstar * mstar + (P0 * m0 - Pstar * mstar) * np.exp(-ts)
t_flow = ts[np.argmax(et / Pt >= 0.5)]
for w in (0.1, 0.01, 0.001):
    print(f'(c) w={w}: time to mean 0.5  moment-CAVI={first_time(cavi_moment, w):.4f}  fisher_natural={first_time(natural, w):.4f}  exact Fisher flow={t_flow:.4f}')
# (d) ch.05 numbers
J = 0.5; a_, c_ = 0.5, 0.25
f = lambda x: (x + x**3) / 4
print('(d) xbar-dot at A, B:', J * c_, J * c_ * (1 - a_**2), ' claimed 1/8, 3/32')
m_, aa = 0.25, 0.25
print('    response means:', (f(m_ + aa) + f(m_ - aa)) / 2 * 256, (f(m_) + f(m_)) / 2 * 256, ' (x256) claimed 20, 17')

# (e) Gaussian non-invariance of the 2.0 smooth peer-KL Fisher flow with a sender role.
x = np.linspace(-14, 14, 40001); dx = x[1] - x[0]
npdf = lambda x, m, s: np.exp(-0.5 * ((x - m) / s)**2) / (s * np.sqrt(2 * np.pi))
qi, qj, qk, r = npdf(x, 0, 1), npdf(x, 0.5, 1), npdf(x, 1.0, 1.3), npdf(x, 0, 2)
def fisher_residual(include_incoming):
    h = 2 * np.log(qi) - np.log(r) - np.log(qj)          # local KL + receiver KL(q_i||q_j)
    if include_incoming:
        h = h - qk / qi                                   # sender role in KL(q_k||q_i)
    score = -(h - (qi * h).sum() * dx)                    # d/dt log q_i
    basis = np.stack([x, x**2 - 1])                       # centered Gaussian scores at N(0,1)
    G = (qi * basis[:, None] * basis[None]).sum(-1) * dx
    coef = np.linalg.solve(G, (qi * basis * score).sum(-1) * dx)
    res = score - coef @ basis
    return np.sqrt((qi * res**2).sum() * dx) / np.sqrt((qi * score**2).sum() * dx)
print('(e) relative Fisher norm of the non-Gaussian part of d/dt log q_i:',
      'without sender role', f'{fisher_residual(False):.2e},', 'with sender role', f'{fisher_residual(True):.3f}')
