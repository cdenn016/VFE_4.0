"""L2-gauge float64 checks for g11_latent_links.tex."""
import numpy as np
from scipy.linalg import expm, sqrtm
rng = np.random.default_rng(11)
inv = np.linalg.inv; det = np.linalg.det
def spd(K):
    A = rng.standard_normal((K, K)); return A@A.T + 0.5*np.eye(K)
def rgl(K, s=0.6):
    while True:
        A = np.eye(K) + s*rng.standard_normal((K, K))
        if det(A) > 0.2: return A
def Lex(B, T):
    n = B.shape[0]; M = np.zeros((2*n, 2*n)); M[:n, :n] = B; M[n:, n:] = B; M[:n, n:] = T
    return expm(M)[:n, n:]
def kl(m0, S0, m1, S1):
    K = len(m0); S1i = inv(S1); dm = m1 - m0
    return 0.5*(np.trace(S1i@S0) + dm@S1i@dm - K + np.log(det(S1)/det(S0)))
K = 3
mi, mj = rng.standard_normal(K), rng.standard_normal(K); Si, Sj = spd(K), spd(K); w = 0.5*rng.standard_normal((K, K))
E = expm(w); Ei = expm(-w)
form = 0.5*np.trace(inv(Sj)@Ei@Si@Ei.T) + 0.5*(Ei@mi - mj)@inv(Sj)@(Ei@mi - mj) - K/2 + 0.5*np.log(det(Sj)/det(Si)) + np.trace(w)
print("law-level KL formula vs direct:", abs(form - kl(mi, Si, E@mj, E@Sj@E.T)))
Ui, Uj = rgl(K), rgl(K)  # non-orthogonal frames
loc = kl(Ui@mi, Ui@Si@Ui.T, (Ui@E@inv(Uj))@(Uj@mj), (Ui@E@inv(Uj))@(Uj@Sj@Uj.T)@(Ui@E@inv(Uj)).T)
print("frame (local-coordinate) invariance, non-orth U:", abs(loc - form))
A = inv(Sj)@(Ei@(Si + np.outer(mi, mi)) - np.outer(mj, mi))
grad = np.eye(K) - Lex(-w.T, A)
f = lambda ww: kl(mi, Si, expm(ww)@mj, expm(ww)@Sj@expm(ww).T)
h = 1e-6; T = rng.standard_normal((K, K))
print("law-level gradient vs FD:", abs(np.trace(grad.T@T) - (f(w + h*T) - f(w - h*T))/(2*h)))
mu = rng.standard_normal(K); S = spd(K); ups = rng.standard_normal((K, K)); t = 1e-3
kf = (ups@mu)@inv(S)@(ups@mu) + np.trace(ups@ups) + np.trace(inv(S)@ups@S@ups.T)
print("orbit Fisher: 2KL/t^2 - k (O(t)):", abs(2*kl(mu, S, expm(t*ups)@mu, expm(t*ups)@S@expm(t*ups).T)/t**2 - kf), " k =", kf)
# saddle (iv)
s2 = 0.7**2; a = 0.3; Jm = np.array([[0., -1], [1, 0]]); D = np.diag([1., -1])
mj2 = np.array([1., 0]); r0 = -(1 + 2*s2)*np.exp(a)*np.array([1., 0])  # mu_i - o_e
l = lambda M: 0.5*(r0 - M@mj2)@(r0 - M@mj2) + 0.5*np.trace(M@(s2*np.eye(2))@M.T)
w0 = a*np.eye(2) + np.pi*Jm
gfd = np.array([[(l(expm(w0 + h*np.outer(np.eye(2)[p], np.eye(2)[q]))) - l(expm(w0 - h*np.outer(np.eye(2)[p], np.eye(2)[q]))))/(2*h) for q in range(2)] for p in range(2)])
tt = -0.05
print("saddle: |grad|:", np.abs(gfd).max(), " l(w0+tD)-l(w0) at t=-0.05:", l(expm(w0 + tt*D)) - l(expm(w0)), " predicted s2 e^{2a} t^3/pi^2:", s2*np.exp(2*a)*tt**3/np.pi**2)
# cycle determinant, unanchored, m=3 agents, K=2
K2 = 2; Ws = [spd(K2) for _ in range(3)]; ws = [0.4*rng.standard_normal((K2, K2)) for _ in range(3)]
Th = [expm(x) for x in ws]
Tb = np.zeros((3*K2, 3*K2))
for aa in range(3):
    Tb[aa*K2:(aa+1)*K2, aa*K2:(aa+1)*K2] = np.eye(K2)
    b = (aa + 1) % 3; Tb[aa*K2:(aa+1)*K2, b*K2:(b+1)*K2] = -Th[aa]
Wd = np.zeros((3*K2, 3*K2))
for aa in range(3): Wd[aa*K2:(aa+1)*K2, aa*K2:(aa+1)*K2] = Ws[aa]
Lam = Tb.T@Wd@Tb
print("cycle det formula rel:", abs(det(Lam) - det(np.eye(K2) - Th[0]@Th[1]@Th[2])**2*np.prod([det(x) for x in Ws]))/det(Lam))
# induced action edge derivatives (iv), K=3 isotropic
ai, aj, we = 1.3, 0.8, 0.9; ws_ = ai*we/(ai + we); up = rng.standard_normal((K, K)); up = up + up.T
def Sind(tv):
    M = expm(tv*up); L = np.block([[(ai + we)*np.eye(K), -we*M], [-we*M.T, aj*np.eye(K) + we*M.T@M]])
    return 0.5*np.log(det(L))
t = 1e-4
d1 = (Sind(t) - Sind(-t))/(2*t); d2 = (Sind(t) - 2*Sind(0) + Sind(-t))/t**2
print("induced (iv) d1:", abs(d1 - ws_*np.trace(up)/(aj + ws_)), " d2:", abs(d2 - 2*ws_*aj*np.sum(up**2)/(aj + ws_)**2))
sk = rng.standard_normal((K, K)); sk = sk - sk.T
def Sind2(tv):
    M = expm(tv*sk); L = np.block([[(ai + we)*np.eye(K), -we*M], [-we*M.T, aj*np.eye(K) + we*M.T@M]])
    return 0.5*np.log(det(L))
print("induced (iv) skew direction constant:", abs(Sind2(0.9) - Sind2(0)))
# reciprocal pair K=1 flat slice
a0 = 0.7; om = 0.37; gg = np.exp(om); gp = 1/gg
L = np.array([[a0 + 1 + gp**2, -(gg + gp)], [-(gg + gp), a0 + 1 + gg**2]])
print("reciprocal pair det:", abs(det(L) - (a0**2 + 4*a0*np.cosh(om)**2)))
# hopping triangle (v): isotropic isometric, G general; check leading term and remainder bound
G = spd(K); Gh = np.real(sqrtm(G))
def iso():
    Sk = rng.standard_normal((K, K)); Sk = Sk - Sk.T
    return inv(G)@Sk*0.7   # G-orthogonal generator
wv = [iso() for _ in range(3)]
aa_ = [3.0, 2.5, 3.3]; ww = [0.5, 0.4, 0.6]
# triangle e0:1->0, e1:2->1, e2:0->2 ; residual z_i - Theta z_j
edges = [(0, 1), (1, 2), (2, 0)]  # (i, j)
def SindT(scale):
    Lm = np.zeros((3*K, 3*K))
    for i in range(3): Lm[i*K:(i+1)*K, i*K:(i+1)*K] += aa_[i]*G
    for (i, j), x, wt in zip(edges, wv, ww):
        M = expm(scale*x); Wm = wt*G
        Lm[i*K:(i+1)*K, i*K:(i+1)*K] += Wm; Lm[j*K:(j+1)*K, j*K:(j+1)*K] += M.T@Wm@M
        Lm[i*K:(i+1)*K, j*K:(j+1)*K] -= Wm@M; Lm[j*K:(j+1)*K, i*K:(i+1)*K] -= M.T@Wm
    return 0.5*np.linalg.slogdet(Lm)[1]
adiag = [aa_[0] + ww[0] + ww[2], aa_[1] + ww[0] + ww[1], aa_[2] + ww[1] + ww[2]]
coef = np.prod(ww)/np.prod(adiag)
Thp = expm(wv[0])@expm(wv[1])@expm(wv[2])
Qp = K - np.trace(Thp)
whop = max(ww)/min(adiag)
bound = K/5*((2*whop)**5/(1 - 2*whop) + 2*whop**5/(1 - whop))
print("hopping triangle: |remainder| =", abs(SindT(1) - SindT(0) - coef*Qp), " bound =", bound, " leading =", coef*Qp, " isometry check:", np.abs(expm(wv[0]).T@G@expm(wv[0]) - G).max())
