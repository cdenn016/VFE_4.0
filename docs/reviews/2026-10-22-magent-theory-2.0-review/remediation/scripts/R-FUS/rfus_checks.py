"""R-FUS checks for chapters 6 and 7 remediation (single-threaded, short)."""
import numpy as np
import sympy as sp

# 1. chi is inert; variance derivative of local KL at v=1/n
mi, mj, chi, n, a, c, v = sp.symbols('mi mj chi n a c v', real=True)
shift = mj + chi*(mi - mj)
kl = (mi - shift)**2/2
cost = sp.sqrt(2*kl)/(1-chi)
print("1a cost^2 - (mi-mj)^2:", sp.simplify(cost**2 - (mi-mj)**2))
A = sp.Matrix([[a, -c], [-c, a]])
Ecov = v*sp.eye(2)
Pcov = A.inv()/n
KLv = sp.Rational(1, 2)*((Pcov.inv()*Ecov).trace() - 2 + sp.log(Pcov.det()/Ecov.det()))
print("1b dKL/dv at v=1/n (per sector, both sectors share v):",
      sp.simplify(sp.diff(KLv, v).subs(v, 1/n)), " expected n(a-1) for two sectors")

# 2. finite-time complete synchronization for random data, explicit Euler with small step
rng = np.random.default_rng(0)
def run(mu0, nmass, a, c, lam, dt=2e-5, T=6.0, tol=2e-3):
    mu = mu0.copy(); N = nmass.sum(); t = 0.0
    R0 = np.sqrt((nmass[:, None]*mu0**2).sum())
    kap = a - c
    t0 = max(0.0, np.log(4*c*R0/(lam*N))/kap) if 4*c*R0 > lam*N else 0.0
    Dt0 = None; tsync = None
    while t < T:
        q, s = mu[:, 0], mu[:, 1]
        sq = np.sign(q[:, None]-q[None, :]); ss = np.sign(s[:, None]-s[None, :])
        dq = -a*q + c*s - lam*(sq*nmass[None, :]).sum(1)
        ds = -a*s + c*q - lam*(ss*nmass[None, :]).sum(1)
        if Dt0 is None and t >= t0:
            Dt0 = max(np.ptp(q), np.ptp(s))
        if tsync is None and max(np.ptp(q), np.ptp(s)) < tol:
            tsync = t
        mu = mu + dt*np.stack([dq, ds], 1); t += dt
    bound = t0 + 2*Dt0/(lam*N)
    return tsync, bound, np.abs(mu).max()
for k in range(3):
    mu0 = rng.normal(0, 2, size=(4, 2)); nm = rng.integers(1, 4, size=4).astype(float)
    print("2 sync time, bound:", run(mu0, nm, 1.0, 0.5, 0.3))

# 3. dyadic depth-3 hierarchy in tau frame (sticky event-driven)
def sticky(z, m, lam):
    blocks = [[zi, mi, [i]] for i, (zi, mi) in enumerate(zip(z, m))]
    tau = 0.0; hist = []
    while len(blocks) > 1:
        blocks.sort(key=lambda b: b[0]); Mtot = sum(b[1] for b in blocks)
        left = np.cumsum([0]+[b[1] for b in blocks[:-1]])
        vel = [lam*((Mtot - left[i] - b[1]) - left[i]) for i, b in enumerate(blocks)]
        dts = [(blocks[i+1][0]-blocks[i][0])/(vel[i]-vel[i+1]) for i in range(len(blocks)-1)]
        i = int(np.argmin(dts)); d = dts[i]; tau += d
        for b, vv in zip(blocks, vel): b[0] += vv*d
        B = [blocks[i][0], blocks[i][1]+blocks[i+1][1], blocks[i][2]+blocks[i+1][2]]
        blocks = blocks[:i] + [B] + blocks[i+2:]; hist.append((round(tau, 4), sorted(B[2])))
    return hist
G = [1.0, 1.0*4*1.1, 1.0*4*1.1*4*1.1*1.2]  # N/2=4
z = [0, G[0], G[0]+G[1], 2*G[0]+G[1]]; z = z + [zz + z[-1] + G[2] for zz in z]
print("3 dyadic history:", sticky(np.array(z, float), np.ones(8), 1.0))

# 4. spectrum injectivity for generic reference; product invariance of CGF differences
Aref = np.array([0.6, 0.4])
for kap in ([0.75, 0.25], [0.25, 0.75]):
    print("4a spectrum atoms (mass, value):", list(zip(Aref, np.log(np.array(kap)/Aref).round(3))))
def Lam(kap, Aref, b):
    return np.log((Aref*(kap/Aref)**b).sum())
k1 = [np.array([0.7, 0.3]), np.array([0.2, 0.8])]
U2 = np.array([0.5, 0.5]); psi, psi2 = np.array([0.5, 0.5]), np.array([0.9, 0.1])
bs = np.linspace(0.05, 1, 20)
d1 = [Lam(np.kron(k1[0], psi), np.kron(U2, U2), b)-Lam(np.kron(k1[1], psi), np.kron(U2, U2), b) for b in bs]
d2 = [Lam(np.kron(k1[0], psi2), np.kron(U2, U2), b)-Lam(np.kron(k1[1], psi2), np.kron(U2, U2), b) for b in bs]
print("4b max |difference of CGF differences|:", np.max(np.abs(np.array(d1)-np.array(d2))))
# evolution of CGF: Lam^t(b) = Lam^0(ab) - b Lam^0(a)
kap0 = np.array([0.1, 0.3, 0.6]); A3 = np.array([0.2, 0.5, 0.3]); t = 0.7; aa = np.exp(-t)
kapt = A3*(kap0/A3)**aa; kapt /= kapt.sum()
print("4c CGF evolution residual:", max(abs(Lam(kapt, A3, b)-(Lam(kap0, A3, aa*b)-b*Lam(kap0, A3, aa))) for b in bs))
# range invariance
l0 = np.log(kap0/A3); lt = np.log(kapt/A3)
print("4d range0", l0.min(), l0.max(), "range_t", lt.min(), lt.max(), "width ratio", np.ptp(lt)/np.ptp(l0), "a", aa)
# 5. private-transmitting identity channel: equal spectra, different peer cost
k1v, k2v = np.array([.75, .25]), np.array([.25, .75])
print("5 KL(k1||k1)=0, KL(k1||k2)=", (k1v*np.log(k1v/k2v)).sum(), "0.5 log 3 =", 0.5*np.log(3))
# 6. proximal tie threshold for two agents, pure pair term
h, lam = 0.1, 0.7
for d0 in [0.13, 0.14, 0.15]:
    ds = np.linspace(-1, 1, 200001); obj = lam*np.abs(ds) + (ds-d0)**2/(4*h)
    print("6 d0", d0, "argmin d", round(ds[np.argmin(obj)], 5), "threshold 2h lam =", 2*h*lam)
