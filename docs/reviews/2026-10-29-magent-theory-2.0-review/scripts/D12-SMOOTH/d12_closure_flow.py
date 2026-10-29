r"""D12-SMOOTH check 1: chapter 7 (07_smooth_closure.tex) receiver-reconstruction architecture.

Independent finite-table reconstruction. The fine energy is computed from the definition
(local KL against R_i = R_i^c (x) A^q (x) A^s and peer KL against q_j K_ij with the
four-index kernel K_ij((z1,v1),(z,v)) = B_ij[mu](z1,z) kappa_{i,z}(v)), never from the
cancellation identity. Gradients are complex-step (exact to rounding). Checks:
  A. smooth:energy-split, smooth:split-flow (velocity level, 3 agents, Fisher rows,
     law-dependent public channels depending on source and receiver marginals).
  B. trajectory level (2 agents): fine Fisher flow vs retained flow forced by the explicit
     spectral solution; smooth:cgf-completion product example and a control.
  C. spectral identities: smooth:entropy-rate, smooth:spectrum-inverse, smooth:cgf-flow,
     range bounds of smooth:wellposed (i), smooth:renyi.
  D. smooth:channel-variants: fixed reconstruction B (x) A gives rate 2; private-transmitting
     identity channel counterexample (peer cost (1/2)log 3, different spectral velocity).
Numerical agreement supports identities; it proves nothing about general spaces.
"""
import time
import numpy as np

t_start = time.time()
rng = np.random.default_rng(20261029)
H_CS = 1e-30  # complex-step size


def rprob(*shape):
    x = rng.uniform(0.5, 1.5, size=shape)
    return x / x.sum(axis=-1, keepdims=True)


def softmax_rows(L):
    E = np.exp(L)
    return E / E.sum(axis=1, keepdims=True)


class Model:
    """Two-sector population: belief K = Z x V, model M = W x U."""

    def __init__(self, N, nZ, nV, nW, nU, Tq, Ts, Aq=None, As=None):
        self.N, self.nZ, self.nV, self.nW, self.nU = N, nZ, nV, nW, nU
        self.Tq, self.Ts = Tq, Ts
        self.Aq = [rprob(nV) for _ in range(N)] if Aq is None else Aq
        self.As = [rprob(nU) for _ in range(N)] if As is None else As
        self.rc = [rprob(nW) for _ in range(N)]
        self.Gc = [rprob(nW, nZ) for _ in range(N)]  # G^c(w, z), normalized over z
        self.src = [[j for j in range(N) if j != i] for i in range(N)]
        self.piq = [rprob(N - 1) for _ in range(N)]
        self.pis = [rprob(N - 1) for _ in range(N)]
        self.ThQ = {(i, j): rng.normal(0, .5, (nZ, nZ)) for i in range(N) for j in self.src[i]}
        self.ThS = {(i, j): rng.normal(0, .5, (nW, nW)) for i in range(N) for j in self.src[i]}
        self.fZ, self.gZ, self.gZ2 = rng.normal(0, 1, nZ), rng.normal(0, 1, (nZ, nZ)), rng.normal(0, 1, (nZ, nZ))
        self.fW, self.gW, self.gW2 = rng.normal(0, 1, nW), rng.normal(0, 1, (nW, nW)), rng.normal(0, 1, (nW, nW))
        self.chan = "recon"

    # law-dependent public channels: depend on the source AND receiver public laws
    def Bq(self, i, j, mu_i, mu_j):
        return softmax_rows(self.ThQ[(i, j)] + 0.8 * (mu_j @ self.fZ) * self.gZ + 0.6 * (mu_i @ self.fZ) * self.gZ2)

    def Bs(self, i, j, s_i, s_j):
        return softmax_rows(self.ThS[(i, j)] + 0.7 * (s_j @ self.fW) * self.gW + 0.5 * (s_i @ self.fW) * self.gW2)

    def fine_shapes(self):
        N = self.N
        return [(self.nZ, self.nV)] * N + [(self.nW, self.nU)] * N + [(N - 1,)] * N + [(N - 1,)] * N

    def coarse_shapes(self):
        N = self.N
        return [(self.nZ,)] * N + [(self.nW,)] * N + [(N - 1,)] * N + [(N - 1,)] * N

    @staticmethod
    def unpack(x, shapes):
        out, k = [], 0
        for sh in shapes:
            n = int(np.prod(sh))
            out.append(x[k:k + n].reshape(sh))
            k += n
        return out

    def fine_energy(self, x):
        N = self.N
        T = self.unpack(x, self.fine_shapes())
        q, s, be, ga = T[:N], T[N:2 * N], T[2 * N:3 * N], T[3 * N:]
        mu = [qi.sum(1) for qi in q]
        kap = [qi / qi.sum(1)[:, None] for qi in q]
        sg = [si.sum(1) for si in s]
        vk = [si / si.sum(1)[:, None] for si in s]
        E = 0.0
        for i in range(N):
            P = q[i][:, :, None, None] * s[i][None, None, :, :]
            R = (self.Gc[i].T)[:, None, :, None] * self.rc[i][None, None, :, None]
            R = R * self.Aq[i][None, :, None, None] * self.As[i][None, None, None, :]
            E = E + np.sum(P * np.log(P / R))
            for a, j in enumerate(self.src[i]):
                B = self.Bq(i, j, mu[i], mu[j])
                Bm = self.Bs(i, j, sg[i], sg[j])
                if self.chan == "recon":      # receiver conditional reconstruction
                    K4 = B[:, None, :, None] * kap[i][None, None, :, :]
                    K4s = Bm[:, None, :, None] * vk[i][None, None, :, :]
                elif self.chan == "fixedA":   # fixed reconstruction by the private reference
                    K4 = B[:, None, :, None] * self.Aq[i][None, None, None, :] * np.ones((1, 1, self.nZ, 1))
                    K4s = Bm[:, None, :, None] * self.As[i][None, None, None, :] * np.ones((1, 1, self.nW, 1))
                elif self.chan == "privid":   # private-transmitting identity (needs common V, A)
                    K4 = B[:, None, :, None] * np.eye(self.nV)[None, :, None, :]
                    K4s = Bm[:, None, :, None] * np.eye(self.nU)[None, :, None, :]
                K4 = np.broadcast_to(K4, (self.nZ, self.nV, self.nZ, self.nV))
                K4s = np.broadcast_to(K4s, (self.nW, self.nU, self.nW, self.nU))
                nu = np.einsum("ab,abcd->cd", q[j], K4)
                nus = np.einsum("ab,abcd->cd", s[j], K4s)
                E = E + be[i][a] * np.sum(q[i] * np.log(q[i] / nu))
                E = E + ga[i][a] * np.sum(s[i] * np.log(s[i] / nus))
            E = E + self.Tq * np.sum(be[i] * np.log(be[i] / self.piq[i]))
            E = E + self.Ts * np.sum(ga[i] * np.log(ga[i] / self.pis[i]))
        return E

    def coarse_energy(self, x):
        N = self.N
        T = self.unpack(x, self.coarse_shapes())
        mu, sg, be, ga = T[:N], T[N:2 * N], T[2 * N:3 * N], T[3 * N:]
        E = 0.0
        for i in range(N):
            P = mu[i][:, None] * sg[i][None, :]
            R = self.Gc[i].T * self.rc[i][None, :]
            E = E + np.sum(P * np.log(P / R))
            for a, j in enumerate(self.src[i]):
                nu = mu[j] @ self.Bq(i, j, mu[i], mu[j])
                nus = sg[j] @ self.Bs(i, j, sg[i], sg[j])
                E = E + be[i][a] * np.sum(mu[i] * np.log(mu[i] / nu))
                E = E + ga[i][a] * np.sum(sg[i] * np.log(sg[i] / nus))
            E = E + self.Tq * np.sum(be[i] * np.log(be[i] / self.piq[i]))
            E = E + self.Ts * np.sum(ga[i] * np.log(ga[i] / self.pis[i]))
        return E


def cs_grad(f, x):
    g = np.zeros(x.size)
    xc = x.astype(complex)
    for k in range(x.size):
        xc[k] += 1j * H_CS
        g[k] = f(xc).imag / H_CS
        xc[k] -= 1j * H_CS
    return g


def fisher_logvel(x, g, shapes):
    """d/dt log u = -(g - E_u g) per table (unit mobility)."""
    out, k = np.zeros_like(x), 0
    for sh in shapes:
        n = int(np.prod(sh))
        u, gg = x[k:k + n], g[k:k + n]
        out[k:k + n] = -(gg - np.sum(u * gg))
        k += n
    return out


def renorm_log(y, shapes):
    k = 0
    y = y.copy()
    for sh in shapes:
        n = int(np.prod(sh))
        m = y[k:k + n].max()
        y[k:k + n] -= m + np.log(np.sum(np.exp(y[k:k + n] - m)))
        k += n
    return y


def rand_state(M, kind="fine"):
    shapes = M.fine_shapes() if kind == "fine" else M.coarse_shapes()
    parts = []
    for sh in shapes:
        u = rng.uniform(0.3, 1.7, size=sh)
        parts.append((u / u.sum()).ravel())
    return np.concatenate(parts)


def split_fine(M, x):
    N = M.N
    T = M.unpack(x, M.fine_shapes())
    return T[:N], T[N:2 * N], T[2 * N:3 * N], T[3 * N:]


def kl_rows(k, A):
    return np.sum(k * np.log(k / A[None, :]), axis=1)


def coarse_from_fine(M, x):
    q, s, be, ga = split_fine(M, x)
    return np.concatenate([qi.sum(1) for qi in q] + [si.sum(1) for si in s] + [b for b in be] + [g for g in ga])


def retained_logvel(M, xc, Hq, Hs):
    """Retained velocity: public laws forced by H, rows from the coarse energy."""
    N = M.N
    g = cs_grad(M.coarse_energy, xc)
    shapes = M.coarse_shapes()
    T = M.unpack(g, shapes)
    for i in range(N):
        T[i] = T[i] + Hq[i]
        T[N + i] = T[N + i] + Hs[i]
    g2 = np.concatenate([t.ravel() for t in T])
    return fisher_logvel(xc, g2, shapes)


def fine_traj(M, x0, Tend, dt):
    shapes = M.fine_shapes()
    y = np.log(x0)
    nsteps = int(round(Tend / dt))
    f = lambda yy: fisher_logvel(np.exp(yy), cs_grad(M.fine_energy, np.exp(yy)), shapes)
    for _ in range(nsteps):
        k1 = f(y); k2 = f(y + dt / 2 * k1); k3 = f(y + dt / 2 * k2); k4 = f(y + dt * k3)
        y = renorm_log(y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), shapes)
    return np.exp(y)


def explicit_conditional(kap0, A, t):
    a = np.exp(-t)
    w = kap0 ** a * A[None, :] ** (1 - a)
    return w / w.sum(1, keepdims=True)


def retained_traj(M, x0, Tend, dt):
    """Retained flow: public laws + rows forced by H(z,t) from the explicit spectral solution."""
    q, s, be, ga = split_fine(M, x0)
    kq0 = [qi / qi.sum(1)[:, None] for qi in q]
    ks0 = [si / si.sum(1)[:, None] for si in s]
    shapes = M.coarse_shapes()

    def f(yy, t):
        xc = np.exp(yy)
        Hq = [kl_rows(explicit_conditional(kq0[i], M.Aq[i], t), M.Aq[i]) for i in range(M.N)]
        Hs = [kl_rows(explicit_conditional(ks0[i], M.As[i], t), M.As[i]) for i in range(M.N)]
        return retained_logvel(M, xc, Hq, Hs)

    y = np.log(coarse_from_fine(M, x0)); t = 0.0
    for _ in range(int(round(Tend / dt))):
        k1 = f(y, t); k2 = f(y + dt / 2 * k1, t + dt / 2); k3 = f(y + dt / 2 * k2, t + dt / 2); k4 = f(y + dt * k3, t + dt)
        y = renorm_log(y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), shapes); t += dt
    return np.exp(y)


def build_state(M, kq, ks, mu, sg, be, ga):
    q = [mu[i][:, None] * kq[i] for i in range(M.N)]
    s = [sg[i][:, None] * ks[i] for i in range(M.N)]
    return np.concatenate([qi.ravel() for qi in q] + [si.ravel() for si in s] + list(be) + list(ga))


print("=" * 70)
print("Part A: velocity-level checks of smooth:energy-split and smooth:split-flow")
print("=" * 70)
MA = Model(N=3, nZ=3, nV=4, nW=2, nU=3, Tq=0.7, Ts=0.9)
worst_E = worst_mu = worst_ell = worst_row = 0.0
for trial in range(3):
    x = rand_state(MA, "fine")
    q, s, be, ga = split_fine(MA, x)
    Hq = [kl_rows(qi / qi.sum(1)[:, None], MA.Aq[i]) for i, qi in enumerate(q)]
    Hs = [kl_rows(si / si.sum(1)[:, None], MA.As[i]) for i, si in enumerate(s)]
    xc = coarse_from_fine(MA, x)
    Ef = MA.fine_energy(x).real
    Ec = MA.coarse_energy(xc).real
    Epriv = sum(np.sum(qi.sum(1) * Hq[i]) for i, qi in enumerate(q))
    Epriv += sum(np.sum(si.sum(1) * Hs[i]) for i, si in enumerate(s))
    worst_E = max(worst_E, abs(Ef - (Ec + Epriv)))
    gf = cs_grad(MA.fine_energy, x)
    vf = fisher_logvel(x, gf, MA.fine_shapes())
    vr = retained_logvel(MA, xc, Hq, Hs)
    VF = MA.unpack(vf, MA.fine_shapes())
    VR = MA.unpack(vr, MA.coarse_shapes())
    N = MA.N
    for i in range(N):
        kap = q[i] / q[i].sum(1)[:, None]
        ell = np.log(kap / MA.Aq[i][None, :])
        pred = VR[i][:, None] + (Hq[i][:, None] - ell)
        worst_ell = max(worst_ell, np.max(np.abs(VF[i] - pred)))
        dmu_f = np.sum(q[i] * VF[i], axis=1) / q[i].sum(1)
        worst_mu = max(worst_mu, np.max(np.abs(dmu_f - VR[i])))
        kaps = s[i] / s[i].sum(1)[:, None]
        ells = np.log(kaps / MA.As[i][None, :])
        preds = VR[N + i][:, None] + (Hs[i][:, None] - ells)
        worst_ell = max(worst_ell, np.max(np.abs(VF[N + i] - preds)))
    for k in range(2 * N, 4 * N):
        worst_row = max(worst_row, np.max(np.abs(VF[k] - VR[k])))
print(f"energy split |E_fine - (E_c + private)|         max = {worst_E:.2e}")
print(f"public log-velocity, fine projection vs retained  max = {worst_mu:.2e}")
print(f"fine dlog q vs dlog mu + (H - ell), both sectors  max = {worst_ell:.2e}")
print(f"Fisher row log-velocities, fine vs coarse          max = {worst_row:.2e}")
print("(baseline: velocities are O(1); complex-step gradients exact to about 1e-15)")
print(f"     part A time {time.time()-t_start:.1f}s")

print()
print("=" * 70)
print("Part B: trajectory level (2 agents, product private reference A1 (x) A2)")
print("=" * 70)
A1 = np.array([0.35, 0.65]); A2 = np.array([0.55, 0.45])
Aprod = np.kron(A1, A2)
MB = Model(N=2, nZ=2, nV=4, nW=2, nU=2, Tq=0.8, Ts=0.6, Aq=[Aprod, Aprod])
Tend, dt = 0.5, 0.01
mu0 = [rprob(2), rprob(2)]; sg0 = [rprob(2), rprob(2)]
be0 = [np.array([1.0]), np.array([1.0])]; ga0 = [np.array([1.0]), np.array([1.0])]
k1z = [rprob(2, 2), rprob(2, 2)]
psi, psi2 = np.array([0.5, 0.5]), np.array([0.9, 0.1])  # different spectra against A2
ks0 = [rprob(2, 2), rprob(2, 2)]


def kq_product(k1, ps_by_z):
    return np.array([np.kron(k1[z], ps_by_z[z]) for z in range(2)])


xA = build_state(MB, [kq_product(k1z[0], [psi, psi]), kq_product(k1z[1], [psi, psi])], ks0, mu0, sg0, be0, ga0)
xB = build_state(MB, [kq_product(k1z[0], [psi2, psi2]), kq_product(k1z[1], [psi, psi])], ks0, mu0, sg0, be0, ga0)
xC = build_state(MB, [kq_product(k1z[0], [psi, psi2]), kq_product(k1z[1], [psi, psi])], ks0, mu0, sg0, be0, ga0)
t0 = time.time()
fA = fine_traj(MB, xA, Tend, dt)
rA = retained_traj(MB, xA, Tend, dt)
qA, sA, _, _ = split_fine(MB, fA)
cA = coarse_from_fine(MB, fA)
print(f"[B1] fine vs retained public laws at T={Tend}: max |diff| = {np.max(np.abs(cA - rA)):.2e}")
kexp = explicit_conditional(kq_product(k1z[0], [psi, psi]), Aprod, Tend)
print(f"[B1] fine kappa_0(T) vs explicit exponential arc:  max |diff| = {np.max(np.abs(qA[0] / qA[0].sum(1)[:, None] - kexp)):.2e}")
fB = fine_traj(MB, xB, Tend, dt)
fC = fine_traj(MB, xC, Tend, dt)
cB, cC = coarse_from_fine(MB, fB), coarse_from_fine(MB, fC)
print(f"[B2] cgf product example (psi vs psi2 at every z): public max |diff| = {np.max(np.abs(cA - cB)):.2e}")
print(f"[B2] control (psi at z=0, psi2 at z=1):             public max |diff| = {np.max(np.abs(cA - cC)):.2e}")
vA = np.log(np.kron(k1z[0][0], psi) / Aprod)
vB = np.log(np.kron(k1z[0][0], psi2) / Aprod)
print(f"[B2] spectra differ at z=0: sorted atoms A {np.round(np.sort(vA),4)} vs B {np.round(np.sort(vB),4)}")
print(f"     part B time {time.time()-t0:.1f}s")

print()
print("=" * 70)
print("Part C: spectral identities")
print("=" * 70)
A = rprob(6); k0 = rprob(6)
ell0 = np.log(k0 / A)
Z = lambda a: np.sum(A * np.exp(a * ell0))
Lam0 = lambda b: np.log(Z(b))
def kap_t(t):
    a = np.exp(-t); w = A * np.exp(a * ell0); return w / w.sum()
def Hent(t):
    k = kap_t(t); return np.sum(k * np.log(k / A))
worst = 0.0
for t in [0.1, 0.5, 1.3, 3.0]:
    hh = 1e-5
    dH = (Hent(t + hh) - Hent(t - hh)) / (2 * hh)
    k = kap_t(t); l = np.log(k / A)
    var = np.sum(k * l ** 2) - np.sum(k * l) ** 2
    worst = max(worst, abs(dH + var))
print(f"[C1] smooth:entropy-rate dH/dt + Var_kappa(ell): max = {worst:.2e} (central diff, h=1e-5)")
t = 0.8; a = np.exp(-t)
ellt = np.log(kap_t(t) / A)
rec = ellt / a - np.log(np.sum(A * np.exp(ellt / a)))
print(f"[C2] smooth:spectrum-inverse: max |recovered - ell0| = {np.max(np.abs(rec - ell0)):.2e}")
worst = 0.0
for b in [0.1, 0.4, 0.9, 1.0]:
    Lt = np.log(np.sum(A * np.exp(b * ellt)))
    worst = max(worst, abs(Lt - (Lam0(a * b) - b * Lam0(a))))
print(f"[C3] smooth:cgf-flow Lambda^t(b) = Lambda^0(ab) - b Lambda^0(a): max = {worst:.2e}")
m0, M0 = ell0.min(), ell0.max()
ok = all((np.log(kap_t(tt) / A).min() >= m0 - 1e-14) and (np.log(kap_t(tt) / A).max() <= M0 + 1e-14) for tt in np.linspace(0, 5, 51))
width_ratio = max(abs((np.ptp(np.log(kap_t(tt) / A)) / np.ptp(ell0)) - np.exp(-tt)) for tt in np.linspace(0, 5, 51))
print(f"[C4] range stays in [m0,M0]: {ok}; width ratio vs e^-t: max dev = {width_ratio:.2e}")
Hd = max(abs(Hent(tt) - (np.exp(-tt) * (np.sum(A * ell0 * np.exp(np.exp(-tt) * ell0)) / Z(np.exp(-tt))) - Lam0(np.exp(-tt)))) for tt in [0.2, 1.0, 2.5])
print(f"[C5] H(t) = a Lambda0prime(a) - Lambda0(a): max = {Hd:.2e}")
b = 0.37
Dren = np.log(np.sum(k0 ** b * A ** (1 - b))) / (b - 1)
print(f"[C6] smooth:renyi Lambda(b) - (b-1) D_b: {Lam0(b) - (b - 1) * Dren:.2e}  (D_b = {Dren:.4f} >= 0)")

print()
print("=" * 70)
print("Part D: smooth:channel-variants")
print("=" * 70)
MA.chan = "fixedA"
worst = 0.0
for trial in range(2):
    x = rand_state(MA, "fine")
    q, s, be, ga = split_fine(MA, x)
    gf = cs_grad(MA.fine_energy, x)
    VF = MA.unpack(fisher_logvel(x, gf, MA.fine_shapes()), MA.fine_shapes())
    for i in range(MA.N):
        mu = q[i].sum(1); kap = q[i] / mu[:, None]; ell = np.log(kap / MA.Aq[i][None, :])
        H = kl_rows(kap, MA.Aq[i])
        dmu = np.sum(q[i] * VF[i], axis=1) / mu
        dkap = VF[i] - dmu[:, None]
        worst = max(worst, np.max(np.abs(dkap - 2 * (H[:, None] - ell))))
print(f"[D1] fixed reconstruction B (x) A: |dlog kappa - 2(H - ell)| max = {worst:.2e}")
MA.chan = "recon"
k1 = np.array([0.75, 0.25])
def peer(k1, k2):
    return np.sum(k1 * np.log(k1 / k2))
print(f"[D2] peer cost KL((3/4,1/4)||(1/4,3/4)) = {peer(k1, k1[::-1]):.12f}; (1/2)log3 = {0.5*np.log(3):.12f}")
def dlogk1(k1, k2):
    A = np.array([0.5, 0.5])
    h = np.log(k1 / A) + np.log(k1 / k2) - k2 / k1
    return -(h - np.sum(k1 * h))
v_same, v_swap = dlogk1(k1, k1), dlogk1(k1, k1[::-1])
ell = np.log(k1 / 0.5)
sv_same = np.sum(0.5 * 2 * ell * v_same); sv_swap = np.sum(0.5 * 2 * ell * v_swap)
print(f"[D2] d/dt int f(ell) dA with f=r^2: same-kappa {sv_same:.6f}, swapped-kappa {sv_swap:.6f} (same spectra, different velocity)")
print(f"\ntotal time {time.time()-t_start:.1f}s")
