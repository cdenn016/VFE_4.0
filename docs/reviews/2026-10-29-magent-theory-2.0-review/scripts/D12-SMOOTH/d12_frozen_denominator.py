r"""D12-SMOOTH check 6: proof of smooth:peer-cancellation-rule, 07:87,
"Holding kappa_i fixed only in the denominator would violate that identity and would define
another force."  Compare, at the same state, the complex-step gradient of the fine energy with
the live receiver conditional in K_ij and with kappa_i frozen at its current value. First
order: dF/dkappa_frozen = -int (dq_i/dnu)(kappa_i/kappa_frozen) d(delta kappa) = 0 at matching
values, so the instantaneous force should coincide; second derivatives (curvature) and the
force after displacement from a held snapshot should differ.
"""
import numpy as np

src = open(__file__.replace("d12_frozen_denominator.py", "d12_closure_flow.py")).read()
exec(src.split('print("=" * 70)\nprint("Part A')[0])

M = Model(N=3, nZ=3, nV=4, nW=2, nU=3, Tq=0.7, Ts=0.9)
x = rand_state(M, "fine")
q, s, be, ga = split_fine(M, x)
kap_frozen = [qi / qi.sum(1)[:, None] for qi in q]
vk_frozen = [si / si.sum(1)[:, None] for si in s]


def energy_frozen(xx, kq=kap_frozen, ks=vk_frozen):
    N = M.N
    T = M.unpack(xx, M.fine_shapes())
    qq, ss, bb, gg = T[:N], T[N:2 * N], T[2 * N:3 * N], T[3 * N:]
    mu = [qi.sum(1) for qi in qq]; sg = [si.sum(1) for si in ss]
    E = 0.0
    for i in range(N):
        P = qq[i][:, :, None, None] * ss[i][None, None, :, :]
        R = (M.Gc[i].T)[:, None, :, None] * M.rc[i][None, None, :, None]
        R = R * M.Aq[i][None, :, None, None] * M.As[i][None, None, None, :]
        E = E + np.sum(P * np.log(P / R))
        for a, j in enumerate(M.src[i]):
            K4 = np.broadcast_to(M.Bq(i, j, mu[i], mu[j])[:, None, :, None] * kq[i][None, None, :, :], (M.nZ, M.nV, M.nZ, M.nV))
            K4s = np.broadcast_to(M.Bs(i, j, sg[i], sg[j])[:, None, :, None] * ks[i][None, None, :, :], (M.nW, M.nU, M.nW, M.nU))
            nu = np.einsum("ab,abcd->cd", qq[j], K4); nus = np.einsum("ab,abcd->cd", ss[j], K4s)
            E = E + bb[i][a] * np.sum(qq[i] * np.log(qq[i] / nu)) + gg[i][a] * np.sum(ss[i] * np.log(ss[i] / nus))
        E = E + M.Tq * np.sum(bb[i] * np.log(bb[i] / M.piq[i])) + M.Ts * np.sum(gg[i] * np.log(gg[i] / M.pis[i]))
    return E


g_live = cs_grad(M.fine_energy, x)
g_frz = cs_grad(energy_frozen, x)
v_live = fisher_logvel(x, g_live, M.fine_shapes())
v_frz = fisher_logvel(x, g_frz, M.fine_shapes())
print(f"instantaneous Fisher velocity, live vs frozen-at-current kappa: max |diff| = {np.max(np.abs(v_live - v_frz)):.2e}")
# Hessian along a tangent direction d (mass-preserving within each table)
d = np.zeros_like(x); k = 0
rng2 = np.random.default_rng(5)
for sh in M.fine_shapes():
    n = int(np.prod(sh)); w = rng2.normal(size=n); w -= w.mean(); d[k:k + n] = w * x[k:k + n]; k += n
eps = 1e-4
H_live = (cs_grad(M.fine_energy, x + eps * d) - cs_grad(M.fine_energy, x - eps * d)) / (2 * eps)
H_frz = (cs_grad(energy_frozen, x + eps * d) - cs_grad(energy_frozen, x - eps * d)) / (2 * eps)
print(f"second variation along a tangent direction: |d.H_live.d| = {abs(d @ H_live):.4f}, |d.H_frozen.d| = {abs(d @ H_frz):.4f}, diff = {abs(d @ H_live - d @ H_frz):.2e}")
# force after displacement with the snapshot held
x2 = x + 0.05 * d
x2 = np.concatenate([p.ravel() / p.sum() for p in M.unpack(x2, M.fine_shapes())])
v2_live = fisher_logvel(x2, cs_grad(M.fine_energy, x2), M.fine_shapes())
v2_frz = fisher_logvel(x2, cs_grad(energy_frozen, x2), M.fine_shapes())
print(f"after displacement with the snapshot held: max |diff| = {np.max(np.abs(v2_live - v2_frz)):.2e}")
