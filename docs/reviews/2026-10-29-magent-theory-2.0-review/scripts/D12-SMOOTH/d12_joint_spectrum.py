r"""D12-SMOOTH check 4: joint spectrum for private-transmitting identity channels.

smooth:channel-variants (07:257) suggests the joint law Xi of v -> (ell_{k,z}(v))_{k,z} under a
common private reference A as the retained state for identity private channels and says its
closure is not established; smooth:formation-open (07:321) says a coupled architecture with an
exact compressing completion "is not available". Here the full channel is
K_ij((z1,v1),(z,v)) = B_ij[mu](z1,z) delta_{v1}(v), so private data enter every peer cost,
row and public force. Claim tested: the fine velocity at a private point v is a function of
the vector (ell_{k,z}(v))_{k,z} and of A-integrals, so Xi (with public laws and rows) closes.
Configurations (3 agents, public bit, V = 4 points, A = (0.3,0.2,0.1,0.4), Fisher rows):
  X : ell_{k,z} = log(ex, ew, ex, ey) for every agent k and public value z,
  X2: the same with ex and ey exchanged for ALL agents at once  -> same Xi, different laws,
  X3: exchanged for agent 0 only -> same per-agent spectra, different Xi.
"""
import numpy as np

src = open(__file__.replace("d12_joint_spectrum.py", "d12_closure_flow.py")).read()
exec(src.split('print("=" * 70)\nprint("Part A')[0])

A = np.array([0.3, 0.2, 0.1, 0.4])
N = 3
M = Model(N=N, nZ=2, nV=4, nW=2, nU=1, Tq=0.6, Ts=0.8, Aq=[A] * N)
M.chan = "privid"
rng3 = np.random.default_rng(11)


def logvec(ex, ey):
    ew = (1 - 0.4 * ex - 0.4 * ey) / 0.2
    return np.log(np.array([ex, ew, ex, ey])), np.log(np.array([ey, ew, ey, ex]))


L, Lsw = {}, {}
for kk in range(N):
    for z in range(2):
        ex, ey = rng3.uniform(0.6, 1.6), rng3.uniform(0.4, 1.0)
        L[(kk, z)], Lsw[(kk, z)] = logvec(ex, ey)
mu0 = [rprob(2) for _ in range(N)]
sg0 = [rprob(2) for _ in range(N)]
ks0 = [np.ones((2, 1)) for _ in range(N)]
be0 = [rprob(N - 1) for _ in range(N)]
ga0 = [rprob(N - 1) for _ in range(N)]


def kq_from(Ld, swap_agents):
    out = []
    for kk in range(N):
        rows = []
        for z in range(2):
            l = Lsw[(kk, z)] if kk in swap_agents else Ld[(kk, z)]
            rows.append(A * np.exp(l))
        out.append(np.array(rows))
    return out


xX = build_state(M, kq_from(L, []), ks0, mu0, sg0, be0, ga0)
xX2 = build_state(M, kq_from(L, [0, 1, 2]), ks0, mu0, sg0, be0, ga0)
xX3 = build_state(M, kq_from(L, [0]), ks0, mu0, sg0, be0, ga0)


def joint_spectrum(x):
    q, s, be, ga = split_fine(M, x)
    vecs = np.array([[np.log(q[kk][z, v] / q[kk][z].sum() / A[v]) for kk in range(N) for z in range(2)] for v in range(4)])
    atoms = {}
    for v in range(4):
        key = tuple(np.round(vecs[v], 9))
        atoms[key] = atoms.get(key, 0.0) + A[v]
    return sorted((k, round(m, 12)) for k, m in atoms.items())


def per_agent_spectra(x):
    q, s, be, ga = split_fine(M, x)
    out = []
    for kk in range(N):
        for z in range(2):
            l = np.log(q[kk][z] / q[kk][z].sum() / A)
            d = {}
            for r, m in zip(np.round(l, 9), A):
                d[r] = d.get(r, 0.0) + m
            out.append(sorted((r, round(m, 12)) for r, m in d.items()))
    return out


print("t = 0: joint spectra X == X2:", joint_spectrum(xX) == joint_spectrum(xX2), "; X == X3:", joint_spectrum(xX) == joint_spectrum(xX3))
print("t = 0: per-agent spectra X == X2 == X3:", per_agent_spectra(xX) == per_agent_spectra(xX2) == per_agent_spectra(xX3))
print("t = 0: fine laws X vs X2 max |diff|:", f"{np.max(np.abs(xX - xX2)):.3f}")
gX = cs_grad(M.fine_energy, xX); gX3 = cs_grad(M.fine_energy, xX3)
vX = coarse_from_fine(M, xX * np.exp(0)); 
vcX = coarse_from_fine(M, xX * fisher_logvel(xX, gX, M.fine_shapes()))
vcX3 = coarse_from_fine(M, xX3 * fisher_logvel(xX3, gX3, M.fine_shapes()))
print(f"t = 0: public+row velocity X vs X3 max |diff| = {np.max(np.abs(vcX - vcX3)):.3e} (private data feed back)")
fX = fine_traj(M, xX, 0.5, 0.01)
fX2 = fine_traj(M, xX2, 0.5, 0.01)
fX3 = fine_traj(M, xX3, 0.5, 0.01)
cX, cX2, cX3 = (coarse_from_fine(M, f) for f in (fX, fX2, fX3))
print(f"T = 0.5: public laws and rows  max |X - X2| = {np.max(np.abs(cX - cX2)):.2e}   max |X - X3| = {np.max(np.abs(cX - cX3)):.2e}")
print("T = 0.5: joint spectra X == X2:", joint_spectrum(fX) == joint_spectrum(fX2))
q, s, be, ga = split_fine(M, fX)
print("T = 0.5: ell(v=1) == ell(v=3) preserved in X:", all(abs(np.log(q[kk][z, 0] / A[0]) - np.log(q[kk][z, 2] / A[2])) < 1e-9 for kk in range(N) for z in range(2)))
print("T = 0.5: fine laws X vs X2 still differ, max |diff| =", f"{np.max(np.abs(fX - fX2)):.3f}")
