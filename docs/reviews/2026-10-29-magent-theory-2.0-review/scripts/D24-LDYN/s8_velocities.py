"""D24-LDYN script 8: equal dictionary derivative rows (plug-in) but different Fisher velocities
(it2:per-agent-flow (iii) and the remark after it); law-level rows and velocities differ further."""
import numpy as np
from tangent_lib import Model
rng = np.random.default_rng(99)
edges = [(1, 0), (0, 1), (2, 1), (0, 2), (2, 0)]
M = Model(3, 2, edges, rng, sigma2=1.1, lam=0.9, ranks=[1, 2, 1, 1, 2])
J, eta = M.global_J_eta()
D = np.zeros_like(J)
for bl in M.blocks:
    D[np.ix_(bl, bl)] = J[np.ix_(bl, bl)]
Hp, bp, Ap = M.per_agent_H_b('plug')
Sg, Vg, it = M.law_level_fixed_point(0.0)
Hl, bl_, Al = M.per_agent_H_b('law', Sg, Vg)
wcols = np.concatenate(M.iw)
mu = rng.normal(size=M.n)
gG, gP, gL = J @ mu - eta, Hp @ mu - bp, Hl @ mu - bl_
vG, vP, vL = -np.linalg.solve(D, gG), -np.linalg.solve(Ap, gP), -np.linalg.solve(Al, gL)
print("dictionary derivative rows: |gG - gP| =", np.abs(gG[wcols] - gP[wcols]).max(), " |gG - gL| =", np.abs(gG[wcols] - gL[wcols]).max())
print("dictionary velocities:      |vG - vP| =", np.abs(vG[wcols] - vP[wcols]).max(), " |vG - vL| =", np.abs(vG[wcols] - vL[wcols]).max(), " (scale |vG| =", np.abs(vG[wcols]).max(), ")")
print("mobility blocks differ: max over agents |J_ii - A_i| =", max(np.abs(J[np.ix_(bl, bl)] - Ap[np.ix_(bl, bl)]).max() for bl in M.blocks))
# agent without outgoing edges: global and plug-in mobilities coincide
M2 = Model(3, 2, [(1, 0), (2, 0), (2, 1)], rng, sigma2=1.0, lam=0.5)
J2, eta2 = M2.global_J_eta()
H2, b2, A2 = M2.per_agent_H_b('plug')
print("agent 0 (no outgoing edge): |J_00 - A_0| =", np.abs(J2[np.ix_(M2.blocks[0], M2.blocks[0])] - A2[np.ix_(M2.blocks[0], M2.blocks[0])]).max(),
      "; agent 2 (sender only): |J_22 - A_2| =", np.abs(J2[np.ix_(M2.blocks[2], M2.blocks[2])] - A2[np.ix_(M2.blocks[2], M2.blocks[2])]).max())
