r"""Reuse and shared-label laws at t = 0 for the depth-2 family (definitions taken
from depth2_flow_check.py), to quantify eq:boundary-tree-reuse before the contrast
decays.  Phase rotation by pi/2 at node L; the node-L descriptor (unordered pair)
is unchanged by construction."""
import os
import numpy as np
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "depth2_flow_check.py")).read()
pre, post = src.split("# ---------------- set up ----------------")
exec(pre)
setup = post.split('print("== 1.')[0]
exec(setup)
phase_q2 = [phase_q[0], phase_q[1] + 1, phase_q[2]]
bits_q2 = path_bits(phase_q2, nr)
Qa = lift(f, bits_q); Qb = lift(f, bits_q2)
def reuse(Q0, Q1):
    pa, pb = Q0 * refq, Q1 * refq
    return 0.5 * (np.multiply.outer(pa, pa) + np.multiply.outer(pb, pb))
print("t=0 TV(reuse two-draw law, rotated):", 0.5 * np.abs(reuse(Qa[0], Qa[1]) - reuse(Qb[0], Qb[1])).sum())
ma = (Qa[0] + Qa[1]) / 2 * refq; mb = (Qb[0] + Qb[1]) / 2 * refq
print("t=0 TV(fresh two-draw law, rotated):", 0.5 * np.abs(np.multiply.outer(ma, ma) - np.multiply.outer(mb, mb)).sum())
Sd = lift(g, bits_s)
J1 = sum(np.multiply.outer(Qa[i] * refq, Sd[i] * refs) for i in range(N)) / 4
J2 = sum(np.multiply.outer(Qb[i] * refq, Sd[i] * refs) for i in range(N)) / 4
print("t=0 TV(shared belief/model label law, belief-rotated):", 0.5 * np.abs(J1 - J2).sum())
print("t=0 TV(child LL law, rotated):", 0.5 * np.abs((Qa[0] - Qb[0]) * refq).sum())
