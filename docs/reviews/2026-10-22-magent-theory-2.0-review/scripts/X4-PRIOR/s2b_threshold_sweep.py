r"""X4-PRIOR: threshold sweep of the chapter 12 limit-field events (six agents, A=256, R=1000A).
For each (eps, delta) report the pair-event time, the parent-event time, and whether the third pair ever
qualifies to join a core block before the parent event."""
import numpy as np
from scipy.integrate import solve_ivp
import s2_open_formation_scales as m

A, R = 256.0, 256000.0
u0 = np.array([-A - 2, -A + 2, A - 2, A + 2, R - 2, R + 2])
tmax = np.log(A) + 6
ts = np.linspace(0, tmax, 3001)
sol = solve_ivp(m.field, (0, tmax), u0, t_eval=ts, rtol=1e-10, atol=1e-12, method='LSODA')
for eps, dl in [(0.1, 0.05), (0.02, 0.01), (0.05, 0.02), (0.15, 0.05), (0.2, 0.1), (0.24, 0.05)]:
    m.EPS, m.DEL = eps, dl
    pt = qt = None; bad = False
    for k, t in enumerate(sol.t):
        W = m.weights(sol.y[:, k])
        if pt is None:
            singles = [[i] for i in range(6)]
            if all(m.qualifies(W, singles, singles[a], singles[b]) for a, b in ((0, 1), (2, 3), (4, 5))):
                pt = t
        else:
            blocks = [[0, 1], [2, 3], [4, 5]]
            if m.qualifies(W, blocks, blocks[0], blocks[2]) or m.qualifies(W, blocks, blocks[1], blocks[2]):
                bad = True
            if m.qualifies(W, blocks, blocks[0], blocks[1]):
                qt = t; break
    print("eps=%.3f delta=%.3f  t_pair=%s  t_parent=%s  third-pair join before parent: %s" % (eps, dl, None if pt is None else round(pt, 3), None if qt is None else round(qt, 3), bad))
