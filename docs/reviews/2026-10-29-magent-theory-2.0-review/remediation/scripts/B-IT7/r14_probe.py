r"""B-IT7 probe (log evidence only): states of the standard's counterfactual class for Item 7.
Trajectories of the full flow from widened initial states u (group spread scaled by f) along which the
first-stage event has not occurred by t = 2.3 (last actual handoff on the 0.05 grid); the four groups are then
handed off at that time and the left parent gap is compared with d_u + 0.005 = 0.968."""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
from controls import SCtl, o, y0, lam, rho, ps, g, delta, Delta
cent = np.repeat(o.reshape(4, 3).mean(1), 3)
for f in (1.0, 1.5, 2.0, 3.0):
    u = cent + f * (y0 - cent)
    P = SCtl(o, lam, rho, ps, g, delta, Delta, 2.0, mode="full"); P.y_init = u.copy()
    y = P.run(u, 2.3)
    pools = y.reshape(4, 3).mean(1); spread = np.ptp(y.reshape(4, 3), 1)
    print(f"f={f}: events by 2.3 {[(t, len(b)) for t, _, b in P.hist]}; left gap at 2.3 {pools[1]-pools[0]:.4f};"
          f" within-group ranges {np.round(spread, 3)}")
