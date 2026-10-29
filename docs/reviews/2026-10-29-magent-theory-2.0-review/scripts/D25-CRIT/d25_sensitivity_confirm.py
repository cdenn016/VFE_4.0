r"""D25-CRIT check 9b: confirm two sign flips of check 9 with smaller steps (s = 0.005, 0.01) and a
direct check that, for supercritical data, small asymmetric equilibria exist BELOW T_c and are
local minima of the full potential (T_b(s) decreasing).  Float64.  Output: d25_sensitivity_confirm.json."""
import json
import numpy as np
import d25_sensitivity as m   # re-runs check 9 (0.5 s), then reuses its lambdified functions
out = {}
for name, prm in (("A_omega=1.0", (1.0, 1.25, 23/15, 1.0)), ("P0=1.5", (1.5, 1.25, 23/15, 0.75)),
                  ("published", (1.0, 1.25, 23/15, 0.75))):
    x, r = m.newton(m.x_pub.copy(), 0.0, prm)
    H = np.asarray(m.f_gxx(x, 0.0, *prm), dtype=float)
    g = np.asarray(m.f_g(x, *prm), dtype=float).ravel()
    Tc = float(g @ np.linalg.solve(H, g) / 4)
    est, Tb = [], []
    for sv in (0.005, 0.01, 0.05, 0.1):
        xx = x.copy()
        for z in np.linspace(0, sv, 41)[1:]:
            xx, _ = m.newton(xx, z, prm)
        L = np.log((1 + 2*sv)/(1 - 2*sv))
        est.append((m.f_dl(xx, *prm) + Tc*L)/sv**3)
        Tb.append(float(-m.f_dl(xx, *prm)/L))
    out[name] = {"T_c": Tc, "c_richardson_0.005_0.01": float((4*est[0]-est[1])/3),
                 "Tb(s)-T_c at s=0.005,0.01,0.05,0.1": [t - Tc for t in Tb]}
json.dump(out, open("d25_sensitivity_confirm.json", "w"), indent=1)
print(json.dumps(out, indent=1))
