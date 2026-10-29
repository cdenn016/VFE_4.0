r"""D25-CRIT check 10 (extension): exact sixth-order coefficient of the stationary potential
Phi(s, T_c) = G(X(s), s) + T_c h(s) of it2:critical-pitchfork, from the exact jets X'(0), X''(0),
X'''(0) of d25_pitchfork_exact.json (truncation error of X is O(s^4), hence O(s^8) in Phi because
G_x(X(s), s) = 0).  Gives the branch temperature T_b(s) = T_c - Phi_s(s,T_c)/h'(s)
= T_c - (c/4) s^2 + e4 s^4 + O(s^6) and the leading-order fold estimate s_f^2 = c/(8 e4).
Output: d25_sixth_order.json."""
import json, time
from pathlib import Path
import sympy as sp
from d25_model import A_FUN, DELTA, H_ENT, XS, XSTAR, T_C_CLAIM, s
OUT = Path(__file__).resolve().parent
t0 = time.perf_counter()
A = json.loads((OUT / "d25_pitchfork_exact.json").read_text(encoding="utf-8"))
p, X2, X3 = ([sp.Rational(z) for z in A[k]] for k in ("p", "X2", "X3"))
c = sp.Rational(A["c"])
Tc = T_C_CLAIM
curve = {z: XSTAR[i] + p[i]*s + X2[i]*s**2/2 + X3[i]*s**3/6 for i, z in enumerate(XS)}
Gs = (A_FUN + s*DELTA).subs(curve)
ser = sp.series(Gs, s, 0, 7).removeO()
hs = sp.series(H_ENT, s, 0, 7).removeO()
phi = sp.expand(ser + Tc*hs)
coef = [sp.nsimplify(phi.coeff(s, k)) for k in range(7)]
Phi6 = coef[6]*720
Phi4 = coef[4]*24
e4 = c/3 - Phi6/480
res = {"Phi_ssss_check_equals_6c": bool(Phi4 == 6*c), "Phi_ssssss": str(Phi6), "Phi_ssssss_float": float(Phi6),
       "odd_coefficients_zero": [str(coef[k]) for k in (1, 3, 5)], "s2_coefficient": str(coef[2]),
       "e4_in_Tb": str(e4), "e4_float": float(e4),
       "fold_estimate_s_f": float(sp.sqrt(c/(8*e4))) if (c/(8*e4)) > 0 else None,
       "fold_estimate_T_f_minus_T_c": float(-(c/4)*(c/(8*e4)) + e4*(c/(8*e4))**2) if (c/(8*e4)) > 0 else None,
       "wall_seconds": round(time.perf_counter()-t0, 2)}
(OUT / "d25_sixth_order.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
print(json.dumps(res, indent=1))
