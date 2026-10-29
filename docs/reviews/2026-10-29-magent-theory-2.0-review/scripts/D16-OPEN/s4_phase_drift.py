r"""D16-OPEN s4: do anchor phases stay constant on the product reference family?

open:loss-information states: "On the product reference family the phases are constant".
The product reference family (open:product-reference) has ARBITRARY positive angular profiles,
whose factors evolve by f_t = f_0^{rho}/norm, rho = e^{-2t} (open:angular-flow).
For a von Mises factor the phase is fixed.  For an asymmetric profile, arg E_{f_t} e^{i theta}
moves with rho.  We also check the anchor moment can pass close to zero (regular-chart caveat),
and the rotation inertia E_f[(d_theta log f)^2] used for the per-record Fisher information.
"""
import numpy as np

M = 4096
th = 2 * np.pi * np.arange(M) / M


def stats(logf0, rho):
    f = np.exp(rho * logf0); f /= f.mean()
    u = (f * np.exp(1j * th)).mean()
    dl = rho * np.gradient(logf0, th[1] - th[0])
    inertia = (f * dl ** 2).mean()
    return np.angle(u), abs(u), inertia


out = []
# von Mises: phase fixed
lvm = 1.7 * np.cos(th - 0.9)
ph = [stats(lvm, r)[0] for r in (1, 0.5, 0.1, 0.01)]
out.append("von Mises (kappa=1.7, chi=0.9): phase at rho=1,0.5,0.1,0.01: %s" % np.round(ph, 12))
# asymmetric two-harmonic profile
l2 = 1.0 * np.cos(th) + 2.0 * np.cos(2 * th - 1.0)
rows = []
for r in (1.0, 0.7, 0.5, 0.3, 0.1, 0.03, 0.01):
    p, a, I = stats(l2, r)
    rows.append((r, p, a, I))
out.append("profile log f0 = cos(theta) + 2 cos(2 theta - 1):")
for r, p, a, I in rows:
    out.append("   rho=%.2f  t=%.4f  phase=%+.6f  |u|=%.4e  rotation inertia=%.4e" % (r, -np.log(r) / 2, p, a, I))
out.append("   phase change between t=0 and t=1.15 (rho=0.1): %.4f rad" % (rows[4][1] - rows[0][1]))
# profile whose first moment changes sign along the flow (anchor moment through zero)
l3 = -0.3 * np.cos(th) + 2.5 * np.cos(2 * th) + 1.2 * np.cos(3 * th)
vals = []
for r in np.linspace(1, 0.01, 400):
    f = np.exp(r * l3); f /= f.mean()
    vals.append(((f * np.exp(1j * th)).mean().real, r))
vals = np.array(vals)
sgn = np.where(np.diff(np.sign(vals[:, 0])) != 0)[0]
out.append("profile log f0 = -0.3cos + 2.5cos2 + 1.2cos3 (axially symmetric, u real): Re u at rho=1: %.4e, at rho=0.01: %.4e;"
           " sign changes at rho ~ %s (anchor moment vanishes at a finite time)" %
           (vals[0, 0], vals[-1, 0], np.round(vals[sgn, 1], 3)))
print("\n".join(out))
