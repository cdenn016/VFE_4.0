r"""Exact string replacements in the chapter 15 continuum witness (each must occur once)."""
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8").read()
R = [
    (r"recovers the chosen pointwise reciprocal Fisher flow.",
     r"recovers the pointwise restricted-family flow (\cref{def:dyn-restricted-flow}) of the chosen reciprocal potential."),
    (r"Take $C=S^1$ with measure", r"Take $\mathcal C=S^1$ with measure"),
    (r""" V(P)=2+\tfrac12\langle P,PH\rangle_F,\qquad
 H=\begin{pmatrix}""", r""" V(P)=2+\tfrac12\langle P,P\mathsf H\rangle_F,\qquad
 \mathsf H=\begin{pmatrix}"""),
    (r" 2+\tfrac12\langle P,PH\rangle_F+", r" 2+\tfrac12\langle P,P\mathsf H\rangle_F+"),
    (r" \boxed{P_t=-PH+\lambda D_xD_xP,\qquad \alpha=\alpha_*.}", r" \boxed{P_t=-P\mathsf H+\lambda D_xD_xP,\qquad \alpha=\alpha_*.}"),
    (r"The vector field in this particular symmetric example is horizontal: $H=H^T$ gives $\langle PH,JP\rangle_F=0$, and differentiating $\langle D_xP,JP\rangle_F=0$ covariantly gives $\langle D_xD_xP,JP\rangle_F=0$.",
     r"The vector field in this particular symmetric example is horizontal: $\mathsf H=\mathsf H^T$ gives $\langle P\mathsf H,JP\rangle_F=0$, and differentiating $\langle D_xP,JP\rangle_F=0$ covariantly gives $\langle D_xD_xP,JP\rangle_F=0$. The same degeneracy matters for computation in these law coordinates: the vertical component of the right side is a cancellation among the terms containing $\alpha$, and it is not damped, so a discretization that does not preserve this cancellation exactly can grow off the horizontal family. Integration in the adapted chart, or exact computation of the vertical part, avoids this, and $\langle P_t,JP\rangle_F=0$ is a check."),
    (r""" \tau=\frac{\sqrt5-1}{2},\qquad
 \kappa=5-\tau=\frac{11-\sqrt5}{2},\qquad
 v=(1,\tau,-1,-\tau)^T,\qquad E_0=e_1v^T.""", r""" \varsigma=\frac{\sqrt5-1}{2},\qquad
 \kappa=5-\varsigma=\frac{11-\sqrt5}{2},\qquad
 v=(1,\varsigma,-1,-\varsigma)^T,\qquad E_0=e_1v^T."""),
    (r"The identity $\tau^2+\tau=1$ gives $Hv=\kappa v$ and", r"The identity $\varsigma^2+\varsigma=1$ gives $\mathsf Hv=\kappa v$ and"),
    (r"It is pure gauge; a one-dimensional base gives no curvature witness.",
     r"It is pure gauge. On a one-dimensional base the curvature vanishes identically, and the gauge-invariant content of $A_*$ is its holonomy around the base circle, $\exp(J\oint\alpha_*\mathrm dx)$; here $\oint\alpha_*\mathrm dx=-\oint\varphi_x\mathrm dx=0$ for every periodic phase, so the holonomy is trivial."),
    (r"The existence for every finite time in this explicit sector is a property of its formula. It does not turn the general local theorem into a global theorem, and singleton source rows do not test the nontrivial row part of that theorem.",
     r"The existence for every finite time in this explicit sector is a property of its formula. It does not turn the general local theorem into a global theorem. The witness is the linear radial sector of the system: along the ansatz the shape moves on one ray of the quotient, where the quotient metric is flat and the Christoffel terms vanish; the references are rotation invariant, so $V_r=0$ and the nonbasic correction is absent; $\varphi_t=0$, so there is no frame dynamics; the connection is pure gauge; and the singleton source rows have no row dynamics. It therefore tests none of the nonlinear, frame, nonbasic, or row parts of the theorem, which the fixed-reference example above exercises only in its chart equations."),
]
for a, b in R:
    n = s.count(a)
    if n != 1:
        sys.exit(f"count {n} for: {a[:70]}")
    s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
print("ok")
