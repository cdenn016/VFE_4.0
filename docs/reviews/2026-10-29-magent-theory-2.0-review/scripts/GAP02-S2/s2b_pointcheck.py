"""GAP02-S2 s2b: float64 point evaluation of the s2 certificate quantities on fine grids (sanity check of
sharpness; not a proof). Prints true minima of k - |f_a|/a on the face b = eps a/4, of Bdot near B = 1/10,
and of the first-pair weight and contrast on Omega."""
import numpy as np

def core(a, b):
    w = np.exp(-2*b*b); v0 = np.exp(-2*a*a); vp = np.exp(-2*(a+b)**2); vm = np.exp(-2*(a-b)**2)
    sp, sm = v0+vp, v0+vm; Zp, Zm = 1+w+sp, 1+w+sm
    cp = w*(1/Zp+1/Zm); c0 = v0*(1/Zp+1/Zm); Gp, Gm = (a+b)*vp/Zp, (a-b)*vm/Zm
    fa = -a-2*a*c0-2*(Gp+Gm); fb = -b-2*b*cp-2*(Gp-Gm)
    B = (sp/Zp+sm/Zm)/2
    dBa = ((1+w)*(-4*a*v0-4*(a+b)*vp)/Zp**2 + (1+w)*(-4*a*v0-4*(a-b)*vm)/Zm**2)/2
    dBb = ((-4*(a+b)*vp*(1+w)+4*b*w*sp)/Zp**2 + (4*(a-b)*vm*(1+w)+4*b*w*sm)/Zm**2)/2
    return fa, fb, B, dBa*fa+dBb*fb, w/Zp, w/Zm, (w-sp)/Zp, (w-sm)/Zm

a_s = np.sqrt(np.log(2))
for eps in (1.0, 0.25):
    a = np.linspace(a_s, 4, 200001); b = eps*a/4
    fa, fb, *_ = core(a, b)
    k = -fb/b; marg = k - (-fa)/a
    i = np.argmin(marg)
    print(f"eps={eps}: min (k - |f_a|/a) on face = {marg[i]:.6f} at a = {a[i]:.4f}")
    A, Bq = np.meshgrid(np.linspace(a_s, 4, 1601), np.linspace(0, 1, 401), indexing="ij")
    Bq = Bq*eps*A/4
    fa, fb, B, Bd, pp, pm, cpp, cpm = core(A, Bq)
    near = np.abs(B-0.1) < 1e-3
    print(f"   min Bdot on |B-1/10|<1e-3: {Bd[near].min():.6f}; crossing a range [{A[near].min():.5f}, {A[near].max():.5f}]")
    print(f"   min first-pair weight {min(pp.min(), pm.min()):.6f}, min contrast {min(cpp.min(), cpm.min()):.6f}")
