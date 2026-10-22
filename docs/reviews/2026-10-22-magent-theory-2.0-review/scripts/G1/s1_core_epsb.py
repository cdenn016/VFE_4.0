r"""G1 s1: late core comparison of prop open:routing (chapter 12, lines 163-207).

Checks
  A. open:core against the four-agent limit field open:limit (float64).
  B. b = 0 comparison constants: a_c = sqrt(log 9 / 2), a_* = sqrt(log 2), S, dB/dt at a_c.
  C. Float64 grid scan of an invariant-region certificate for the within-pair half-gap b
     at the start of the late comparison (a = 4).  Region
         Omega = {a_* <= a <= 4, 0 <= b <= beta(a)},  beta(a) = eps (a/4)^m.
     Conditions:
       (I)   inward on b = beta(a):  k(a,b) > m |f_a|/a,  k = -f_b/b
       (III) dB/dt > 0 on every point of Omega where the parent row B equals 1/10
       (IV)  B(a_*, b) > 1/10 for 0 <= b <= beta(a_*)
       (V)   first-pair weights p_pm >= 1/10 + s and contrasts (w - s_pm)/Z_pm >= 1/20 + s
     (f_a < 0, indeed f_a <= -a, holds identically; b = 0 is invariant.)
  D. Interval verification (numpy outward-padded interval arithmetic, mean-value form for
     G_+ - G_-) of (I), (III), (IV), (V) for the selected (eps, m).
  E. Comparison with the open:small-gap bound b(t_A) <= 2 (A/4)^(-1/4) and the sharpened
     bound b(t_A) <= 2 (A/4)^(-(1-68 e^-8)/2).

Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 20 python s1_core_epsb.py
"""
from __future__ import annotations

import math

import numpy as np

SLACK = 6e-3  # margin reserved for third-pair exterior (< 1e-3) and later perturbations


# ---------------------------------------------------------------- A. core vs limit field
def limit_field(u: np.ndarray) -> np.ndarray:
    r"""\dot u_i = -u_i + \sum_j (w_ij + w_ji)(u_j - u_i), w_ij = n_ij / sum_k n_ik."""
    x = u[None, :] - u[:, None]  # x[i, j] = u_j - u_i
    n = np.exp(-0.5 * x**2)
    w = n / n.sum(axis=1, keepdims=True)
    return -u + ((w + w.T) * x).sum(axis=1)


def core_field(a: float, b: float) -> tuple[float, float]:
    r"""open:core: (\dot a, \dot b) for u = (-a-b, -a+b, a-b, a+b)."""
    w = math.exp(-2 * b * b)
    v0 = math.exp(-2 * a * a)
    vp = math.exp(-2 * (a + b) ** 2)
    vm = math.exp(-2 * (a - b) ** 2)
    Zp = 1 + w + v0 + vp
    Zm = 1 + w + v0 + vm
    cp = w * (1 / Zp + 1 / Zm)
    c0 = v0 * (1 / Zp + 1 / Zm)
    Gp = (a + b) * vp / Zp
    Gm = (a - b) * vm / Zm
    fa = -a - 2 * a * c0 - 2 * (Gp + Gm)
    fb = -b - 2 * b * cp - 2 * (Gp - Gm)
    return fa, fb


rng = np.random.default_rng(1)
res = 0.0
for _ in range(400):
    a = rng.uniform(0.3, 6.0)
    b = rng.uniform(0.0, 2.5)
    u = np.array([-a - b, -a + b, a - b, a + b])
    du = limit_field(u)
    fa, fb = core_field(a, b)
    res = max(res, abs((du[3] + du[2]) / 2 - fa), abs((du[3] - du[2]) / 2 - fb),
              abs(du[0] + du[3]), abs(du[1] + du[2]))
print(f"A. open:core vs limit field, 400 random (a,b): max residual {res:.2e}")

# ---------------------------------------------------------------- B. b = 0 comparison
a_c = math.sqrt(math.log(9) / 2)
a_s = math.sqrt(math.log(2))  # B = 1/5  <=>  e^{-2a^2} = 1/4


def Bbar(a: float) -> float:
    v = math.exp(-2 * a * a)
    return v / (1 + v)


def Bbar_dot(a: float) -> float:
    v = math.exp(-2 * a * a)
    return 4 * a * a * v * (1 + 4 * Bbar(a)) / (1 + v) ** 2


# S = int_{a_*}^{4} da / (a (1 + 4 B(a))) by composite Simpson in log a
nS = 20000
s = np.linspace(math.log(a_s), math.log(4.0), nS + 1)
aa = np.exp(s)
vv = np.exp(-2 * aa**2)
integrand = 1.0 / (1 + 4 * vv / (1 + vv))
hS = (s[-1] - s[0]) / nS
S_val = hS / 3 * (integrand[0] + integrand[-1] + 4 * integrand[1:-1:2].sum() + 2 * integrand[2:-1:2].sum())
t_c = None
s2 = np.linspace(math.log(a_c), math.log(4.0), nS + 1)
aa2 = np.exp(s2)
vv2 = np.exp(-2 * aa2**2)
i2 = 1.0 / (1 + 4 * vv2 / (1 + vv2))
h2 = (s2[-1] - s2[0]) / nS
t_c = h2 / 3 * (i2[0] + i2[-1] + 4 * i2[1:-1:2].sum() + 2 * i2[2:-1:2].sum())
print(f"B. a_c = {a_c:.6f}, B(a_c) = {Bbar(a_c):.6f}, dB/dt(a_c) = {Bbar_dot(a_c):.6f}")
print(f"   a_* = {a_s:.6f}, B(a_*) = {Bbar(a_s):.6f}; time 4 -> a_c = {t_c:.6f}; S = {S_val:.6f}"
      f" <= log(4/a_*) = {math.log(4 / a_s):.6f}")
print(f"   min dB/dt over a in [a_*, 4]: at a=4 {Bbar_dot(4.0):.3e} (tiny: no global positivity margin;"
      f" transversality is needed only near the crossing)")


# ---------------------------------------------------------------- C. float64 grid scan
def quantities(a: np.ndarray, b: np.ndarray) -> dict[str, np.ndarray]:
    w = np.exp(-2 * b * b)
    v0 = np.exp(-2 * a * a)
    vp = np.exp(-2 * (a + b) ** 2)
    vm = np.exp(-2 * (a - b) ** 2)
    sp, sm = v0 + vp, v0 + vm
    Zp, Zm = 1 + w + sp, 1 + w + sm
    cp = w * (1 / Zp + 1 / Zm)
    c0 = v0 * (1 / Zp + 1 / Zm)
    Gp, Gm = (a + b) * vp / Zp, (a - b) * vm / Zm
    fa = -a - 2 * a * c0 - 2 * (Gp + Gm)
    fb = -b - 2 * b * cp - 2 * (Gp - Gm)
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(b > 0, -fb / np.where(b > 0, b, 1.0), np.nan)
    # b -> 0 limit of k
    c = 1 + w + v0
    E = v0
    phip_a = E * ((1 - 4 * a * a) * c + E) / (c + E) ** 2
    k0 = 1 + 2 * cp + 4 * phip_a
    k = np.where(b > 0, k, k0)
    B = 0.5 * (sp / Zp + sm / Zm)
    dBa = 0.5 * ((1 + w) * (-4 * a * v0 - 4 * (a + b) * vp) / Zp**2
                 + (1 + w) * (-4 * a * v0 - 4 * (a - b) * vm) / Zm**2)
    dBb = 0.5 * ((-4 * (a + b) * vp * (1 + w) + 4 * b * w * sp) / Zp**2
                 + (4 * (a - b) * vm * (1 + w) + 4 * b * w * sm) / Zm**2)
    Bdot = dBa * fa + dBb * fb
    pp, pm = w / Zp, w / Zm
    conp, conm = (w - sp) / Zp, (w - sm) / Zm
    return dict(fa=fa, fb=fb, k=k, B=B, Bdot=Bdot, pp=pp, pm=pm, conp=conp, conm=conm)


def scan(eps: float, m: float, na: int = 1500, nb: int = 300) -> dict[str, float]:
    a = np.linspace(a_s, 4.0, na)
    beta = eps * (a / 4.0) ** m
    # (I) on the boundary
    q = quantities(a, beta)
    I_margin = float(np.min(q["k"] - m * (-q["fa"]) / a))
    # region grid
    A2 = np.repeat(a[:, None], nb, axis=1)
    Bfrac = np.linspace(0, 1, nb)[None, :]
    B2 = beta[:, None] * Bfrac
    qq = quantities(A2, B2)
    near = np.abs(qq["B"] - 0.1) < 2e-3
    III_margin = float(np.min(qq["Bdot"][near])) if near.any() else float("nan")
    IV_margin = float(np.min(qq["B"][0, :]) - 0.1)
    V_p = float(min(qq["pp"].min(), qq["pm"].min()) - 0.1)
    V_c = float(min(qq["conp"].min(), qq["conm"].min()) - 0.05)
    return dict(I=I_margin, III=III_margin, IV=IV_margin, Vp=V_p, Vc=V_c)


print("C. float64 grid scan (margins must all be > 0; V needs > SLACK =", SLACK, ")")
best: dict[float, float] = {}
for m in (0.25, 0.5, 0.75, 1.0, 1.25):
    lo, hi = 0.0, 3.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        r = scan(mid, m, na=600, nb=120)
        ok = r["I"] > 0 and r["III"] > 0 and r["IV"] > 0 and r["Vp"] > SLACK and r["Vc"] > SLACK
        lo, hi = (mid, hi) if ok else (lo, mid)
    best[m] = lo
    r = scan(lo * 0.999, m, na=600, nb=120)
    print(f"   m={m:4.2f}: largest eps passing ~ {lo:.4f}; margins at 0.999*eps: "
          + ", ".join(f"{kk}={vv:.4f}" for kk, vv in r.items()))

# ---------------------------------------------------------------- D. interval verification
EXP_PAD = 4


def pad(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(EXP_PAD):
        lo = np.nextafter(lo, -np.inf)
        hi = np.nextafter(hi, np.inf)
    return lo, hi


class Iv:
    """Vectorized outward-padded interval [lo, hi]."""

    def __init__(self, lo, hi=None):
        lo = np.asarray(lo, dtype=float)
        hi = lo if hi is None else np.asarray(hi, dtype=float)
        self.lo, self.hi = lo, hi

    def __add__(self, o):
        o = o if isinstance(o, Iv) else Iv(o)
        return Iv(*pad(self.lo + o.lo, self.hi + o.hi))

    __radd__ = __add__

    def __neg__(self):
        return Iv(-self.hi, -self.lo)

    def __sub__(self, o):
        o = o if isinstance(o, Iv) else Iv(o)
        return Iv(*pad(self.lo - o.hi, self.hi - o.lo))

    def __rsub__(self, o):
        return Iv(o) - self

    def __mul__(self, o):
        o = o if isinstance(o, Iv) else Iv(o)
        c = np.stack([self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi])
        return Iv(*pad(c.min(axis=0), c.max(axis=0)))

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = o if isinstance(o, Iv) else Iv(o)
        assert np.all(o.lo > 0), "division by interval containing nonpositive values"
        return self * Iv(*pad(1.0 / o.hi, 1.0 / o.lo))

    def __rtruediv__(self, o):
        return Iv(o) / self

    def sq(self):
        lo2, hi2 = self.lo**2, self.hi**2
        lo = np.where((self.lo <= 0) & (self.hi >= 0), 0.0, np.minimum(lo2, hi2))
        return Iv(*pad(lo, np.maximum(lo2, hi2)))

    def exp(self):
        return Iv(*pad(np.exp(self.lo), np.exp(self.hi)))


def interval_quantities(a: Iv, b: Iv) -> dict[str, Iv]:
    w = (-2.0 * b.sq()).exp()
    v0 = (-2.0 * a.sq()).exp()
    ap, am = a + b, a - b
    vp = (-2.0 * ap.sq()).exp()
    vm = (-2.0 * am.sq()).exp()
    sp, sm = v0 + vp, v0 + vm
    c = 1.0 + w + v0
    Zp, Zm = c + vp, c + vm
    iZp, iZm = 1.0 / Zp, 1.0 / Zm
    cp = w * (iZp + iZm)
    c0 = v0 * (iZp + iZm)
    Gp, Gm = ap * vp * iZp, am * vm * iZm
    fa = -1.0 * a - 2.0 * a * c0 - 2.0 * (Gp + Gm)
    # mean-value form: (G_+ - G_-)/b = int_{-1}^{1} phi_c'(a + s b) ds, enclosed by
    # summing hulls of phi_c' over NSUB equal pieces of s in [-1, 1].
    NSUB = 16
    sedges = np.linspace(-1.0, 1.0, NSUB + 1)
    integral = Iv(np.zeros_like(a.lo))
    for j in range(NSUB):
        sj = Iv(np.full_like(a.lo, sedges[j]), np.full_like(a.lo, sedges[j + 1]))
        r = a + sj * b
        E = (-2.0 * r.sq()).exp()
        phip = E * ((1.0 - 4.0 * r.sq()) * c + E) / (c + E).sq()
        integral = integral + (sedges[j + 1] - sedges[j]) * phip
    k = 1.0 + 2.0 * cp + 2.0 * integral
    fb = -1.0 * b * k
    B = 0.5 * (sp * iZp + sm * iZm)
    dBa = 0.5 * ((1.0 + w) * (-4.0 * a * v0 - 4.0 * ap * vp) * iZp * iZp
                 + (1.0 + w) * (-4.0 * a * v0 - 4.0 * am * vm) * iZm * iZm)
    dBb = 0.5 * ((-4.0 * ap * vp * (1.0 + w) + 4.0 * b * w * sp) * iZp * iZp
                 + (4.0 * am * vm * (1.0 + w) + 4.0 * b * w * sm) * iZm * iZm)
    Bdot = dBa * fa + dBb * fb
    pp, pm = w * iZp, w * iZm
    conp, conm = (w - sp) * iZp, (w - sm) * iZm
    absfa_over_a = 1.0 + 2.0 * c0 + 2.0 * (Gp + Gm) / a
    return dict(k=k, absfa_over_a=absfa_over_a, B=B, Bdot=Bdot, pp=pp, pm=pm, conp=conp, conm=conm)


def certify(eps: float, m: float, na: int = 1600, nb: int = 40) -> dict[str, float]:
    edges = np.linspace(a_s, 4.0, na + 1)
    alo, ahi = edges[:-1], edges[1:]
    beta = lambda x: eps * (x / 4.0) ** m  # noqa: E731
    out: dict[str, float] = {}
    # (I) boundary boxes
    qa = interval_quantities(Iv(alo, ahi), Iv(beta(alo), beta(ahi)))
    gI = qa["k"] - m * qa["absfa_over_a"]
    out["I_lower"] = float(gI.lo.min())
    # region boxes: a in [alo, ahi], b in [j, j+1] * beta(ahi)/nb
    A_lo = np.repeat(alo[:, None], nb, axis=1)
    A_hi = np.repeat(ahi[:, None], nb, axis=1)
    fr = np.linspace(0, 1, nb + 1)
    B_lo = beta(ahi)[:, None] * fr[None, :-1]
    B_hi = beta(ahi)[:, None] * fr[None, 1:]
    qq = interval_quantities(Iv(A_lo, A_hi), Iv(B_lo, B_hi))
    hit = (qq["B"].lo <= 0.1) & (qq["B"].hi >= 0.1)
    out["III_boxes_touching_B=0.1"] = int(hit.sum())
    out["III_lower_Bdot"] = float(qq["Bdot"].lo[hit].min())
    a_cross_min = float(A_lo[hit].min())
    out["min_a_on_crossing_boxes"] = a_cross_min
    # since f_a <= -a, the crossing occurs at time <= log(4 / a_cross_min) after t_A
    out["crossing_time_upper"] = math.log(4.0 / a_cross_min)
    out["V_p_lower"] = float(min(qq["pp"].lo.min(), qq["pm"].lo.min()) - 0.1)
    out["V_con_lower"] = float(min(qq["conp"].lo.min(), qq["conm"].lo.min()) - 0.05)
    # (IV) at a = a_*
    fr2 = np.linspace(0, 1, 401)
    q4 = interval_quantities(Iv(np.full(400, a_s)), Iv(beta(a_s) * fr2[:-1], beta(a_s) * fr2[1:]))
    out["IV_lower"] = float(q4["B"].lo.min() - 0.1)
    # start at a = 4 below threshold
    q0 = interval_quantities(Iv(np.full(400, 4.0)), Iv(eps * fr2[:-1], eps * fr2[1:]))
    out["start_B_upper"] = float(q0["B"].hi.max())
    return out


CHOICES = ((0.5, 0.5), (1.0, 1.0))  # (m, eps) certified below
for m_sel, eps_sel in CHOICES:
    cert = certify(eps_sel, m_sel)
    ok = (cert["I_lower"] > 0 and cert["III_lower_Bdot"] > 0 and cert["IV_lower"] > 0
          and cert["V_p_lower"] > SLACK and cert["V_con_lower"] > SLACK and cert["start_B_upper"] < 0.1)
    print(f"D. interval certificate eps={eps_sel}, m={m_sel}: {'PASS' if ok else 'FAIL'}")
    for kk, vv in cert.items():
        print(f"     {kk}: {vv}")
    print(f"     crossing_time_upper < S = {S_val:.6f}: {cert['crossing_time_upper'] < S_val}")

# ---------------------------------------------------------------- E. compare with open:small-gap
def text_bound(A: float) -> float:
    return 2 * (A / 4) ** (-0.25)


rate = 1 - 68 * math.exp(-8)
print(f"E. open:small-gap at A=4096: 2(A/4)^(-1/4) = {text_bound(4096):.4f}")
print(f"   sharpened: bdot <= -(1-68e^-8) b = -{rate:.5f} b gives b(t_A) <= 2(A/4)^(-{rate/2:.5f}) = "
      f"{2 * (4096 / 4) ** (-rate / 2):.4f}")
for m_sel, eps_sel in CHOICES:
    A_need = 4 * (2 / eps_sel) ** 4
    print(f"   m={m_sel}: eps_b = {eps_sel}: text bound needs A >= 4 (2/eps_b)^4 = {A_need:.1f};"
          f" text bound at 4096 {'meets' if text_bound(4096) <= eps_sel else 'does NOT meet'} it")
