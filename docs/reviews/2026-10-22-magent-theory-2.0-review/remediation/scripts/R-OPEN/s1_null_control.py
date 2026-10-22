"""R-OPEN s1: limit-field vs null-flow membership histories (open:limit, open:selector),
the proof constants at A=4096, and the forced final merge."""
import numpy as np

EPS, DEL = 0.1, 0.05

def rows(u):
    d = u[:, None] - u[None, :]
    n = np.exp(-d**2 / 2)
    return n / n.sum(1, keepdims=True)

def field(u):
    w = rows(u)
    x = u[None, :] - u[:, None]  # x_ij = u_j - u_i
    return -u + ((w + w.T) * x).sum(1)

def null(u):
    return -u

def blockrow(w, I, J):
    return w[np.ix_(I, J)].sum() / len(I)

def step_select(part, w):
    merged = True
    while merged:
        merged = False
        for a in range(len(part)):
            for b in range(a + 1, len(part)):
                I, J = part[a], part[b]
                ok = True
                for (P, Q) in ((I, J), (J, I)):
                    B = blockrow(w, P, Q)
                    ext = sum(blockrow(w, P, K) for k, K in enumerate(part) if K is not I and K is not J)
                    if not (B > EPS and B - ext > DEL):
                        ok = False
                if ok:
                    part = [K for K in part if K is not I and K is not J] + [sorted(I + J)]
                    return part, True
    return part, False

def run(u0, F, tmax, dt=1e-3):
    u = u0.copy(); part = [[i] for i in range(len(u))]; t = 0.0; ev = []
    while t < tmax:
        k1 = F(u); k2 = F(u + dt/2*k1); k3 = F(u + dt/2*k2); k4 = F(u + dt*k3)
        u = u + dt/6*(k1 + 2*k2 + 2*k3 + k4); t += dt
        ch = True
        while ch:
            part, ch = step_select(part, rows(u))
            if ch: ev.append((round(t, 3), [list(p) for p in part]))
    return ev

for A, R, tmax in [(8, 200, 12), (64, 2000, 15), (4096, 4e5, 18)]:
    u0 = np.array([-A-2, -A+2, A-2, A+2, R-2, R+2], float)
    print("A=%g R=%g" % (A, R))
    print("  full:", run(u0, field, tmax))
    print("  null:", run(u0, null, tmax))

# proof constants at A=4096 via the reduced core (a,b) of open:core
def core(a, b):
    w = np.exp(-2*b*b); v0 = np.exp(-2*a*a); vp = np.exp(-2*(a+b)**2); vm = np.exp(-2*(a-b)**2)
    Zp = 1 + w + v0 + vp; Zm = 1 + w + v0 + vm
    cp = w*(1/Zp + 1/Zm); c0 = v0*(1/Zp + 1/Zm)
    da = -a - 2*a*c0 - 2*((a+b)*vp/Zp + (a-b)*vm/Zm)
    db = -b - 2*b*cp - 2*((a+b)*vp/Zp - (a-b)*vm/Zm)
    return da, db
a, b, t, dt = 4096.0, 2.0, 0.0, 1e-4
while a > 4:
    da, db = core(a, b); a += dt*da; b += dt*db; t += dt
tA, bA = t, b
# b=0 comparison from a=4 until B=1/5
a, t2 = 4.0, 0.0
while True:
    v = np.exp(-2*a*a); B = v/(1+v)
    if B >= 0.2: break
    a += dt*(-a*(1+4*B)); t2 += dt
S = t2; L = tA + S
print("tA=%.4f b(tA)=%.3e S=%.4f L=%.4f" % (tA, bA, S, L))
for D in (12, 33):
    R = (D + 4096 + 6)*np.exp(L)
    h2 = 0.5/(0.1283*R**3)
    print("D=%d R=%.4e  tube(1/2) needs T=h^2 < %.2e ; admissibility h<1/R -> T<%.2e" % (D, R, h2, 1/R**2))
# early-phase sharpened constants
print("1-68e^-8 =", 1-68*np.exp(-8), " 1+4e^-32+4e^-8 =", 1+4*np.exp(-32)+4*np.exp(-8))
print("A_min for a(tau)>=8:", 8/np.exp(-(1+4*np.exp(-32)+4*np.exp(-8))*np.log(4)/(1-68*np.exp(-8))))
# curvature limit
from scipy.special import iv
def I(v): return v*iv(1, v)/iv(0, v)
def Ip(v, e=1e-6): return (I(v+e)-I(v-e))/(2*e)
k, lam = 1.5, 1.0
for t in (0, 1, 2, 3, 6):
    r = np.exp(-2*t); at = I(r*k); bt = I(r*lam)
    print("t=%d F=%.5f  h_chichi=%.3e" % (t, -bt*r*Ip(r*k)/(at+bt)**2, at*bt/(at+bt)))
print("F_inf=", -2*lam**2*k/(k**2+lam**2)**2)
