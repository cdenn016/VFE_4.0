r"""D16-OPEN s6: the null control on the EXACT scaled binary field at a fixed h (focus question 3).

The 2026-10-22 report notes that no lens ran the chapter 12 null control on the exact field at
fixed h; open:null-control reports limit-field integrations only.  Here, at the proof's own
reference (A = 4096, third pair at R = (D_R + A + 6) e^L with D_R = 12, L = t_A + S), we run:
  full exact field  : open:binary-field in scaled coordinates u = x/h, T = h^2 (RK4),
  null exact field  : xdot = -(1-x^2) atanh x, solved exactly: atanh x(t) = e^{-t} atanh x(0),
  full limit field  : open:limit,
  null limit field  : u(t) = e^{-t} u(0),
all read by the exact rows of each field and the selector open:selector (eps=1/10, delta=1/20).
Stable evaluation: KL(mu_i||mu_j)/h^2 is the Bregman divergence of psi(x) = KL(Ber||unif),
expanded in h*Delta, Delta = u_i - u_j, with exact coefficients psi^(n)(x_j); differences of
atanh and tanh use the subtraction formulas.  Two values of h: 1e-9 and 2e-11 (the latter
inside the radius-1/2 tube condition of open:constants).  Margins of the null reference are
printed at its events.  Numerical evidence only.
"""
import numpy as np

EPS, DEL = 0.1, 0.05
A = 4096.0
tA = np.log(A / 4)                         # limit core reaches a = 4 at ~ log(A/4)
S = 1.426696
L = tA + S
R = (12 + A + 6) * np.exp(L)
u0 = np.array([-A - 2, -A + 2, A - 2, A + 2, R - 2, R + 2])


def breg_over_h2(D, y, h):
    # KL(mu_x||mu_y)/h^2 with x = y + h D: sum_n psi^(n)(y) (hD)^n / n! / h^2
    s = 1 - y * y
    p2 = 1 / s; p3 = 2 * y / s ** 2; p4 = (2 + 6 * y * y) / s ** 3; p5 = (24 * y + 24 * y ** 3) / s ** 4
    e = h * D
    return D * D * (p2 / 2 + p3 * e / 6 + p4 * e * e / 24 + p5 * e ** 3 / 120)


def rows_from(Dmat, y, h):
    Dv = breg_over_h2(Dmat, y[None, :], h)          # [i,j] receiver i, source j
    Dv = Dv - Dv.min(1, keepdims=True)
    n = np.exp(-Dv)
    return n / n.sum(1, keepdims=True)


def rows_limit(u):
    d = u[:, None] - u[None, :]
    n = np.exp(-d ** 2 / 2)
    return n / n.sum(1, keepdims=True)


def full_exact(u, h):
    x = h * u
    Dm = u[:, None] - u[None, :]
    B = rows_from(Dm, x, h)
    at = np.arctanh(h * Dm / (1 - x[:, None] * x[None, :]))   # atanh x_i - atanh x_j
    term = (1 - x * x) * (-np.arctanh(x) - (B * at).sum(1)) / h
    inc = (B * (u[:, None] - u[None, :])).sum(0)                 # inc_i = sum_k B[k,i] (u_k - u_i)
    return term + inc


def full_limit(u):
    W = rows_limit(u)
    X = u[None, :] - u[:, None]
    return -u + ((W + W.T) * X).sum(1)


def blockrow(B, I, J):
    return B[np.ix_(I, J)].sum() / len(I)


def qualify(part, B):
    nb = len(part)
    if nb < 2:
        return None
    Pm = np.zeros((6, nb))
    for k, p in enumerate(part):
        Pm[p, k] = 1.0
    BB = (Pm.T @ B @ Pm) / Pm.sum(0)[:, None]          # block rows B_IJ
    off = BB - np.diag(np.diag(BB))
    tot = off.sum(1, keepdims=True)
    contr = off - (tot - off)                          # B_IJ - sum_{K != I,J} B_IK
    Q = (off > EPS) & (contr > DEL)
    Q = Q & Q.T
    idx = np.argwhere(np.triu(Q, 1))
    if len(idx) == 0:
        return None
    return int(idx[0][0]), int(idx[0][1])


def qualify_slow(part, B):
    for a in range(len(part)):
        for b in range(a + 1, len(part)):
            I, J = part[a], part[b]
            ok = True
            for P, Q in ((I, J), (J, I)):
                ext = [k for p in part if p is not I and p is not J for k in p]
                bpq = blockrow(B, P, Q)
                bext = blockrow(B, P, ext) if ext else 0.0
                if not (bpq > EPS and bpq - bext > DEL):
                    ok = False
            if ok:
                return a, b
    return None


def run(rows_at, tmax, dt, label):
    part = [[i] for i in range(6)]
    ev = []
    t = 0.0
    n = int(round(tmax / dt))
    for s in range(1, n + 1):
        t = s * dt
        B = rows_at(t)
        while True:
            q = qualify(part, B)
            if q is None:
                break
            a, b = q
            I, J = part[a], part[b]
            part = [p for k, p in enumerate(part) if k not in (a, b)] + [sorted(I + J)]
            ev.append((round(t, 4), sorted(I + J)))
    return ev


def integrate(field, u0, tmax, dt):
    traj = [u0.copy()]
    u = u0.copy()
    for _ in range(int(round(tmax / dt))):
        k1 = field(u); k2 = field(u + dt / 2 * k1); k3 = field(u + dt / 2 * k2); k4 = field(u + dt * k3)
        u = u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        traj.append(u.copy())
    return np.array(traj)


out = ["A=%g  L=%.5f  R=%.6e" % (A, L, R)]
tmax, dt = 18.0, 2e-3
# limit fields
trL = integrate(full_limit, u0, tmax, dt)
evL = run(lambda t: rows_limit(trL[int(round(t / dt))]), tmax, dt, "full limit")
evL0 = run(lambda t: rows_limit(np.exp(-t) * u0), tmax, dt, "null limit")
out.append("full limit : %s" % evL)
out.append("null limit : %s" % evL0)
for h in (1e-9, 2e-11):
    tr = integrate(lambda u: full_exact(u, h), u0, tmax, dt)
    def rows_full(t, tr=tr, h=h):
        u = tr[int(round(t / dt))]
        return rows_from(u[:, None] - u[None, :], h * u, h)
    th0 = np.arctanh(h * u0)
    def rows_null(t, h=h):
        al = np.exp(-t) * th0
        x = np.tanh(al)
        # differences (x_i - x_j)/h via tanh subtraction
        dal = np.exp(-t) * np.arctanh(h * (u0[:, None] - u0[None, :]) / (1 - h * h * u0[:, None] * u0[None, :]))
        Dm = np.sinh(dal) / (np.cosh(al)[:, None] * np.cosh(al)[None, :]) / h
        return rows_from(Dm, x, h)
    evF = run(rows_full, tmax, dt, "full exact")
    evN = run(rows_null, tmax, dt, "null exact")
    out.append("h=%g full exact : %s" % (h, evF))
    out.append("h=%g null exact : %s" % (h, evN))
    same = [e[1] for e in evF] == [e[1] for e in evN]
    out.append("   same partition sequence (full vs null, exact field): %s" % same)
    # deviation of the far pair from the limit trajectory (G1-01 common-mode effect)
    out.append("   max |u_far(exact) - u_far(limit)| on [0,L]: %.3e" %
               np.abs(tr[:int(L / dt), 4:] - trL[:int(L / dt), 4:]).max())
    # margins of the null exact reference at its parent event
    tpar = [e[0] for e in evN if len(e[1]) == 4][0]
    B = rows_null(tpar)
    P1, P2, P3 = [0, 1], [2, 3], [4, 5]
    pw = min(B[0, 1], B[1, 0], B[2, 3], B[3, 2])
    ext = max(B[0, 2:].sum(), B[1, 2:].sum(), B[2, [0, 1, 4, 5]].sum(), B[3, [0, 1, 4, 5]].sum())
    cf = max(blockrow(B, P1 + P2, P3), blockrow(B, P3, P1 + P2))
    Bp = rows_null(tpar + 0.01); Bm = rows_null(tpar - 0.01)
    dB = (blockrow(Bp, P1, P2) - blockrow(Bm, P1, P2)) / 0.02
    out.append("   null exact margins at parent event t=%.3f: min first-pair weight %.4f, max first-pair exterior %.4f,"
               " core-third mass %.2e, dB_P1P2/dt %.4f" % (tpar, pw, ext, cf, dB))
print("\n".join(out))
