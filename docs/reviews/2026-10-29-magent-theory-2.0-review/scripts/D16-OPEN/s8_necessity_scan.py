r"""D16-OPEN s8: where does the null flow fail to reproduce the membership history? (extension probe)

Core u0 = (-a-b, -a+b, a-b, a+b) with b = 2 and ratio r = b/a, optionally with a far third pair
(R-2, R+2).  Limit field open:limit vs null limit u(t) = e^{-t} u0, read by open:selector
(eps=1/10, delta=1/20).  Events are detected on a time grid and their times refined by linear
interpolation of the qualifying margin; histories are compared as sets of formed blocks
(an order difference between nearly simultaneous disjoint events is reported separately, since
it only reflects event times).  Openness is probed by random independent perturbations of all
initial coordinates.  Exploratory numerics of the limit field only; nothing here is a proof.
"""
import numpy as np

EPS, DEL = 0.1, 0.05
rng = np.random.default_rng(1)


def rows(u):
    d = u[:, None] - u[None, :]
    n = np.exp(-d ** 2 / 2)
    return n / n.sum(1, keepdims=True)


def field(u):
    W = rows(u)
    X = u[None, :] - u[:, None]
    return -u + ((W + W.T) * X).sum(1)


def qualify(part, B, n):
    nb = len(part)
    if nb < 2:
        return None
    Pm = np.zeros((n, nb))
    for k, p in enumerate(part):
        Pm[p, k] = 1.0
    BB = (Pm.T @ B @ Pm) / Pm.sum(0)[:, None]
    off = BB - np.diag(np.diag(BB))
    tot = off.sum(1, keepdims=True)
    Q = (off > EPS) & (off - (tot - off) > DEL)
    Q = Q & Q.T
    idx = np.argwhere(np.triu(Q, 1))
    return None if len(idx) == 0 else (int(idx[0][0]), int(idx[0][1]))


def history(u0, null, tmax, dt=5e-3):
    n = len(u0)
    u = u0.copy(); part = [[i] for i in range(n)]; blocks = []
    for s in range(1, int(tmax / dt) + 1):
        t = s * dt
        if null:
            uu = np.exp(-t) * u0
        else:
            k1 = field(u); k2 = field(u + dt / 2 * k1); k3 = field(u + dt / 2 * k2); k4 = field(u + dt * k3)
            u = u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4); uu = u
        B = rows(uu)
        while True:
            q = qualify(part, B, n)
            if q is None:
                break
            a, b = q
            new = tuple(sorted(part[a] + part[b]))
            part = [p for k, p in enumerate(part) if k not in (a, b)] + [list(new)]
            blocks.append((round(t, 3), new))
    return blocks


def summary(h):
    return frozenset(b for _, b in h)


out = []
for far, R, tmax in ((False, None, 6.0), (True, 60.0, 6.0)):
    for r in (0.40, 0.47, 0.48, 0.49, 0.51):
        a = 2.0 / r
        u0 = [-a - 2, -a + 2, a - 2, a + 2] + ([R - 2, R + 2] if far else [])
        u0 = np.array(u0)
        hF, hN = history(u0, False, tmax), history(u0, True, tmax)
        nd_set, nd_order, trials = 0, 0, 6
        for _ in range(trials):
            up = u0 + rng.uniform(-0.1, 0.1, len(u0))
            pF, pN = history(up, False, tmax), history(up, True, tmax)
            if summary(pF) != summary(pN):
                nd_set += 1
            elif [b for _, b in pF] != [b for _, b in pN]:
                nd_order += 1
        out.append("%s r=%.2f  full=%s  null=%s  | perturbed (+-0.1, %d trials): block sets differ %d, only order differs %d"
                   % ("6-agent(R=60)" if far else "4-agent", r, hF, hN, trials, nd_set, nd_order))
print("\n".join(out))
