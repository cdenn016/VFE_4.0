"""Ch.12 checks: open:limit vs open:core, open:angular-flow, and the 32/3 generated-loss example."""
import numpy as np

def limit_field(u):
    D = (u[:, None] - u[None, :])**2 / 2
    w = np.exp(-D); w /= w.sum(1, keepdims=True)         # self entry included
    S = w + w.T
    return -u + (S * (u[None, :] - u[:, None])).sum(1)

rng = np.random.default_rng(11)
worst = 0.0
for _ in range(200):
    a, b = rng.uniform(0, 4), rng.uniform(0, 3)
    u = np.array([-a - b, -a + b, a - b, a + b])
    du = limit_field(u)
    adot, bdot = (du[2] + du[3]) / 2, (du[3] - du[2]) / 2
    w, v0, vp, vm = np.exp(-2 * b * b), np.exp(-2 * a * a), np.exp(-2 * (a + b)**2), np.exp(-2 * (a - b)**2)
    Zp, Zm = 1 + w + v0 + vp, 1 + w + v0 + vm
    cp, c0 = w * (1 / Zp + 1 / Zm), v0 * (1 / Zp + 1 / Zm)
    ca = -a - 2 * a * c0 - 2 * ((a + b) * vp / Zp + (a - b) * vm / Zm)
    cb = -b - 2 * b * cp - 2 * ((a + b) * vp / Zp - (a - b) * vm / Zm)
    worst = max(worst, abs(adot - ca), abs(bdot - cb))
print('open:core residual max over 200 random (a,b):', worst)

# scaled exact binary field -> limit field (T = h^2, x = h u), stable evaluation
def exact_scaled(u, h):
    x = h * u
    d = lambda p, q: 0.5 * ((1 + p) * (np.log1p(p) - np.log1p(q)) + (1 - p) * (np.log1p(-p) - np.log1p(-q)))
    D = d(x[:, None], x[None, :]) / h**2
    beta = np.exp(-(D - D.min(1, keepdims=True))); beta /= beta.sum(1, keepdims=True)
    at = np.arctanh(x)
    xd = (1 - x**2) * (-at - (beta * (at[:, None] - at[None, :])).sum(1)) + (beta.T * (x[None, :] - x[:, None])).sum(1)
    return xd / h
u = np.array([-6.0, -2.0, 2.0, 6.0, 9.0, 13.0])
for h in (1e-2, 1e-3, 1e-4, 1e-6):
    print(f'h={h:g}: max|exact scaled field - limit field| = {np.abs(exact_scaled(u, h) - limit_field(u)).max():.3e}')

# 32/3 example: two identity-coupled laws on (z,y) in {0,1}^2, uniform references
def flow_rhs(q1, q2):
    h1 = 2 * np.log(q1) - np.log(q2) - q2 / q1
    h2 = 2 * np.log(q2) - np.log(q1) - q1 / q2
    return -q1 * (h1 - (q1 * h1).sum()), -q2 * (h2 - (q2 * h2).sum())
def evolve(q1, q2, t, n=400):
    dt = t / n
    for _ in range(n):
        k1 = flow_rhs(q1, q2); k2 = flow_rhs(q1 + dt / 2 * k1[0], q2 + dt / 2 * k1[1])
        k3 = flow_rhs(q1 + dt / 2 * k2[0], q2 + dt / 2 * k2[1]); k4 = flow_rhs(q1 + dt * k3[0], q2 + dt * k3[1])
        q1 = q1 + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]); q2 = q2 + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return q1, q2
k1c = np.array([[0.75, 0.25], [0.75, 0.25]]); k2c = np.array([[0.25, 0.75], [0.25, 0.75]])   # rows z, cols y
def law(mu, kap): return (mu[:, None] * kap)
def hidden_loss(t, eps=1e-5):
    out = []
    for s in (eps, -eps):
        mu2 = np.array([0.5 + s, 0.5 - s])
        out.append(evolve(law(np.array([0.5, 0.5]), k1c), law(mu2, k2c), t))
    loss = 0.0
    q0 = evolve(law(np.array([0.5, 0.5]), k1c), law(np.array([0.5, 0.5]), k2c), t)
    for i in range(2):
        qp, qm = out[0][i], out[1][i]
        kp, km = qp / qp.sum(1, keepdims=True), qm / qm.sum(1, keepdims=True)
        aX = (np.log(kp) - np.log(km)) / (2 * eps)
        loss += (q0[i] * aX**2).sum()
    return loss
vals = []
for t in (0.0025, 0.00125, 0.000625):
    vals.append(hidden_loss(t) / t**2)
    print(f't={t}: (fine - retained Fisher)/t^2 = {vals[-1]:.5f}  (claimed 32/3 = {32 / 3:.5f})')
print('Richardson extrapolation (error O(t)):', 2 * vals[-1] - vals[-2])

# numerical hazard: naive log-ratio evaluation of d(x,z)/T at T = h^2
def d_naive(p, q):
    return 0.5 * ((1 + p) * np.log((1 + p) / (1 + q)) + (1 - p) * np.log((1 - p) / (1 - q)))
def d_stable(p, q):
    return 0.5 * ((1 + p) * (np.log1p(p) - np.log1p(q)) + (1 - p) * (np.log1p(-p) - np.log1p(-q)))
for h in (1e-4, 1e-6, 1e-8):
    uu, vv = 3.0, 1.0
    exact = (uu - vv)**2 / 2
    print(f'h={h:g}: naive d/h^2={d_naive(h * uu, h * vv) / h**2:.10f}, stable d/h^2={d_stable(h * uu, h * vv) / h**2:.10f}, limit={exact}')
