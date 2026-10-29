r"""GAP02-S2 s3b: float64 readout (RK4, dt = 1e-3; numerical observation, not a proof) of the event
margins of the decoupled reference of open:routing at A = 34 (core) and the far pair, in the form
open:persistence needs: for each crossing observable, min time derivative while the observable is in the
band |O - 1/10| <= 0.01; window margins of the contrasts above 1/20 after entry.
"""
import math
import numpy as np

def rows(u):
    x = u[None, :] - u[:, None]; n = np.exp(-0.5 * x * x); return n / n.sum(axis=1, keepdims=True)

def field(u):
    x = u[None, :] - u[:, None]; w = rows(u); return -u + ((w + w.T) * x).sum(axis=1)

def rk4(y, T, dt):
    n = int(round(T / dt)); out = [y.copy()]
    for _ in range(n):
        k1 = field(y); k2 = field(y + dt/2*k1); k3 = field(y + dt/2*k2); k4 = field(y + dt*k3)
        y = y + dt/6*(k1 + 2*k2 + 2*k3 + k4); out.append(y.copy())
    return np.array(out)

A, dt = 34.0, 1e-3
L = math.log(A/4) + math.log(4/math.sqrt(math.log(2)))
core = rk4(np.array([-A-2, -A+2, A-2, A+2]), L, dt)
pair = rk4(np.array([-2.0, 2.0]), L, dt)
t = np.arange(core.shape[0]) * dt
W = np.array([rows(u) for u in core]); Wp = np.array([rows(u) for u in pair])
p12, p21 = W[:, 0, 1], W[:, 1, 0]
con12 = W[:, 0, 1] - W[:, 0, 2] - W[:, 0, 3]
B = 0.5 * (W[:, 0, 2] + W[:, 0, 3] + W[:, 1, 2] + W[:, 1, 3])
conP = B - 0.0  # exterior (third pair) is absent in the decoupled reference; the six-agent exterior < 1e-3
def band_rate(o):
    d = np.gradient(o, dt); m = np.abs(o - 0.1) <= 0.01
    return d[m].min(), t[m].min(), t[m].max()
for name, o in (("core p12", p12), ("core p21", p21), ("far pair p", Wp[:, 0, 1]), ("parent B", B)):
    r = band_rate(o); print(f"{name:10s}: min dO/dt on band = {r[0]:.4f}, band times [{r[1]:.4f}, {r[2]:.4f}]")
i_in = np.argmax(p12 > 0.11); i_B = np.argmax(B > 0.11)
print(f"first-pair contrast - 1/20 on [entry, L]: min = {(con12[i_in:] - 0.05).min():.4f}")
print(f"first-pair weight - 1/10 on [entry, L]: min = {(np.minimum(p12, p21)[i_in:] - 0.1).min():.4f}")
print(f"parent contrast (B - 1/20) after band exit, before L: min = {(B[i_B:] - 0.05).min():.4f} (six-agent exterior < 1e-3 not included)")
print(f"far-pair contrast after entry: min = {(Wp[np.argmax(Wp[:,0,1] > 0.11):, 0, 1] - 0.05).min():.4f}")
print(f"L = {L:.4f}; first-pair band ends {t[np.argmax(p12 > 0.11)]:.4f}; parent band ends {t[i_B]:.4f}")
