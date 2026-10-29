"""XC3-SCOPE: stability under perturbation of the partition-sequence difference found in ch11_null_convention.py.
Under the chapter 13 null convention (labels, priors 1/N and row entropy kept, Gibbs rows at T),
compute, after the first join {1,2}, the first-crossing times of the two competing pairs
({1,2},3) and (3,4) for the configuration u=(1,1.2,1.45,1.79) and random perturbations of it
inside the admitted class form:gaps.  Compare with the full flow, where the order is fixed by
the initial centroid distances (0.35 vs 0.34)."""
import numpy as np
from scipy.integrate import solve_ivp
import importlib.util, sys, pathlib
here = pathlib.Path(__file__).parent
spec = importlib.util.spec_from_file_location("nc", here / "ch11_null_convention.py")
nc = importlib.util.module_from_spec(spec); spec.loader.exec_module(nc)

def crossing(times, f, delta):
    for k in range(1, len(times)):
        if f[k-1] > delta >= f[k]:
            return times[k-1] + (f[k-1]-delta)/(f[k-1]-f[k])*(times[k]-times[k-1])
    return np.inf

def compare(u, T=1.0, a0=1.0, kappa=1.0, delta=0.1, tmax=3.0):
    N = 4; P_init = [np.eye(2)]*N
    y0 = np.concatenate([np.array(u, float)] + [P.ravel() for P in P_init])
    times = np.linspace(0, tmax, 30001)
    sol = solve_ivp(nc.make_rhs("B", a0, kappa, N, T, list(u), P_init), (0, tmax), y0,
                    t_eval=times, rtol=1e-10, atol=1e-12)
    b = sol.y[:N] - a0
    t12 = crossing(times, b[1]-b[0], delta)
    tA = crossing(times, b[2]-0.5*(b[0]+b[1]), delta)   # ({1,2},3)
    tB = crossing(times, b[3]-b[2], delta)              # (3,4)
    return t12, tA, tB

def gaps_ok(u, delta=0.1):
    s = sorted(u); d = np.diff(s)
    return (delta < min(d[0], d[2]) < d[1]) and (max(d[0], d[2]) < d[1] + 0.5*min(d[0], d[2]))

if __name__ == "__main__":
    u = [1.0, 1.2, 1.45, 1.79]
    print("admitted (form:gaps):", gaps_ok(u))
    t12, tA, tB = compare(u)
    print("ch13-convention null: t({1,2})=%.4f  t(({1,2},3))=%.4f  t((3,4))=%.4f" % (t12, tA, tB))
    print("full flow: t({1,2})=%.4f  t(({1,2},3))=%.4f  t((3,4))=%.4f" % (0.5*np.log(2), 0.5*np.log(3.5), 0.5*np.log(3.4)))
    rng = np.random.default_rng(1)
    flips = 0; total = 0; margins = []
    for k in range(12):
        v = list(np.array(u) + rng.uniform(-0.004, 0.004, 4)); v[0] = 1.0 + rng.uniform(-0.004, 0.004)
        if not gaps_ok(v):
            continue
        total += 1
        _, a, bb = compare(v)
        s = sorted(v); d = np.diff(s)
        full_first_is_34 = (d[2] < d[1] + 0.5*d[0])
        null_first_is_34 = bb < a
        flips += int(full_first_is_34 and not null_first_is_34)
        margins.append(a - bb)
    print("perturbations in admitted class: %d, partition-order flips: %d" % (total, flips))
    print("null margins t(({1,2},3)) - t((3,4)) (negative = ({1,2},3) first):", np.round(margins, 4))
