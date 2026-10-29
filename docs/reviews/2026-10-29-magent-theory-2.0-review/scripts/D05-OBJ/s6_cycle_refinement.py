"""D05-OBJ s6: refinements for the per-agent examples.

1. Limit cycle of the law-dependent per-agent example of s4: interpolated up-crossing times and
   successive maxima of l_1 from two different initial conditions.
2. prop:dyn-nonreciprocal inside def:dyn-per-agent-flow: rows evolve (T = 1), equal evaluators
   rho_i = rho, priors pi = (1/(1+w), w/(1+w)); the equilibrium is l = rho, rows = pi, and the
   linearization keeps the complex pair of eq:dyn-three-cycle.
Single-threaded, < 10 s.
"""
import numpy as np
from scipy.integrate import solve_ivp
def sig(z): return 1/(1+np.exp(-z))
def klb(p, r): return p*np.log(p/r) + (1-p)*np.log((1-p)/(1-r))

gain, sgn = 4.0, np.array([1.0, 1.0, -1.0]); pis, pip = 0.2, 0.8; L0 = np.log(pip/pis)
def lam(l): return sgn*gain*np.tanh(np.roll(l, -1))
def per_agent(t, z):
    l, Lr = z[:3], z[3:]; bp = sig(Lr); lm = lam(l)
    dl = -(1-bp)*l - bp*(l - lm)
    dL = -(klb(sig(l), sig(lm)) - klb(sig(l), 0.5) + Lr - L0)
    return np.concatenate([dl, dL])
def up_cross(t, z): return z[0]
up_cross.direction = 1
for z0 in (np.array([0.01, 0, 0, L0, L0, L0]), np.array([2.0, -3.0, 1.0, 0.0, -1.0, 2.0])):
    sol = solve_ivp(per_agent, (0, 600), z0, events=up_cross, rtol=1e-11, atol=1e-13, dense_output=True)
    tc = sol.t_events[0]; tc = tc[tc > 300]
    per = np.diff(tc)
    tt = np.linspace(tc[0], tc[-1], 200001); l1 = sol.sol(tt)[0]
    peaks = [l1[(tt >= tc[k]) & (tt < tc[k+1])].max() for k in range(len(tc)-1)]
    print("start", np.round(z0, 2), ": period mean %.6f spread %.2e; successive maxima of l_1 mean %.6f spread %.2e; |z| max %.3f"
          % (per.mean(), np.ptp(per), np.mean(peaks), np.ptp(peaks), np.abs(sol.y).max()))

print()
for w in (0.5, 2.0):
    rho = 0.7; L0w = np.log(w)                 # log(pi_peer/pi_self) with pi = (1/(1+w), w/(1+w))
    def full(z):
        l, Lr = z[:3], z[3:]; bp = sig(Lr); lp = np.roll(l, -1)
        dl = -(1-bp)*(l - rho) - bp*(l - lp)
        dL = -(klb(sig(l), sig(lp)) - klb(sig(l), sig(rho)) + Lr - L0w)
        return np.concatenate([dl, dL])
    zs = np.concatenate([np.full(3, rho), np.full(3, L0w)])
    J = np.array([(full(zs + 1e-6*e) - full(zs - 1e-6*e))/2e-6 for e in np.eye(6)]).T
    ev = np.sort_complex(np.linalg.eigvals(J))
    print("w = %.1f: field at (rho, log w) = %.1e; eigenvalues %s; predicted complex pair (-1-3w/2 +- i sqrt3 w/2)/(1+w) = %.5f +- %.5f i"
          % (w, np.linalg.norm(full(zs)), np.round(ev, 5), (-1-1.5*w)/(1+w), np.sqrt(3)*w/2/(1+w)))
