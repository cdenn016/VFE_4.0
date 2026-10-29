"""D05-OBJ s4: per-agent dynamics (def:dyn-per-agent-flow, prop:dyn-nonreciprocal).

1. The three-cycle of eq:dyn-three-cycle: spectrum, and the Euclidean strict Lyapunov function.
2. Fixed rows with positive self weight b and fixed channels: contraction of max_i osc(theta_i - theta_i')
   at rate >= b (Birkhoff), on random ternary examples.
3. A per-agent example inside def:dyn-per-agent-flow (rows evolving at unit mobility) with a
   constant-in-input law-dependent channel: unique equilibrium, unstable, bounded orbits, so no
   continuous strict Lyapunov function; numerically a limit cycle.  The Fisher gradient flow of
   F_tot = sum_i F_i with the same data converges and dissipates F_tot.
Single-threaded, < 15 s.
"""
import numpy as np
from scipy.integrate import solve_ivp
rng = np.random.default_rng(3)
def sig(z): return 1/(1+np.exp(-z))
def klb(p, r): return p*np.log(p/r) + (1-p)*np.log((1-p)/(1-r))

print("1. three-cycle (1+w) l_i' = -(l_i - rho_i) - w (l_i - l_{i+1})")
w = 2.0; S = np.roll(np.eye(3), 1, axis=1)          # (S l)_i = l_{i+1}
A = (-(1+w)*np.eye(3) + w*S)/(1+w)
ev = np.linalg.eigvals(A); print("   eigenvalues:", np.round(np.sort_complex(ev), 6))
print("   predicted: -1/(1+w) =", -1/(1+w), " and (-1-3w/2 +- i sqrt3 w/2)/(1+w) =", (-1-1.5*w)/(1+w), "+-", np.sqrt(3)*w/2/(1+w), "i")
print("   eigenvalues of A + A^T (all < 0 => V = |l - l*|^2 is a strict global Lyapunov function):", np.round(np.linalg.eigvalsh(A + A.T), 6))
rho = np.array([1.0, -0.5, 2.0]); lstar = np.linalg.solve(A, -rho/(1+w))
sol = solve_ivp(lambda t, l: A @ l + rho/(1+w), (0, 20), [3, -4, 0.5], max_step=0.01, rtol=1e-10, atol=1e-12)
V = np.sum((sol.y - lstar[:, None])**2, axis=0)
print("   along a trajectory: max increment of V =", float(np.max(np.diff(V))), " V(0) =", V[0], " V(20) =", V[-1])

print("\n2. fixed rows, fixed positive channels, ternary alphabet, 4 agents: contraction in max-osc")
def run_contraction(trial):
    N, K = 4, 3; b = 0.15 + 0.3*rng.random()
    srcs = {i: rng.choice([j for j in range(N) if j != i], size=2, replace=False) for i in range(N)}
    rows = {i: np.concatenate([[b], (1-b)*(lambda v: v/v.sum())(rng.random(2)+0.1)]) for i in range(N)}
    logG = {i: np.log((lambda v: v/v.sum())(rng.random(K)+0.1)) for i in range(N)}
    ch = {(i, j): (lambda M: M/M.sum(1, keepdims=True))(rng.random((K, K))**3 + 0.01) for i in range(N) for j in srcs[i]}
    def field(t, th):
        th = th.reshape(N, K); q = np.exp(th - th.max(1, keepdims=True)); q /= q.sum(1, keepdims=True)
        out = np.empty_like(th)
        for i in range(N):
            tgt = rows[i][0]*logG[i] + sum(rows[i][1+a]*np.log(q[j] @ ch[(i, j)]) for a, j in enumerate(srcs[i]))
            v = -th[i] + tgt; out[i] = v - v.mean()
        return out.ravel()
    th1 = rng.normal(0, 3, N*K); th2 = rng.normal(0, 3, N*K)
    ts = np.linspace(0, 12, 121)
    s1 = solve_ivp(field, (0, 12), th1, t_eval=ts, rtol=1e-10, atol=1e-12)
    s2 = solve_ivp(field, (0, 12), th2, t_eval=ts, rtol=1e-10, atol=1e-12)
    d = s1.y - s2.y
    M = np.array([max(np.ptp(d[i*K:(i+1)*K, k]) for i in range(N)) for k in range(len(ts))])
    rate = -np.polyfit(ts[20:100], np.log(M[20:100]), 1)[0]
    worst_ratio = float(np.max(M[1:]/(M[0]*np.exp(-b*ts[1:]))))
    return b, rate, worst_ratio
for trial in range(4):
    b, rate, wr = run_contraction(trial)
    print("   self weight b = %.3f  fitted contraction rate = %.3f  max of M(t)/(M(0) e^{-bt}) = %.4f" % (b, rate, wr))

print("\n3. per-agent dynamics with a constant-in-input law-dependent channel (rows evolve, T = 1)")
gain, sgn = 4.0, np.array([1.0, 1.0, -1.0]); pis, pip = 0.2, 0.8
L0 = np.log(pip/pis)
def lam(l): return sgn*gain*np.tanh(np.roll(l, -1))     # logit of the compared law nu_i = Ber(sig(s_i g tanh l_{i+1}))
def per_agent(t, z):
    l, Lr = z[:3], z[3:]
    bp = sig(Lr); lm = lam(l)
    dl = -(1-bp)*l - bp*(l - lm)                            # G_i = Ber(1/2): self logit 0
    cp = klb(sig(l), sig(lm)); cs = klb(sig(l), 0.5)
    dL = -(cp - cs + Lr - L0)
    return np.concatenate([dl, dL])
def Ftot(z):
    l, Lr = z[:3], z[3:]; bp = sig(Lr); bs = 1 - bp; p = sig(l)
    return float(np.sum(bs*klb(p, 0.5) + bp*klb(p, sig(lam(l))) + bs*np.log(bs/pis) + bp*np.log(bp/pip)))
def num_grad(fun, z, h=1e-6):
    gvec = np.zeros_like(z)
    for k in range(len(z)):
        e = np.zeros_like(z); e[k] = h; gvec[k] = (fun(z+e) - fun(z-e))/(2*h)
    return gvec
z0 = np.concatenate([np.zeros(3), np.full(3, L0)])
J = np.array([(per_agent(0, z0 + 1e-6*e) - per_agent(0, z0 - 1e-6*e))/2e-6 for e in np.eye(6)]).T
print("   equilibrium z* = (l = 0, L = log(pi_p/pi_s)); field there:", np.round(per_agent(0, z0), 12))
c = pip*gain
print("   Jacobian eigenvalues:", np.round(np.sort_complex(np.linalg.eigvals(J)), 5))
print("   predicted law block: -1-c =", -1-c, ", -1+c/2 +- i c sqrt3/2 =", -1+c/2, "+-", c*np.sqrt(3)/2, "i  (c = pi_p g =", c, ")")
sol = solve_ivp(per_agent, (0, 400), z0 + np.array([0.01, 0, 0, 0, 0, 0]), max_step=0.02, rtol=1e-9, atol=1e-11)
late = sol.t > 300; l1 = sol.y[0, late]; tt = sol.t[late]
up = np.where((l1[:-1] < 0) & (l1[1:] >= 0))[0]
periods = np.diff(tt[up])
print("   late-time amplitude of l_1: [%.4f, %.4f]; up-crossing periods: mean %.4f, spread %.2e" % (l1.min(), l1.max(), periods.mean(), np.ptp(periods)))
Fl = np.array([Ftot(sol.y[:, k]) for k in np.where(late)[0]])
print("   F_tot along the per-agent limit cycle oscillates in [%.4f, %.4f]" % (Fl.min(), Fl.max()))
# reciprocal comparison: Fisher gradient flow of F_tot in (l, L) (Bernoulli Fisher flow in log odds: x' = -dF/dp)
def grad_flow(t, z):
    l, Lr = z[:3], z[3:]
    gl = num_grad(Ftot, z)                                # derivative in logit coordinates
    p = sig(l); b = sig(Lr)
    dl = -gl[:3]/(p*(1-p))                                # -dF/dp = -(dF/dl)/(p(1-p))
    dL = -gl[3:]/(b*(1-b))
    return np.concatenate([dl, dL])
sol2 = solve_ivp(grad_flow, (0, 200), z0 + np.array([0.01, 0, 0, 0, 0, 0]), max_step=0.05, rtol=1e-9, atol=1e-11)
Fv = np.array([Ftot(sol2.y[:, k]) for k in range(sol2.y.shape[1])])
print("   gradient flow of F_tot: max increment of F_tot = %.2e, F_tot(0) = %.6f, F_tot(200) = %.6f, |field| at t=200 = %.2e"
      % (np.max(np.diff(Fv)), Fv[0], Fv[-1], np.linalg.norm(grad_flow(0, sol2.y[:, -1]))))
print("   endpoint of the gradient flow l =", np.round(sol2.y[:3, -1], 4))
