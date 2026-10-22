r"""L06-ERA script 7: sketch check for the persistent-heterogeneity extension.

Same system as s02, but agent i uses evaluator coupling J_i = log 3 + eta_i
(heterogeneous constitutive data).  The gradient gains -(J_i - J) y_i in the x_i
component and -(J_i - J) x_i in the y_i component.  Report in-box separations at
late times: they should converge to a positive limit instead of zero.
"""
import numpy as np
from scipy.integrate import solve_ivp
import s02_erasure_simulation as s2

eta = np.array([0.002, -0.0015, -0.002, 0.0015])


def rhs(t, th, T):
    g = s2.grad(th, T)
    x, y = th[0:4], th[12:16]
    g[0:4] += -eta * y
    g[12:16] += -eta * x
    return -(1 - th**2) * g


th0 = s2.initial()
sol = solve_ivp(rhs, (0, 150), th0, t_eval=[0, 10, 50, 100, 150], args=(1.0,),
                method='LSODA', rtol=1e-10, atol=1e-13)
X, Y = sol.y[0:4], sol.y[12:16]
sgn = np.array([1, -1, 1, -1])
inbox = np.all((sgn[:, None] * X > 0.49) & (sgn[:, None] * X < 0.51) & (sgn[:, None] * Y > 0.49) & (sgn[:, None] * Y < 0.51))
for k, t in enumerate(sol.t):
    print(f't={t:6.1f}: in-box separations + {np.hypot(X[0,k]-X[2,k], Y[0,k]-Y[2,k]):.4e}, '
          f'- {np.hypot(X[1,k]-X[3,k], Y[1,k]-Y[3,k]):.4e}')
print('boxes respected at sampled times:', inbox)
print('gate closed at end:', s2.alpha(sol.y[8:12, -1].mean()) == 0, s2.alpha(sol.y[20:24, -1].mean()) == 0)
