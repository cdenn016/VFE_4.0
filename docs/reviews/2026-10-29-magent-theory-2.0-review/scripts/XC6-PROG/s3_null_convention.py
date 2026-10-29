r"""XC6-PROG s3: the (F5) verdict for a row-reading selector depends on an unstated convention.

Four Bernoulli agents (log odds l_i), trivial model sector, Bernoulli references with log odds
rho_i, identity peer channels, nonself sources with uniform priors 1/3, exact Gibbs rows at
temperature T (reduced potential E_red = sum KL(q_i||r_i) - T sum log Z_i), unit Fisher mobility.
In log odds the Fisher flow is  dl_i/dt = - dE_red/dp_i  with
    dE_red/dp_i = (l_i - rho_i) + sum_j beta_ij (l_i - l_j) + sum_k beta_ki (p_i - p_k)/(p_i(1-p_i)).
Null flow (inh:null-flow): delete every peer cost, so dl_i/dt = -(l_i - rho_i).

Selector: join {i,j} when beta_ij > theta and beta_ji > theta (mutual preferential attention),
theta = 0.6 > prior 1/3; blocks once joined are kept (persistence is not the point here).

Two readings of "the same selector along Phi^0":
  literal  : the rows are part of the configuration and follow the null potential; with peer
             costs deleted the Gibbs rows equal the priors 1/3, so the selector never fires;
  readout  : the rows are recomputed as Gibbs rows of the full peer costs at the null laws
             (the convention of chapters/12_open_formation.tex, open:null-control).
"""
import numpy as np

T = 0.05
theta = 0.6
rho = np.array([2.0, 2.3, -2.0, -2.4])
l0 = np.array([0.30, -0.25, 0.05, -0.45])
N = 4


def p_of(l: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-l))


def kl_ber(p: float, q: float) -> float:
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def gibbs_rows(l: np.ndarray) -> np.ndarray:
    p = p_of(l)
    B = np.zeros((N, N))
    for i in range(N):
        w = np.array([0.0 if j == i else np.exp(-kl_ber(p[i], p[j]) / T) / 3.0 for j in range(N)])
        B[i] = w / w.sum()
    return B


def full_field(l: np.ndarray) -> np.ndarray:
    p = p_of(l)
    B = gibbs_rows(l)
    v = np.zeros(N)
    for i in range(N):
        g = l[i] - rho[i]
        g += sum(B[i, j] * (l[i] - l[j]) for j in range(N) if j != i)
        g += sum(B[k, i] * (p[i] - p[k]) / (p[i] * (1 - p[i])) for k in range(N) if k != i)
        v[i] = -g
    return v


def null_field(l: np.ndarray) -> np.ndarray:
    return -(l - rho)


def run(fieldfun, rows_mode: str, tend: float = 8.0, dt: float = 1e-3):
    l = l0.copy()
    joined = set()
    events = []
    for step in range(int(tend / dt) + 1):
        t = step * dt
        if rows_mode == "literal":
            B = np.full((N, N), 1.0 / 3.0)
            np.fill_diagonal(B, 0.0)
        else:
            B = gibbs_rows(l)
        for i in range(N):
            for j in range(i + 1, N):
                if (i, j) not in joined and B[i, j] > theta and B[j, i] > theta:
                    joined.add((i, j))
                    events.append((round(t, 3), (i + 1, j + 1)))
        k1 = fieldfun(l)
        k2 = fieldfun(l + 0.5 * dt * k1)
        k3 = fieldfun(l + 0.5 * dt * k2)
        k4 = fieldfun(l + dt * k3)
        l = l + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return events, l


B0 = gibbs_rows(l0)
print("initial Gibbs rows (full costs):")
print(np.round(B0, 3))
mutual0 = [(i + 1, j + 1) for i in range(N) for j in range(i + 1, N) if B0[i, j] > theta and B0[j, i] > theta]
print(f"initially selected pairs: {mutual0}  (empty means initially unpartitioned)")
ev_full, lf = run(full_field, "readout")
ev_null_readout, ln = run(null_field, "readout")
ev_null_literal, _ = run(null_field, "literal")
print(f"full flow          : events {ev_full}; final l = {np.round(lf, 3)}")
print(f"null, readout rows : events {ev_null_readout}; final l = {np.round(ln, 3)}")
print(f"null, literal rows : events {ev_null_literal}")
P_full = [e[1] for e in ev_full]
print(f"P = P0 under readout convention: {P_full == [e[1] for e in ev_null_readout]}; "
      f"P = P0 under literal convention: {P_full == [e[1] for e in ev_null_literal]}")
