r"""D09-INH check: clause (B) of inh:inherited-parent against the intertwining branch of (D),
the proposed output-intertwining repair, and the companion's description of the smallest
exact parent for random dynamics.

  1. In the inh:intertwining example, the joint kernel x -> law(D_a(O_a), R(X')) cannot factor
     through R for ANY response, because its R(X') marginal T(x, R^{-1}.) is not constant on
     the fiber {1,2}.  So (B) with A_I nonempty forces strong lumpability and the intertwining
     branch of (D) is never used.
  2. Repair: require Lambda T_o = T^c_o Lambda for the output-indexed subkernels
     T_o(x,y) = P(O = o, X' = y | X = x).  Test two responses on the same chain:
       (a) a fiber-measurable response O ~ K(R(x), .) drawn independently of the transition;
       (b) a revealing response O = 1{X = 1}.
     For each, compare the true joint law of (Z_0, O_0, Z_1, O_1, Z_2) from nu Lambda with the
     parent prediction built from Lambda (filter Lambda(Z_n, .), response Lambda K, kernel T^c).
  3. Probabilistic bisimulation (coarsest strongly lumpable refinement) versus agreement of
     future interface laws ("trace equivalence") on a 7-state chain: two states with identical
     future label laws are not merged by any strongly lumpable refinement of the label partition.
"""
import itertools
import numpy as np

T = np.array([[0.3, 0.5, 0.2], [0.4, 0.2, 0.4], [0.25, 0.25, 0.5]])
Rm = np.array([0, 0, 1])                       # fibers a = {1,2} (index 0,1), b = {3}
Phi = np.eye(2)[Rm]                            # 3x2 deterministic kernel
Lam = np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])
Tc = Lam @ T @ Phi
print("1. next-retained-state kernel rows on fiber a:", (T @ Phi)[0], (T @ Phi)[1],
      "-> any joint kernel with this marginal is non-constant on the fiber; (B) fails for every response")

def subkernels(resp):
    """resp: 3 x n_o matrix of response probabilities given X, independent of the transition."""
    return [np.diag(resp[:, o]) @ T for o in range(resp.shape[1])]

resp_a = np.array([[0.7, 0.3], [0.7, 0.3], [0.2, 0.8]])   # depends on R(x) only
resp_b = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])   # O = 1{X = state 1}

for name, resp in (("(a) fiber-measurable response", resp_a), ("(b) revealing response", resp_b)):
    Ts = subkernels(resp)
    res = max(np.abs(Lam @ To - (Lam @ To @ Phi) @ Lam).max() for To in Ts)
    # true joint law of (Z0,O0,Z1,O1,Z2) from nu Lambda
    nu = np.array([0.6, 0.4])
    p0 = nu @ Lam
    P = [np.diag((Rm == z).astype(float)) for z in (0, 1)]
    true, pred = {}, {}
    for z0, o0, z1, o1, z2 in itertools.product((0, 1), (0, 1), (0, 1), (0, 1), (0, 1)):
        f = p0 @ P[z0] @ Ts[o0] @ P[z1] @ Ts[o1] @ P[z2]
        true[(z0, o0, z1, o1, z2)] = f.sum()
        # parent prediction: filter Lambda(Z_n,.), response Lambda K, next state T^c, as if O and
        # Z' were generated from Lambda(Z_n,.) jointly (the best a memoryless parent can do)
        J = lambda z, o, w: (Lam[z] @ Ts[o] @ Phi)[w]
        pred[(z0, o0, z1, o1, z2)] = nu[z0] * J(z0, o0, z1) * J(z1, o1, z2)
    tv = 0.5 * sum(abs(true[k] - pred[k]) for k in true)
    print(f"2. {name}: output-intertwining residual max|Lambda T_o - T^c_o Lambda| = {res:.3e};"
          f" TV(true joint law of (Z0,O0,Z1,O1,Z2), Lambda-parent prediction) = {tv:.3e}")

# ---------- 3. bisimulation vs trace equivalence ----------
# states: 0 = x, 1 = x', 2 = y, 3 = y1, 4 = y2, 5 = u, 6 = v ; labels A A B B B C D
lab = ['A', 'A', 'B', 'B', 'B', 'C', 'D']
T7 = np.zeros((7, 7))
T7[0, 2] = 1.0
T7[1, 3] = 0.5; T7[1, 4] = 0.5
T7[2, 5] = 0.5; T7[2, 6] = 0.5
T7[3, 5] = 1.0
T7[4, 6] = 1.0
T7[5, 5] = 1.0; T7[6, 6] = 1.0

def refine(T, part):
    while True:
        new = []
        for blk in part:
            groups = {}
            for s in blk:
                key = tuple(np.round([T[s, b].sum() for b in part], 12))
                groups.setdefault(key, []).append(s)
            new.extend(groups.values())
        if len(new) == len(part):
            return new
        part = new

part0 = [[i for i in range(7) if lab[i] == L] for L in sorted(set(lab))]
bis = refine(T7, part0)
print("3. coarsest strongly lumpable refinement of the label partition:", sorted(sorted(b) for b in bis))

def trace_law(s0, n):
    law = {}
    for path in itertools.product(range(7), repeat=n):
        pr = 1.0; prev = s0
        for s in path:
            pr *= T7[prev, s]; prev = s
        if pr > 0:
            key = (lab[s0],) + tuple(lab[s] for s in path)
            law[key] = law.get(key, 0) + pr
    return law

lx, lxp = trace_law(0, 4), trace_law(1, 4)
print("   future label-sequence laws from x and x' (4 steps) equal:",
      set(lx) == set(lxp) and all(abs(lx[k] - lxp[k]) < 1e-12 for k in lx), "->", lx)
print("   x and x' in the same refinement block:", any(0 in b and 1 in b for b in bis))

# ---------- 4. canonical lumping ignores the joint response/transition condition (B) ----------
# Chain of chapters/05_inheritance.tex:557-563: x = (u,v), fresh fair xi, u' = u xor xi, v' fresh,
# response O = xi xor v.  Interface Q(x) = (u, response kernel of x) -- the response kernel is
# uniform for every x, so B0 = {u = 0}, {u = 1}.
st = [(u, v) for u in (0, 1) for v in (0, 1)]
Tuv = np.zeros((4, 4)); Juv = np.zeros((4, 2, 4))      # Juv[x, o, y] = P(O=o, X'=y | X=x)
for i, (u, v) in enumerate(st):
    for xi in (0, 1):
        for v2 in (0, 1):
            j = st.index((u ^ xi, v2))
            Tuv[i, j] += 0.25
            Juv[i, xi ^ v, j] += 0.25
B0 = [[0, 1], [2, 3]]
lumped = refine(Tuv, [list(b) for b in B0])
print("4. inh:canonical-lumping on the (u,v) chain with interface (u, response kernel):", sorted(sorted(b) for b in lumped))
def refine_joint(J, part):
    while True:
        new = []
        for blk in part:
            groups = {}
            for s in blk:
                key = tuple(np.round([J[s, o, b].sum() for o in range(J.shape[1]) for b in part], 12))
                groups.setdefault(key, []).append(s)
            new.extend(groups.values())
        if len(new) == len(part):
            return new
        part = new
joint = refine_joint(Juv, [list(b) for b in B0])
print("   refinement by the joint (response, next block) kernel required by clause (B):", sorted(sorted(b) for b in joint))

# ---------- 5. intertwining is sufficient, not necessary, for a memoryless parent ----------
T5 = np.array([[0.6, 0.0, 0.4], [0.0, 0.6, 0.4], [0.5, 0.0, 0.5]])   # strongly lumpable for {1,2},{3}
R5 = np.array([0, 0, 1]); F5 = np.eye(2)[R5]
print("5. strongly lumpable:", np.allclose((T5 @ F5)[0], (T5 @ F5)[1]))
sols = []
for lam in np.linspace(0, 1, 101):
    L5 = np.array([[lam, 1 - lam, 0], [0, 0, 1.0]])
    Tc5 = L5 @ T5 @ F5
    if np.abs(L5 @ T5 - Tc5 @ L5).max() < 1e-12:
        sols.append(round(lam, 3))
print("   lambda in [0,1] with Lambda T = T^c Lambda (Lambda(a,.) = (lambda, 1-lambda, 0), Lambda(b,.) = delta_3):", sols)
print("   initial law delta_2 is not nu Lambda for these Lambda, yet the retained process is Markov (strong lumpability)")
