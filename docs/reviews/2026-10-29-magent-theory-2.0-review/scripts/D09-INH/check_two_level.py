r"""D09-INH check: the finite two-level construction, chapters/05_inheritance.tex:745-821.

Recomputes in exact rational arithmetic:
  * q_a, s_a, the evaluator K(m,{1}) = 1/4 + m/2, and nu_a = s_a K = Ber(1/2 + a/8);
  * the 16x16 fine kernel from (inh:finite-dynamics) with r' ~ nu_a;
  * strong lumpability of R10(a,b,c,r) = (a,b,r) and R21(a,b,r) = (a,r);
  * the closed forms (inh:first-finite-parent) for T^1 and the stated T^2;
  * path-law reproduction (retained configuration and record histories) from a
    random non-product initial law, by exhaustive summation over hidden paths;
  * that no other coordinate-dropping retention of (a,b,c,r) keeping r is lumpable
    except those listed, to confirm "the update of a reads only a ..." structure.
"""
from fractions import Fraction as Fr
import itertools
import random

def f(p, bit):
    return p if bit == 1 else 1 - p

def ber(p):
    return {0: 1 - p, 1: p}

q = {a: ber(Fr(1 + 2 * a, 4)) for a in (0, 1)}
s = {a: ber(Fr(2 + a, 4)) for a in (0, 1)}
K = {m: ber(Fr(1, 4) + Fr(m, 2)) for m in (0, 1)}
nu = {a: {r: sum(s[a][m] * K[m][r] for m in (0, 1)) for r in (0, 1)} for a in (0, 1)}
print("nu_a(1):", {a: nu[a][1] for a in (0, 1)}, " claimed 1/2 + a/8:", {a: Fr(1, 2) + Fr(a, 8) for a in (0, 1)})
print("q_a(1):", {a: q[a][1] for a in (0, 1)}, " s_a(1):", {a: s[a][1] for a in (0, 1)})

states = list(itertools.product((0, 1), repeat=4))  # (a,b,c,r)

def T(x, y):
    a, b, c, r = x
    a2, b2, c2, r2 = y
    return (f(Fr(1, 4), a2 ^ a) * f(Fr(1, 3), b2 ^ b ^ a) * f(Fr(1, 5), c2 ^ c ^ b) * nu[a][r2])

# stochasticity
assert all(sum(T(x, y) for y in states) == 1 for x in states)

def lump_check(R, states, T):
    img = sorted(set(R(x) for x in states))
    ok = True
    Tc = {}
    for x in states:
        row = {z: sum(T(x, y) for y in states if R(y) == z) for z in img}
        key = R(x)
        if key in Tc:
            if Tc[key] != row:
                ok = False
        else:
            Tc[key] = row
    return ok, Tc, img

R10 = lambda x: (x[0], x[1], x[3])
R21 = lambda z: (z[0], z[2])
R20 = lambda x: R21(R10(x))
ok1, T1, img1 = lump_check(R10, states, T)
ok2, T2, img2 = lump_check(R20, states, T)
print("R10 strongly lumpable:", ok1, " |image| =", len(img1))
print("R20 = R21 o R10 strongly lumpable:", ok2, " |image| =", len(img2))

# closed forms
def T1_closed(z, w):
    a, b, r = z
    a2, b2, r2 = w
    return f(Fr(1, 4), a2 ^ a) * nu[a][r2] * f(Fr(1, 3), b2 ^ b ^ a)

def T2_closed(z, w):
    a, r = z
    a2, r2 = w
    return f(Fr(1, 4), a2 ^ a) * nu[a][r2]

print("T^1 matches (inh:first-finite-parent):", all(T1[z][w] == T1_closed(z, w) for z in img1 for w in img1))
print("T^2 matches stated closed form:", all(T2[z][w] == T2_closed(z, w) for z in img2 for w in img2))
# T^2 is also the lumping of T^1 under R21 (composition)
ok21 = True
for z in img1:
    for w2 in img2:
        tot = sum(T1[z][w] for w in img1 if R21(w) == w2)
        if tot != T2[R21(z)][w2]:
            ok21 = False
print("T^1 lumps to T^2 under R21 (inh:composition):", ok21)

# path-law reproduction from a random non-product initial law
random.seed(7)
weights = [Fr(random.randint(1, 20)) for _ in states]
tot = sum(weights)
p0 = {x: w / tot for x, w in zip(states, weights)}
n = 3  # transitions
def fine_retained_law(R):
    law = {}
    for path in itertools.product(states, repeat=n + 1):
        pr = p0[path[0]]
        for k in range(n):
            pr *= T(path[k], path[k + 1])
            if pr == 0:
                break
        if pr == 0:
            continue
        key = tuple(R(x) for x in path)
        law[key] = law.get(key, 0) + pr
    return law

def parent_law(R, Tc, img):
    nu0 = {}
    for x in states:
        nu0[R(x)] = nu0.get(R(x), 0) + p0[x]
    law = {}
    for path in itertools.product(img, repeat=n + 1):
        pr = nu0.get(path[0], 0)
        for k in range(n):
            pr *= Tc[path[k]][path[k + 1]]
        if pr != 0:
            law[path] = pr
    return law

for name, Rm, Tc, img in (("R10", R10, T1, img1), ("R20", R20, T2, img2)):
    lf = fine_retained_law(Rm)
    lp = parent_law(Rm, Tc, img)
    keys = set(lf) | set(lp)
    err = max(abs(lf.get(k, 0) - lp.get(k, 0)) for k in keys)
    # record history (r coordinate) marginal
    rec_f, rec_p = {}, {}
    for k, v in lf.items():
        rk = tuple(z[-1] for z in k)
        rec_f[rk] = rec_f.get(rk, 0) + v
    for k, v in lp.items():
        rk = tuple(z[-1] for z in k)
        rec_p[rk] = rec_p.get(rk, 0) + v
    err_r = max(abs(rec_f.get(k, 0) - rec_p.get(k, 0)) for k in set(rec_f) | set(rec_p))
    print(f"{name}: exact max |fine retained path law - parent path law| over {n + 1}-state histories = {err};"
          f" record-history residual = {err_r}")

# which coordinate-subset retentions (keeping r) are lumpable?
print("Lumpable coordinate projections (subsets of {a,b,c}, r always kept):")
names = "abc"
for k in range(0, 4):
    for sub in itertools.combinations(range(3), k):
        Rm = lambda x, sub=sub: tuple(x[i] for i in sub) + (x[3],)
        ok, _, img = lump_check(Rm, states, T)
        print("   keep", "".join(names[i] for i in sub) + "r", "->", ok)
