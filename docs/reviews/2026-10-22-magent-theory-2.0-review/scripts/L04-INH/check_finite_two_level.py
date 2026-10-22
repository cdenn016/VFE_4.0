r"""Exact check of the finite two-level construction (chapters/05_inheritance.tex, Sec. 'A finite two-level construction').

Fine state x=(a,b,c,r) in {0,1}^4; a'=a+xi, b'=b+a+eta, c'=c+b+upsilon (mod 2),
xi~Ber(1/4), eta~Ber(1/3), upsilon~Ber(1/5), r'~nu_a=Ber(1/2+a/8) independently.
Checks: nu_a; strong lumpability of R10(a,b,c,r)=(a,b,r) and R21(a,b,r)=(a,r);
the displayed parent kernels; path-law agreement for a random initial law;
and the skew-product (autonomy) structure: retained transitions never read discarded coordinates.
All arithmetic in exact rationals.
"""
from fractions import Fraction as Fr
from itertools import product
import random

def f(p: Fr, bit: int) -> Fr:
    return p if bit == 1 else 1 - p

q = {a: Fr(1 + 2 * a, 4) for a in (0, 1)}
s = {a: Fr(2 + a, 4) for a in (0, 1)}
Kev = {m: Fr(1, 4) + Fr(m, 2) for m in (0, 1)}
nu = {a: (1 - s[a]) * Kev[0] + s[a] * Kev[1] for a in (0, 1)}
assert nu[0] == Fr(1, 2) and nu[1] == Fr(5, 8), nu
print("nu_a(1):", nu)

S = list(product((0, 1), repeat=4))

def T(x, y) -> Fr:
    a, b, c, r = x
    a2, b2, c2, r2 = y
    return (f(Fr(1, 4), a2 ^ a) * f(Fr(1, 3), b2 ^ b ^ a) * f(Fr(1, 5), c2 ^ c ^ b)
            * f(nu[a], r2))

for x in S:
    assert sum(T(x, y) for y in S) == 1

R10 = lambda x: (x[0], x[1], x[3])
R21 = lambda z: (z[0], z[2])
S1 = sorted(set(map(R10, S)))
S2 = sorted(set(map(R21, S1)))

def lumpable(T, S, R, Sc):
    rows = {}
    for x in S:
        row = tuple(sum(T(x, y) for y in S if R(y) == z) for z in Sc)
        if R(x) in rows and rows[R(x)] != row:
            return False, None
        rows[R(x)] = row
    return True, rows

ok1, rows1 = lumpable(T, S, R10, S1)
print("R10 strongly lumpable:", ok1)
T1 = lambda z, w: rows1[z][S1.index(w)]
for z in S1:
    for w in S1:
        a, b, r = z
        a2, b2, r2 = w
        assert T1(z, w) == f(Fr(1, 4), a2 ^ a) * f(nu[a], r2) * f(Fr(1, 3), b2 ^ b ^ a)
ok2, rows2 = lumpable(T1, S1, R21, S2)
print("R21 strongly lumpable:", ok2)
for z in S2:
    for w in S2:
        a, r = z
        a2, r2 = w
        assert rows2[z][S2.index(w)] == f(Fr(1, 4), a2 ^ a) * f(nu[a], r2)
print("displayed parent kernels T^1, T^2 reproduced exactly")

# path-law agreement for a random (correlated) initial law, horizon 4
random.seed(1)
w = [Fr(random.randint(1, 9)) for _ in S]
p0 = {x: wi / sum(w) for x, wi in zip(S, w)}
def path_law_fine(n):
    out = {}
    for path in product(S, repeat=n):
        pr = p0[path[0]]
        for u, v in zip(path, path[1:]):
            pr *= T(u, v)
        key = tuple(R21(R10(x)) for x in path)
        out[key] = out.get(key, 0) + pr
    return out
def path_law_parent(n):
    pc = {}
    for x in S:
        z = R21(R10(x)); pc[z] = pc.get(z, 0) + p0[x]
    out = {}
    for path in product(S2, repeat=n):
        pr = pc[path[0]]
        for u, v in zip(path, path[1:]):
            pr *= rows2[u][S2.index(v)]
        out[path] = pr
    return out
n = 3
PF, PP = path_law_fine(n), path_law_parent(n)
maxdiff = max(abs(PF.get(k, 0) - PP.get(k, 0)) for k in set(PF) | set(PP))
print(f"max |fine - parent| over retained paths of length {n}: {maxdiff}")

# autonomy: retained coordinates' marginal transition never depends on discarded coordinates
# (this is what strong lumpability means for coordinate projections; see report L04-INH-05)
print("coordinate projection lumpable <=> retained block autonomous: verified by construction above")
