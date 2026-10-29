r"""Exact symbolic recomputation of imem:local-retention-failure (parity example).

Lens D08-IMEM2.  Scope: appendices/interacting_memory.tex lines 617-653 and
companion/c04_interacting_memory.tex lines 213-225.

Preparation: X0 fair, B independent with P(B=0)=rho, X1 = X0 xor B; A_i, Y_i
from the 3/4 kernels of imem:binary-preparation; D0 = * (uninformative);
P(D1 = X1 | X1) = 3/4.  Public record C = X0 xor X1 = B.  Agent 1 samples
H from P(Y1 | D1, C) (identity encoder).  rho is a sympy symbol, so every
equality below is an identity in rho on (0,1).
"""
from __future__ import annotations

import itertools
from typing import Callable, Dict, List, Tuple

import sympy as sp

rho = sp.symbols("rho", positive=True)
q3, q1 = sp.Rational(3, 4), sp.Rational(1, 4)
half = sp.Rational(1, 2)

checks: List[Tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, bool(ok), detail))


def ch(a: int, b: int) -> sp.Expr:
    return q3 if a == b else q1


# atoms e = (B, X0, X1, A0, A1, Y0, Y1, D1)
joint: Dict[Tuple[int, ...], sp.Expr] = {}
for b, x0, a0, a1, y0, y1, d1 in itertools.product((0, 1), repeat=7):
    x1 = x0 ^ b
    p = (rho if b == 0 else 1 - rho) * half
    p *= ch(a0, x0) * ch(a1, x1) * (q3 if y0 == 1 - x0 else q1) * (q3 if y1 == 1 - x1 else q1)
    p *= ch(d1, x1)
    joint[(b, x0, x1, a0, a1, y0, y1, d1)] = p
IB, IX0, IX1, IA0, IA1, IY0, IY1, ID1 = range(8)


def P(pred: Callable[[Tuple[int, ...]], bool], J=None) -> sp.Expr:
    J = joint if J is None else J
    return sp.simplify(sum((p for e, p in J.items() if pred(e)), sp.Integer(0)))


def Pc(num: Callable, den: Callable, J=None) -> sp.Expr:
    return sp.simplify(P(lambda e: num(e) and den(e), J) / P(den, J))


check("normalized", sp.simplify(P(lambda e: True) - 1) == 0)
check("P(C=0)=rho, P(C=1)=1-rho (both positive for 0<rho<1)",
      sp.simplify(P(lambda e: e[IB] == 0) - rho) == 0)

# Local laws given C (and D1 for agent 1) do not depend on C
loc0 = {}
loc1 = {}
for c in (0, 1):
    for z in itertools.product((0, 1), repeat=3):
        loc0[(c, z)] = Pc(lambda e, z=z: (e[IA0], e[IX0], e[IY0]) == z, lambda e, c=c: e[IB] == c)
        for d in (0, 1):
            loc1[(c, d, z)] = Pc(lambda e, z=z: (e[IA1], e[IX1], e[IY1]) == z,
                                 lambda e, c=c, d=d: e[IB] == c and e[ID1] == d)
check("agent 0 local law P(A0,X0,Y0|C) identical for C=0,1 and free of rho",
      all(sp.simplify(loc0[(0, z)] - loc0[(1, z)]) == 0 and not loc0[(0, z)].has(rho)
          for z in itertools.product((0, 1), repeat=3)))
check("agent 1 local law P(A1,X1,Y1|C,D1=d) identical for C=0,1, each d, free of rho",
      all(sp.simplify(loc1[(0, d, z)] - loc1[(1, d, z)]) == 0 and not loc1[(0, d, z)].has(rho)
          for d in (0, 1) for z in itertools.product((0, 1), repeat=3)))
lawZ1D1 = {c: [Pc(lambda e, w=w: (e[IA1], e[IX1], e[IY1], e[ID1]) == w, lambda e, c=c: e[IB] == c)
               for w in itertools.product((0, 1), repeat=4)] for c in (0, 1)}
check("whole local law of (A1,X1,Y1,D1) independent of C",
      all(sp.simplify(u - v) == 0 for u, v in zip(lawZ1D1[0], lawZ1D1[1])))

# Cross-agent dependence differs
cov = {c: sp.simplify(Pc(lambda e: e[IX0] == 1 and e[IX1] == 1, lambda e, c=c: e[IB] == c) - q1)
       for c in (0, 1)}
check("Cov(X0,X1|C=0)=+1/4, Cov(X0,X1|C=1)=-1/4", cov == {0: q1, 1: -q1}, str(cov))

# Posterior-sampling message from agent 1: law P(Y1 | D1, C)
s = {(d, c): Pc(lambda e: e[IY1] == 1, lambda e, d=d, c=c: e[ID1] == d and e[IB] == c)
     for d in (0, 1) for c in (0, 1)}
check("P(H=1|D1=0,C)=5/8 and P(H=1|D1=1,C)=3/8 for both C",
      all(s[(0, c)] == sp.Rational(5, 8) and s[(1, c)] == sp.Rational(3, 8) for c in (0, 1)), str(s))
jointH: Dict[Tuple[int, ...], sp.Expr] = {}
for e, p in joint.items():
    sv = s[(e[ID1], e[IB])]
    jointH[e + (1,)] = p * sv
    jointH[e + (0,)] = p * (1 - sv)
IH = 8
lik = {(x, c): Pc(lambda e: e[IH] == 1, lambda e, x=x, c=c: e[IX1] == x and e[IB] == c, jointH)
       for x in (0, 1) for c in (0, 1)}
check("P(H=1|X1=0,C)=9/16, P(H=1|X1=1,C)=7/16",
      all(lik[(0, c)] == sp.Rational(9, 16) and lik[(1, c)] == sp.Rational(7, 16) for c in (0, 1)), str(lik))
pH = {c: Pc(lambda e: e[IH] == 1, lambda e, c=c: e[IB] == c, jointH) for c in (0, 1)}
check("P(H=1|C)=1/2 for both parities", pH == {0: half, 1: half}, str(pH))

post = {c: Pc(lambda e: e[IX0] == 1, lambda e, c=c: e[IB] == c and e[IH] == 1, jointH) for c in (0, 1)}
check("P(X0=1|C=0,H=1)=7/16, P(X0=1|C=1,H=1)=9/16",
      post == {0: sp.Rational(7, 16), 1: sp.Rational(9, 16)}, str(post))
outA = {c: Pc(lambda e: e[IA0] == 1, lambda e, c=c: e[IB] == c and e[IH] == 1, jointH) for c in (0, 1)}
outY = {c: Pc(lambda e: e[IY0] == 1, lambda e, c=c: e[IB] == c and e[IH] == 1, jointH) for c in (0, 1)}
check("P(A0=1|C,H=1) = 15/32, 17/32", outA == {0: sp.Rational(15, 32), 1: sp.Rational(17, 32)}, str(outA))
check("P(Y0=1|C,H=1) = 17/32, 15/32", outY == {0: sp.Rational(17, 32), 1: sp.Rational(15, 32)}, str(outY))

# Agent 0 local law after the message differs between C=0 and C=1 (fiber test fails)
loc0H = {c: [Pc(lambda e, z=z: (e[IA0], e[IX0], e[IY0]) == z, lambda e, c=c: e[IB] == c and e[IH] == 1, jointH)
             for z in itertools.product((0, 1), repeat=3)] for c in (0, 1)}
check("receiver local joint law after H=1 differs between the two histories",
      any(sp.simplify(u - v) != 0 for u, v in zip(loc0H[0], loc0H[1])))
# Sender invariance in the extended (public-record) protocol
loc1H_ok = all(
    sp.simplify(Pc(lambda e, z=z: (e[IA1], e[IX1], e[IY1]) == z,
                   lambda e, c=c, d=d, h=h: e[IB] == c and e[ID1] == d and e[IH] == h, jointH)
                - loc1[(c, d, z)]) == 0
    for c in (0, 1) for d in (0, 1) for h in (0, 1) for z in itertools.product((0, 1), repeat=3))
check("sender's local law unchanged by its own message (both C, both d, both h)", loc1H_ok)

# The receiver's NEXT outgoing law (reply) also differs: P(Y0=1|C,H=1)
check("receiver's next sampling law differs (17/32 vs 15/32)", outY[0] != outY[1])

# Each conditional preparation Lambda_0(.|C=c) is itself a preparation of the declared
# form imem:preparation with the SAME local banks: X1 = X0 xor c is a coupling, D0 a
# constant kernel, D1 a kernel of X1 alone.  (Checked: local laws above are C-free.)
# The record C itself is a function of BOTH cues, so it is neither a private record
# L_i(u_i, dd_i) nor a posterior-sampled message.
dep = all(sp.simplify(Pc(lambda e: e[IB] == 1, lambda e, x=x: e[IX0] == x) - (1 - rho)) == 0 and
          sp.simplify(Pc(lambda e: e[IB] == 1, lambda e, x=x: e[IX1] == x) - (1 - rho)) == 0
          for x in (0, 1))
check("C is independent of each single cue (not a function of either cue alone)", dep)

width = max(len(c[0]) for c in checks)
for name, ok, detail in checks:
    print(f"{'PASS' if ok else 'FAIL'}  {name.ljust(width)}  {detail}")
print(f"\n{sum(ok for _, ok, _ in checks)}/{len(checks)} checks passed")
