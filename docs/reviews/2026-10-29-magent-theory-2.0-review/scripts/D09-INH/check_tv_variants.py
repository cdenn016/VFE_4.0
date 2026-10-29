r"""D09-INH: reconcile the 2026-10-22 L04-INH-03 number 53/2000 with the current text's 1/10.

Computes, for the inh:intertwining example from delta_1 versus the T^c chain from delta_a,
(i) total variation of retained path laws (Z_0..Z_n), (ii) total variation of the one-time
marginal law of Z_n, (iii) max absolute difference of single path probabilities.
"""
from fractions import Fraction as Fr
import itertools
T = [[Fr(3,10),Fr(5,10),Fr(2,10)],[Fr(4,10),Fr(2,10),Fr(4,10)],[Fr(1,4),Fr(1,4),Fr(1,2)]]
R = ['a','a','b']; Tc = {('a','a'):Fr(7,10),('a','b'):Fr(3,10),('b','a'):Fr(1,2),('b','b'):Fr(1,2)}
for n in range(1,6):
    fine = {}
    for path in itertools.product(range(3), repeat=n):
        pr = T[0][path[0]]
        for k in range(n-1): pr *= T[path[k]][path[k+1]]
        key = ('a',)+tuple(R[x] for x in path); fine[key] = fine.get(key,0)+pr
    par = {}
    for path in itertools.product('ab', repeat=n):
        key = ('a',)+path; pr = Fr(1)
        for k in range(n): pr *= Tc[(key[k], key[k+1])]
        par[key] = pr
    tvp = sum(abs(fine.get(k,0)-par.get(k,0)) for k in set(fine)|set(par))/2
    mf = {z: sum(v for k,v in fine.items() if k[-1]==z) for z in 'ab'}
    mp = {z: sum(v for k,v in par.items() if k[-1]==z) for z in 'ab'}
    tvm = sum(abs(mf[z]-mp[z]) for z in 'ab')/2
    mx = max(abs(fine.get(k,0)-par.get(k,0)) for k in set(fine)|set(par))
    print(f"n={n} transitions: path-law TV = {tvp}, marginal TV of Z_n = {tvm} = {float(tvm):.5f}, max single-path gap = {mx}")
