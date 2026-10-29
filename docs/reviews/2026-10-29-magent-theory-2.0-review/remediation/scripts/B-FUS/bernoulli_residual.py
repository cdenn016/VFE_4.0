"""Residual n KL(q x s || R) - KL(law of (S_k,S_m) | recognition || reference) for n=2
Bernoulli children with reference R(k,m) = Ber(rho)(m) e^{Jkm}/(1+e^{Jm}).
Gaussian-location analogue is the constant (n-1)C; here we test state dependence."""
import itertools, math
J, rho, n = 0.7, 0.4, 2
def R(k, m):
    pm = rho if m else 1 - rho
    g1 = math.exp(J*m)/(1 + math.exp(J*m))
    return pm*(g1 if k else 1 - g1)
def Q(k, m, p, pp):
    return (p if k else 1-p)*(pp if m else 1-pp)
def res(p, pp):
    kl1 = sum(Q(k,m,p,pp)*math.log(Q(k,m,p,pp)/R(k,m)) for k in (0,1) for m in (0,1))
    PT, RT = {}, {}
    for ch in itertools.product([(0,0),(0,1),(1,0),(1,1)], repeat=n):
        T = (sum(c[0] for c in ch), sum(c[1] for c in ch))
        PT[T] = PT.get(T,0) + math.prod(Q(*c,p,pp) for c in ch)
        RT[T] = RT.get(T,0) + math.prod(R(*c) for c in ch)
    klT = sum(PT[T]*math.log(PT[T]/RT[T]) for T in PT)
    return n*kl1 - klT
for st in [(0.5,0.5),(0.2,0.7),(0.05,0.05),(0.9,0.9)]:
    print(st, '%.9f' % res(*st))
