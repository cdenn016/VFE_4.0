"""Calibration example: anchored linear consensus (four Gaussian location agents).
Full flow xdot_i = -(x_i-c_i) - 2w sum_{j!=i}(x_i-x_j), w=1/4; null xdot=-(x-c);
frozen-environment xdot_i = -(x_i-c_i) - 2w sum_{j!=i}(x_i-x_j(0)).
Selector: join two current blocks when count-weighted centroid gap reaches delta from above.
All flows linear: integrate with matrix exponential on a fine time grid + bisection."""
import numpy as np
from scipy.linalg import expm
c=np.array([0.0,0.6,1.3,1.95]); N=4; w=0.25
A_full=-(np.eye(N)+2*w*(N*np.eye(N)-np.ones((N,N))))
EIG={'full':np.linalg.eigh(A_full),'null':np.linalg.eigh(-np.eye(N)),'frozen':np.linalg.eigh(-(1+2*w*(N-1))*np.eye(N))}
def traj(kind,x0,t):
    if kind=='full': A=A_full; b=c
    elif kind=='null': A=-np.eye(N); b=c
    else:
        A=-(1+2*w*(N-1))*np.eye(N); b=c+2*w*(x0.sum()-x0)
    xs=np.linalg.solve(-A,b)  # equilibrium
    lam,V=EIG[kind]
    return xs+V@(np.exp(lam*t)*np.linalg.solve(V,x0-xs))
def run(kind,x0,delta,T=4.0,dt=2e-3):
    blocks=[[i] for i in range(N)]; ev=[]
    def gaps(x):
        out={}
        for a in range(len(blocks)):
            for b in range(a+1,len(blocks)):
                out[(a,b)]=abs(x[blocks[a]].mean()-x[blocks[b]].mean())
        return out
    t=0.0; g=gaps(x0)
    if min(g.values())<=delta: return None  # outside selector domain
    while t<T and len(blocks)>1:
        t2=t+dt; x=traj(kind,x0,t2); g2=gaps(x)
        hit=[k for k,v in g2.items() if v<=delta]
        if hit:
            k=min(hit,key=lambda k:g2[k]); a,b=k
            lo,hi=t,t2
            for _ in range(50):
                m=0.5*(lo+hi); xm=traj(kind,x0,m)
                if abs(xm[blocks[a]].mean()-xm[blocks[b]].mean())<=delta: hi=m
                else: lo=m
            new=sorted(blocks[a]+blocks[b]); blocks=[bl for i,bl in enumerate(blocks) if i not in (a,b)]+[new]
            ev.append((round(hi,4),tuple(new))); t=hi; g2=gaps(traj(kind,x0,t))
            if g2 and min(g2.values())<=delta: return ('cascade',ev)
            continue
        t=t2
    return ev
x0=c.copy()
for d in [0.45,0.5,0.55]:
    print('delta',d,'full',run('full',x0,d),'null',run('null',x0,d),'frozen',run('frozen',x0,d))
print('limits full pair gaps',np.diff(np.linalg.solve(-A_full,c)))
xs=np.linalg.solve(-A_full,c); print('limit parent centroid gap',xs[2:].mean()-xs[:2].mean())
# scan delta
ok=[]
for d in np.arange(0.40,0.62,0.005):
    f=run('full',x0,d); n=run('null',x0,d)
    ok.append((round(d,3),[e[1] for e in f] if isinstance(f,list) else f,[e[1] for e in n] if isinstance(n,list) else n))
for r in ok: print(r)
# perturbations
rng=np.random.default_rng(0); same=0
for k in range(15):
    xp=c+rng.uniform(-0.01,0.01,4)
    f=[e[1] for e in run('full',xp,0.5)]; n=run('null',xp,0.5); fr=[e[1] for e in run('frozen',xp,0.5)]
    print('pert',k,f,n,fr)
