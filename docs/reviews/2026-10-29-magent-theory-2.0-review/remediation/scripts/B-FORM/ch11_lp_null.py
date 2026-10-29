"""Chapter 11: label-preserving null (N1) versus full flow and renormalized null (N2).
N1 law (derived by hand): alpha_i' = -(1+b_i)(alpha_i-alpha0),
P_i' = -(1+b_i) P_i + I + b_i C_i^0, b_i = a e^{-D_ii/T}/(a e^{-D_ii/T}+1-a), a=1/N,
D_ii = KL(W(alpha_i,P_i) || W(alpha0, Lam_i^0)), Lam_i^0 = I + (kappa/N)(alpha_i P_i^{-1} + M0_{-i}),
C_i^0 = Lam_i^0 + (kappa/N)(alpha_i P_i^{-1} - alpha0 Lam_i^{-1}).
Selector: block score = mean shape - alpha0 from fine states; join when distance first reaches delta from above.
"""
import numpy as np
from scipy.special import gammaln, digamma
N=4
def lg2(a): return 0.5*np.log(np.pi)+gammaln(a)+gammaln(a-0.5)
def ps2(a): return digamma(a)+digamma(a-0.5)
def KLW(a,P,b,Q):
    Pi=np.linalg.inv(P)
    return (a-b)*ps2(a)-lg2(a)+lg2(b)+b*np.log(np.linalg.det(P)/np.linalg.det(Q))+a*(np.trace(Q@Pi)-2)
def field(state,mode,kap,a0,T,M0):
    al,P=state
    dal=np.zeros(N); dP=np.zeros_like(P); I=np.eye(2)
    for i in range(N):
        Pi=np.linalg.inv(P[i]); Lam=I+(kap/N)*(al[i]*Pi+M0[i]); Li=np.linalg.inv(Lam)
        C=Lam+(kap/N)*(al[i]*Pi-a0*Li)
        if mode=='rn': b=1.0
        else:
            D=KLW(al[i],P[i],a0,Lam); e=np.exp(-D/T)/N; b=e/(e+1-1/N)
        dal[i]=-(1+b)*(al[i]-a0); dP[i]=-(1+b)*P[i]+I+b*C
    return dal,dP
def run(u,P0,kap,a0,T,delta,mode,tmax=6.0,h=2e-3):
    al=np.array(u,float); P=np.array([P0.copy() for _ in range(N)])
    M0tot=sum(al[j]*np.linalg.inv(P[j]) for j in range(N))
    M0=[M0tot-al[i]*np.linalg.inv(P[i]) for i in range(N)]
    blocks=[(i,) for i in range(N)]; hist=[]; t=0.0
    def score(al,B): return np.mean(al[list(B)])-a0
    def step(al,P):
        if mode=='full': return -2*(al-a0),None
        return field((al,P),mode,kap,a0,T,M0)
    def rk4(al,P,h):
        if mode=='full':
            k=lambda x:-2*(x-a0)
            k1=k(al);k2=k(al+h/2*k1);k3=k(al+h/2*k2);k4=k(al+h*k3)
            return al+h/6*(k1+2*k2+2*k3+k4),P
        a1,p1=step(al,P);a2,p2=step(al+h/2*a1,P+h/2*p1);a3,p3=step(al+h/2*a2,P+h/2*p2);a4,p4=step(al+h*a3,P+h*p3)
        return al+h/6*(a1+2*a2+2*a3+a4),P+h/6*(p1+2*p2+2*p3+p4)
    margins=[]
    while t<tmax and len(blocks)>1:
        an,Pn=rk4(al,P,h)
        ev=None
        for x in range(len(blocks)):
            for y in range(x+1,len(blocks)):
                d0=abs(score(al,blocks[x])-score(al,blocks[y])); d1=abs(score(an,blocks[x])-score(an,blocks[y]))
                if d0>delta and d1<=delta:
                    lo,hi=0.0,h
                    for _ in range(50):
                        m=(lo+hi)/2; am,_p=rk4(al,P,m)
                        if abs(score(am,blocks[x])-score(am,blocks[y]))>delta: lo=m
                        else: hi=m
                    te=t+hi
                    if ev is None or te<ev[0]: ev=(te,x,y)
        if ev is None:
            al,P=an,Pn; t+=h; continue
        te,x,y=ev; al,P=rk4(al,P,te-t); t=te
        others=[abs(score(al,blocks[p])-score(al,blocks[q]))-delta for p in range(len(blocks)) for q in range(p+1,len(blocks)) if {p,q}!={x,y}]
        margins.append(min(others) if others else np.inf)
        nb=tuple(sorted(blocks[x]+blocks[y])); blocks=[b for k,b in enumerate(blocks) if k not in (x,y)]+[nb]
        hist.append((round(te,4),tuple(i+1 for i in nb)))
    return hist,margins
if __name__=='__main__':
    I=np.eye(2)
    u=(1.2807,1.5057,1.8665,2.3057)
    for mode in ['full','rn','lp']:
        print(mode,run(u,I,0.0,1.0,1.0,0.1,mode))
    print('chapter example kappa=1, lp:',run((1,1.2,1.8,2.1),I,1.0,1.0,1.0,0.1,'lp'))
    print('chapter example kappa=1, rn:',run((1,1.2,1.8,2.1),I,1.0,1.0,1.0,0.1,'rn'))
    rng=np.random.default_rng(0); cnt=0; mins=[]
    for k in range(12):
        uu=np.array(u)+rng.uniform(-4e-3,4e-3,4)
        A=rng.uniform(-1e-2,1e-2,(2,2)); P0=I+(A+A.T)/2
        kap=rng.choice([0.0,0.01]); T=rng.uniform(0.9,1.1); de=rng.uniform(0.098,0.102)
        hf,_=run(uu,P0,kap,1.0,T,de,'full'); hl,ml=run(uu,P0,kap,1.0,T,de,'lp')
        sf=[b for _,b in hf]; sl=[b for _,b in hl]
        cnt+= (sl[1]==(1,2,3) and sf[1]==(3,4)); mins.append(min(ml))
        print(k,'kap',kap,'T%.3f d%.4f'%(T,de),sf,sl,'lp margin %.4f'%min(ml))
    print('structural difference in',cnt,'of 12; min lp margin %.4f'%min(mins))
