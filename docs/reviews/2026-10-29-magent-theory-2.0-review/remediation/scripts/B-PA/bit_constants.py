"""Chapter 13 Bernoulli construction (pa:bit-flow): partition sequences of the
full flow, the label-preserving null (N1), the renormalized null (N2) and the
frozen-environment control (N3) over selector constants c; plus the
four-state construction on its product-content family. Single-threaded RK4."""
import numpy as np, sys
PART=sys.argv[1]
sig=lambda x:1/(1+np.exp(-x))
def kl(l,lp):
    p=sig(l);pp=sig(lp)
    return p*np.log(p/pp)+(1-p)*np.log((1-p)/(1-pp))
a,p=0.2,0.4
o=np.array([1.,1.,-1.])
def rows(l,lpeer):
    # self cost: KL to uniform evaluator; peer costs to lpeer
    w=np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            w[i,j]=(a*np.exp(-kl(l[i],0.0)) if i==j else p*np.exp(-kl(l[i],lpeer[j])))
    return w/w.sum(1,keepdims=True)
def field(l,mode,eps,l0):
    if mode=='full':
        B=rows(l,l); return -l+(B*(1-np.eye(3)))@l+eps*o
    if mode=='fr':
        out=np.zeros(3)
        for i in range(3):
            lp=l0.copy(); lp[i]=l[i]; B=rows(lp,lp)
            out[i]=-l[i]+sum(B[i,j]*l0[j] for j in range(3) if j!=i)+eps*o[i]
        return out
    if mode=='n1':
        b0=np.array([a*np.exp(-kl(x,0))/(a*np.exp(-kl(x,0))+2*p) for x in l]); return -b0*l+eps*o
    if mode=='n2': return -l+eps*o
def part(l,c):
    import itertools
    par=list(range(3))
    def f(i):
        while par[i]!=i: i=par[i]
        return i
    for i,j in itertools.combinations(range(3),2):
        if np.sign(l[i])==np.sign(l[j])!=0 and min(abs(l[i]),abs(l[j]))/max(abs(l[i]),abs(l[j]))>c:
            par[f(i)]=f(j)
    g={}
    for i in range(3): g.setdefault(f(i),[]).append(i+1)
    return '|'.join(''.join(map(str,v)) for v in sorted(g.values()))
def run(mode,eps,u0,c,T=80,dt=0.05):
    l=eps*np.array(u0,float); l0=l.copy(); seq=[part(l,c)]; mx=0
    for k in range(int(T/dt)):
        f=lambda x:field(x,mode,eps,l0)
        k1=f(l);k2=f(l+dt/2*k1);k3=f(l+dt/2*k2);k4=f(l+dt*k3)
        l=l+dt/6*(k1+2*k2+2*k3+k4); s=part(l,c)
        if s!=seq[-1]: seq.append(s)
        if l[2]>0: mx=max(mx,l[2]/max(l[0],l[1]))
    return seq,l/eps,mx
if PART=='bern':
 for eps in [0.01,0.05]:
    for c in [0.13,0.2,0.25,0.3,0.33,0.34,0.36,0.4,0.6,0.9]:
        res={m:run(m,eps,[4,.5,-1],c) for m in ['full','n1','n2','fr']}
        print(f"eps={eps} c={c}: "+"; ".join(f"{m}:{'>'.join(r[0])}" for m,r in res.items()))
    r=run('full',eps,[4,.5,-1],0.25); print(' final u full',np.round(r[1],4),'sup agent-3 ratio',round(r[2],5))
    for m in ['n1','n2','fr']: print(' final u',m,np.round(run(m,eps,[4,.5,-1],0.25)[1],4))
if PART=='pert':
 rng=np.random.default_rng(0); agree={'lp':0,'rn':0,'fr':0}; n=20
 for t in range(n):
     u0=np.array([4,.5,-1])+rng.uniform(-.05,.05,3)
     P=run('full',0.02,u0,0.25,T=80,dt=0.05)[0]
     agree['lp']+=P==run('n1',0.02,u0,0.25,T=80,dt=0.05)[0]
     agree['rn']+=P==run('n2',0.02,u0,0.25,T=80,dt=0.05)[0]
     agree['fr']+=P==run('fr',0.02,u0,0.25,T=80,dt=0.05)[0]
 print('perturbations eps=0.02 c=1/4: P equal to control in',agree,'of',n)
 
# four-state construction on the product-content family
J=1.5; rq=np.cosh(J)/(1+np.cosh(J)); rs=0.5
def klb(m,mp):
    p1,p2=(1+m)/2,(1+mp)/2
    return p1*np.log(p1/p2)+(1-p1)*np.log((1-p1)/(1-p2))
def ent(m):
    p1=(1+m)/2; return -(p1*np.log(p1)+(1-p1)*np.log(1-p1))
def frows(m,n,mp,np_):
    # content self cost: -H(q)-J m n + log(4coshJ) with q on 4 states (H includes log2 of uniform x)
    Dq=-(ent(m)+np.log(2))-J*m*n+np.log(4*np.cosh(J)); Dp=klb(m,mp)
    Ds=np.log(2)-ent(n); Dsp=klb(n,np_)  # KL(s||uniform on 4) = log4 - H(s)=log2-H_b
    wb=np.array([rq*np.exp(-Dq),(1-rq)*np.exp(-Dp)]); wg=np.array([rs*np.exp(-Ds),(1-rs)*np.exp(-Dsp)])
    return wb/wb.sum(), wg/wg.sum(), Dq, Ds
eta,h=0.02,2e-4; H=np.array([h,-h])
def f4(x,mode,x0):
    u,v=x[:2],x[2:]; m,n=np.tanh(u),np.tanh(v); out=np.zeros(4)
    for i in range(2):
        j=1-i
        mj,nj=(np.tanh(x0[j]),np.tanh(x0[2+j])) if mode=='fr' else (m[j],n[j])
        uj,vj=(x0[j],x0[2+j]) if mode=='fr' else (u[j],v[j])
        b,g,Dq,Ds=frows(m[i],n[i],mj,nj)
        if mode in('full','fr'):
            out[i]=-u[i]+b[1]*uj+b[0]*J*n[i]+H[i]; out[2+i]=-v[i]+g[1]*vj+b[0]*J*m[i]+H[i]
        else:
            b0=1.0 if mode=='n2' else rq*np.exp(-Dq)/(rq*np.exp(-Dq)+1-rq)
            g0=1.0 if mode=='n2' else rs*np.exp(-Ds)/(rs*np.exp(-Ds)+1-rs)
            out[i]=-b0*u[i]+b0*J*n[i]+H[i]; out[2+i]=-g0*v[i]+b0*J*m[i]+H[i]
    return out
def sel(x,c):
    m=np.tanh(x); return 'pair' if (np.all(m>c) or np.all(m<-c)) else '1|2'
if PART=='four':
 for mode in ['full','n1','n2','fr']:
     x=np.arctanh(eta*np.array([3,-1,3,-1.])); x0=x.copy(); dt=0.02; seq=[(sel(x,eta/(8*np.sqrt(2))),0.0)]
     for k in range(int(60/dt)):
         f=lambda y:f4(y,mode,x0)
         k1=f(x);k2=f(x+dt/2*k1);k3=f(x+dt/2*k2);k4=f(x+dt*k3); x=x+dt/6*(k1+2*k2+2*k3+k4)
         s=sel(x,eta/(8*np.sqrt(2)))
         if s!=seq[-1][0]: seq.append((s,round((k+1)*dt,2)))
     print('four-state',mode,seq,'final m,n',np.round(np.tanh(x),4))
 