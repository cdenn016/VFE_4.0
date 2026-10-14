"""L4-attention a03: Supplement G g03 / g10 spot checks."""
import numpy as np
rng=np.random.default_rng(2)
def spd(k): A=rng.normal(size=(k,k)); return A@A.T+k*np.eye(k)
# g03 gain absorption (ii): P2 = -L12^{-1} L11 reframes into I({1,2})
for K in (2,3):
    for trial in range(200):
        L=spd(2*K); L11,L12,L22=L[:K,:K],L[:K,K:],L[K:,K:]
        if (-1)**K*np.linalg.det(L12)<=0: continue
        P2=-np.linalg.solve(L12,L11); P=np.block([[np.eye(K),np.zeros((K,K))],[np.zeros((K,K)),P2]])
        Lr=P.T@L@P; W=-Lr[:K,K:]
        A1=Lr[:K,:K]-W; A2=Lr[K:,K:]-W
        ok=(np.abs(W-W.T).max()<1e-9 and np.linalg.eigvalsh((W+W.T)/2).min()>0 and np.abs(A1).max()<1e-9
            and np.linalg.eigvalsh((A2+A2.T)/2).min()>-1e-9 and np.linalg.det(P2)>0)
        if not ok: print('g03 (ii) FAIL', K); break
    else: print('g03 (ii) construction verified for K=',K)
# g03 law-level witness (iii): v-structure triangle sign for odd K
for K in (1,3):
    Li=spd(K); Oj=rng.normal(size=(K,K)); Ojp=rng.normal(size=(K,K))
    Lij=-Li@Oj; Lijp=-Li@Ojp; Ljjp=Oj.T@Li@Ojp
    prod=np.linalg.det(Lij)*np.linalg.det(Ljjp)*np.linalg.det(Lijp.T)
    print(f'g03 (iii) K={K}: sign of triangle product', np.sign(prod), ' membership requires', (-1)**(3*K))
# g10 determinant factorization
K=3; mx=0
for _ in range(200):
    Te=rng.normal(size=(K,K))+2*np.eye(K); Tf=rng.normal(size=(K,K))+2*np.eye(K)
    Se=spd(K); Sf=spd(K)
    de=np.hstack([np.eye(K),-Te]); df=np.hstack([-Tf,np.eye(K)])
    Lam=de.T@np.linalg.solve(Se,de)+df.T@np.linalg.solve(Sf,df)
    rhs=np.linalg.det(np.eye(K)-Tf@Te)**2/(np.linalg.det(Se)*np.linalg.det(Sf))
    mx=max(mx,abs(np.linalg.det(Lam)-rhs)/abs(rhs))
print('g10 det factorization max rel err', mx)
