"""L4-attention a05: two-head comparison-record receiver (rows (i,1),(i,2) of
def:obs-attention-label-model).  Exact posterior mean via the joint label table
(closed form) vs Monte Carlo; mean-field label factorization gives the
multi-head operator form sum_h sum_j b^h_j Sig+ S_h^T Wt_h S^K_hj k_j."""
import numpy as np
from scipy.special import softmax
from scipy.stats import multivariate_normal as mvn
rng=np.random.default_rng(5)
def spd(k): A=rng.normal(size=(k,k)); return A@A.T+k*np.eye(k)
K=3; Sig=spd(K)*.5; mu=rng.normal(size=K); H=2; d=[2,1]
SQ=[rng.normal(size=(dh,K)) for dh in d]; Wt=[spd(dh) for dh in d]
ks=[rng.normal(size=K) for _ in range(3)]
SK=[[rng.normal(size=(dh,K)) for _ in range(3)] for dh in d]
pi=[np.ones(3)/3, np.ones(3)/3]
Sp=np.linalg.inv(np.linalg.inv(Sig)+sum(S.T@W@S for S,W in zip(SQ,Wt)))
# exact: joint table over (j1,j2); stacked record o=(S1 k - SK1 k_j1, S2 k - SK2 k_j2) at 0
Sst=np.vstack(SQ); Nst=np.block([[np.linalg.inv(Wt[0]),np.zeros((2,1))],[np.zeros((1,2)),np.linalg.inv(Wt[1])]])
Spred=Nst+Sst@Sig@Sst.T
lt=np.zeros((3,3)); means=np.zeros((3,3,K))
for a in range(3):
    for b in range(3):
        kk=np.concatenate([SK[0][a]@ks[a],SK[1][b]@ks[b]])
        lt[a,b]=np.log(pi[0][a]*pi[1][b])+mvn(mean=Sst@mu-kk,cov=Spred).logpdf(np.zeros(3))
        means[a,b]=Sp@(np.linalg.solve(Sig,mu)+SQ[0].T@Wt[0]@SK[0][a]@ks[a]+SQ[1].T@Wt[1]@SK[1][b]@ks[b])
w=np.exp(lt-lt.max()); w/=w.sum(); mean_ex=np.einsum('ab,abk->k',w,means)
M=500000; X=rng.multivariate_normal(mu,Sig,size=M); jw=np.zeros(M); acc=np.zeros(K)
for a in range(3):
    for b in range(3):
        l1=mvn(mean=np.zeros(2),cov=np.linalg.inv(Wt[0])).pdf(X@SQ[0].T-SK[0][a]@ks[a])
        l2=mvn(mean=np.zeros(1),cov=np.linalg.inv(Wt[1])).pdf(X@SQ[1].T-SK[1][b]@ks[b])
        ww=pi[0][a]*pi[1][b]*l1*l2; jw+=ww
mean_mc=(jw[:,None]*X).sum(0)/jw.sum()
print('exact two-head mean', np.round(mean_ex,4), ' MC', np.round(mean_mc,4))
r1=w.sum(1); r2=w.sum(0); print('label table factorizes? max|w - r1 r2^T| =', np.abs(w-np.outer(r1,r2)).max())
# mean-field operator form with the exact marginals plugged in:
mf=Sp@np.linalg.solve(Sig,mu)+sum(sum(bh[j]*Sp@SQ[h].T@Wt[h]@SK[h][j]@ks[j] for j in range(3)) for h,bh in enumerate([r1,r2]))
print('multi-head operator form with exact head marginals', np.round(mf,4), ' |diff to exact|', np.abs(mf-mean_ex).max())
# per-head single-head softmax rows (each head alone, prop:obs-selective-receiver-update)
for h in range(2):
    Sph=np.linalg.inv(Wt[h])+SQ[h]@Sig@SQ[h].T
    lw=[np.log(1/3)+mvn(mean=SQ[h]@mu-SK[h][j]@ks[j],cov=Sph).logpdf(np.zeros(d[h])) for j in range(3)]
    print(f'head {h}: single-head row', np.round(softmax(lw),4), ' exact coupled marginal', np.round([r1,r2][h],4))
