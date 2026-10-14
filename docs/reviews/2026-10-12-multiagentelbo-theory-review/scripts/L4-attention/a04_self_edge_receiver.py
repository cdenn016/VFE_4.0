"""L4-attention a04: comparison-record receiver update with an included self edge
(S^K_ii = S^Q_i, identity twist).  prop:obs-selective-receiver-update conditions on
k_j fixed by y_{-i}; for j=i the source state is k_i itself.  Brute-force the exact
posterior and compare with the proposition's formula applied naively to j=i."""
import numpy as np
from scipy.special import softmax
from scipy.stats import multivariate_normal as mvn
rng=np.random.default_rng(3)
def spd(k): A=rng.normal(size=(k,k)); return A@A.T+k*np.eye(k)
K=2; d=2; Sig=spd(K); mu=rng.normal(size=K); SQ=np.eye(d)@np.eye(K); W=np.eye(d); tau=1.0; Wt=W/tau
kj=[rng.normal(size=K)*1.5 for _ in range(2)]; pi=np.array([1/3,1/3,1/3])  # sources: self, j1, j2
M=600000; ks=rng.multivariate_normal(mu,Sig,size=M)
nz=mvn(mean=np.zeros(d),cov=np.linalg.inv(Wt))
lik_self=nz.pdf(ks@SQ.T-ks@SQ.T)            # record o = S^Q k_i - S^Q k_i + noise: uninformative
lik=[nz.pdf(ks@SQ.T-SQ@k) for k in kj]
L=np.vstack([lik_self]+lik); jw=pi[:,None]*L
w_mc=jw.sum(1)/jw.sum(); mean_mc=(jw.sum(0)[:,None]*ks).sum(0)/jw.sum()
# exact closed form with self edge
Sp=np.linalg.inv(np.linalg.inv(Sig)+SQ.T@Wt@SQ); Spred=np.linalg.inv(Wt)+SQ@Sig@SQ.T
lw_self=np.log(pi[0])+nz.logpdf(np.zeros(d))
lw=[np.log(p)+mvn(mean=SQ@mu-SQ@k,cov=Spred).logpdf(np.zeros(d)) for p,k in zip(pi[1:],kj)]
w_ex=softmax(np.array([lw_self]+lw))
mean_ex=w_ex[0]*mu+sum(w*(Sp@np.linalg.solve(Sig,mu)+Sp@SQ.T@Wt@SQ@k) for w,k in zip(w_ex[1:],kj))
# naive application of the proposition with k_i -> mu_i as the self "source state"
r=[SQ@mu-SQ@mu]+[SQ@mu-SQ@k for k in kj]
w_nv=softmax(np.log(pi)+np.array([-.5*x@np.linalg.solve(Spred,x) for x in r]))
mean_nv=Sp@np.linalg.solve(Sig,mu)+sum(w*Sp@SQ.T@Wt@SQ@k for w,k in zip(w_nv,[mu]+kj))
print('MC weights          ', np.round(w_mc,4))
print('exact (self handled)', np.round(w_ex,4))
print('naive proposition   ', np.round(w_nv,4))
print('MC mean', np.round(mean_mc,4), ' exact', np.round(mean_ex,4), ' naive', np.round(mean_nv,4))
print('self-edge boost factor det(Spred Wt)^{1/2} =', np.sqrt(np.linalg.det(Spred@Wt)))
