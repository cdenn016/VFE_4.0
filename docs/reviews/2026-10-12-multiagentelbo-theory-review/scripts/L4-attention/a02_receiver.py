"""L4-attention a02: dot-product reduction, exact receiver update, Woodbury,
limit, example (b), mean-field (CAVI) comparison, head coupling, cross ratio."""
import numpy as np
from scipy.special import softmax, logsumexp
from scipy.stats import multivariate_normal as mvn
rng = np.random.default_rng(1)
def spd(k): A=rng.normal(size=(k,k)); return A@A.T+k*np.eye(k)

# --- (1) dot-product reduction (cor:obs-selective-softmax b)
dc=3; SigK=spd(dc); SigQ=spd(dc); muQ=rng.normal(size=dc); muK=rng.normal(size=(4,dc))
def KL(m0,S0,m1,S1):
    S1i=np.linalg.inv(S1); k=len(m0)
    return .5*(np.trace(S1i@S0)+(m1-m0)@S1i@(m1-m0)-k+np.log(np.linalg.det(S1)/np.linalg.det(S0)))
P=np.linalg.inv(SigK)
rem=[KL(muQ,SigQ,m,SigK)-(-muQ@P@m+.5*m@P@m) for m in muK]
print('[1] dot-product remainder spread over sources', np.ptp(rem))

# --- (2) exact receiver update, brute force on a grid-free basis:
# joint over (k_i, J) given k_j fixed; compute posterior exactly by Gaussian algebra
# independently: posterior weight = int N(k;mu,Sig) N(0; SQ k - SK_j k_j, Wt^{-1}) dk
K=3; d=2; Sig=spd(K); mu=rng.normal(size=K); SQ=rng.normal(size=(d,K)); W=spd(d); tau=0.8
Wt=W/tau; SK=[rng.normal(size=(d,K)) for _ in range(4)]; kj=[rng.normal(size=K) for _ in range(4)]
pi=rng.dirichlet(np.ones(4))
Sp=np.linalg.inv(np.linalg.inv(Sig)+SQ.T@Wt@SQ); Spred=np.linalg.inv(Wt)+SQ@Sig@SQ.T
r=[SQ@mu-S@k for S,k in zip(SK,kj)]
w=softmax(np.log(pi)+np.array([-.5*x@np.linalg.solve(Spred,x) for x in r]))
# Monte Carlo-free check via importance integration with many samples
M=400000; ks=rng.multivariate_normal(mu,Sig,size=M)
lik=np.array([mvn(mean=np.zeros(d),cov=np.linalg.inv(Wt)).pdf((ks@SQ.T)-(S@k)) for S,k in zip(SK,kj)])  # (4,M)
jw=pi[:,None]*lik
wmc=jw.sum(1)/jw.sum()
mean_mc=(jw.sum(0)[:,None]*ks).sum(0)/jw.sum()
Lval=[Sp@SQ.T@Wt@S for S in SK]
mean_f=Sp@np.linalg.solve(Sig,mu)+sum(wj*L@k for wj,L,k in zip(w,Lval,kj))
print('[2] weights formula', np.round(w,5), ' MC', np.round(wmc,5))
print('[2] mean formula', np.round(mean_f,4), ' MC', np.round(mean_mc,4))
print('[2] Sig+Sig^-1 = I - Sig+ SQ^T Wt SQ residual', np.abs(Sp@np.linalg.inv(Sig)-(np.eye(K)-Sp@SQ.T@Wt@SQ)).max())
print('[2] Woodbury residual', np.abs(Sp@SQ.T@Wt-Sig@SQ.T@np.linalg.inv(Spred)).max())
# limit Wt^{-1}->0
for s in [1e-2,1e-4,1e-6]:
    Wt2=W/s; Sp2=np.linalg.inv(np.linalg.inv(Sig)+SQ.T@Wt2@SQ)
    Re=Sig@SQ.T@np.linalg.inv(SQ@Sig@SQ.T)
    print(f'[2] tau={s}: |Lval - Re SK|', np.abs(Sp2@SQ.T@Wt2@SK[0]-Re@SK[0]).max())
# exact weights are a dot product with metric Spred^{-1} (j-independent) + key-norm bias
Pp=np.linalg.inv(Spred); q=SQ@mu
logit_dot=np.log(pi)+np.array([q@Pp@(S@k)-.5*(S@k)@Pp@(S@k) for S,k in zip(SK,kj)])
print('[2] exact weights = softmax(log pi + q^T Spred^{-1} key - 1/2|key|^2) diff', np.abs(softmax(logit_dot)-w).max())

# example (b)
Spred_s=1+1; w2=np.exp(-.5*4/Spred_s); print('[2b] w_i2', w2/(1+w2), np.exp(-1)/(1+np.exp(-1)), ' beta^P_2', np.exp(-2)/(1+np.exp(-2)))

# --- (3) mean-field CAVI vs exact single-site update when sources are uncertain
# sources k_j ~ N(mu_j, Sig_j) independent; mean-field q(k_i)q(J_i)prod q(k_j)
muj=[rng.normal(size=K) for _ in range(4)]; Sigj=[spd(K)*0.3 for _ in range(4)]
# CAVI fixed point for q(k_i)=N(m,Sp), q(J)=b, with q(k_j) held at their priors (sources not updated)
m=mu.copy()
for it in range(200):
    logits=np.log(pi)-np.array([.5*(SQ@m-S@mj)@Wt@(SQ@m-S@mj)+.5*np.trace(Wt@SQ@Sp@SQ.T)+.5*np.trace(Wt@S@Sj@S.T)
                                for S,mj,Sj in zip(SK,muj,Sigj)])
    b=softmax(logits)
    m_new=Sp@(np.linalg.solve(Sig,mu)+SQ.T@Wt@sum(bj*S@mj for bj,S,mj in zip(b,SK,muj)))
    if np.abs(m_new-m).max()<1e-14: break
    m=m_new
# transformer-shaped form of the CAVI mean
m_tf=Sp@np.linalg.solve(Sig,mu)+sum(bj*Sp@SQ.T@Wt@S@mj for bj,S,mj in zip(b,SK,muj))
print('[3] CAVI mean equals Sig+Sig^-1 mu + sum b_j Lval_j mu_j (plug-in source means):', np.abs(m-m_tf).max())
# exact posterior with sources integrated: weights use source-dependent predictive cov
Spj=[np.linalg.inv(Wt)+SQ@Sig@SQ.T+S@Sj@S.T for S,Sj in zip(SK,Sigj)]
lw=np.log(pi)+np.array([mvn(mean=SQ@mu-S@mj,cov=C).logpdf(np.zeros(d)) for S,mj,C in zip(SK,muj,Spj)])
wex=softmax(lw)
wplug=softmax(np.log(pi)+np.array([-.5*(SQ@mu-S@mj)@np.linalg.solve(Spred,SQ@mu-S@mj) for S,mj in zip(SK,muj)]))
print('[3] exact (sources integrated) weights', np.round(wex,4))
print('[3] plug-in Spred weights           ', np.round(wplug,4))
print('[3] CAVI weights                    ', np.round(b,4))

# --- (4) head coupling example and cross-ratio criterion
a=1.7; keys=[0.,a]
def mass(j,l):
    Ccov=np.eye(2)+np.ones((2,2)); return mvn(mean=[-keys[j],-keys[l]],cov=Ccov).pdf([0,0])
p=np.array([[mass(j,l) for l in range(2)] for j in range(2)]); p/=p.sum()
print('[4] odds ratio', p[0,0]*p[1,1]/(p[0,1]*p[1,0]), ' formula', np.exp(a*a/3))
f=lambda kj,kl: np.exp(-(kj**2+kl**2)/3+kj*kl/3)
q=np.array([[f(keys[j],keys[l]) for l in range(2)] for j in range(2)]); q/=q.sum()
print('[4] mass formula residual', np.abs(p-q).max())
# remark example: query cross-cov diag(1,0) but zero pairing
S=np.array([[1,0,0],[0,1,0],[1,0,0],[0,0,1.]]); Srec=S@S.T+np.eye(4); L=np.linalg.inv(Srec)
L12=L[:2,2:]; print('[4] Lambda12', np.round(L12,4), ' pairing (0,a)L12(0,b)=', np.array([0,1.])@L12@np.array([0,1.]))
# brute-force table for remark example with a=1.1,b=0.9
A_=[np.zeros(2),np.array([0,1.1])]; B_=[np.zeros(2),np.array([0,.9])]
tab=np.array([[mvn(mean=np.concatenate([-A_[j],-B_[l]]),cov=Srec).pdf(np.zeros(4)) for l in range(2)] for j in range(2)])
print('[4] remark table cross ratio', tab[0,0]*tab[1,1]/(tab[0,1]*tab[1,0]))
