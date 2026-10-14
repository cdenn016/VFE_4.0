"""L4-attention a07: cor:obs-selective-kernel-gauge on finite spaces; bijections = permutations."""
import numpy as np
rng=np.random.default_rng(9)
def stoch(a,b): M=rng.random((a,b)); return M/M.sum(1,keepdims=True)
def perm(n): P=np.eye(n)[rng.permutation(n)]; return P   # P[x,:] = delta_{g(x)}
nS,nm,np_,nO=4,3,5,4
E,N,D=stoch(nS,nm),stoch(nm,np_),stoch(np_,nO); K=E@N@D
gj,gi,phm,php=perm(nS),perm(nO),perm(nm),perm(np_)
# re-expressed kernels: E'(s',.) = (phi-)_# E(gj^{-1}s',.)  ->  E' = gj^T E phm  (row s' picks gj^{-1}s')
Ep=gj.T@E@phm; Np=phm.T@N@php; Dp=php.T@D@gi
Kp=Ep@Np@Dp
print('(b) K\' = gj^T K gi residual', np.abs(Kp-gj.T@K@gi).max())
q=rng.dirichlet(np.ones(nS)); qp=q@gj
print('(b) q\'K\' = (gi)_# (qK) residual', np.abs(qp@Kp-(q@K)@gi).max())
# (c) link Theta as permutation of S (receiver fiber identified with S here, nS==nO)
Th=perm(nS); F=Th.T@K           # F(s,.) = K(Theta^{-1}s,.)
print('(c) K = F(Theta s) residual', np.abs(Th@F-K).max())
# represented link law Theta' = g_i Theta g_j^{-1}, written for row-action matrices
# with row-action convention X[x,:]=delta_{f(x)}, composition f∘h corresponds to H@F
Thp=gj.T@Th@gi
Fp=Thp.T@Kp
print('(c) F\'(s\',.) = (gi)_# F(gi^{-1}s\',.) residual', np.abs(Fp-gi.T@F@gi).max())
