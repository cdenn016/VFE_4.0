r"""L08-CND check 1b: Monte Carlo standard error for the p=2 Wishart KL comparison in s1 (d)."""
import numpy as np
from scipy.special import gammaln, digamma
rng = np.random.default_rng(11)
def lG2(x): return 0.5*np.log(np.pi)+gammaln(x)+gammaln(x-0.5)
def psi2(x): return digamma(x)+digamma(x-0.5)
def samp(nu,P,n):
    u=rng.gamma(nu,1.0,n); v=rng.gamma(nu-0.5,1.0,n); z=rng.normal(0,np.sqrt(0.5),n)
    Z=np.empty((n,2,2)); Z[:,0,0]=u; Z[:,0,1]=Z[:,1,0]=np.sqrt(u)*z; Z[:,1,1]=v+z**2
    w,V=np.linalg.eigh(P); R=V@np.diag(w**-0.5)@V.T
    return R@Z@R
def logW(X,nu,P): return nu*np.log(np.linalg.det(P))-lG2(nu)+(nu-1.5)*np.log(np.linalg.det(X))-np.einsum('ij,nji->n',P,X)
P=np.array([[1.3,0.4],[0.4,0.7]]); Q=np.array([[0.8,-0.3],[-0.3,1.9]]); a,b=0.9,1.4
X=samp(a,P,1000000)
d=logW(X,a,P)-logW(X,b,Q)
kl=(a-b)*psi2(a)-lG2(a)+lG2(b)+b*np.log(np.linalg.det(P)/np.linalg.det(Q))+a*(np.trace(Q@np.linalg.inv(P))-2)
print("closed form", kl, " MC", d.mean(), " SE", d.std()/np.sqrt(len(d)), " z", (d.mean()-kl)/(d.std()/np.sqrt(len(d))))
ld=np.log(np.linalg.det(X)); print("E log det X MC", ld.mean(), " psi2(a)-log det P", psi2(a)-np.log(np.linalg.det(P)), " SE", ld.std()/1000)
