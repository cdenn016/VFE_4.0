"""L4-attention a06: with fixed averaged energies the replicator flow of
cor:obs-attention-replicator has the closed form beta(t) ∝ beta0^{e^{-rt}} beta*^{1-e^{-rt}}
(an e-geodesic), so r_i is the exponential rate in log-odds."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import softmax
rng=np.random.default_rng(7); n=4; pi=rng.dirichlet(np.ones(n)); d=rng.normal(size=n); tau=.6; r=1.7
b0=rng.dirichlet(np.ones(n)); bstar=softmax(np.log(pi)-d/tau)
def f(t,b):
    c=np.log(b/pi)+d/tau; return -r*b*(c-b@c)
sol=solve_ivp(f,(0,3),b0,rtol=1e-12,atol=1e-14,dense_output=True)
err=0
for t in np.linspace(0,3,13):
    cf=b0**np.exp(-r*t)*bstar**(1-np.exp(-r*t)); cf/=cf.sum(); err=max(err,np.abs(cf-sol.sol(t)).max())
print('closed-form replicator max err', err)
