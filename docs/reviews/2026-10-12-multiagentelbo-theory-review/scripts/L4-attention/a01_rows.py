"""L4-attention a01: row free energy, softmax stationarity, constant-row gap,
replicator dissipation, reduced Hessian / Schur complement, witness (c06)."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
rng = np.random.default_rng(0)

# --- (1) recognition optimum: unique minimizer of KL(b||pi)+ sum b d / tau
n = 5; pi = rng.dirichlet(np.ones(n)); d = rng.normal(size=n)*2; tau = 0.7
F = lambda b: np.sum(b*np.log(b/pi)) + b@d/tau
bstar = softmax(np.log(pi) - d/tau)
def Fz(z):
    b = softmax(z); return F(b)
res = minimize(Fz, np.zeros(n), method='BFGS', options={'gtol':1e-12})
print('[1] softmax optimum vs BFGS max|diff| =', np.abs(softmax(res.x)-bstar).max())
print('[1] min value vs R^Q=-log sum pi e^{-d/tau}:', F(bstar), -logsumexp(np.log(pi)-d/tau))
# KKT: log(b/pi)+d/tau constant on support
kkt = np.log(bstar/pi)+d/tau; print('[1] KKT spread', kkt.max()-kkt.min())
# row functional identity (cor:obs-selective-softmax a)
b = rng.dirichlet(np.ones(n))
lhs = b@d + tau*np.sum(b*np.log(b/pi))
rhs = tau*np.sum(b*np.log(b/bstar)) - tau*logsumexp(np.log(pi)-d/tau)
print('[1] row-functional identity residual', lhs-rhs)

# --- (2) constant-row gap on finite Y
nY = 7; qY = rng.dirichlet(np.ones(nY)); D = rng.normal(size=(nY,n))*1.5
dbar = qY@D
RQ = -logsumexp(np.log(pi)-dbar/tau)
RP = -logsumexp(np.log(pi)[None,:]-D/tau, axis=1)
bQ = softmax(np.log(pi)-dbar/tau)
bP = softmax(np.log(pi)[None,:]-D/tau, axis=1)
EKL = qY@np.sum(bQ[None,:]*np.log(bQ[None,:]/bP),axis=1)
print('[2] R^Q - E R^P =', RQ-qY@RP, ' E KL =', EKL, ' residual', RQ-qY@RP-EKL)
# bounds min D/tau <= R^P <= sum pi D/tau
print('[2] bounds hold:', np.all(D.min(1)/tau <= RP+1e-15), np.all(RP <= D@pi/tau+1e-15))
# random-source logit shift: D=1/2 r^T W r, r=y_i - A y_j, Y_i,Y_j uncorrelated
K=2; W=np.array([[2.,.3],[.3,1.]]); A=[rng.normal(size=(K,K)) for _ in range(3)]
mi=rng.normal(size=K); Ci=np.diag([.5,.2]); mj=[rng.normal(size=K) for _ in range(3)]
Cj=[np.diag(rng.uniform(.1,1,K)) for _ in range(3)]
dex=[]; dmean=[]; shift=[]
for a,m,C in zip(A,mj,Cj):
    rbar=mi-a@m; covr=Ci+a@C@a.T
    dex.append(.5*rbar@W@rbar+.5*np.trace(W@covr)); dmean.append(.5*rbar@W@rbar)
    shift.append(np.trace(W@a@C@a.T)/2)
dex=np.array(dex); dmean=np.array(dmean); shift=np.array(shift)
print('[2] random-source row: softmax(-dex/tau) vs softmax(-(dmean+shift)/tau) diff',
      np.abs(softmax(-dex/tau)-softmax(-(dmean+shift)/tau)).max())

# --- (3) replicator = Shahshahani natural gradient; dissipation
r=1.3; bb=rng.dirichlet(np.ones(n))
c=np.log(bb/pi)+d/tau; bdot=-r*bb*(c-bb@c)
grad=c+1
print('[3] dF/dt=grad.bdot', grad@bdot, ' -r Var', -r*(bb@(c-bb@c)**2), ' sum bdot', bdot.sum())
# Shahshahani: natural grad v minimizes <g,v> + 1/2 sum v^2/b on tangent: v_j = -b_j(g_j - gbar)
gbar=bb@grad; v=-bb*(grad-gbar); print('[3] natural-grad vs replicator/r', np.abs(v-bdot/r).max())

# --- (4) reduced Hessian / Schur complement in R^2 with nonlinear energies
tau=0.9; pi4=np.array([.2,.5,.3])
def E(x):
    return np.array([np.sin(x[0])+x[1]**2, (x[0]-1)**2+x[0]*x[1], np.exp(.3*x[1])+.5*x[0]**2])
def Rf(x): return -logsumexp(np.log(pi4)-E(x)/tau)
x0=np.array([.3,-.4]); h=1e-4
def gE(x):
    return np.array([[np.cos(x[0]),2*x[1]],[2*(x[0]-1)+x[1],x[0]],[x[0],.3*np.exp(.3*x[1])]])
def HE(x):
    return [np.array([[-np.sin(x[0]),0],[0,2]]), np.array([[2,1],[1,0]]), np.array([[1,0],[0,.09*np.exp(.3*x[1])]])]
bet=softmax(np.log(pi4)-E(x0)/tau); G=gE(x0); gb=bet@G
Hform=sum(bet[j]*HE(x0)[j] for j in range(3))/tau - (G-gb).T@np.diag(bet)@(G-gb)/tau**2
Hfd=np.zeros((2,2)); I2=np.eye(2)
for a in range(2):
    for b_ in range(2):
        Hfd[a,b_]=(Rf(x0+h*I2[a]+h*I2[b_])-Rf(x0+h*I2[a]-h*I2[b_])-Rf(x0-h*I2[a]+h*I2[b_])+Rf(x0-h*I2[a]-h*I2[b_]))/(4*h*h)
print('[4] Hessian formula vs FD max|diff|', np.abs(Hform-Hfd).max())
# Schur complement over tangent space {sum w=0}
Hxx=sum(bet[j]*HE(x0)[j] for j in range(3))/tau; Hxb=G.T/tau; Hbb=np.diag(1/bet)
Bt=np.array([[1,0],[-1,1],[0,-1.]])  # basis of tangent space
Hxw=Hxb@Bt; Hww=Bt.T@Hbb@Bt
schur=Hxx-Hxw@np.linalg.solve(Hww,Hxw.T)
print('[4] Schur complement vs formula max|diff|', np.abs(schur-Hform).max())

# --- (5) witness: X~N(0,tau/omega), J in {-1,1}, O|x,j ~ N(x - j delta, tau), o=0
for (om,de,ta) in [(1,2,1),(0.5,1.3,0.8)]:
    f=lambda x: (om+1)*x**2/(2*ta)-np.log(np.cosh(de*x/ta))
    # direct negative log posterior
    g=lambda x: om*x**2/(2*ta) - np.log(.5*np.exp(-(0-x+de)**2/(2*ta))+.5*np.exp(-(0-x-de)**2/(2*ta)))
    xs=np.linspace(-1,1,5); off=g(xs)-f(xs)
    h2=1e-4; curv=(g(h2)-2*g(0)+g(-h2))/h2**2
    print(f'[5] (om,de,ta)={om,de,ta}: const offset spread', off.max()-off.min(), ' curvature', curv, ' formula', (om+1-de**2/ta)/ta)
    # constant-row Gaussian recognition curvature in m
    vv=0.37
    def Fm(m):
        Ed=np.array([((m-j*de)**2+vv)/2 for j in (-1,1)])
        return om*(m**2+vv)/(2*ta) - logsumexp(np.log([.5,.5])-Ed/ta)
    print('    constant-row m-curvature', (Fm(h2)-2*Fm(0)+Fm(-h2))/h2**2)
    # evidence at o=0
    from scipy.integrate import quad
    ev=quad(lambda x: np.exp(-om*x**2/(2*ta))/np.sqrt(2*np.pi*ta/om)*(.5*np.exp(-(x-de)**2/(2*ta))+.5*np.exp(-(x+de)**2/(2*ta)))/np.sqrt(2*np.pi*ta),-50,50)[0]
    s2=ta*(1+1/om); print('    evidence quad', ev, ' formula', np.exp(-de**2/(2*s2))/np.sqrt(2*np.pi*s2))

# --- (6) self-edge blocking corollary
N,nb=6,2
eta=np.full((N,N),1/(N*(N-1))); np.fill_diagonal(eta,0)
blk=np.repeat(np.arange(N//nb),nb)
etac=np.zeros((N//nb,N//nb))
for i in range(N):
    for j in range(N): etac[blk[i],blk[j]]+=eta[i,j]
bc=etac/etac.sum(1,keepdims=True)
print('[6] beta^c_II', bc[0,0], (nb-1)/(N-1), ' beta^c_IJ', bc[0,1], nb/(N-1))

# --- (7) softmax parameter-path derivative (eq rg-attention-beta-softmax)
def row(t):
    p=np.array([1+.3*t,2-t*t,1.5]); p=p/p.sum(); Ee=np.array([np.sin(t),t**2,1-t]); ta=1+.2*t
    return softmax(np.log(p)-Ee/ta), p, Ee, ta
t0=.4; hh=1e-6
bd=(row(t0+hh)[0]-row(t0-hh)[0])/(2*hh)
b0,p0,E0,t_=row(t0)
lp=lambda t: np.log(row(t)[1]); Ef=lambda t: row(t)[2]; tf=lambda t: row(t)[3]
dlp=(lp(t0+hh)-lp(t0-hh))/(2*hh); dE=(Ef(t0+hh)-Ef(t0-hh))/(2*hh); dt=(tf(t0+hh)-tf(t0-hh))/(2*hh)
chi=dlp-dE/t_+E0*dt/t_**2
print('[7] softmax path derivative residual', np.abs(bd-b0*(chi-b0@chi)).max())
