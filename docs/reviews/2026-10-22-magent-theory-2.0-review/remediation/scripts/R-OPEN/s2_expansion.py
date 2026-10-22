"""R-OPEN s2: series of h^-2 d(hu,hv) and the closed-cluster m-independence."""
import sympy as sp
h,u,v=sp.symbols('h u v',real=True)
def d(x,y):
    return sum((1+z*x)/2*sp.log((1+z*x)/(1+z*y)) for z in (1,-1))
s=sp.series(d(h*u,h*v)/h**2,h,0,5).removeO()
print(sp.factor(sp.expand(s)))
print(sp.simplify(sp.expand(s) - ((u-v)**2/2 + h**2*(u-v)**2*(u**2+2*u*v+3*v**2)/12) ).series(h,0,4))
# scaled far-agent field: h^-1 (1-h^2u^2) atanh(hu)
print(sp.series((1-h**2*u**2)*sp.atanh(h*u)/h,h,0,5))
# two clusters of m coincident agents at +-a (limit field): B and da/dt
a,m=sp.symbols('a m',positive=True)
V=sp.exp(-2*a**2)
w_in=1/(m*(1+V)); w_out=V/(m*(1+V))
B=m*w_out
adot=-a + m*(2*w_out)*(-2*a)
print(sp.simplify(B), sp.simplify(adot + a*(1+4*V/(1+V))))
