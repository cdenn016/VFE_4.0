p = r"C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/chapters/19_curvature_hierarchy.tex"
s = open(p, encoding="utf-8").read()
a = r"This is the squared quotient norm of the class of $\mathcal V\omega$ in $H^1$. It vanishes"
b = r"This is the squared quotient norm of the class of $\mathcal V\omega$ in $H^1$, the squared norm of its harmonic representative in the $R^{-1}$-weighted discrete Hodge decomposition in degree one \citep[Theorem~3.1]{HansenGhrist2019}, whose graph form goes back to \citet{Eckmann1944}; see also \citet{Lim2020}. It vanishes"
assert s.count(a) == 1
open(p, "w", encoding="utf-8").write(s.replace(a, b))
print("ok")
