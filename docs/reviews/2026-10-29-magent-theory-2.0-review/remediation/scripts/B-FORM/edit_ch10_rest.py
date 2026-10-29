p = r'C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/chapters/10_boundary_closure.tex'
d = open(p, 'rb').read().decode('utf-8')
n0 = d.count('\r\n')


def rep(a, b):
    global d
    assert d.count(a) == 1, (d.count(a), a[:70])
    d = d.replace(a, b)


rep(r"This also establishes measurability of the descended updates.",
    r"On a finite base this also gives measurability of the descended updates, since the chart is then finite dimensional and the profile flow is continuous; on a general base the record statements below assume, as \cref{def:dyn-record-interrupted} does, that the flow is jointly Borel in time and state.")

rep(r"Fix a normalized positive record kernel $L_e^q(do\mid r)$ with bounded positive density on the admitted update domain, applied as a readout",
    r"Fix a normalized positive record kernel $L_e^q(do\mid r)$ with bounded positive density on the admitted update domain, jointly Borel in $(o,r)$, applied as a readout")

rep(r"Induction, using the inherited continuous flow between records, proves equality of every finite history law under policies choosing the next admitted experiment from the observed history.",
    r"Induction, using the inherited continuous flow between records, proves equality of every finite history law under policies choosing the next admitted experiment from the observed history; the policies are Borel kernels from finite histories to admitted experiments and waiting times, and the history laws are those of \cref{def:adaptive-history}.")

rep(r"increased by $\Vert\log L\Vert_\infty$.",
    r"increased by $\Vert\log L\Vert_\infty$, where $\Vert\log L\Vert_\infty$ is the sum of the bounds in the two sectors when both carry evidence. The added term is the case of \cref{eq:dyn-evidence-potential} with $o_i=o$ for every agent and record kernels that read only the base coordinates, $\ell_i(o\mid k,m)=L^q(o\mid r(k))L^s(o\mid w(m))$.")

rep(r"with $J\notin\{2,6\}$; this includes the range $|J|<1/4$ used in \cref{ch:public-compression}.",
    r"for any real $J$; this includes the range $|J|<1$ used in \cref{ch:public-compression}.")

rep(r"Its leading coefficient vanishes only at $J=2$ and $J=6$, so the determinant is nonzero for sufficiently small nonzero $\varepsilon$, and remains nonzero nearby.",
    r"Its leading coefficient vanishes only at $J=2$ and $J=6$. At a general base point $(m,n,a,e)=\varepsilon(c_m,c_n,c_a,c_e)$ the same block gives the leading coefficient $\varepsilon^4Q(c;J)$ with $Q=[2(6c_m-Jc_n)c_a-2Jc_mc_e][2(6c_n-Jc_m)c_e-2Jc_nc_a]-4J^2c_mc_nc_ac_e$, which equals $-32$ at $J=2$, $c=(1,0,1,1)$ and $-36$ at $J=6$, $c=(1,0,1,1/2)$. For every real $J$ there is therefore a base point at which the determinant is nonzero for sufficiently small nonzero $\varepsilon$, and it remains nonzero nearby.")

rep(r"The proposition is local: global convergence of the contrast to zero, and whether contrast can vanish at finite time, are not established here.",
    r"The proposition is local. Two global statements follow from the lemmas above.")

open(p, 'wb').write(d.encode('utf-8'))
print(n0, d.count('\r\n'))
