import re
p=r'C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/chapters/10_boundary_closure.tex'
d=open(p,'rb').read().decode('utf-8')
lines=d.split('\r\n')
# lines are 0-indexed; file lines 166..187 -> idx 165..186
cnt=0
for k in range(165,187):
    n=lines[k].count('\pi'); cnt+=n
    lines[k]=lines[k].replace('\pi','\varpi')
print('pi replaced',cnt)
d='\r\n'.join(lines)
def rep(a,b):
    global d
    assert d.count(a)==1,(d.count(a),a[:60]); d=d.replace(a,b)
rep(r"with the same rule for model rows.",
 r"with the same rule for model rows. The product is $\omega_1\omega_2=(\Phi_2\circ\Phi_1,\varpi_2\circ\varpi_1)$, componentwise in each sector; with this law the formulas define a left action, $\omega_1\cdot(\omega_2\cdot x)=(\omega_1\omega_2)\cdot x$. The finite-dimensional statement \cref{inh:equivariant-descent} does not apply here: $\mathcal S$ is not a Lie group, it acts on a Banach chart, and the sets used below are fixed-point sets of actions that are not free, so no quotient manifold or quotient metric is asserted.")
rep(r"acts on $\operatorname{Fix}(H)$ commuting with the flow, and the flow descends to the orbit space $\operatorname{Fix}(H)/K$.",
 r"acts on $\operatorname{Fix}(H)$ commuting with the flow, and the flow descends to the orbit space $\operatorname{Fix}(H)/K$. More generally, if $X$ is a flow-invariant union of fixed-point sets, the flow descends to $X$ modulo the equivalence relation induced on $X$ by $\mathcal S$.")
rep(r"so orbits are carried to orbits. For (iii)",
 r"so orbits are carried to orbits; the same commutation applies to any two points of $X$ related by an element of $\mathcal S$. For (iii)")
rep(r"by rotations of the arcs through rationally independent irrational fractions of their length. This map is ergodic on each cell for the normalized Haar product \citep{Walters1982}",
 r"by rotations of the arcs through fractions $c_v$ of their length such that $1$ and the $c_v$ are linearly independent over the rationals. This map is ergodic on each cell for the normalized Haar product by the Kronecker--Weyl criterion \citep[Chapter 1]{Walters1982}; rational independence of the $c_v$ alone does not suffice, since $c_1=\sqrt2-1$ and $c_2=2-\sqrt2$ leave the character $e^{2\pi i(x_1+x_2)}$ invariant")
rep(r"Phase rotations $\theta_v\mapsto\theta_v+\alpha'_v(r)-\alpha_v(r)$ map $\operatorname{Fix}(H_\alpha)$ onto $\operatorname{Fix}(H_{\alpha'})$ with the profile unchanged, and orientation changes act in the same way in the coin realization. By \cref{lem:boundary-equivariance}, the flow descends to the quotient by these maps,",
 r"Phase rotations $\theta_v\mapsto\theta_v+\alpha_v(r)-\alpha'_v(r)$ map $\operatorname{Fix}(H_\alpha)$ onto $\operatorname{Fix}(H_{\alpha'})$ with the profile unchanged, and orientation changes act in the same way in the coin realization. These maps do not form one subgroup acting on a single $\operatorname{Fix}(H_\alpha)$, since phase rotations change $\alpha$ and $\Phi_\tau$ depends on it. The union $X=\bigcup_\alpha\operatorname{Fix}(H_\alpha)$ is flow invariant, and by the general form of \cref{lem:boundary-equivariance} (ii) the flow descends to $X$ modulo the relation $(\alpha,f)\sim(\alpha',f(r,\tau(r)u))$,")
rep(r"Applying $\Psi_t$ pointwise to the second configuration gives a path whose spectra",
 r"Two routine identifications make this precise. A continuously differentiable curve in $L^\infty$ has a jointly measurable version that is absolutely continuous in $t$ for almost every $\xi$ and satisfies the pointwise equation, so $x(t)$ is the pointwise flow almost everywhere; conversely, since $\Psi_t$ is locally Lipschitz in $v$ uniformly on the common essential range of the values, which is bounded and bounded away from zero, $\xi\mapsto\Psi_t(r,v(\xi))$ is a continuously differentiable curve in $L^\infty$. Applying $\Psi_t$ pointwise to the second configuration gives a path whose spectra")
rep(r"A base-only response or likelihood reads only $(r,\mu^q)$ and multiplies the value vector at $r$ by a common factor, which acts on the spectra.",
 r"A base-only response reads only $(r,\mu^q)$. A base-only assimilation sends the value vector at $r$ to $(L(o\mid r)v_i/Z_i)_i$ with $Z_i=\int L(o\mid r)q_{i,0}(r)dA(r)$; these factors depend on the agent unless the base marginals agree, but they are determined by $\mu^q$, so the update acts on the spectra.")
rep(r"Discarding the join phase under the fresh interface therefore needs this conjugacy condition, which holds automatically on the symmetric family. Whether such two-block families are invariant under the fine flow, and when the flow preserves the conjugacy condition, is not established here.",
 r"Discarding the join phase under the fresh interface therefore needs this conjugacy condition, which holds automatically on the symmetric family. The two-block family is invariant: it is $\operatorname{Fix}(H'_\alpha)$ for the subgroup $H'_\alpha\le H_\alpha$ of elements whose permutation preserves the two root branches, by the argument of \cref{prop:boundary-fixed-family} without the root node element, so \cref{lem:boundary-equivariance} (i) applies. When the flow preserves the conjugacy condition, so that phase compression at a heterogeneous join holds along the whole trajectory and not only at one time, is not established here.")
rep(r"The proofs use only a group of agent permutations preserving the edges, priors, and evaluator that acts transitively on each set of sibling subtrees, together with public coordinates that realize at each node a uniformly random arrangement of its children, fair given the base and ancestor coordinates.",
 r"The proofs use, at each node, a group of permutations of the children that acts 2-transitively on them, for example the full symmetric group, which gives an iterated wreath product of symmetric groups on the leaves, together with public coordinates that realize at each node a uniformly random element of that group, fair given the base and ancestor coordinates. Transitivity alone does not suffice: with cyclic arrangements of three children the two sibling costs of a receiver differ in general, so class-constant rows are not preserved.")
open(p,'wb').write(d.encode('utf-8'))
print(d.count('\r\n'))
