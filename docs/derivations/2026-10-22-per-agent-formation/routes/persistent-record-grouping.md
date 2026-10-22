<!-- rigorous-theory-search-metadata {"contract_id":"contract-sha256-0324b4d85a47c03878c8c5ffc8caeec2c30ab48cb28b2f143c858ee2931b6dca","schema_version":"rigorous-theory-search/v1","target_digest":"0324b4d85a47c03878c8c5ffc8caeec2c30ab48cb28b2f143c858ee2931b6dca"} -->
# Interaction-dependent grouping with persistent records

This construction establishes an open-class, persistent F5 grouping mechanism under the revised per-agent free energy. It does not yet provide a compressed parent. Its source graph, evaluator, record likelihood and priors are identical for all three agents; the different fixed records are observed data, not different constitutive rules. No numerical experiment is needed for the existence proof.

## Normalized model and complete field

Take three agents with content and presentation spaces both equal to \(\{-1,+1\}\), counting measure, uniform presentation reference \(r_i\), and evaluator \(G_i(m,k)=1/2\). Peer channels are identity. Let the positive source priors in both sectors be

\[
\pi_{ii}=a=\frac15,\qquad \pi_{ij}=b=\frac25\quad(j\ne i).
\]

Use exact Gibbs rows at unit temperature and unit Fisher mobility. For \(\epsilon>0\), the common normalized record likelihood is

\[
\ell_\epsilon(o\mid k,m)
=\frac{\exp(\epsilon ok/2)}{2\cosh(\epsilon/2)},
\qquad o\in\{-1,+1\}.
\tag{R1}
\]

For each \(k,m\), summing over \(o\) gives one. Every probability is positive. Condition on the record vector \(o=(1,1,-1)\), and keep these records in each objective throughout the evolution. The local generative model is exactly

\[
P_i(m,k,J=j,L=l,o_i)
=\pi_{il}\rho_{il}(m)\pi_{ij}\kappa_{ij}(k)\ell_\epsilon(o_i\mid k,m),
\]

where \(\rho_{ii}=r_i\), \(\rho_{il}=s_l\) for nonself \(l\), \(\kappa_{ii}=G_i\), and \(\kappa_{ij}=q_j\) for nonself \(j\). Its normalization is immediate by summing successively over \(o_i,k,m,J,L\). Its variational recognition is \(s_iq_i\beta_i\gamma_i\). The resulting free energy is the sum of the two mixture-source KL sums, the two row divergences, and \(-\mathbb E_{q_i}\log\ell_\epsilon(o_i\mid K)\). The latter depends on the content law and is not an assimilation impulse.

Parameterize the positive laws by log odds,

\[
q_i(k)=\frac{e^{l_i k/2}}{2\cosh(l_i/2)},\qquad
s_i(m)=\frac{e^{n_i m/2}}{2\cosh(n_i/2)}.
\]

Write \(q_0=s_0\) for the uniform law. The Gibbs content row has costs \(D_{ii}=D(q_i\Vert q_0)\), \(D_{ij}=D(q_i\Vert q_j)\). Each peer law is data for the receiving agent, so there is no incoming source-role derivative. The envelope identity for an exactly minimized normalized row removes derivatives of \(\beta^*\) from the law force. Direct differentiation in the probability \(q_i(+1)\), followed by the Bernoulli Fisher inverse, gives

\[
\dot l_i=-l_i+\sum_{j\ne i}\beta_{ij}(l)l_j+\epsilon o_i,
\qquad
\dot n_i=-n_i+\sum_{j\ne i}\gamma_{ij}(n)n_j.
\tag{R2}
\]

The source-reference log odds are zero. The record contribution is \(+\epsilon o_i\) because the log-likelihood ratio of \(k=+1\) to \(k=-1\) is \(\epsilon o_i\). Both fields are analytic on all finite log odds. They obey \(\max_i|\dot l_i|\le2\max_i|l_i|+\epsilon\) and the analogous bound without \(\epsilon\) for \(n\), so solutions exist for every finite time and remain strictly positive as laws.

The model sector is nontrivial and may vary independently, but its evaluator is constant in \(m\); its dynamics decouples from the content sector. Thus this is a special normalized agent species, not a claim that generic generative-model inference is independent of perception.

## Exact null control

The prescribed null deletes every nonself peer cost. There is no hidden law dependence of the identity channels or constant evaluators to freeze. Priors, records, self costs and exact row reduction stay unchanged. Consequently

\[
\beta_{ii}^0(l_i)
=\frac{a e^{-D(q_i\Vert q_0)}}{a e^{-D(q_i\Vert q_0)}+2b},
\qquad
\dot l_i^0=-\beta_{ii}^0(l_i^0)l_i^0+\epsilon o_i.
\tag{R3}
\]

The nonself null row entries remain positive, but their deleted costs contribute no derivative. Replacing this null by a self-only model with a renormalized self prior would be a different control. At \(l_i=0\), the vector field is \(\epsilon o_i\). Uniqueness and the inward-pointing boundary therefore make \(l_1>0,l_2>0,l_3<0\) forward invariant under the null. Its model-sector equation is \(\dot n_i^0=-\gamma_{ii}^0(n_i^0)n_i^0\).

## Stable interaction-dependent consensus of signs

Put \(l=\epsilon u\). The Bernoulli divergence is analytic and has no constant or linear term in the pair of log odds at zero. Uniformly on bounded \(u\)-sets,

\[
\beta_{ij}(\epsilon u)=\pi_{ij}+O(\epsilon^2),\qquad
\dot u=A u+o+O(\epsilon^2),
\quad
A=-\frac75 I+\frac25\mathbf1\mathbf1^\top.
\tag{R4}
\]

The error is analytic with its first derivatives. The eigenvalue of \(A\) on the mean direction is \(-1/5\); its two transverse eigenvalues are \(-7/5\). The limiting equilibrium is

\[
u_*=-A^{-1}o=\left(\frac{15}7,\frac{15}7,\frac57\right).
\tag{R5}
\]

The implicit function theorem gives an equilibrium \(u_\epsilon=u_*+O(\epsilon^2)\) for all sufficiently small positive \(\epsilon\). It is locally asymptotically stable, since its derivative is \(A+O(\epsilon^2)\). A direct uniform local attraction argument is also available: choose a small fixed ball around \(u_*\) on which the symmetric part of the derivative is negative definite for every sufficiently small \(\epsilon\); recenter at \(u_\epsilon\). The derivative of \(\|u-u_\epsilon\|^2\) is strictly negative away from the equilibrium, so a smaller ball is forward invariant and attracting.

The limiting affine system converges to \(u_*\) from

\[
u(0)=(4,1/2,-1).
\tag{R6}
\]

Choose one finite time at which that limiting trajectory lies strictly inside the smaller attraction ball. Continuous dependence of the analytic vector field and its solutions on \(\epsilon\) and the initial point puts the exact nonlinear trajectory there for all sufficiently small \(\epsilon\), and for a nonempty open neighborhood of (R6). It remains there and converges to \(u_\epsilon\). Independent small presentation-law perturbations do not change this statement because the two sectors decouple.

## A predeclared selector and the partition comparison

Connect two agents when their nonzero content log odds have the same sign and

\[
\frac{\min(|l_i|,|l_j|)}{\max(|l_i|,|l_j|)}>\frac14.
\tag{R7}
\]

Blocks are the connected components of this undirected graph; the inequality is strict, and no edge is present when either log odds is zero. This is a permutation-covariant state selector. It is fixed before any retention and uses no supplied block labels. At (R6), the only same-sign pair has ratio \(1/8\), so every block is initially a singleton. These strict initial conditions hold on an open neighborhood.

At (R5), every log odds is positive and the smallest pair ratio is \(1/3>1/4\). Choose the attraction ball above small enough that all these inequalities remain strict. The full flow therefore eventually selects the three-agent block and keeps it for all subsequent time. Under the null, agent 3 remains negative while agents 1 and 2 remain positive; its partition never contains the three-agent block. Hence the two partition sequences differ even after every event time is deleted. Their difference holds on a nonempty open class of independently variable positive belief and model laws with the fixed record vector. Include all simultaneous permutations of the record and initial-law neighborhoods to obtain a relabeling-invariant admitted class.

This proves F1, state-based selection with eventual persistence, and F5 for this candidate. The lossless product of all child laws supplies a trivial inherited state, but that state fails F4. The threshold and source architecture are supplied constitutive choices; no optimality or uniqueness of this selector is claimed.

## Limits and failure tests

The construction's peer forces overcome conflicting persistent record evidence for one agent. Its null comparison is exact. The proof provides a sufficiently small positive \(\epsilon\), not a numerical witness interval. A displayed finite simulation would not replace that argument.

The result does not establish inherited compression, equality of joint response histories after compression, a nonflat collective connection, endogenous selection of a sufficient interface, or recursive formation. In particular a smooth connected sample-frame group acts trivially on the finite alphabets, so nonzero connection curvature needs an appropriate continuous realization. Any compressed extension must include the records \(o\), their effect on future forces, and every admitted future observation. If a protocol exposes each \(q_i\) directly, preserving it rules out losing those laws. The stable grouping calculation cannot override that obstruction.
