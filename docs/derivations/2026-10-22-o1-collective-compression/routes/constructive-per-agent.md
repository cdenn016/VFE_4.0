<!-- rigorous-theory-search-metadata {"contract_id":"contract-sha256-f83979013626f813aee847ebad2dcca2c0fa4f2687c0daf601404aed35c64a6a","schema_version":"rigorous-theory-search/v1","target_digest":"f83979013626f813aee847ebad2dcca2c0fa4f2687c0daf601404aed35c64a6a"} -->
# Collective relative-angle compression under own-agent dynamics

Date: 2026-10-22. This route modifies one mechanism from the current angular construction: peers transmit the second angle, and the fixed boundary protocol reads its pairwise relative values. Each primitive angle is then operationally exposed. Only a simultaneous physical rotation of the complete connected population remains invisible. The resulting one-dimensional quotient cannot factor into independently compressive retentions for any proper partition on the admitted open family. The common symmetry and relative protocol are declared constitutive data; interactions select the block on which the quotient becomes a local parent. The symmetry is not claimed to arise dynamically.

The route gives the normalized model, complete own Fisher and null equations, fixed normalized joint response kernels, exact original-time quotient field, persistent-record closure, and interaction-dependent grouping. For geometry it uses a section-valued parent anchored at one fixed contextual basepoint. The companion [geometry-history.md](geometry-history.md) develops the actual joint-history Fisher calculation and nonflat window. A pointwise contextual phase erasure is not substituted for that construction.

The supplied target is existential and allows general probability laws, an open class or a declared invariant family, and genuinely collective loss as an alternative to newly generated invisibility. The assumptions introduced here are a three-agent complete graph, the explicit exponential family below, exact Gibbs rows at unit temperature, a fixed relative-angle protocol, small positive parameters, and coherent updates of a compact contextual section. No probability law over configurations is assumed. No second-stage recursion or other future extension is addressed.

The current sources resumed are Chapter 3, lines 312–355, for the normalized own-agent model; Chapter 5, the canonical-parent, boundary-response, memory and formation/null sections; Chapter 13, especially its exact angular field and the explicitly unresolved primitive-phase quotient; and the preceding [integrated-grouping bridge](../../2026-10-22-per-agent-formation/routes/integrated-grouping-bridge.md) and [operational-compression route](../../2026-10-22-per-agent-formation/routes/operational-compression.md). The older primitive phase retention is not reused as the collective loss.

## 1. Fixed normalized probability model

There are three identical agent species. Their belief space is
\(\mathcal K=\{-1,1\}\times\mathbb T\times\mathbb T\), with reference
\(\mu_0=(\delta_{-1}+\delta_1)/2\otimes H\otimes H\), where \(H\) is normalized Haar measure. Their model space is \(\{-1,1\}\) with counting measure. Write

\[
e_\varphi=(\cos\varphi,\sin\varphi),\qquad
h(V)=\log\int_{\mathbb T}e^{V\cdot e_\varphi}H(d\varphi)=\log I_0(|V|),
\qquad M(V)=\nabla h(V).
\tag{C1}
\]

Each positive belief density is

\[
q_i(z,\varphi,\psi)=Z_i^{-1}
\exp\left[l_i z/2+A_i\cdot e_\varphi+zC_i\cdot e_\varphi+B_i\cdot e_\psi\right],
\tag{C2}
\]

with \(A_i,C_i,B_i\in\mathbb R^2\), and

\[
Z_i=I_0(|B_i|)
\frac{e^{l_i/2}I_0(|A_i+C_i|)+e^{-l_i/2}I_0(|A_i-C_i|)}2.
\tag{C3}
\]

The model law is \(s_i(w)=e^{n_iw/2}/[2\cosh(n_i/2)]\). Use fixed positive priors \(\pi_{ii}=a=1/5\), \(\pi_{ij}=p=2/5\) for \(j\ne i\), in both source sectors. The self model reference is uniform. For fixed \(\epsilon>0\), set \(J=\epsilon^3\) and use the fixed self evaluator

\[
G(w;z,\varphi,\psi)=\frac{e^{Jz\cos\varphi}}{I_0(J)}.
\tag{C4}
\]

This is normalized relative to \(\mu_0\), independent of \(w\), and independent of all recognition laws.

The complete peer graph has the same channel on every directed edge:

\[
K((z,\varphi,\psi),d(z',\varphi',\psi'))
=\delta_z(dz')H(d\varphi')\delta_\psi(d\psi').
\tag{C5}
\]

It copies the public sign and the physical second angle and resamples only the marked first angle. Its compared peer law has density

\[
\nu_j(z,\varphi,\psi)
=\frac{e^{L_jz/2}}{\cosh(L_j/2)}
\frac{e^{B_j\cdot e_\psi}}{I_0(|B_j|)},\qquad
L_j=l_j+h(A_j+C_j)-h(A_j-C_j),
\tag{C6}
\]

relative to \(\mu_0\). The sign probability itself is \(e^{L_jz/2}/[2\cosh(L_j/2)]\); the factor of two differs because \(\mu_0\) uses a uniform sign reference. Model peers use the identity channel on \(w\).

A record \((o,\omega)\in\{-1,1\}\times\mathbb T\) has density, relative to counting measure times Haar measure,

\[
\ell_{r,\lambda}(o,\omega\mid z,\varphi,\psi,w)
=\frac{e^{ro z/2}}{2\cosh(r/2)}
\frac{e^{\lambda\cos(\omega-\varphi)}}{I_0(\lambda)}.
\tag{C7}
\]

It is positive and normalized for every latent state. The initial episode uses \(r=\lambda=\epsilon\), with observed binary pattern \(o=(1,1,-1)\); its angular records remain arbitrary in \(\mathbb T^3\). Later admitted evidence transactions use the same likelihood family with finite nonnegative strengths. They are distinct from read-only relative-angle queries.

Given the current peer laws, the local joint law is

\[
P_i(w,k,j,r',o,\omega)
=\pi_{ir'}R_{ir'}(w)\pi_{ij}T_{ij}(k)
\ell_{r,\lambda}(o,\omega\mid k,w),
\tag{C8}
\]

where \(R_{ii}=1/2\), \(R_{ir'}=s_{r'}\) for \(r'\ne i\), \(T_{ii}=G\), and \(T_{ij}=\nu_j\) for \(j\ne i\). All measures are the references just declared. Integrating successively over the record, belief state, belief label, model state and model label proves normalization. A finite retained record word uses the product of its conditional likelihoods given the same latent pair. Its signed and vector coefficients are

\[
H_i=\sum_t r_{i,t}o_{i,t},\qquad
\Lambda_i=\sum_t\lambda_{i,t}e_{\omega_{i,t}}.
\tag{C9}
\]

The full ordered record word is also retained. These coefficients suffice for its persistent objective, but do not replace accessible history.

## 2. Exact Gibbs coefficients, own dynamics, and null

With product recognition \(Q_i=s_iq_i\beta_i\gamma_i\), define

\[
F_i=\mathbb E_{Q_i}\left[
\log\frac{s_i(W)q_i(K)\beta_{iJ}\gamma_{iR}}
{\pi_{iR}R_{iR}(W)\pi_{iJ}T_{iJ}(K)\mathcal L_i(h_i\mid K,W)}\right].
\tag{C10}
\]

Every density is positive; finite natural parameters on compact angle spaces imply absolute continuity and finite expectations. The model and variational product structure are explicit assumptions. No correlated recognition identity is inferred from them.

Let \(D_{ii}=\mathbb E_{q_i}[\log q_i-\log G]\), \(D_{ij}=\mathbb E_{q_i}[\log q_i-\log\nu_j]\), and \(E_{ii}=D(s_i\Vert1/2)\), \(E_{ij}=D(s_i\Vert s_j)\). The record expectation is

\[
\mathbb E_{q_i}[-\log\mathcal L_i]
=\text{record constant}-\frac{H_i}2\mathbb E_{q_i}Z
-\Lambda_i\cdot\mathbb E_{q_i}e_\varphi.
\tag{C11}
\]

Thus \(F_i\) is the two row-weighted cost sums, the two row KL divergences, and (C11). Constrained row minimization gives

\[
\beta_{ij}=\frac{\pi_{ij}e^{-D_{ij}}}{\sum_k\pi_{ik}e^{-D_{ik}}},\qquad
\gamma_{ij}=\frac{\pi_{ij}e^{-E_{ij}}}{\sum_k\pi_{ik}e^{-E_{ik}}}.
\tag{C12}
\]

The multiplier equation is \(D_{ij}+\log(\beta_{ij}/\pi_{ij})+1+\xi_i=0\); positive priors and finite costs exclude boundary minima, and the Hessian \(\operatorname{diag}(1/\beta_{ij})\) is positive on normalized tangents. This establishes the unique unit-temperature Gibbs optimum. Its envelope derivative is exactly \(\sum_j\beta_{ij}dD_{ij}\).

Put \(m_i=\tanh(L_i/2)\), \(t_i=\mathbb E_{q_i}[Z\cos\varphi]\), and \(S_i=\mathbb E_{q_i}\log q_i\). Direct expectation gives

\[
\begin{aligned}
D_{ii}&=S_i-Jt_i+\log I_0(J),\\
D_{ij}&=S_i-L_jm_i/2+\log\cosh(L_j/2)
-B_j\cdot M(B_i)+h(B_j).
\end{aligned}
\tag{C13}
\]

The common entropy cancels from the full row. Its exact unnormalized weights are

\[
w_{ii}=\frac{a e^{Jt_i}}{I_0(J)},\qquad
w_{ij}=\frac{p\exp[L_jm_i/2+B_j\cdot M(B_i)]}
{\cosh(L_j/2)I_0(|B_j|)},\qquad
\beta_{ij}=\frac{w_{ij}}{\sum_kw_{ik}}.
\tag{C14}
\]

The new cross term depends on relative physical phases. Unlike the preceding primitive-phase construction, a change in one \(B_j\)'s orientation generally changes rows and peer forces.

Use the own-law Fisher metric \(\int (vv'/q_i)d\mu_0+\sum_w\zeta(w)\zeta'(w)/s_i(w)\) on zero-mass tangents. In natural coordinates its belief block is the covariance of
\((z,e_\varphi,ze_\varphi,e_\psi)\), with first natural coordinate \(l/2\). These seven nonconstant statistics are linearly independent modulo constants, as follows by separating sign parity and Fourier modes, so the covariance is positive definite for every positive density. The exact centered logarithmic Fisher velocity lies in their span. Coefficient matching gives, in the original time,

\[
\begin{aligned}
\dot l_i&=-l_i+\sum_{j\ne i}\beta_{ij}L_j+H_i,&
\dot A_i&=-A_i+\Lambda_i,\\
\dot C_i&=-C_i+\beta_{ii}Je_1,&
\dot B_i&=-B_i+\sum_{j\ne i}\beta_{ij}B_j,\\
\dot n_i&=-n_i+\sum_{j\ne i}\gamma_{ij}n_j.
\end{aligned}
\tag{C15}
\]

Thus (C2) is an exactly invariant family, not an approximate moment closure. Every nonself source is data in its receiver's own derivative; no sender-objective derivative is added. The state dependence of the Gibbs weights remains in (C15) and its Jacobian.

The prescribed null removes the nonself costs without removing labels, priors, or row entropy. The fixed kernels and evaluator have no hidden law dependence to freeze. Define

\[
\beta_{ii}^0=\frac{a e^{-D_{ii}}}{a e^{-D_{ii}}+1-a},\qquad
\gamma_{ii}^0=\frac{a e^{-E_{ii}}}{a e^{-E_{ii}}+1-a}.
\tag{C16}
\]

Its exact equations are

\[
\begin{aligned}
\dot l_i^0&=-\beta_{ii}^0l_i^0+H_i,&
\dot A_i^0&=-\beta_{ii}^0A_i^0+\Lambda_i,\\
\dot C_i^0&=\beta_{ii}^0(Je_1-C_i^0),&
\dot B_i^0&=-\beta_{ii}^0B_i^0,\\
\dot n_i^0&=-\gamma_{ii}^0 n_i^0.
\end{aligned}
\tag{C17}
\]

Angular entropy cannot be canceled from (C16) after deletion. Both dynamics use the same row convention and persistent records.

## 3. Fixed operational protocol, before retention

The protocol admits fresh draws of \((Z_i,\Phi_i)\) at any primitive port and fresh model draws \(W_i\). A direct marked-angle query reports its physical lab-angle label, obtained by the inverse local embedding, and stores that label in history. Explicitly, let the known retained constitutive rotation \(E_i(c)\in SO(2)\) embed lab directions in the local marked frame. The reported label \(\omega_i^{\mathrm{lab}}\) is determined by \(e_{\omega_i^{\mathrm{lab}}}=E_i(c)^{-1}e_{\Phi_i}\). In the original trivialization \(E_i(c)=I\), this is exactly the displayed \(\Phi_i\) readout. Under a passive marked frame change, \(E_i\mapsto R_gE_i\) and \(e_{\Phi_i}\mapsto R_ge_{\Phi_i}\), so the reported and stored lab label stays fixed. The known embedding still identifies the full local marked law from these query distributions.

The protocol admits relative-angle queries on every unordered pair of distinct ports of the complete graph. A pair query samples \(\Psi_i,\Psi_j\) independently from their current recognition marginals and returns \(R_{ij}=\Psi_i-\Psi_j\) modulo \(2\pi\). Its normalized density is

\[
k_{ij}(r)=\int_{\mathbb T}v_{B_i}(u+r)v_{B_j}(u)H(du),\qquad
v_B(u)=\frac{e^{B\cdot e_u}}{I_0(|B|)}.
\tag{C18}
\]

When several differences share samples, the protocol first samples the requested vector of primitive states from the product recognition law, then returns the requested differences of that vector. It uses this normalized joint pushforward, not a fictitious product of independent edge laws. Every new transaction uses fresh samples; shared samples occur only within a single joint query.

Relative-angle queries are read-only; their outputs are nevertheless appended to the complete accessible history. Evidence transactions select a primitive port and strengths in (C7), draw the record from the normalized generative marginal of (C8), append its outcome, and add its exact likelihood to (C9). This choice of generative observation law is declared; it is not equated to a recognition-predictive law. Source labels may be recorded externally when the chosen transaction includes them, without changing the agent's source-label conditioning convention. The admissible outputs never reveal an absolute \(\psi\) angle or a raw absolute-angle peer sample. Individual \(\psi\) laws remain exposed through their relative responses to other ports.

Action labels, strengths, waiting times, query results, evidence records, and selection events are retained. Finite adaptive action choices use only that complete history and fixed physical/lab protocol data, with optional external randomization; they do not read unrecorded state coordinates. Their conditional action kernels are unchanged by a passive marked frame change. Absolute-angle evidence likelihoods are outside this fixed protocol from its declaration, not removed at handoff. All later finite evidence words have finite coefficients. No law over the law-valued configuration is needed to define these kernels.

Selection is evaluated at two predeclared census times \(0,T\), using the public graph rule in Section 7. The same times and rule are used for full and null. The field and all probes remain unchanged between censuses. On the strict-margin class constructed below, the census times and selected partitions are constant across the contextual patch. Exact continuously observed threshold-crossing times are not included in the geometric experiment, since a context-dependent deterministic timestamp would require a separate differentiability analysis of the history family.

## 4. A decisive collective identifiability lemma

For \(B_i=b_i e_{\theta_i}\), \(b_i>0\), write

\[
\mu_i=\mathbb E e^{i\Psi_i}=r(b_i)e^{i\theta_i},\qquad
r(b)=I_1(b)/I_0(b).
\tag{C19}
\]

Differentiating the circle partition function gives \(r'(b)=\operatorname{Var}_{v_b}(\cos\psi)>0\), and \(r(0)=0\). Therefore \(r(b)>0\) for \(b>0\), and \(r\) is injective on that domain. The first relative response moment is

\[
c_{ij}=\mathbb E e^{iR_{ij}}=\mu_i\overline{\mu_j}.
\tag{C20}
\]

For three nonzero factors and distinct \(i,j,k\),

\[
r(b_i)=\sqrt{\frac{|c_{ij}||c_{ik}|}{|c_{jk}|}}.
\tag{C21}
\]

Thus, at each fixed context, the pair responses identify every \(b_i\) and every relative phase. Two triples have the same relative response laws only if their \(B_i\) differ by one simultaneous rotation; the converse holds for every joint difference query by rotating all sampled angles together. The direct \((Z,\Phi)\) laws identify \(l,A,C\), since equality of their positive densities makes the difference of their log densities a constant and the five nonconstant sign/Fourier coefficients must vanish. The model draws identify \(n\). Hence the instantaneous fixed-context protocol has exactly a one-dimensional common-phase ambiguity in this family, after its other parameters are included. Across a whole contextual section, the geometry construction below retains contextual variations of that phase and erases only one context-constant rotation.

This ambiguity is genuinely collective in a stronger block-local sense. Let \(I\) be any proper nonempty block, and hold an outside port \(j\notin I\) fixed with \(B_j\ne0\). For every \(i\in I\), its cross-boundary moment determines

\[
\mu_i=c_{ij}/\overline{\mu_j},
\tag{C22}
\]

and hence \(B_i\) itself. Direct public/marked and model queries identify the remaining parameters using the known retained lab-to-local embedding. Therefore any block-local retention that preserves this fixed boundary protocol for all allowed outside states is injective on the primitive laws in this open family. In particular the whole-population quotient below cannot be implemented by independently erasing a phase inside each block of a proper partition. Introducing nonlocal shared reference data across those blocks would be an additional collective retention, not such a factorization.

The complete graph and identical rules do not encode a desired subpartition. The selected block is the whole three-agent population. Before selection, every primitive and every proper block is exposed by its cross-boundary responses. A simultaneous global physical symmetry was already present, but no permanently invisible primitive factor supplied the loss.

## 5. Physical symmetry and a section-valued canonical parent

Let \(R_\chi\) denote plane rotation and \(\mathsf J=\begin{pmatrix}0&-1\\1&0\end{pmatrix}\). The physical action sends every \(B_i\) to \(R_\chi B_i\), leaving \(l,A,C,n\), records and their coefficients fixed. Equations (C13)–(C17) are equivariant because \(M(R_\chi B)=R_\chi M(B)\), and all row terms use rotation-invariant scalar products and norms. The fixed kernels (C5) commute with a simultaneous shift of their input and output second angles. Every admitted response kernel is invariant under this action. This physical action is distinct from the marked comparison gauge acting on \(\varphi\) and its constitutive directions. Under a passive marked frame change, each stored record keeps its fixed numerical label \(\omega\); its local direction embedding \(e_\omega\) rotates together with \(A,C\) and the evaluator direction. Section 8 fixes this convention at the level of the normalized joint experiment.

The complete state for the geometric construction is a regular contextual section \(x:c\mapsto(l_i,A_i,C_i,B_i,n_i)(c)\) over a fixed compact contextual patch, together with complete history. The same realized history is applied coherently to every evaluation context. In particular, conditioning a section on a history does not give each contextual point a different realized record word. Pointwise own-agent equations still evolve each section value.

Fix a contextual basepoint \(c_0\) before any retention. Suppose all \(B_i(c)\) lie in one open half-plane, uniformly on the admitted compact patch. Let

\[
S(c_0)=\sum_iB_i(c_0),\qquad
U=S(c_0)/|S(c_0)|,\qquad
\widehat B_i(c)=R_{-\arg U}B_i(c).
\tag{C23}
\]

The rotation can be written directly from the unit vector \(U\), so no branch of \(\arg\) is required. Retain the complete sections \(l,A,C,\widehat B,n\), all primitive port and action labels, source data, records, event information, and constitutive parameters. The discarded variable is the single common physical phase of the whole section. A pointwise rotation \(\arg S(c)\) is not used.

For each context, the parent belief law is the product of the normalized laws (C2) with \(B_i\) replaced by \(\widehat B_i\), and its model law is the product of the \(s_i\). Its self evaluator may be written as the normalized product of (C4). Its inherited primitive source channels are the same fixed kernels in their canonical coordinates; projecting to a primitive source port, applying (C5), and appending normalized factors constructs whole-parent output channels when required. These internal law and channel data descend through a common physical-rotation representative and the same fixed relative protocol. No equality of the unrotated child \(\psi\) laws is asserted, and no primitive source-output action exposing an absolute \(\psi\) sample or law is admitted before or after handoff. Primitive port labels and every relative-pair request remain available at all stages. This parent retains the directed port dynamics. It is not asserted to be a fresh single-source agent or the Fisher gradient of a newly chosen sum of objectives.

The quotient is physically noninjective even after memory is retained. A common rotation of the entire section changes the child diagnostic \(\mathbb E_{q_i}e^{i\Psi_i}\) in (C19), while leaving the retained value and every admissible history kernel unchanged. The orbit is one-dimensional and free on the nonzero-anchor domain. It is not a passive marked frame change. The diagnostic distinguishes the physical loss, but is not an admitted future absolute-angle boundary action.

## 6. Explicit inherited original-time field and joint histories

Write

\[
V_i(B)=-B_i+\sum_{j\ne i}\beta_{ij}B_j,\qquad
V_S=\sum_iV_i,\qquad
\Omega(x)=\frac{\det(S(c_0),V_S(c_0))}{|S(c_0)|^2}.
\tag{C24}
\]

All omitted arguments in \(V_i\) are the retained \(l,A,C,n\), record coefficients and source data. The same expression evaluated on canonical variables gives the same scalar \(\Omega\). Differentiating (C23) gives the exact parent equation

\[
\dot{\widehat B}_i(c)=V_i(\widehat B)(c)
-\Omega(\widehat x)\mathsf J\widehat B_i(c).
\tag{C25}
\]

Every other parameter follows its equation in (C15), with weights evaluated on canonical variables. The subtraction is the derivative of the retention's moving representative, not an independently fitted parent force or a time change. For the null use \(V_i^0=-\beta_{ii}^0B_i\) in (C24)–(C25), and the other equations in (C17). This also shows why the anchor sum may rotate even when individual null phases are fixed: differing radial rates can change the sum's direction.

The anchor remains nonzero at every finite time in the admitted cone. Fix a unit vector \(v\) with \(v\cdot B_i(c,0)>0\) for all ports and contexts. In the full system the projected variables obey
\(\dot b_i=-b_i+\sum_{j\ne i}\beta_{ij}b_j\). Positivity is preserved and variation of constants gives
\(b_i(c,t)\geq e^{-t}b_i(c,0)>0\). Under the null they equal their initial values times \(\exp[-\int\beta_{ii}^0]\). Thus (C23)–(C25) remain regular on every finite interval, uniformly over compact initial cone patches. Evidence transactions in (C7) do not alter these projection equations.

Normalized response kernels descend directly. For relative queries, replacing every sampled angle by its canonical angle preserves all differences, including joint dependencies among differences sharing samples. Public/marked and model draws are unchanged. Evidence likelihoods ignore \(\psi\); generative source marginals and optional source-label readouts are unchanged by rotating all internal second-angle samples. Given an output, the full word is appended, (C9) is updated, and the same coherent operation is applied to the entire retained section. These operations commute with the physical action and are determined by the retained state and recorded outcome. Therefore the joint kernel of the response and the next retained state, rather than only their separate marginals, factors through the parent.

An induction over the causal product of waiting-time flows, adaptive action kernels, response kernels and retained updates proves equality of every admitted finite joint response-record history, with its dependence on next states. No preparation distribution is needed for this pointwise inheritance statement. If a statistical experiment supplies a normalized preparation and history law, the same kernel identity holds after integration against that law. The complete stored history and the fixed census events do not restore the common phase: every global rotation gives the same retained record word and the same conditional history laws.

Finite continuation with finitely many finite-strength records is regular. The \(A\) and \(C\) equations have bounded forcing on each such horizon. The largest \(|B_i|\) is nonincreasing because \(\sum_{j\ne i}\beta_{ij}<1\). Since \(|L_i-l_i|\leq2|C_i|\), the maximum public log odds has at most linear growth, and the model log odds satisfy the analogous bound. Hence no natural parameter escapes at a finite time. Compact-context derivatives have the corresponding local integral bounds on finite regular tubes.

## 7. Interaction-dependent persistent grouping

The initial episode is held fixed for this persistence theorem: \(H_i=\epsilon o_i\), \(o=(1,1,-1)\), and \(\Lambda_i=\epsilon e_{\omega_i}\). Later evidence remains covered by exact inheritance, but may change selection or geometry. The initial data have \(l/\epsilon\) near \((4,1/2,-1)\), \(|A_i|<1/20\), \(|C_i|<\epsilon^3\), and \(B_i=\delta b_i\), where \(b_i\) lies in a fixed compact subset of a common open half-plane with nonzero amplitudes. The relative \(b_i\) phases vary independently in small open intervals; there is no imposed common-phase relation. Choose \(\delta>0\) sufficiently small that \(|B_i|<1/20\). These conditions are relatively open in the invariant finite-dimensional family and extend uniformly to compact contextual sections.

For later use, both full and null trajectories remain in a simple bounded region. Assume \(\epsilon\leq1/100\) and

\[
|l_i|\leq12\epsilon,\qquad |A_i|\leq1/10,\qquad
|C_i|\leq2\epsilon^3,\qquad |B_i|\leq1/10.
\tag{C26}
\]

The sufficient-statistic vector in (C2) has squared norm 4, so its log-partition Hessian has operator norm at most 4. The integral Taylor formula for the exponential-family KL therefore yields

\[
D_{ii}\leq2|(l_i/2,A_i,C_i,B_i)-(0,0,Je_1,0)|^2
\leq72\epsilon^2+\frac1{25}+18\epsilon^6<\log2.
\tag{C27}
\]

Since peer costs are nonnegative,

\[
\beta_{ii}\geq\frac{a e^{-D_{ii}}}{a e^{-D_{ii}}+1-a}\geq\frac19,
\qquad \frac19\leq\beta_{ii}^0\leq\frac15.
\tag{C28}
\]

The \(A\) and \(C\) norm boundaries point inward in both flows. The upper Dini derivative of \(\max_i|B_i|\) is at most \(-\max_i|B_i|/9\). Moreover,

\[
|L_i-l_i|\leq2|C_i|(|A_i|+|C_i|)
\leq\frac25\epsilon^3+8\epsilon^6<\epsilon/8,
\tag{C29}
\]

because \(\nabla h(0)=0\) and \(\|\nabla^2h\|\leq1\). At a full-flow boundary \(|l_i|=12\epsilon\), its outward derivative is at most
\(-12\epsilon/9+\epsilon+\epsilon/8<0\). This proves invariance of (C26), using (C27) only on that region and closing the first-exit argument.

For the null sharpen the public bound to \(1/4\leq o_i l_i/\epsilon\leq12\). Its signed equation is
\(d(o_i l_i/\epsilon)/dt=1-\beta_{ii}^0(o_i l_i/\epsilon)\), which points inward at both endpoints by (C28). Consequently

\[
o_iL_i(t)\geq\epsilon/8>0
\quad\text{for every finite null time}.
\tag{C30}
\]

The first two public signs remain positive and the third remains negative indefinitely. This is a uniform invariant-region conclusion, not an extrapolated finite-time approximation.

For the full system scale \(u=l/\epsilon\), \(c=C/\epsilon^3\), \(b=B/\delta\), leaving \(A\) unscaled. On bounded sets, (C6) gives \(L/\epsilon=u+O_{C^1}(\epsilon^2)\), while \(m=O(\epsilon)\), \(t=O(\epsilon)\), and
\(B_j\cdot M(B_i)-h(B_j)=O_{C^1}(\delta^2)\). Hence

\[
\beta_{ij}=\pi_{ij}+O_{C^1}(\epsilon^2+\delta^2),
\tag{C31}
\]

and the scaled exact system is

\[
\begin{aligned}
\dot u&=Mu+o+O_{C^1}(\epsilon^2+\delta^2),\\
\dot c_i&=-c_i+a e_1+O_{C^1}(\epsilon^2+\delta^2),\\
\dot A_i&=-A_i+\epsilon e_{\omega_i},\\
\dot b_i&=-b_i+\sum_{j\ne i}\pi_{ij}b_j
+O_{C^1}(\epsilon^2+\delta^2),
\end{aligned}
\qquad M=-\frac75I+\frac25\mathbf1\mathbf1^\top.
\tag{C32}
\]

The last line uses \(M\) on each plane component. The two-parameter zero-limit field is affine and has a symmetric negative-definite derivative, with eigenvalues among \(-1/5,-7/5,-1\). Its equilibrium is

\[
u_*=(15/7,15/7,5/7),\qquad c_{i,*}=a e_1,
\qquad A_{i,*}=b_{i,*}=0.
\tag{C33}
\]

The maps extend analytically to \((\epsilon,\delta)=(0,0)\); the factors divided by \(\epsilon\) or \(\delta\) in (C32) have already been canceled by their exact expressions. The implicit function theorem applies because the displayed derivative is invertible. It gives a nearby equilibrium for all sufficiently small positive \(\epsilon,\delta\). A fixed small ball around it is attracting: the symmetric part of the exact derivative remains negative definite, so the squared distance has a strictly negative derivative away from the equilibrium. The equilibrium has \(B=0\); the admitted finite-time trajectories retain nonzero \(B\) by the cone argument, while approaching that limiting boundary. No phase erasure at a finite time is used.

The limiting affine trajectories enter the attraction ball after one finite time, uniformly over compact initial \(u,c,A,b\) patches and the complete angular-record torus. Finite-time continuous dependence puts the exact trajectories inside a smaller attracting ball for sufficiently small positive parameters and a nonempty relative-open initial neighborhood. Choose the ball so that all public \(L_i\) are positive and all pair ratios exceed \(1/4\); at (C33) the smallest ratio is \(1/3\). This gives a common finite entrance bound \(T\), chosen as part of the construction before running either flow. The same argument gives uniform \(T\) over a compact contextual patch with fixed small nonzero marked concentrations, as required by the geometry route.

At census times \(0,T\), connect \(i,j\) if their public odds have the same nonzero sign and

\[
\frac{\min(|L_i|,|L_j|)}{\max(|L_i|,|L_j|)}>\frac14,
\tag{C34}
\]

and select connected components. Initially the only same-sign pair has limiting ratio \(1/8\), so the strict singleton condition survives small perturbations. At \(T\), the full graph is complete and stays complete with the initial episode fixed. The null never has the three-agent block at any time by (C30), including at its identical census time. Thus the partition sequences differ after timestamps are removed. The finite census schedule does not modify the vector field, delete a probe, or erase any state; its decisions still depend on the observed state, and its null control cannot reproduce the full block.

All rules are homogeneous on the complete graph. Taking simultaneous permutations of initial laws and record values gives a relabeling-covariant admitted class. The initial compact cone patch may also be rotated globally; the selection uses only invariant public data. Neither membership labels nor a supplied hierarchy are encoded in the graph.

## 8. Geometry interface and exact scope

The declared gauge is the marked \(S^1\) comparison action on \(\varphi\). All record labels in \(H\) are fixed under passive frame changes, including the numerical evidence angles \(\omega\) and the lab-angle labels of every earlier direct marked query defined in Section 3. Their local direction embeddings rotate: \(E_i e_\omega\mapsto R_gE_i e_\omega\), together with \(A,C\), the evaluator direction and the current first-angle sample. In the transformed frame of the original trivialization, the angular factor in (C7) is \(\exp[\lambda(R_g e_\omega)\cdot e_\varphi]/I_0(\lambda)\). This remains normalized on the same fixed record-label space. A fresh \(Y_\varphi\) in the geometric comparison experiment is the current local sample coordinate and rotates; it is distinct from a previously reported lab-angle label stored in \(H\). The physical \(\psi\) coordinate is unchanged by this marked action.

Let a normalized initial record experiment generate \(H\) conditionally on a prepared contextual section, and let all subsequent section updates use that same history word. Changing the marked frame and all the displayed constitutive embeddings leaves the history law \(\Pi_c(H)\) invariant by change of variables in its normalized generative kernels. The marked update is covariant, every Gibbs cost is invariant, and hence \(B^H\) and its anchor are unchanged at fixed \(H\). The joint marked sample action is precisely \((H,Y_\varphi,Y_\psi)\mapsto(H,R_gY_\varphi,Y_\psi)\). At fixed time and history, the canonicalization (C23) rotates second-angle samples by one angle \(\alpha_t(H)\) evaluated at \(c_0\). It is constant in the evaluation context \(c\). Thus the joint map

\[
(H,Y)\longmapsto(H,T_{-\alpha_t(H)}Y)
\tag{C35}
\]

has unit Haar Jacobian and no derivative in \(c\) at fixed \(H\). It pushes the fine joint history/conditional-law experiment to the actual canonical parent experiment. Since \(\alpha_t(H)\) is invariant under the marked action at each fixed record word, this map commutes with that action. The joint contextual score includes \(X\log\Pi_c(H)\), which can be nonzero. The history law has zero marked orbit score under the fixed-label convention, but its contextual score is retained in full. Changing variables therefore preserves every contextual and marked orbit score inner product. In particular the entire collective quadratic cost \(\mathbb E(s_X+\xi s_{\mathrm{orb}})^2\), its actual minimizer class, and its minimum agree. Replacing the joint experiment by a product of one-step marginals would not prove this assertion.

A pointwise anchor \(\alpha(c,H)\) would add a discarded \(\psi\)-orbit score proportional to \(d_c\alpha\) and can fail this identity. The present route instead retains the entire contextual section modulo a single context-constant physical rotation. If a parent were required to carry only its state at one context with no section data, this construction would not establish its geometry obligation.

For the nonflat window the marked patch can be the existing two-coordinate section
\(A_1(0)=r e_\chi\), \(A_2(0)=d e_1\), with \(0<r_-\leq r\leq r_+\), fixed \(d>0\), and all marks small but independent of \(\epsilon,\delta\). At the two-parameter zero limit the physical \(\psi\) laws are uniform, rows equal the fixed priors, and the marked solution is \(A_i(t)=e^{-t}A_i(0)\). Its common marked connection has the explicit nonzero curvature and positive-rank minimized tensor obtained by minimizing the marked phase Fisher costs; the derivation and finite-parameter continuation are recorded in [geometry-history.md](geometry-history.md). The common finite \(T\) above can be used for the same compact post-selection window. Positive \(\delta\) retains a nonzero physical phase fiber at every finite time even though the limiting calculation uses \(\delta=0\).

The supported constructive result is a genuinely collective one-dimensional physical quotient with exact own-agent dynamics, fixed joint probes, complete persistent memory, and a full/null grouping mechanism on a declared regular family. The full result depends on the section-valued geometry construction, the predeclared relative protocol, and block-local inheritance when excluding proper-partition quotients. It does not claim that interactions created a new statistical symmetry, that absolute-angle probes can be preserved, or that a second inherited formation has been constructed.

The decisive failure tests are concrete. An admitted absolute \(\psi\) probe or absolute \(\psi\)-record likelihood breaks the common physical symmetry. A zero anchor leaves this regular chart. Independent phase canonicalization of proper blocks violates cross-edge responses by (C22). Pointwise context-dependent anchoring can lose Fisher score. A record/action mechanism using unretained absolute angle, or an incoherent context-by-context update with different histories, invalidates the joint-history argument. None of these modifications is included in the stated model.

All calculations in this route are analytical. No script, simulation, GPU query, Lean work, manuscript edit, or publication was performed. This assigned route file is the only output written for the O1 task.
