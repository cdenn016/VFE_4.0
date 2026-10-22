import sys
p = sys.argv[1]
s = open(p, encoding="utf8").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, a[:80]
    s = s.replace(a, b)


rep(r"""Assume $\KL(s\Vert t)<\infty$. Set $\mu(dm,do)=s(dm)K(m,do)$ and $\nu(dm,do)=t(dm)K(m,do)$, and""",
    r"""Assume $\KL(s\Vert t)<\infty$. Set $\mathbb P_s(dm,do)=s(dm)K(m,do)$ and $\mathbb P_t(dm,do)=t(dm)K(m,do)$, and""")
rep(r"""giving $\KL(\mu\Vert\nu)=\KL(s\Vert t)$.""", r"""giving $\KL(\mathbb P_s\Vert\mathbb P_t)=\KL(s\Vert t)$.""")
rep(r"""Finiteness makes the decomposition unambiguous.
\end{proof}
""", r"""Finiteness makes the decomposition unambiguous.
\end{proof}

\Cref{thm:predictive-chain} is the chain rule for relative entropy \citep{KullbackLeibler1951,CoverThomas2006}, written for a joint law built from a common kernel. Without the finiteness assumption the identity still holds in $[0,\infty]$, both sides being infinite together \citep{Gray1990}. With a deterministic kernel $K(m,\cdot)=\delta_{T(m)}$ it decomposes a divergence into the divergence of the pushforwards and the expected divergence of the conditional laws given $T$, the form used in \cref{cor:pythagorean}.
""")
rep(r"""Its vanishing is an exact sufficiency condition.""",
    r"""Its vanishing is an exact sufficiency condition: $\KL(sK\Vert tK)=\KL(s\Vert t)$ exactly when the Bayesian inverse of $K$ computed at $t$ returns $s$ from $sK$, the classical case of Petz's recovery theorem \citep{Petz1988}.""")
rep(r"""induces the message law $\nu_j=(T_j)_\#q_j$. Different laws can have the same message law.""",
    r"""induces the message law $\mu_j=(T_j)_\#q_j$. Different laws can have the same message law. Messages travel from source to receiver. The reciprocal flow of \cref{ch:dynamics} also moves each source by the likelihood ratio of its receivers, so a communication model that realizes that flow must include this reverse message; the per-agent dynamics of \cref{def:dyn-per-agent-flow} needs only the forward one.""")
rep(r"""The observation selects among possibilities already related by the receiver's model.
""", r"""The observation selects among possibilities already related by the receiver's model. Under the reciprocal flow such an update is an assimilation event in the sense of \cref{def:dyn-record-interrupted}; its effect persists in the objective only in the free-energy pictures or in the evidence-augmented potential \cref{eq:dyn-evidence-potential}.
""")
rep(r"""Let $q,p$ be laws with $\KL(q\Vert p)<\infty$, and set $\nu=T_\#q$.""", r"""Let $q,p$ be laws with $q\ll p$, and set $\mu=T_\#q$.""")
rep(r"""\widehat q(dk)=\int p(dk\mid h)\nu(dh).""", r"""\widehat q(dk)=\int p(dk\mid h)\mu(dh).""")
rep(r"""Then $T_\#\widehat q=\nu$ and""", r"""Then $T_\#\widehat q=\mu$ and, in $[0,\infty]$,""")
rep(r"""=\int\KL\bigl(q(\cdot\mid h)\Vert p(\cdot\mid h)\bigr)\nu(dh).""", r"""=\int\KL\bigl(q(\cdot\mid h)\Vert p(\cdot\mid h)\bigr)\mu(dh).""")
rep(r"""agree for $\nu$-almost every $h$.""", r"""agree for $\mu$-almost every $h$.""")
rep(r"""Absolute continuity gives $\nu\ll T_\#p$, so the conditional version of $p$ is defined at every $h$ needed up to a $\nu$-null set.""",
    r"""Absolute continuity gives $\mu\ll T_\#p$, so the conditional version of $p$ is defined at every $h$ needed up to a $\mu$-null set.""")
rep(r"""Hence its mixture has pushforward $\nu$.""", r"""Hence its mixture has pushforward $\mu$.""")
rep(r"""This reconstruction preserves the communicated marginal and supplies the discarded conditional from $p$.""",
    r"""The reconstruction \cref{eq:conditional-lift} is Jeffrey's rule of conditioning the prior $p$ on the marginal $\mu$ \citep{DiaconisZabell1982}. It preserves the communicated marginal and supplies the discarded conditional from $p$.""")
rep(r"""is needed before applying it across different fibers.
""", r"""is needed before applying it across different fibers.

\corollaryheading{Pythagorean identity and information projection}{cor:pythagorean}
Under the hypotheses of \cref{prop:conditional-lift}, in $[0,\infty]$,
\begin{equation}
\KL(q\Vert p)=\KL(q\Vert\widehat q)+\KL(\widehat q\Vert p),
\qquad
\KL(\widehat q\Vert p)=\KL(\mu\Vert T_\#p).
\label{eq:pythagorean}
\end{equation}
Among all laws $r$ with $T_\#r=\mu$ and $\KL(\mu\Vert T_\#p)<\infty$, the reconstruction $\widehat q$ is the unique minimizer of $\KL(r\Vert p)$.
\begin{proof}
Apply the deterministic-kernel case of the chain rule to $(q,p)$, to $(\widehat q,p)$, and to $(r,p)$. The pushforwards are $\mu$ and $T_\#p$ in each case. The conditional term vanishes for $\widehat q$, equals the right side of \cref{eq:conditional-lift-loss} for $q$, and is nonnegative for $r$, with equality only when $r$ has the conditional laws of $p$, that is, when $r=\widehat q$.
\end{proof}

This is the Pythagorean identity for information projections \citep{Csiszar1975}, and the minimization property is the characterization of Jeffrey's rule by minimum relative entropy \citep{DiaconisZabell1982}. What \cref{prop:conditional-lift} adds to these sources is its use as a receiver mechanism: the loss term measures exactly what a receiver's prior conditional fails to restore.
""")
rep(r"""$L_\alpha$ has a $\sigma(C)$-measurable version. No finite-divergence""",
    r"""$L_\alpha$ has a $\sigma(C)$-measurable version, that is, exactly when $C$
is sufficient for the family $\{P_0\}\cup\{P_\alpha\}$. No finite-divergence""")
rep(r"""sense \citep{Kallenberg2021}.
""", r"""sense \citep{Kallenberg2021}. The equivalence in \cref{prop:common-family-recovery} is the Halmos--Savage factorization criterion for a family dominated by one of its members \citep{HalmosSavage1949}; the common reference conditional is the kernel that defines sufficiency, compare the adequate subfields of \citet{Skibinsky1967}, and the statement is an instance of abstract sufficiency in the Markov category of standard Borel spaces \citep{Fritz2020}. What is used below is the explicit density \cref{eq:common-family-density} of the reconstruction when descent fails.
""")
rep(r"""Apply \cref{prop:common-family-recovery} to that map.
\end{proof}
""", r"""Apply \cref{prop:common-family-recovery} to that map.
\end{proof}

Descent of the likelihood ratios alone does not suffice. Let $P_0$ be uniform on $\{(0,0),(1,1)\}$ for $(X,E)$ and let $C$ be constant. The family $\{P_0\}$ has a trivially descending likelihood ratio, but the lift draws $X$ uniformly and independently of $E$, so it does not reconstruct $P_0$; here $X$ and $E$ are dependent given $Z$.
""")
rep(r"""An encoder $T(du\mid X)$ gives""", r"""An encoder $\mathcal T(du\mid X)$ gives""")
rep(r"""For any predictive decoder $D(dy\mid u,z)$ with""", r"""For any predictive decoder $\Delta(dy\mid u,z)$ with""")
rep(r"""P(Y\mid X,Z)\Vert D(Y\mid U,Z)""", r"""P(Y\mid X,Z)\Vert\Delta(Y\mid U,Z)""")
rep(r"""P(Y\mid U,Z)\Vert D(Y\mid U,Z)""", r"""P(Y\mid U,Z)\Vert\Delta(Y\mid U,Z)""")
rep(r"""\mathcal D(T)=I(Y;X\mid U,Z)""", r"""\mathcal D(\mathcal T)=I(Y;X\mid U,Z)""")
rep(r"""use $I(Y;U\mid X,Z)=0$.
\end{proof}
""", r"""use $I(Y;U\mid X,Z)=0$.
\end{proof}

The decomposition \cref{eq:decoder-decomposition} is the logarithmic-loss identity of source coding with soft reconstructions, in which the optimal reconstruction is a conditional law and the distortion is a conditional mutual information \citep{CourtadeWeissman2014}.
""")
rep(r"""I(X;U\mid Z)+\lambda\mathcal D(T)""", r"""I(X;U\mid Z)+\lambda\mathcal D(\mathcal T)""")
open(p, "w", encoding="utf8").write(s)
print("ok")
