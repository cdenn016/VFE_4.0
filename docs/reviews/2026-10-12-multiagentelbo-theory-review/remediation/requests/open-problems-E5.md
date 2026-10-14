# Open-problem table requests from E5 (ca_open_problems.tex, owned by the dispatcher)

Both rows change only to match the settling conditions now written in `c14_interpretation.tex`
(finding L10-09: a test with stated error probabilities, not mere dependence of the law on the
hypothesis). Labels and items are unchanged.

## Row: Single-point testability, `\cref{open:phil-testability}`

Replace the settling-condition cell with:

For a named target system at $c_*$ whose data the model does not generate, observed with sampling units and a noise model through a declared record kernel that reads the links of a closed walk or the coordinates of a candidate block: a test fixed before the data, with an estimator of gauge-typed parameters, a held-out statistic, an acceptance tolerance, and baselines without link holonomy or without formation, whose failure would falsify nontrivial holonomy on that walk or formation of that block, with a proof that under the kernel and sample size the test rejects under the baseline and accepts under the hypothesis, each with probability at least one minus an error probability stated before the data, the two error probabilities summing to less than one; negatively, a proof that under the kernel every statistic of the records has the same law for two link configurations differing only in the conjugacy class of the walk's holonomy, or whether or not the block is coherent at the declared resolution.

## Row: Physical-law identification, `\cref{claim:physical-law-identification}`

Replace the settling-condition cell with:

For a named target system whose data the model does not generate, observed through a declared record kernel that under the model's own law separates two objects of the coarse-graining differing in a gauge-invariant datum, with an observable, an estimator, and a validation and uncertainty protocol set before the data whose decision has two error probabilities, stated in advance and summing to less than one: held-out evidence, robust across admissible blocking schemes, that an object of the coarse-graining meets a baseline margin with probability-level closure; or held-out evidence under the same protocol that the margin is not met.

Nontriviality (R16): the positive branch needs a target system and a test whose two error
probabilities sum to less than one, which no existing statement supplies; the negative branch
(equal laws for every statistic) is exactly the case in which no such test exists at any sample size,
and no existing statement establishes it.
