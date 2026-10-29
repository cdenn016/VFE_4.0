from edit_lib import Doc

d = Doc('companion/c09_public_compression.tex')
d.rep(r"\chapter{A collective trajectory is not a complete interface}", r"\chapter{A Collective Trajectory Is Not a Complete Interface}")
d.rep(r"The bound $|J|<1$ is exactly the condition for the common reference to be stable.",
      r"The bound $|J|<1$ is exactly the condition for the common reference to be exponentially stable, which the construction uses; at $|J|=1$ it is still stable, but only at a slower than exponential rate.")
d.rep(r"If the peer costs are switched off, each agent relaxes on its own with the common rates, the differences decay no faster than the average, and the construction below loses its foundation.",
      r"If the peer costs are switched off, each agent relaxes on its own with the common rates, and the differences decay no faster than the average. Exact collective coordinates still exist then: linearizing each agent's own relaxation and averaging the linearized coordinates gives one, and many others. What is lost is uniqueness, and the property that configurations sharing a collective coordinate approach each other faster than the collective signal decays. The gap is what makes the construction below the only one of its kind.")
d.rep(r"Measured against the formation standard, the collective phase is a retention of a supplied block rather than a formation: nothing is selected, the ordinary sampling response is not inherited, and the fine differences it compresses decay.",
      r"Measured against the formation standard, the collective phase is a retention of a supplied block rather than a formation: nothing is selected, the ordinary sampling response is not inherited, and the fine differences it compresses decay. A channel aimed at one member does see the discarded differences, so the retention compresses in the sense of the standard, but only transiently.")
d.save()
