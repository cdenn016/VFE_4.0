# CONTENT.md and SPEC.md requests from E5

No construction or result changed mathematically. Wording that the dispatcher may want to align
(findings L6-01, L6-02, L10-02):

1. `prop:phil-seeded-contrast-formation` is retitled in `c14_interpretation.tex` from "Seeded contrast
   formation with idle source learning" to "Seeded contrast formation with a learning-independent
   lower bound" (label unchanged). CONTENT.md:721-722 ("Its probability is supplied by the
   preparation's matching seed, and source learning is idle.") and SPEC.md:82-83 ("whose probability
   comes from the preparation's matching seed with source learning idle") repeat the overstated
   reading. Suggested replacement: "Its lower bound is supplied by the preparation's matching seed
   and is the same with or without learning; the proof uses learning only on the seed interval, the
   learned and frozen probabilities of contrast formation are not compared, and the bound is a
   positivity statement of size at most 3e-30 for every initial mean."

2. `cor:phil-joint-selection`: the clause "for any prescribed a > 7/2, d_n <= T_n^a" now reads "for any
   prescribed positive caps h_n, in particular h_n = T_n^a with any a > 0, d_n <= h_n" (the proof in
   `ca_joint_selection.tex:159-179` already permits arbitrary positive caps). If CONTENT.md or SPEC.md
   quotes the a > 7/2 cap for the corollary, align it; 7/2 remains the threshold of
   `prop:phil-temperature-tracking`.

3. `conj:phil-contrast-formation` is now described in c01 and c14 as the continuous-time learning
   rule for the mixture proportions of the source prior, an instance of the parameter-learning locus
   that does not separate the program from a hierarchical model with the same learned prior. CONTENT.md
   line 723 ("rebases the participatory reading on formation") is still accurate.

4. Author conventions 1 to 3 are stated as defaults in c01's standing-conventions paragraph. SPEC.md
   section 0 and R6 ("equals it unless a model adds copies"; "is the reversal closure of the union of
   these supports") should say "contains the reversal closure", with the copy set a structural datum,
   distinct copies for mutually attending pairs, identity links on agent-level self edges, and the
   2-cycles in the declared loop family once per endpoint.
