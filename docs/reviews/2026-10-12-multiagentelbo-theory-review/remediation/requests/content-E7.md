# CONTENT.md requests from E7

New results (all ESTABLISHED, proofs in place):

- `prop:restrict-gaussian-evidence` (g04_restrictions.tex, Supplement G): Gaussian evidence in information form,
  log p(o|X) = h_o^T o - (1/2) o^T Lambda_oo o + A(h_Y - Lambda_Yo o, Lambda_YY) - A(h, Lambda), under the
  observation-free joint precision hypothesis of `prop:restrict-conditional-precision-observation-free`; the block
  optimum bound is this value minus the determinant gap.
- `rem:rg-pressure-native-ising` (b08_lattice_pressure.tex, Supplement B): the native free partition of
  `def:rg-pressure-native` equals sum_S t^{|S|} b^{|d_2 S|} over plaquette sets (all real b, t), and for 0<b,t<1 is a
  nearest-neighbor ferromagnetic Ising model on the (L-1)x(L-1) plaquette grid with coupling (1/2) log(1/b) and
  negative fields (1/2) log t + (n/2) log b. No analyticity consequence is claimed.
- `rem:rg-thermo-native-raw-nonconvex` (b09_lattice_thermo_fisher.tex, Supplement B): lim_{j->0+} d^2 p/dj^2 = 4b^6 - 5b^8,
  negative for b^2 > 4/5, so the raw pressure is not convex; only H = Hess A is positive semidefinite.
