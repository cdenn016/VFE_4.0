# R-ERA remediation log (chapters 08, 09, companion c09)

Files edited (M = Research/manuscripts/MAgent_Theory_2.0): `M/chapters/08_smooth_erasure.tex`, `M/chapters/09_public_compression.tex`, `M/companion/c09_public_compression.tex`. Nothing else touched. Not compiled (per brief); environment, brace and dollar balance checked by `scripts/R-ERA/s03_tex_balance.py`; display punctuation completed by `s04_punctuate_displays.py` (five commas added).

Scripts (single-threaded, each under 10 s) in `remediation/scripts/R-ERA/`:
- `s01_erasure_null_flow.py`: null-flow control for ch. 8 (reuses the L06-ERA s02 model). Pairing history full flow: crossed (1,0,3,2) then box (2,3,0,1), switch at t = 0.0108; null flow: same two pairings, switch at t = 0.0221; gate closure 0.0164 vs 0.0333; null flow keeps public signs.
- `s02_phase_and_probes.py`: (a) binary phase coefficient by forward-backward truncation for J = 0.3, 0.5, 0.8, 0.95: (p-m)/(m a^2) = 0.542645, 0.500089, 0.428657, 0.388974 at s = 0.01 against c = 0.542553, 0.5, 0.428571, 0.388889 (O(s^2) convergence); log-density and mean-parameter averages give the same limit (|dp| <= 8e-17); Pi_T stable to 1e-15 for T = 4..12. (b) ||Df|| <= (1+|J|) on 2000 random 6-point samples (max ratio 0.99974). (c) witness intercepts -0.056633012265 (both), velocities 1/9 and 0.131149 = (5 sqrt5 - 10)/9. (d) deterministic merge: image velocities equal and equal to r - nu; nondeterministic kernel: image velocities differ (-0.0318 vs -0.0362).
- `s03_tex_balance.py`, `s04_punctuate_displays.py`: TeX hygiene.

| Finding | Status | What changed (file, label) / reason |
|---|---|---|
| L06-ERA-01 | applied (c08 part deferred to its owner) | 08: theorem retitled "Changing and persistent source preferences" (`erase:pairs`, label kept); new `erase:pair-convergence` (LaSalle + unique nondegenerate symmetric equilibrium in Q for fixed T, small eps; box-mates equal, boxes opposite, exponential convergence, eta_I -> Dirac). Proof reconstructed: uniqueness at eps=0 via x=g(g(x)) argument and atanh(x)/x monotone, Jacobian det 16/9 - (log 3)^2 > 0, C^1 perturbation O(eps^2) at fixed T, S_4 and reversal equivariance. Chapter title changed to "...Persistent Source Preferences". README/c08 wording not in my files. |
| L06-ERA-02 | applied | 08: lemma `erase:gate-entry` now states the gate is an autonomous relaxation perturbed by at most B rho M_chi/Delta; text after lemma calls the gate mean an internal clock; text after theorem states the preference change is a switch between scales rho sigma^2 and eps^2, and that box membership is the supplied initial well of the bistable single-agent energy. |
| L06-ERA-03 | applied | 08: T put into the parameter ordering; new display `erase:row-bound` (after-closure same-box weight bound with Delta_eps) and the preclosure crossed-partner bound; rows within O((rho+eps^2)/T) of uniform at fixed T; mixed-term expansion stated valid at fixed T as gains -> 0, with a sentence that T <~ eps^2 needs a direct comparison. |
| L06-ERA-04 | applied | 08: separated-energy paragraph identifies the post-closure flow as a product flow and the retention as projection onto a decoupled factor (cites HorstmeyerAtay2016) plus a finite label quotient; new paragraph: single eta_I not autonomous, only the pair closes; parent is not an agent of the children's type, cannot be iterated, and the missing requirement (retention onto a law pair with inherited reciprocal interaction) is stated. |
| L06-ERA-05 | applied | 09: theorem retitled "An asymptotic-phase projection under the reciprocal KL flow"; new paragraph identifies Pi as the asymptotic phase / strong-stable-foliation projection with citations, states that the proof is direct because the diagonal is noncompact in a Banach space, that every fiber converges to one point, and that the interaction contributes the gap kappa_N (temperature independent at linear order). Theorem adds "every trajectory in U converges to the common reference". Companion c09 says the content is the decaying part of the approach to the reference. |
| L06-ERA-06 | applied | 09: |J|<1/4 replaced by |J|<1 (main and companion); L = 1+|J| with the covariance bound written out (new `eq:public-phase-lipschitz`); gap argument rewritten (b renamed omega; omega > L possible since |J| < N/(N-1)); binary example 0<J<1. Basin extension added in `cor:public-phase-unique`. Checked: proof steps reconstructed; s02(a),(b). |
| L06-ERA-07 | applied | 09: new `cor:public-phase-unique` (uniqueness of Lipschitz intertwining retraction, with proof); response-defect paragraph: R - r o Pi vanishes with first derivative on the diagonal, O(||delta||^2) where twice differentiable (I state it under that hypothesis because Pi is only proved C^1 in general), attained by the binary example. |
| L06-ERA-08 | applied | 09: new `cor:public-gibbs-channel` (fixed Markov kernel identifies exactly sum w_j delta_{nu_j}; compression possible precisely on fibers of q -> qK); paragraph deriving the deterministic-channel incoming force nu_dot = beta(r - nu) and its failure for nondeterministic kernels (s02(d)); link to ch. 7 stated as not established. |
| L06-ERA-09 | deferred | 08: precise remark that persistent heterogeneity needs heterogeneous constitutive data, with the J_i variant (first-order separation |eta_i-eta_k|/(8/3 - 2 log 3)) and the polarization gate, both explicitly marked as not proved. A proof would need the IFT step and box-margin bookkeeping. |
| L06-ERA-10 | applied | 08: lemma now says "for as long as the solution exists" with public means unrestricted, and proves global existence on the admitted class of `erase:pairs` (private/gate means compact by the lemma, public by box invariance). I did not adopt the lens's boundary-blowup argument for general public data because it is not written out. |
| L06-ERA-11 | applied | 08: `erase:parent-flow` uniqueness restricted to curves of realizable finite-atom states; following sentence says uniqueness for general measure-valued curves would need a Lipschitz stability estimate. |
| L06-ERA-12 | applied | 09: adapted norm and forward-invariance bootstrap stated in the proof; the binary C^3 claim removed and replaced by a direct derivation of the order-3 coefficient from `eq:public-phase-integral` (symmetry gives g_b = (6-3J) m a^2 + O(|m|a^2 r^2); rates corrected by O(r^2) uniformly), with remainder O(|m| a^2 r^2) in `eq:public-phase-binary-parent`; invariance-equation check kept as a consistency statement. |
| L06-ERA-13 | applied | 09: citations Fenichel1974, Guckenheimer1975, MauroyMezicMoehlis2013, BatesLuZeng2000 (with ElderingKvalheimRevzen2018 kept), GolubitskyStewart2002 (fixed-point subspace), PappasLafferriereSastry2000 (controlled projectability). 08: LaSalle1960 for the invariance principle. |
| L06-ERA-14 | applied | 09: paragraph after the proof of `prop:public-gibbs-identifies`: one generic probe line suffices; restricted families fail only on annihilated directions; Vandermonde nodes merge as children approach. Three-atom witness inserted after `eq:public-restricted-force` (verified s02(c)). Intercept renamed iota_j. |
| L06-ERA-15 | applied | 08: psi -> d_0, p -> sigma_max, zeta -> sigma (also avoids the joint-law symbol zeta_i), gate width delta -> Delta, constants m,M,V -> m_K,M_K,V_K, counting measure M -> \mathsf M. 09: phase mean p -> \bar p (main and companion), decay rate b -> omega, intercept b_j -> iota_j, truncation time t_0. |
| L06-ERA-16 | applied | 09: section retitled "An asymptotic collective phase near a stable reference". |
| X6-LIT-11 | applied | 09: as L06-ERA-13; "asymptotic phase" used in theorem title, section title, and companion; text states the finite-dimensional and compact-manifold theorems do not apply verbatim. |
| X6-LIT-06 (09:111 part) | applied | 09: "arithmetic or the geometric pool" now cites GenestZidek1986 and Abbas2009. Chapter 3/5 parts not mine. |
| X5-IMPL-16 | applied | 09: "no finite-cost evaluation algorithm" replaced by `eq:public-phase-truncation` (error C e^{-(2 omega - L) t_0}/(2 omega - L)) and the forward-backward evaluation statement; chart independence of the limit stated (any smooth symmetric retraction gives the same limit), which also corrects the impression that the average must be in log-density coordinates. Companion updated. |
| X5-IMPL-02 / H8 (09:184-188 part) | applied (scope sentence) | 09: sentence that the incoming force r_theta - f_j is a mixture-geodesic term leaving finite-dimensional exponential families, so the statements are on full density charts. Chapter 3 fix not mine. |
| X5-IMPL-04 (binary-subfamily test) | not applicable here | Proposal for a VFE_4.0 test lane; no manuscript change in my files. The binary coefficient was rechecked by s02(a) with L = 1+|J|. |
| L03-DYN-03 (09:23, :59 part) | applied | 09: new display `eq:public-phase-chart-flow` declares the Fisher-Rao flow in the log-density chart, with smoothness on bounded balls, Picard-Lindelof well-posedness in the Banach space, and the dissipation rate. General chapter-3 definition not mine. |
| L01-FOUND-13 (09 part) | applied | 09: sentence that all statements are pointwise in the initial configuration and use no measurability of Phi_t on the nonseparable L^infinity, so the evaluation/norm Borel mismatch does not arise in this chapter. |
| X2-FRAME-03 (09:192-206 part) | applied | 09: new `def:public-controlled-projection` defines X_I, V_I(x_I;e), R_I, R_ext, bar V_I, J_{I,J}, bar J_{I,J}, boundary-kernel factorization; existing labels `eq:public-controlled-projection`, `eq:public-recursive-joining` kept; paragraph relates them to (F6), cites Pappas et al., and states what has and has not been checked for the phase projection. |
| X2-FRAME-05 (08 part) | applied | 08: sentence after `erase:parent-flow` that the descent holds for every partition, so inheritance does not distinguish the selected pairs. |
| X1-GAUGE-03 / L09-OPEN-06 (08 part) | applied | 08: "public/private/gate" defined through a declared observable algebra (what the peer channel transmits, what it reads through the gated mean, what the admitted responses read); "physical compression" reworded. Chapter 12 and README parts not mine. |
| REPORT 7.2 Lipschitz constant (L06-ERA-06 vs X5-IMPL-04) | decided | Not a conflict of fact. 1+|J| is valid by the covariance bound (checked s02(b)) and is needed to reach |J|<1; adopted. |
| Brief: reciprocal-potential scope and formation status | applied | 08: opening paragraph and new `sec:erase-status` (F1 holds in invariant-family form, F2 holds, F3 holds with empirical-measure qualification, F4 fails, F6 fails; null flow reproduces the pairing sequence and shifts only the switch time, so F5 holds only through an event time and its strong form fails; classification: single-stage selection with exact inheritance by decoupled-factor projection). 09: opening paragraph and new `sec:public-status` (no selection; F2, F5, F6 not addressed; F3 or F4 fails depending on probe class; null flow removes the spectral gap so the proof's rate condition fails). Companion c09 opening and closing paragraphs say the same in companion prose. |

## BIB ENTRIES

```bibtex
@article{LaSalle1960,
  author  = {LaSalle, Joseph P.},
  title   = {Some Extensions of {L}iapunov's Second Method},
  journal = {IRE Transactions on Circuit Theory},
  volume  = {7},
  number  = {4},
  pages   = {520--527},
  year    = {1960},
  doi     = {10.1109/TCT.1960.1086720}
}
@article{Fenichel1974,
  author  = {Fenichel, Neil},
  title   = {Asymptotic Stability with Rate Conditions},
  journal = {Indiana University Mathematics Journal},
  volume  = {23},
  number  = {12},
  pages   = {1109--1137},
  year    = {1974},
  doi     = {10.1512/iumj.1974.23.23090}
}
@article{Guckenheimer1975,
  author  = {Guckenheimer, John},
  title   = {Isochrons and Phaseless Sets},
  journal = {Journal of Mathematical Biology},
  volume  = {1},
  number  = {3},
  pages   = {259--273},
  year    = {1975},
  doi     = {10.1007/BF01273747}
}
@article{MauroyMezicMoehlis2013,
  author  = {Mauroy, Alexandre and Mezi{\'c}, Igor and Moehlis, Jeff},
  title   = {Isostables, Isochrons, and {K}oopman Spectrum for the Action-Angle Representation of Stable Fixed Point Dynamics},
  journal = {Physica D: Nonlinear Phenomena},
  volume  = {261},
  pages   = {19--30},
  year    = {2013},
  doi     = {10.1016/j.physd.2013.06.004}
}
@article{BatesLuZeng2000,
  author  = {Bates, Peter W. and Lu, Kening and Zeng, Chongchun},
  title   = {Invariant Foliations near Normally Hyperbolic Invariant Manifolds for Semiflows},
  journal = {Transactions of the American Mathematical Society},
  volume  = {352},
  number  = {10},
  pages   = {4641--4676},
  year    = {2000},
  doi     = {10.1090/S0002-9947-00-02503-4}
}
@book{GolubitskyStewart2002,
  author    = {Golubitsky, Martin and Stewart, Ian},
  title     = {The Symmetry Perspective: From Equilibrium to Chaos in Phase Space and Physical Space},
  series    = {Progress in Mathematics},
  volume    = {200},
  publisher = {Birkh{\"a}user},
  address   = {Basel},
  year      = {2002},
  doi       = {10.1007/978-3-0348-8167-8}
}
@article{PappasLafferriereSastry2000,
  author  = {Pappas, George J. and Lafferriere, Gerardo and Sastry, Shankar},
  title   = {Hierarchically Consistent Control Systems},
  journal = {IEEE Transactions on Automatic Control},
  volume  = {45},
  number  = {6},
  pages   = {1144--1160},
  year    = {2000},
  doi     = {10.1109/9.863598}
}
@article{GenestZidek1986,
  author  = {Genest, Christian and Zidek, James V.},
  title   = {Combining Probability Distributions: A Critique and an Annotated Bibliography},
  journal = {Statistical Science},
  volume  = {1},
  number  = {1},
  pages   = {114--135},
  year    = {1986},
  doi     = {10.1214/ss/1177013825}
}
@article{Abbas2009,
  author  = {Abbas, Ali E.},
  title   = {A {K}ullback-{L}eibler View of Linear and Log-Linear Pools},
  journal = {Decision Analysis},
  volume  = {6},
  number  = {1},
  pages   = {25--37},
  year    = {2009},
  doi     = {10.1287/deca.1080.0133}
}
```

Verification: Fenichel (IUMJ record, volume 23 issue 12, 1109-1137; the journal page prints 1975 on one listing and 1974 on the article page, the DOI carries 1974), Guckenheimer (Springer DOI and Utah bibliography: 1(3):259-273), LaSalle (IEEE DOI and venue; volume/issue/pages from the standard citation, not re-read), Pappas-Lafferriere-Sastry (IEEE DOI, vol. 45 no. 6, June 2000 confirmed from the PDF header; pages 1144-1160 from memory, consistent with the PDF's page 1150), Mauroy-Mezic-Moehlis and Bates-Lu-Zeng (venues verified by X6-LIT; DOIs from memory), Genest-Zidek, Abbas, Golubitsky-Stewart (from memory; Abbas verified by X6-LIT at venue level). The lead should confirm the unverified page ranges and DOIs before the final build. If another agent adds `Abbas2009` or `GenestZidek1986`, keep one entry.

## NEW LABELS

`erase:row-bound`, `erase:pair-convergence`, `sec:erase-status` (ch. 08); `eq:public-phase-chart-flow`, `eq:public-phase-lipschitz`, `eq:public-phase-truncation`, `cor:public-phase-unique`, `cor:public-gibbs-channel`, `def:public-controlled-projection`, `sec:public-status` (ch. 09). All checked unique across the three documents. No existing label was removed or renamed; external references (`ch:public-compression` from ch. 10 and the outlook, `ch:smooth-erasure` from the outlook) still resolve.

## AUTHOR DECISIONS NEEDED

1. Chapter 8's title changed to "State-Controlled Erasure and Persistent Source Preferences"; the companion c08 (other owner) and README still say "persistent heterogeneous pairs" and should be aligned.
2. Whether to prove the heterogeneous-coupling variant (L06-ERA-09) as a proposition; it is currently a remark marked unproved.
3. The ch. 8 null-flow verdict depends on the selector being "argmin of the reciprocal row cost at the current state" and on (F5) counting event times. The present standard lets a mere shift of the switching time satisfy plain (F5); the author may want (F5) to compare partition sequences only, in which case the erasure construction fails (F5) outright.
