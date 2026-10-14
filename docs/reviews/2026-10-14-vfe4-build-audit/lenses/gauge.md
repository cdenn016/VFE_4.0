# Lens: gauge structure (audit-gauge-theorist, Claude Opus 5.5)

The scratch scripts are in scratchpad/gauge/.

## G1 (high): `fixed_sector_rank` overcounts the fixed sector on deep spanning trees
- **Location:** vfe4/analytics/holonomy.py:36-48, used through observables.py:49 and run_formation.py:43.
- **Evidence:** the function uses rooted path products `maps[v]` with one global singular-value threshold scaled by the stacked map norms. Those norms grow exponentially with tree depth, so O(1) constraint singular values fall under the threshold.
  - It matches the dense connection-Laplacian kernel on 24 small cases.
  - On the real 512-agent mutual ring with K=4 and radius 1 it returns 0, 0, 2, 2 at twist std 0.05, 0.2, 0.5, 1.0. The true rank is 0 in every case: some based two-cycle has σ_min(Hol−I) of 0.735 at std 0.5 and 1.375 at std 1.0.
  - A K=3, n=128 ring returns 1, also wrongly.
  - The active (32,16) radius-2 lattice at std 0.05 is correct (0).
- **Failure scenario:** point mode with `lattice_shape=(512,)`, radius 1 and twist std 0.5 reports a fixed sector of dimension 2 in a network that has none.
- **Fix:** decide the rank from the fundamental-cycle holonomies `I - maps[t]^{-1} T_e maps[s]` per non-tree edge, or apply the tolerance per constraint block.
- **Falsifies:** a dense ker(δᵀδ) on that ring with dimension 2.

## G2 (medium): the holonomy defect reads the belief half of `omega` even when only the model channel communicates
- **Location:** observables.py:61-65, together with model.py:53-54.
- **Evidence:** with `channels=('model',)` and `shared_channels=False`, the reported defect is 0.9975. Adding 5 to every belief-half generator leaves F unchanged (ΔF=0) while the reported defect becomes 4.05e17. When both channels communicate, the model-channel holonomy is never reported.
- **Fix:** report the defect per communicating channel, using the `record_primitives` offset.

## G3 (low): negative-determinant frames cannot be reached from config
- **Evidence:** every configured frame is exp(·) or exp(·)exp(·). The GLReal, interaction_transport and FlatReferenceConnection helpers are used only by tests. The "either determinant component" wording in spec §5 and model-contract.md:13 therefore describes an unreachable capability. F does not depend on frames.
- **Fix:** add a reflection seed option, or reword the docs.

## G4 (low): the base "cocycle" test is a coboundary identity
- **Location:** tests/test_grounding.py:78-81 and test_transport.py:11.
- **Evidence:** Ω=U_t U_s^{-1} composes identically for any frame field, so the test cannot detect the property it is named for. Trivial base holonomy depends on periodicity, which is untested. It holds now: the cycle defect is 4.4e-16.
- **Fix:** test composition around one full period of each torus axis.

## Checked and sound
- ΔF=0 exactly under per-agent and per-point g, including det<0. This holds structurally: the objective never reads frames, because records live in the reference trivialization.
- Θ orientation matches c02 eq:geo-edge-twist in both the records and observation generation.
- Independent primitives for mutual pairs, identity self copies, and both based two-cycles; none for directed windows.
- Base transport telescopes; the spatial record uses P0=I.
- The coherent partition uses the record-space residual |Aμ_r − Bμ_s| and is gauge-inert.
- Connected components are correct.
- Section evaluation applies frame congruence correctly.
