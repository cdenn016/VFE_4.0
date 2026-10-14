# Lens: numerical analysis (audit-numerical-analyst, Claude Opus 5.5)

All probes ran on CPU (torch 2.11.0+cpu) from scratchpad/numerical/p1-p10.py.

## NUM-1 (medium): a rejected sweep is not rolled back, so `failed_state.pt` holds the rejected iterate
- **Location:** inference.py:161-162 and 172-189; simulation.py:107-110.
- **Evidence:** `step` mutates beta, mu, chol, log_pi, links and sweep in place before it checks descent, and nothing restores them. In `_channel_sweep`, mu is assigned (line 161) before `checked_cholesky` runs (line 162). A Cholesky failure therefore leaves new means paired with old factors. Probe p5 forced a failed acceptance check: after the raise, mu had changed by 0.93 and beta by 0.67, and sweep had been incremented.
- **Why it matters:** spec §302 says to reject the update and record the failed block, and §241 says accepted steps satisfy SPD and nonincrease. The persisted state breaks both.
- **Fix:** snapshot the state at the start of `step` and restore it before raising, or persist the last accepted state as the failed-state file.

## NUM-2 (medium): `accepted` and `spd_status` are constants, yet scaling.md:61 reports them as runtime indicators
Also reported independently as C3 (claims lens) and partly as IG-3 (info-geometry lens).

## NUM-3 (low): `checked_cholesky` has two blind spots
**Location:** gaussian.py:8-13.
- **Non-finite input loses the block index.** A batch with an inf diagonal entry returns info 0 with a non-finite L, and the error message reads "Non-SPD Gaussian block at []".
- **The upper triangle is never read.** A NaN in the strict upper triangle passes as SPD, because only the lower triangle is factored. This is latent, since all current inputs are symmetric by construction.
- **Fix:** mark a block bad when `(info != 0) | ~isfinite(L).all(-1,-2)`.

## Checked and sound
- **SPD handling:**
  - No silent repair: no epsilon, jitter, eigenvalue floor or eps*I anywhere.
  - Every update-path Cholesky is checked, and non-SPD BLR proposals halve dt.
  - The `clamp_min` in sections.py:62 is a log-domain underflow guard, not repair.
- **matrix_exp:** every call site receives contiguous input. Objectives are identical at copy_chunk_size 1, 3 and 4096. Second derivatives match a central difference to 2.6e-9.
- **Bump basis:** 20,000 random points show no NaN, and weights sum to 1 within 2.2e-16 (float64) and 1.8e-7 (float32). Values stay finite as 1−r goes to 1e-12, and points outside support are masked.
- **Ragged log-softmax:** stable (per-segment max subtracted), and empty rows are rejected.
- **Acceptance tolerance:** scale-aware. Over 80 sweeps covering every dtype and method there were no false rejections; the largest relative increase was 1e-7, against a float32 tolerance of 1e-5.
- **Float32 conditioning:** κ ≤ 10 at block dimension D=100.
- **Natural-parameter interpolation:** a random search over 2e5 cases found no increase. Moment damping is monotone.
- **Link tolerance:** the link step uses a dtype-blind 1e-9 tolerance. It never bound: zero retries over 25 sweeps in every combination.
- **NaN detection:** a NaN in mu, chol, beta or log_pi is caught by the monitor or by `isfinite(after)`.
- **Determinism:** CPU deterministic mode gives bitwise-equal repeats. CUDA determinism was not verified.
