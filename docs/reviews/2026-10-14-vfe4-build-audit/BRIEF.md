# Deep audit brief: VFE_4.0 build at 3ebd31c

You are one investigator in a parallel deep audit of the VFE_4.0 multi-agent simulation suite. The code is at `C:\Users\chris and christine\Desktop\VFE_4.0` (Git HEAD `3ebd31c776a4e619c760d46bcb19d692f598b607`, clean). The work is read-only: do not edit, stage, or commit any repository file, and write scratch scripts only to the scratchpad directory you are given.

## Inputs

- Requirements: `docs/superpowers/specs/2026-10-13-spatial-multi-agent-design.md` (sections 4 to 14) and `docs/superpowers/plans/2026-10-13-spatial-multi-agent-implementation.md`, revision 2.
- Claims to check: `README.md`, `docs/model-contract.md`, `docs/implementation/build-handoff.md`, `docs/performance/scaling.md`.
- Code: `vfe4/` (about 2,800 lines), `run_simulation.py`, `run_multi_seed.py`, `run_aggregate.py`, `experiments/`, `tests/`.
- Theory, for labels cited in code and docs: `C:\Users\chris and christine\Desktop\Research\manuscripts\MultiAgentELBO_Theory\`. These files are a source to check against, not a target of this audit.
- The active configuration is the `CONFIG` dictionary in `run_simulation.py`, together with `vfe4/presets.py`.

## Rules

1. Read the code, not its comments or docstrings. A docstring that claims a property is a claim to verify.
2. Do not audit default toggle values for theoretical purity. Do check that the mathematically pure path exists and is reachable under some configuration: held links, frozen source prior, full patch covariance, and chromatic CAVI.
3. Prefer executed evidence: tiny CPU scripts (single-digit dimensions, under a second each) and small finite-difference checks. Bare `python` is CPU-only. Do not launch GPU work, large runs, or the full test suite. You may run individual existing test files on CPU.
4. Out of scope:
   - tamper, forgery, and file-corruption concerns;
   - Lean or any formalization;
   - the manuscripts' own correctness;
   - re-deliberating chosen config values.
5. Report each defect as one finding, even when another investigator may also find it. Do not pad: a finding needs a concrete failure scenario. Report fewer than 15 findings.

## Finding format

Return your findings as plain text, each in this form:

```
### <ID>: <title>
Location: file:line (and config path if relevant)
Severity: critical | high | medium | low
Evidence: what the code does, with the executed check and its output where you ran one
Failure scenario: concrete input or config -> wrong output, crash, or false claim
Fix: the smallest change that would correct it
Falsifies: the observation that would show this finding is wrong
```

Severity scale:

- **critical:** a wrong objective, update, or theory-labeled quantity under the active or a pure configuration.
- **high:** a wrong result under a reachable configuration, or a documented claim the code contradicts.
- **medium:** a latent defect, or a missing check on a reachable input.
- **low:** quality or clarity.

End your report with a two-line verdict for your scope, and list what you checked and found sound, so the coverage is visible.
