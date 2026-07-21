# VFE 4.0 Buildout Investigation

- **Date:** 2026-07-21
- **Scope:** Current VFE 4.0 scaffold, V3 engineering patterns, VFE4/MAgent
  theory boundary, click-to-run requirements, gradient strategy, and build
  architecture
- **Result:** Recommend a greenfield zero-dimensional VFE4 package with an
  independent oracle lane and H1--H8 promotion gates

## 1. Investigation method

Three independent lanes were used:

1. V3 launcher/config/data/training/artifact pattern extraction;
2. VFE4 scaffold and repository-architecture analysis; and
3. direct theory-to-software synthesis from the VFE4 and MAgent whitepapers
   plus the Research wiki.

Subagent reports were required to identify sources, exact file/line locations,
copy-ready patterns, allowed interfaces, anti-patterns, confidence, and gaps.
The main synthesis was checked against the current local sources rather than
closed by agent consensus.

No V3 or Research files were edited. The Research vault was queried but not
ingested or modified.

## 2. Initial VFE 4.0 state

Before bootstrap, `C:/Users/chris and christine/Desktop/VFE_4.0` contained:

- the VFE4 whitepaper and nine modules;
- the MAgent exact-ELBO whitepaper, thirteen modules, figures, and oracle code;
- GL(K), PIFB2, shared bibliography, and TikZ manuscript material; and
- one ignored Python bytecode artifact.

There was no valid Git repository, Python package, launcher, `pyproject.toml`,
dependency lock, README, production test tree, dataset interface, checkpoint
schema, run layout, or CI definition. The only Python source was the MAgent
whitepaper's NumPy/SymPy verification material.

The preexisting empty `.git` directory did not form a Git worktree.
`git status` and the verification control plane both failed with "not a git
repository."

## 3. Source identity

### 3.1 Target whitepapers

The local and Research target files matched by SHA-256:

| Artifact | SHA-256 |
|---|---|
| `VFE4_gauge_causal_elbo_whitepaper.tex` | `D733880D3613D32A97B7A12C93FF6C037D0ABDFD9CE4810E411769997DBAD03C` |
| `MAgent_exact_elbo_whitepaper.tex` | `D62CC62E0B37EA3C299BD3C29680CB579251E51C461B1766BEE1E517CDB5D497` |
| `magent_elbo_whitepaper/verification/elbo_oracles.py` | `F54CB46A5CCC623D48E2BF5B21590FB7DF7A3C7D35C0EB7684AE1B0B8FEF4F63` |
| `magent_elbo_whitepaper/verification/test_elbo_oracles.py` | `3607239914491EBFD9A0402D8DAEEF685CE8A172454CFF2789961267FED50651` |

A recursive relative-path/hash comparison of the complete local
`vfe4_whitepaper` and `magent_elbo_whitepaper` trees returned `ALL_MATCH`
against `Research/manuscripts`.

The closeout Research `HEAD` was
`d50e51556c237708b2bba728d57931a721d67395`. The Research checkout contained
unrelated tracked and untracked WIP, so equality is asserted only for the
hash-compared target files and trees, not for the complete checkout.

### 3.2 V3 guide

V3 changed during the investigation:

| Observation | Initial expert read | Closeout refresh |
|---|---|---|
| `HEAD` | `aa5aceab7844d48c800d72397c0ce4550c567ba1` | `b362506d0bf31a8724c807973e6941760449a6e5` |
| `train_vfe3.py` SHA-256 | `9B72E591895608DBB27F51A9A1E606DD8D6B6BC1506C8CCCBC65F86975C207AD` | `8ABB07426AE90DE6AC31E289FA059DAC1A7FC671E7CE2F32FBDD9053534D2DFD` |
| `vfe3/config.py` SHA-256 | `5BF62D42571AC7D9F3FD038D79B71E5A525E6E9048898356704583ED6C73B8AD` | `788D242D02E1C083622E346CAA69C02F755220DCCD668A84AECDB95344379671` |

The closeout checkout remained dirty with user WIP. No fetch was performed, so
no fresh remote-state claim is made.

The click-run and importable-runtime seams survived the refresh:

- editable `config` dictionary:
  `V3_Transformer/train_vfe3.py:74`;
- per-run typed construction:
  `V3_Transformer/train_vfe3.py:599`;
- `main()` and module guard:
  `V3_Transformer/train_vfe3.py:720` and `:773`;
- `train_step`, `evaluate`, and `train`:
  `V3_Transformer/vfe3/train.py:491`, `:819`, and `:1234`; and
- `RunArtifacts` and `finalize_run`:
  `V3_Transformer/vfe3/run_artifacts.py:1306` and `:2729`.

A current launcher scan found no `argparse`, Typer, Click command, or Hydra
surface. Because the source moved during the investigation, implementation
work must refresh V3 locations before copying a pattern. The design relies on
the seams, not on stale line numbers or wholesale source copies.

## 4. Theory-to-code findings

### 4.1 VFE4 is the normative language model

The VFE4 whitepaper already specifies the zero-dimensional language reduction.
The singleton base retains labeled population fibers, typed state/model
channels, same-point internal frames, a causal source graph, normalized
transitions, categorical observations, and one structured recognition law.
It has no nontrivial base connection, base curvature, or base holonomy.

The MAgent whitepaper supplies related exact finite ELBO, CAVI, Gaussian,
source-row, precision-transport, and frame-covariance oracles. MAgent's
configuration-level Gibbs theory and legacy moving-peer population energy
remain separate from the VFE4 state joint.

### 4.2 V3 is not the VFE4 objective

V3's token marginals, deterministic pair-energy grids, source weights, and
separate decode cross-entropy do not instantiate VFE4's correlated joint
recognition and normalized fixed generative model. V3 is an engineering
ancestor and comparison arm. A later initializer can copy compatible tensors
through an explicit one-way map, but it cannot create VFE4 joint precision,
normalized transition, source-posterior, or evidence semantics.

### 4.3 Prior prediction and recognition must be separate

Training recognition can condition on observations according to a declared
filtering or smoothing contract. Held-out prior prediction must be constructed
before the scored target is assimilated and cannot accept target-conditioned
recognition as input.

### 4.4 Sparse precision is the promoted interface

Small oracles can use dense algebra. Production code must expose solves,
factorizations, log determinants, selected inverse blocks, and sampling. It
must not materialize a global dense population covariance on the promoted H8
path.

### 4.5 The model is hybrid, not categorically backpropagation-free

Exact CAVI and analytic Gaussian/source coordinates should be used where they
exist. PyTorch reverse-mode autograd is the default derivative engine for
general E-like and M-like proposals because the objective is scalar and the
learned parameter count is large.

The default block-coordinate schedule freezes model parameters during E-like
updates, materializes an immutable nonaliasing recognition snapshot, then
freezes that snapshot during the M-like update. Detachment alone is not enough
if model and recognition blocks share storage or if recognition is recomputed
inside the M-step.

Hand-derived gradients remain independent oracles. Custom backward kernels are
deferred until profiling identifies a concrete need and analytic,
finite-difference, and double-precision gradient checks agree.

## 5. Oracle evidence

### 5.1 Current successful run

The current command was executed from the matching Research package layout:

```powershell
python -B -m pytest manuscripts\magent_elbo_whitepaper\verification\test_elbo_oracles.py -q -p no:cacheprovider --junitxml=C:\tmp\vfe4-plan-magent-oracles-research-20260721.xml
```

Environment:

- Python `3.14.4`;
- pytest `9.0.2`.

The JUnit record reports:

- tests: `17`;
- failures: `0`;
- errors: `0`;
- skipped: `0`; and
- suite time: `0.506` seconds.

JUnit SHA-256:
`ED7A44898DB20633B3A45CE37F7C04BFD10060E8666BBBB4A966138E66D51699`.

The cases cover finite Gaussian mean-field gaps, zero-coupling controls,
nested state/configuration identities, source-row envelopes, covariance and
precision transport, directed Gaussian normalization, analytic posterior/ELBO
identity, complete-Markov-blanket CAVI, and a complete linear-Gaussian frame
invariance fixture.

These results verify only the executed finite oracle fixtures at the
hash-matched source revision. They do not implement or close VFE4 H1--H8 and do
not prove the general mathematical statements by numerical agreement.

### 5.2 VFE4 layout collection failure

The same test path was attempted from:

1. `VFE_4.0/Manuscripts`; and
2. the `VFE_4.0` root.

Both attempts failed during collection with:

```text
ModuleNotFoundError: No module named 'manuscripts'
```

The files live under uppercase `Manuscripts` while the test imports lowercase
`manuscripts.magent_elbo_whitepaper...`. This is a packaging failure, not an
oracle-formula failure. The build plan must normalize package/import layout or
port the oracles into the independent lowercase `verification` namespace.

## 6. Architecture alternatives

### 6.1 Domain-first production package plus independent oracle lane

**Selected.** It preserves probability/type boundaries, supports click-to-run
entry points, and allows H1 production code to be tested against sealed
NumPy/SymPy references.

### 6.2 Sealed analytic sidecar promoted later

Not selected as the primary architecture because it creates a migration and
provenance boundary between the first verified formulas and production types.

### 6.3 V3 fork with objective replacement

Rejected because it imports state, objective, config, and checkpoint
assumptions that conflict with the new normalized joint.

### 6.4 Registry/plugin system from the first commit

Deferred until H3 identifies stable extension seams. Premature registries
increase invalid configuration combinations and objective drift.

## 7. Recommended promotion order

1. repository, package, click-run launchers, typed config, and provenance;
2. H1/H2 exact finite core;
3. H3 structured-versus-factorized recognition;
4. H4/H5 sparse precision and update coherence;
5. H6 prefix safety and matched language prediction;
6. H7 complete internal-frame covariance;
7. H8 sparse allocation/scale; and
8. separately approved extensions.

## 8. Repository bootstrap record

After the user approved the design and Git initialization:

- the empty placeholder was initialized as a repository on `main`;
- empty root commit `29c798e` established `main` without changing source files;
- branch `codex/vfe4-buildout-spec-20260721` was created;
- `.gitignore` excluded Python caches, LaTeX build outputs, local secrets, and
  generated runs; and
- commit `6993e61` captured the 40 authored manuscript/oracle/figure files and
  ignore rules on the dedicated branch.

No remote is configured, and no external publish action was taken.

## 9. Limits and open obligations

- There is no VFE4 production implementation to test.
- H1--H8 are unimplemented and unverified.
- No V3 training or V3 test suite was run for this investigation.
- No performance, memory, prediction, covariance, or scaling result exists for
  VFE4.
- The language dataset, tokenizer, seeds, compute budget, and effect thresholds
  belong in dated preregistrations before H6, not in this architecture
  investigation.
- Positive-dimensional base geometry, independent graph links,
  configuration-Gibbs theory, and unrolled/implicit inference require separate
  approved extensions.
- Current source claims must be rebound when any relevant manuscript, code,
  dependency, configuration, estimator, or dataset changes.
