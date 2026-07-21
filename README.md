# VFE 4.0

This repository currently implements only the H1 configuration boundary for a
deterministic, evaluation-only verification run. H1 is the only implemented
gate.

The intended user-facing surface is a click-to-run launcher. That launcher is
not implemented yet, and there is no training path in this repository yet.

The initial package validates the frozen H1 configuration, resolves its
artifact path, and produces a reproducible canonical configuration hash.
