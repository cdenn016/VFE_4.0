"""Deterministic production inference interfaces."""

from .h3_optimize import optimize_h3_arm
from .h4_instrumentation import (
    CountingOperationRecorder,
    InstrumentedLinearAlgebra,
    NullOperationRecorder,
    measure_untimed_memory,
)
from .h4_solvers import (
    H4GaussianSolver,
    H4InnovationDiagnostic,
    H4MaterializedProblem,
    H4NativeDiagnostics,
    InformationFormH4Solver,
    MomentFormH4Solver,
    evaluate_h4_native_diagnostics,
    materialize_h4_problem,
    solve_information_form,
    solve_moment_form,
    to_common_terminal_law,
)

__all__ = [
    "optimize_h3_arm",
    "NullOperationRecorder",
    "CountingOperationRecorder",
    "InstrumentedLinearAlgebra",
    "measure_untimed_memory",
    "H4MaterializedProblem",
    "H4InnovationDiagnostic",
    "H4NativeDiagnostics",
    "H4GaussianSolver",
    "InformationFormH4Solver",
    "MomentFormH4Solver",
    "materialize_h4_problem",
    "solve_information_form",
    "solve_moment_form",
    "to_common_terminal_law",
    "evaluate_h4_native_diagnostics",
]
