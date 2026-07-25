"""Complete forward-covariance diagnostics for the frozen H7 laws.

This module is an evidence-only boundary.  It consumes owned H7 law and
density-probe snapshots, evaluates the complete normalized local and
monolithic objectives, and returns an immutable
``H7ObjectiveCovarianceEvaluation``.  It never returns a tensor suitable for
training and it does not call the private post-H6 training-objective seam.

Task 6 owns independent high-precision oracle values and operand-local budget
construction.  Accordingly, this module requires exact prebuilt
``H7BudgetRecord`` instances for every residual; it never invents a tolerance
or substitutes a run-wide constant.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

import torch
from torch import Tensor

from vfe4.numerics.quadrature import probabilists_gauss_hermite
from vfe4.objective.language_elbo import CompleteLanguageELBOFactorTrace
from vfe4.types.h7 import (
    H7AffineComponentSnapshot,
    H7BudgetRecord,
    H7CompleteLawSnapshot,
    H7DensityObservationRecord,
    H7DensityProbeEvaluation,
    H7DensityProbePair,
    H7FactorizedPromotionWitness,
    H7GaussianComponentSnapshot,
    H7GenerativeSnapshot,
    H7GLPlus2Action,
    H7IndependentH1EvidenceRecord,
    H7InitialJointKlRecord,
    H7LawPairSnapshot,
    H7LocalTermRecord,
    H7ObjectiveCovarianceEvaluation,
    H7RecognitionSnapshot,
    H7ResidualCategory,
    H7ResidualRecord,
    H7ScalarReplayAction,
    H7SourceScorerRowSnapshot,
    H7TensorActionSnapshot,
    h7_owned_sha256,
)
from vfe4.validation.h7_fixture import H1_FIXTURE_RAW_SHA256


H7_COMPLETE_LOCAL_TERM_IDS: tuple[str, ...] = (
    "expected_log_emission[1]",
    "expected_log_emission[2]",
    "model_source_kl[1]",
    "state_source_kl[1]",
    "model_transition_kl[1]",
    "state_transition_kl[1]",
    "model_source_kl[2]",
    "state_source_kl[2]",
    "model_transition_kl[2]",
    "state_transition_kl[2]",
    "joint_recognition_entropy",
)

H7_MATRIX_SCORER_RESIDUAL_IDS: tuple[str, ...] = (
    "source_scorer.model.receiver_1.z_covector",
    "source_scorer.model.receiver_1.m_covector",
    "source_scorer.model.receiver_2.z_covector",
    "source_scorer.model.receiver_2.m_covector",
    "source_scorer.state.receiver_1.z_covector",
    "source_scorer.state.receiver_1.m_covector",
    "source_scorer.state.receiver_2.z_covector",
    "source_scorer.state.receiver_2.m_covector",
    "source_scorer.model.receiver_1.raw_score",
    "source_scorer.model.receiver_2.raw_score",
    "source_scorer.state.receiver_1.raw_score",
    "source_scorer.state.receiver_2.raw_score",
)

H7_COMPLETE_LOCAL_INVARIANT_ID = "complete_local_elbo"
H7_COMPLETE_MONOLITHIC_INVARIANT_ID = "complete_monolithic_elbo"
H7_POINTWISE_P_SHIFT_INVARIANT_ID = "complete_pointwise_p_density_shift"
H7_POINTWISE_Q_SHIFT_INVARIANT_ID = "complete_pointwise_q_density_shift"
H7_POINTWISE_LOG_RATIO_INVARIANT_ID = "complete_pointwise_log_ratio"
H7_ENTROPY_SHIFT_INVARIANT_ID = "joint_recognition_entropy_shift"
H7_SCALAR_EVIDENCE_INVARIANT_ID = (
    "scalar_log_evidence_and_elbo_kl_identity"
)
H7_SCALAR_POSTERIOR_KL_INVARIANT_ID = "scalar_posterior_kl_invariance"
H7_MATRIX_EVIDENCE_NOT_APPLICABLE_REASON = (
    "analytic evidence/posterior KL is not applicable to the nonconjugate "
    "h7-v1 categorical-emission matrix fixture"
)

# These two owned identities name the already-existing independent H1
# producer.  They do not define or authorize a second evidence implementation.
# The semantic preimages freeze the exact entry point and the normalization
# conventions implemented by verification.numpy_oracles.h1_elbo.
_H7_INDEPENDENT_H1_PRODUCER_SEMANTIC: Final = MappingProxyType(
    {
        "module": "verification.numpy_oracles.h1_elbo",
        "entrypoint": "h1_evidence_and_posterior_kl",
        "result_type": "H1IdentityRecord",
        "fixture_id": "h1-v1",
        "quadrature_order": 21,
        "convergence_check_order": 17,
    }
)
_H7_INDEPENDENT_H1_NORMALIZATION_SEMANTIC: Final = MappingProxyType(
    {
        "module": "verification.numpy_oracles.h1_elbo",
        "identity_entrypoint": "h1_evidence_and_posterior_kl",
        "generative_evidence_enumerator": "_all_evidence_source_loop",
        "normalization_check": "h1_all_observation_evidences",
        "source_path_order": ((0, 0), (1, 0), (0, 1), (1, 1)),
        "observation_label_base": 1,
        "vocabulary_labels": (1, 2, 3),
        "quadrature_rule": "numpy.polynomial.hermite.hermgauss",
        "standard_normal_node_scale": "sqrt(2)",
        "standard_normal_weight_scale": "1/sqrt(pi)",
        "quadrature_order": 21,
        "convergence_check_order": 17,
        "evidence_normalization": (
            "joint_3_by_3_observation_probability_sums_to_one"
        ),
        "logarithm": "natural",
        "identity_orientation": (
            "elbo_from_identity=log_probability-posterior_kl(Q||p(.|x))"
        ),
    }
)
H7_INDEPENDENT_H1_PRODUCER_IDENTITY_SHA256: Final = h7_owned_sha256(
    "vfe4.h7.independent-h1-producer-identity.v1",
    _H7_INDEPENDENT_H1_PRODUCER_SEMANTIC,
)
H7_INDEPENDENT_H1_NORMALIZATION_IDENTITY_SHA256: Final = h7_owned_sha256(
    "vfe4.h7.independent-h1-normalization-identity.v1",
    _H7_INDEPENDENT_H1_NORMALIZATION_SEMANTIC,
)

_INITIAL_TERM_ID = "K0_joint_z0_m0"
_MATRIX_ANCHOR_PROVENANCE = (
    "raw_fixture_component_mean_and_lower_cholesky_v1"
)
_FLOAT64_EPSILON = torch.finfo(torch.float64).eps


@dataclass(frozen=True)
class _SourcePath:
    path_id: str
    a: tuple[int, int]
    b: tuple[int, int]
    q_probability: float
    p_probability: float

    def __post_init__(self) -> None:
        if (
            type(self.path_id) is not str
            or not self.path_id
            or type(self.a) is not tuple
            or len(self.a) != 2
            or type(self.b) is not tuple
            or len(self.b) != 2
            or any(type(item) is not int or item < 0 for item in (*self.a, *self.b))
            or not math.isfinite(self.q_probability)
            or not math.isfinite(self.p_probability)
            or self.q_probability < 0.0
            or self.p_probability < 0.0
        ):
            raise ValueError("invalid H7 source-path declaration")
        if self.q_probability > 0.0 and self.p_probability <= 0.0:
            raise ValueError("recognition source mass lies outside generative support")


@dataclass(frozen=True)
class _JointMoments:
    mean: Tensor
    covariance: Tensor

    def __post_init__(self) -> None:
        if (
            not isinstance(self.mean, Tensor)
            or not isinstance(self.covariance, Tensor)
            or self.mean.dtype is not torch.float64
            or self.covariance.dtype is not torch.float64
            or self.mean.ndim != 1
            or self.covariance.shape != (self.mean.numel(), self.mean.numel())
            or not bool(torch.isfinite(self.mean).all().item())
            or not bool(torch.isfinite(self.covariance).all().item())
        ):
            raise ValueError("joint moments must be finite float64 tensors")
        torch.linalg.cholesky(self.covariance)


@dataclass(frozen=True)
class _CompleteValues:
    factor_trace: CompleteLanguageELBOFactorTrace
    initial_joint_kl: float
    initial_factor_ids: tuple[str, ...]
    local_terms: Mapping[str, float]
    local_factor_ids: Mapping[str, tuple[str, ...]]
    complete_local: float
    complete_monolithic: float
    q_moments: Mapping[str, _JointMoments]
    p_moments: Mapping[str, _JointMoments]
    paths: tuple[_SourcePath, ...]

    def __post_init__(self) -> None:
        if (
            type(self.factor_trace) is not CompleteLanguageELBOFactorTrace
            or type(self.initial_joint_kl) is not float
            or not math.isfinite(self.initial_joint_kl)
            or type(self.initial_factor_ids) is not tuple
            or len(self.initial_factor_ids) != 1
            or not isinstance(self.local_terms, Mapping)
            or tuple(self.local_terms) != H7_COMPLETE_LOCAL_TERM_IDS
            or not isinstance(self.local_factor_ids, Mapping)
            or tuple(self.local_factor_ids) != H7_COMPLETE_LOCAL_TERM_IDS
            or any(
                type(value) is not float or not math.isfinite(value)
                for value in self.local_terms.values()
            )
            or any(
                type(value) is not tuple or not value
                for value in self.local_factor_ids.values()
            )
            or type(self.complete_local) is not float
            or not math.isfinite(self.complete_local)
            or type(self.complete_monolithic) is not float
            or not math.isfinite(self.complete_monolithic)
            or not isinstance(self.q_moments, Mapping)
            or not isinstance(self.p_moments, Mapping)
            or tuple(self.q_moments) != tuple(path.path_id for path in self.paths)
            or tuple(self.p_moments) != tuple(path.path_id for path in self.paths)
        ):
            raise ValueError("complete H7 objective values are malformed")


def require_h7_complete_term_inventory(
    *,
    initial_term_ids: tuple[str, ...],
    local_term_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """Fail closed unless the complete H7 local inventory is present once."""

    if initial_term_ids != (_INITIAL_TERM_ID,):
        raise ValueError(
            "H7 requires exactly one undecomposed joint initial K0 term"
        )
    if local_term_ids != H7_COMPLETE_LOCAL_TERM_IDS:
        raise ValueError(
            "H7 complete local-term inventory is missing, duplicated, or reordered"
        )
    return local_term_ids


def require_h7_matrix_scorer_residual_inventory(
    invariant_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """Require eight covector and four raw-score residuals in frozen order."""

    if invariant_ids != H7_MATRIX_SCORER_RESIDUAL_IDS:
        raise ValueError(
            "matrix H7 requires the exact twelve source-scorer residuals"
        )
    return invariant_ids


def h7_joint_gaussian_kl(
    q_mean: Tensor,
    q_covariance: Tensor,
    p_mean: Tensor,
    p_covariance: Tensor,
) -> float:
    """Return the raw finite ``KL(N_q || N_p)`` evaluation without a clamp."""

    dimension = _require_gaussian_inputs(
        q_mean,
        q_covariance,
        p_mean,
        p_covariance,
    )
    q_cholesky = torch.linalg.cholesky(q_covariance)
    p_cholesky = torch.linalg.cholesky(p_covariance)
    displacement = (q_mean - p_mean).unsqueeze(1)
    trace_term = torch.trace(
        torch.cholesky_solve(q_covariance, p_cholesky)
    )
    quadratic_term = torch.sum(
        displacement * torch.cholesky_solve(displacement, p_cholesky)
    )
    q_logdet = 2.0 * torch.log(torch.diagonal(q_cholesky)).sum()
    p_logdet = 2.0 * torch.log(torch.diagonal(p_cholesky)).sum()
    value = 0.5 * (
        trace_term
        + quadratic_term
        - dimension
        + p_logdet
        - q_logdet
    )
    result = float(value.item())
    if not math.isfinite(result):
        raise ValueError("joint Gaussian KL is nonfinite")
    return result


def evaluate_h7_complete_covariance(
    original: H7CompleteLawSnapshot,
    transformed: H7CompleteLawSnapshot,
    action: H7TensorActionSnapshot,
    *,
    original_factor_trace: CompleteLanguageELBOFactorTrace,
    transformed_factor_trace: CompleteLanguageELBOFactorTrace,
    density_probe_pairs: tuple[H7DensityProbePair, ...] | None,
    quadrature_orders: tuple[int, int],
    budgets_by_invariant: Mapping[str, H7BudgetRecord],
    scalar_evidence: H7IndependentH1EvidenceRecord | None = None,
) -> H7ObjectiveCovarianceEvaluation:
    """Evaluate one owned original/transformed complete-law pair.

    ``budgets_by_invariant`` must be produced by Task 6.  This function only
    computes observations and binds each to its exact supplied budget.
    """

    if type(original) is not H7CompleteLawSnapshot:
        raise ValueError("original must be an exact H7 complete-law snapshot")
    if type(transformed) is not H7CompleteLawSnapshot:
        raise ValueError("transformed must be an exact H7 complete-law snapshot")
    if type(action) not in (H7ScalarReplayAction, H7GLPlus2Action):
        raise ValueError("action must be an exact owned H7 action")
    law_pair = H7LawPairSnapshot.create(
        original=original,
        transformed=transformed,
        action_sha256=action.action_sha256,
    )
    return evaluate_h7_law_pair_covariance(
        law_pair,
        action,
        original_factor_trace=original_factor_trace,
        transformed_factor_trace=transformed_factor_trace,
        density_probe_pairs=density_probe_pairs,
        quadrature_orders=quadrature_orders,
        budgets_by_invariant=budgets_by_invariant,
        scalar_evidence=scalar_evidence,
    )


def evaluate_h7_law_pair_covariance(
    law_pair: H7LawPairSnapshot,
    action: H7TensorActionSnapshot,
    *,
    original_factor_trace: CompleteLanguageELBOFactorTrace,
    transformed_factor_trace: CompleteLanguageELBOFactorTrace,
    density_probe_pairs: tuple[H7DensityProbePair, ...] | None,
    quadrature_orders: tuple[int, int],
    budgets_by_invariant: Mapping[str, H7BudgetRecord],
    scalar_evidence: H7IndependentH1EvidenceRecord | None = None,
) -> H7ObjectiveCovarianceEvaluation:
    """Construct the immutable Task-5 evaluation from owned evidence."""

    _validate_law_pair_action(law_pair, action)
    for name, trace in (
        ("original_factor_trace", original_factor_trace),
        ("transformed_factor_trace", transformed_factor_trace),
    ):
        if type(trace) is not CompleteLanguageELBOFactorTrace:
            raise ValueError(f"{name} must be an exact complete post-H6 trace")
        trace.__post_init__()
    if quadrature_orders != (41, 51):
        raise ValueError("H7 production quadrature_orders must equal (41, 51)")
    if not isinstance(budgets_by_invariant, Mapping):
        raise ValueError("budgets_by_invariant must be a mapping")
    if any(
        type(key) is not str or type(value) is not H7BudgetRecord
        for key, value in budgets_by_invariant.items()
    ):
        raise ValueError("objective budgets must be exact H7 records")
    scalar_evidence = _require_scalar_evidence_provenance(
        law_pair,
        action,
        scalar_evidence,
    )
    _require_trace_entropy_orientation(
        original_factor_trace,
        transformed_factor_trace,
        action,
    )

    _require_joint_initial_structure(law_pair.original)
    _require_joint_initial_structure(law_pair.transformed)
    promotion_witness = _require_factorized_promotion(law_pair, action)
    probes = _require_corresponding_probe_pairs(
        law_pair,
        action,
        density_probe_pairs,
    )
    original_values = _evaluate_complete_law(
        law_pair.original,
        factor_trace=original_factor_trace,
        quadrature_order=quadrature_orders[1],
    )
    transformed_values = _evaluate_complete_law(
        law_pair.transformed,
        factor_trace=transformed_factor_trace,
        quadrature_order=quadrature_orders[1],
    )
    if tuple(
        (
            path.path_id,
            path.a,
            path.b,
            path.q_probability,
            path.p_probability,
        )
        for path in original_values.paths
    ) != tuple(
        (
            path.path_id,
            path.a,
            path.b,
            path.q_probability,
            path.p_probability,
        )
        for path in transformed_values.paths
    ):
        raise ValueError("source support/order/probabilities changed under H7")
    require_h7_complete_term_inventory(
        initial_term_ids=(_INITIAL_TERM_ID,),
        local_term_ids=tuple(original_values.local_terms),
    )
    require_h7_complete_term_inventory(
        initial_term_ids=(_INITIAL_TERM_ID,),
        local_term_ids=tuple(transformed_values.local_terms),
    )

    used_budget_ids: set[str] = set()
    initial_residual = _make_residual(
        invariant_id=_INITIAL_TERM_ID,
        category="local_term",
        value=abs(
            original_values.initial_joint_kl
            - transformed_values.initial_joint_kl
        ),
        expected_budget_category="local_term",
        budgets=budgets_by_invariant,
        used_budget_ids=used_budget_ids,
    )
    initial_record = H7InitialJointKlRecord.create(
        term_id=_INITIAL_TERM_ID,
        original_factor_ids=original_values.initial_factor_ids,
        transformed_factor_ids=transformed_values.initial_factor_ids,
        original_value=original_values.initial_joint_kl,
        transformed_value=transformed_values.initial_joint_kl,
        residual=initial_residual,
        chain_decomposition=None,
    )

    expected_entropy_shift = _global_log_jacobian(action)
    local_records: list[H7LocalTermRecord] = []
    for term_id in H7_COMPLETE_LOCAL_TERM_IDS:
        original_value = original_values.local_terms[term_id]
        transformed_value = transformed_values.local_terms[term_id]
        residual_value = (
            abs(
                transformed_value
                - original_value
                - expected_entropy_shift
            )
            if term_id == "joint_recognition_entropy"
            else abs(transformed_value - original_value)
        )
        local_records.append(
            H7LocalTermRecord.create(
                term_id=term_id,
                original_factor_ids=original_values.local_factor_ids[term_id],
                transformed_factor_ids=(
                    transformed_values.local_factor_ids[term_id]
                ),
                original_value=original_value,
                transformed_value=transformed_value,
                signed_child_ids=_signed_child_ids(
                    law_pair.original,
                    term_id,
                ),
                residual=_make_residual(
                    invariant_id=term_id,
                    category="local_term",
                    value=residual_value,
                    expected_budget_category="local_term",
                    budgets=budgets_by_invariant,
                    used_budget_ids=used_budget_ids,
                ),
            )
        )

    scorer_residuals = _evaluate_scorer_residuals(
        law_pair,
        action,
        budgets_by_invariant,
        used_budget_ids,
    )
    density_evaluations = _evaluate_density_probes(
        law_pair,
        action,
        probes,
        original_values,
        transformed_values,
        budgets_by_invariant,
        used_budget_ids,
    )
    density_observations = tuple(
        observation
        for evaluation in density_evaluations
        for observation in evaluation.observations
    )
    complete_local = _make_residual(
        invariant_id=H7_COMPLETE_LOCAL_INVARIANT_ID,
        category="absolute",
        value=abs(
            original_values.complete_local
            - transformed_values.complete_local
        ),
        expected_budget_category="complete_objective",
        budgets=budgets_by_invariant,
        used_budget_ids=used_budget_ids,
    )
    complete_monolithic = _make_residual(
        invariant_id=H7_COMPLETE_MONOLITHIC_INVARIANT_ID,
        category="monolithic",
        value=max(
            abs(
                original_values.complete_monolithic
                - transformed_values.complete_monolithic
            ),
            abs(
                original_values.complete_local
                - original_values.complete_monolithic
            ),
            abs(
                transformed_values.complete_local
                - transformed_values.complete_monolithic
            ),
        ),
        expected_budget_category="complete_objective",
        budgets=budgets_by_invariant,
        used_budget_ids=used_budget_ids,
    )
    p_density_shift = _make_residual(
        invariant_id=H7_POINTWISE_P_SHIFT_INVARIANT_ID,
        category="density",
        value=max(
            item.residual.value
            for item in density_observations
            if item.role == "p"
        ),
        expected_budget_category="density",
        budgets=budgets_by_invariant,
        used_budget_ids=used_budget_ids,
    )
    q_density_shift = _make_residual(
        invariant_id=H7_POINTWISE_Q_SHIFT_INVARIANT_ID,
        category="density",
        value=max(
            item.residual.value
            for item in density_observations
            if item.role == "q"
        ),
        expected_budget_category="density",
        budgets=budgets_by_invariant,
        used_budget_ids=used_budget_ids,
    )
    log_ratio = _make_residual(
        invariant_id=H7_POINTWISE_LOG_RATIO_INVARIANT_ID,
        category="density",
        value=max(
            item.residual.value
            for item in density_observations
            if item.role == "log_ratio"
        ),
        expected_budget_category="density",
        budgets=budgets_by_invariant,
        used_budget_ids=used_budget_ids,
    )
    entropy_shift = _make_residual(
        invariant_id=H7_ENTROPY_SHIFT_INVARIANT_ID,
        category="jacobian",
        value=abs(
            (
                transformed_values.local_terms[
                    "joint_recognition_entropy"
                ]
                - original_values.local_terms[
                    "joint_recognition_entropy"
                ]
            )
            - expected_entropy_shift
        ),
        expected_budget_category="density",
        budgets=budgets_by_invariant,
        used_budget_ids=used_budget_ids,
    )

    evidence_record: H7ResidualRecord | None
    posterior_kl_record: H7ResidualRecord | None
    not_applicable_reason: str | None
    if law_pair.original.fixture_id == "h1-v1":
        if scalar_evidence is None:
            raise RuntimeError("validated scalar evidence unexpectedly missing")
        evidence_value = max(
            abs(
                scalar_evidence.original_log_evidence
                - scalar_evidence.transformed_log_evidence
            ),
            abs(
                scalar_evidence.original_log_evidence
                - original_values.complete_local
                - scalar_evidence.original_posterior_kl
            ),
            abs(
                scalar_evidence.transformed_log_evidence
                - transformed_values.complete_local
                - scalar_evidence.transformed_posterior_kl
            ),
        )
        evidence_record = _make_residual(
            invariant_id=H7_SCALAR_EVIDENCE_INVARIANT_ID,
            category="evidence",
            value=evidence_value,
            expected_budget_category="complete_objective",
            budgets=budgets_by_invariant,
            used_budget_ids=used_budget_ids,
        )
        posterior_kl_record = _make_residual(
            invariant_id=H7_SCALAR_POSTERIOR_KL_INVARIANT_ID,
            category="posterior_kl",
            value=abs(
                scalar_evidence.original_posterior_kl
                - scalar_evidence.transformed_posterior_kl
            ),
            expected_budget_category="complete_objective",
            budgets=budgets_by_invariant,
            used_budget_ids=used_budget_ids,
        )
        not_applicable_reason = None
    else:
        evidence_record = None
        posterior_kl_record = None
        not_applicable_reason = H7_MATRIX_EVIDENCE_NOT_APPLICABLE_REASON

    if set(budgets_by_invariant) != used_budget_ids:
        missing = sorted(set(budgets_by_invariant) - used_budget_ids)
        raise ValueError(
            "objective budget inventory contains unused or unknown IDs: "
            + ", ".join(missing)
        )
    return H7ObjectiveCovarianceEvaluation.create(
        original_factor_trace_sha256=original_factor_trace.trace_sha256,
        transformed_factor_trace_sha256=(
            transformed_factor_trace.trace_sha256
        ),
        original_ordered_factor_ids=(
            original_factor_trace.ordered_factor_ids
        ),
        transformed_ordered_factor_ids=(
            transformed_factor_trace.ordered_factor_ids
        ),
        original_ordered_factor_values=(
            original_factor_trace.ordered_factor_values
        ),
        transformed_ordered_factor_values=(
            transformed_factor_trace.ordered_factor_values
        ),
        original_complete_local_value=original_factor_trace.total_value,
        transformed_complete_local_value=transformed_factor_trace.total_value,
        initial_joint_kl=initial_record,
        local_terms=tuple(local_records),
        density_probes=probes,
        density_probe_evaluations=density_evaluations,
        scorer_residuals=scorer_residuals,
        complete_local=complete_local,
        complete_monolithic=complete_monolithic,
        p_density_shift=p_density_shift,
        q_density_shift=q_density_shift,
        log_ratio=log_ratio,
        entropy_shift=entropy_shift,
        scalar_evidence=scalar_evidence,
        factorized_promotion_witness=promotion_witness,
        evidence=evidence_record,
        posterior_kl=posterior_kl_record,
        not_applicable_reason=not_applicable_reason,
    )


def _validate_law_pair_action(
    law_pair: H7LawPairSnapshot,
    action: H7TensorActionSnapshot,
) -> None:
    if type(law_pair) is not H7LawPairSnapshot:
        raise ValueError("law_pair must be an exact owned H7 record")
    law_pair.__post_init__()
    if type(action) not in (H7ScalarReplayAction, H7GLPlus2Action):
        raise ValueError("action must be an exact owned H7 action")
    if law_pair.action_sha256 != action.action_sha256:
        raise ValueError("law pair belongs to another H7 action")
    fixture_id = law_pair.original.fixture_id
    if (
        (fixture_id == "h1-v1" and type(action) is not H7ScalarReplayAction)
        or (fixture_id == "h7-v1" and type(action) is not H7GLPlus2Action)
    ):
        raise ValueError("law-pair fixture/action dimensions disagree")


def _require_scalar_evidence_provenance(
    law_pair: H7LawPairSnapshot,
    action: H7TensorActionSnapshot,
    scalar_evidence: H7IndependentH1EvidenceRecord | None,
) -> H7IndependentH1EvidenceRecord | None:
    if law_pair.original.fixture_id != "h1-v1":
        if scalar_evidence is not None:
            raise ValueError("matrix H7 cannot fabricate scalar evidence values")
        return None
    if type(scalar_evidence) is not H7IndependentH1EvidenceRecord:
        raise ValueError(
            "scalar H7 replay requires independent evidence/posterior-KL values"
        )
    scalar_evidence.__post_init__()
    if (
        scalar_evidence.raw_fixture_sha256 != H1_FIXTURE_RAW_SHA256
        or scalar_evidence.raw_fixture_sha256
        != law_pair.original.raw_fixture_sha256
        or scalar_evidence.action_sha256 != action.action_sha256
    ):
        raise ValueError("scalar evidence provenance does not match fixture/action")
    if (
        scalar_evidence.producer_identity_sha256
        != H7_INDEPENDENT_H1_PRODUCER_IDENTITY_SHA256
    ):
        raise ValueError(
            "scalar evidence producer identity is not the frozen independent H1 "
            "producer"
        )
    if (
        scalar_evidence.normalization_identity_sha256
        != H7_INDEPENDENT_H1_NORMALIZATION_IDENTITY_SHA256
    ):
        raise ValueError(
            "scalar evidence normalization identity is not the frozen H1 "
            "normalization"
        )
    return scalar_evidence


def _require_trace_entropy_orientation(
    original: CompleteLanguageELBOFactorTrace,
    transformed: CompleteLanguageELBOFactorTrace,
    action: H7TensorActionSnapshot,
) -> None:
    def entropy_value(trace: CompleteLanguageELBOFactorTrace) -> float:
        values = tuple(
            value
            for term, value in zip(
                trace.source_trace.ordered_factor_terms,
                trace.ordered_factor_values,
                strict=True,
            )
            if term.partition == "entropy"
        )
        if len(values) != 2:
            raise ValueError("complete trace lacks both H7 entropy factors")
        return math.fsum(values)

    expected_shift = _global_log_jacobian(action)
    if expected_shift == 0.0:
        return
    observed_shift = entropy_value(transformed) - entropy_value(original)
    if observed_shift == 0.0 or observed_shift * expected_shift < 0.0:
        raise ValueError(
            "complete trace reverses or omits the recognition entropy-shift sign"
        )


def _require_joint_initial_structure(law: H7CompleteLawSnapshot) -> None:
    q_mean = law.recognition.initial_joint.mean.value()
    q_covariance = law.recognition.initial_joint.covariance.value()
    if q_mean.numel() % 2:
        raise ValueError("joint initial law cannot be divided into z0/m0 blocks")
    block = q_mean.numel() // 2
    if (
        law.fixture_id == "h7-v1"
        and law.recognition.origin_family == "structured_full_block"
    ):
        cross = q_covariance[:block, block:]
        if not bool(torch.any(cross != 0.0).item()):
            raise ValueError(
                "structured H7 initial law lost nonzero z0/m0 cross covariance"
            )


def _require_factorized_promotion(
    law_pair: H7LawPairSnapshot,
    action: H7TensorActionSnapshot,
) -> H7FactorizedPromotionWitness | None:
    original = law_pair.original.recognition
    transformed = law_pair.transformed.recognition
    if original.origin_family != "factorized_diagonal_within_fiber":
        return None
    if original.representation != "factorized_diagonal_within_fiber":
        raise ValueError("factorized origin must enter Task 5 in factorized form")
    original_candidates = (
        original.initial_joint,
        *(
            item.receiver_law
            for item in (
                *original.model_conditionals,
                *original.state_conditionals,
            )
        ),
    )
    for component in original_candidates:
        covariance = component.covariance.value()
        for block_start in range(0, covariance.shape[0], action.dimension):
            block = covariance[
                block_start : block_start + action.dimension,
                block_start : block_start + action.dimension,
            ]
            if block.shape != (action.dimension, action.dimension):
                raise ValueError(
                    "factorized covariance cannot be divided into fibers"
                )
            if bool(
                torch.any(
                    block != torch.diag(torch.diagonal(block))
                ).item()
            ):
                raise ValueError(
                    "factorized origin has a non-diagonal within-fiber covariance"
                )
    action_values = tuple(item.value() for item in action.elements)
    non_diagonal = any(
        bool(
            torch.any(
                value
                != torch.diag(torch.diagonal(value))
            ).item()
        )
        for value in action_values
    )
    if not non_diagonal:
        if transformed.representation != "factorized_diagonal_within_fiber":
            raise ValueError("diagonal action unexpectedly promoted factorized law")
        return None
    if transformed.representation != "unrestricted_full_block_pushforward":
        raise ValueError(
            "non-diagonal action requires unrestricted factorized pushforward"
        )
    candidates = (
        transformed.initial_joint,
        *(
            item.receiver_law
            for item in (
                *transformed.model_conditionals,
                *transformed.state_conditionals,
            )
        ),
    )
    for component in candidates:
        covariance = component.covariance.value()
        block_width = action.dimension
        for block_start in range(0, covariance.shape[0], block_width):
            block_stop = block_start + block_width
            if block_stop > covariance.shape[0]:
                raise ValueError(
                    "factorized covariance cannot be divided into fibers"
                )
            for row in range(block_start, block_stop):
                for column in range(row + 1, block_stop):
                    value = float(covariance[row, column].item())
                    if value != 0.0:
                        return H7FactorizedPromotionWitness.create(
                            action_sha256=action.action_sha256,
                            component_id=component.component_id,
                            covariance_snapshot_sha256=(
                                component.covariance.snapshot_sha256
                            ),
                            row=row,
                            column=column,
                            value=value,
                            transformed_representation=(
                                "unrestricted_full_block_pushforward"
                            ),
                        )
    raise ValueError(
        "unrestricted factorized pushforward lacks a nonzero off-diagonal witness"
    )


def _evaluate_complete_law(
    law: H7CompleteLawSnapshot,
    *,
    factor_trace: CompleteLanguageELBOFactorTrace,
    quadrature_order: int,
) -> _CompleteValues:
    factor_trace.__post_init__()
    paths = _source_paths(law)
    q_moments = {
        path.path_id: _recognition_joint_moments(law.recognition, path)
        for path in paths
    }
    p_moments = {
        path.path_id: _generative_joint_moments(law.generative, path)
        for path in paths
    }
    by_slot = {
        (term.partition, term.receiver_t): (
            term.factor_identity_sha256,
            value,
        )
        for term, value in zip(
            factor_trace.source_trace.ordered_factor_terms,
            factor_trace.ordered_factor_values,
            strict=True,
        )
    }
    local_slots: dict[str, tuple[tuple[str, int], ...]] = {
        "expected_log_emission[1]": (("emission", 1),),
        "expected_log_emission[2]": (("emission", 2),),
        "model_source_kl[1]": (("model_source", 1),),
        "state_source_kl[1]": (("state_source", 1),),
        "model_transition_kl[1]": (("model_transition", 1),),
        "state_transition_kl[1]": (("state_transition", 1),),
        "model_source_kl[2]": (("model_source", 2),),
        "state_source_kl[2]": (("state_source", 2),),
        "model_transition_kl[2]": (("model_transition", 2),),
        "state_transition_kl[2]": (("state_transition", 2),),
        "joint_recognition_entropy": (
            ("entropy", 1),
            ("entropy", 2),
        ),
    }
    local_factor_ids = {
        term_id: tuple(by_slot[slot][0] for slot in slots)
        for term_id, slots in local_slots.items()
    }
    local_values = {
        term_id: float(math.fsum(by_slot[slot][1] for slot in slots))
        for term_id, slots in local_slots.items()
    }
    initial_factor_id, initial_value = by_slot[("initial", 0)]
    bound_factor_ids = (
        initial_factor_id,
        *(
            factor_id
            for term_id in H7_COMPLETE_LOCAL_TERM_IDS
            for factor_id in local_factor_ids[term_id]
        ),
    )
    if set(bound_factor_ids) != set(factor_trace.ordered_factor_ids):
        raise ValueError("H7 local inventory does not bind all post-H6 factors")
    monolithic_contributions: list[float] = []
    for path in paths:
        if path.q_probability <= 0.0:
            continue
        gaussian_ratio = -h7_joint_gaussian_kl(
            q_moments[path.path_id].mean,
            q_moments[path.path_id].covariance,
            p_moments[path.path_id].mean,
            p_moments[path.path_id].covariance,
        )
        source_ratio = math.log(path.p_probability) - math.log(
            path.q_probability
        )
        emission = math.fsum(
            _expected_log_emission(
                law,
                q_moments[path.path_id],
                receiver_t=receiver_t,
                quadrature_order=quadrature_order,
            )
            for receiver_t in (1, 2)
        )
        monolithic_contributions.append(
            path.q_probability
            * math.fsum((gaussian_ratio, source_ratio, emission))
        )
    return _CompleteValues(
        factor_trace=factor_trace,
        initial_joint_kl=float(initial_value),
        initial_factor_ids=(initial_factor_id,),
        local_terms=MappingProxyType(local_values),
        local_factor_ids=MappingProxyType(local_factor_ids),
        complete_local=factor_trace.total_value,
        complete_monolithic=float(math.fsum(monolithic_contributions)),
        q_moments=MappingProxyType(q_moments),
        p_moments=MappingProxyType(p_moments),
        paths=paths,
    )


def _source_paths(law: H7CompleteLawSnapshot) -> tuple[_SourcePath, ...]:
    if law.fixture_id == "h7-v1":
        return (_SourcePath("matrix-singleton-path", (0, 1), (0, 1), 1.0, 1.0),)
    p_law = law.generative.scalar_source_law
    q_law = law.recognition.scalar_source_law
    if p_law is None or q_law is None:
        raise ValueError("scalar H7 law lacks exact source snapshots")
    if p_law.ordered_paths != q_law.ordered_paths:
        raise ValueError("scalar source path inventories disagree")
    paths: list[_SourcePath] = []
    for declared in p_law.ordered_paths:
        q_probability = 1.0
        p_probability = 1.0
        for index in range(2):
            a = declared.a[index]
            b = declared.b[index]
            q_b = q_law.model_source_probabilities[index].value()
            q_a_given_b = (
                q_law.state_source_probabilities_given_model_source[
                    index
                ].value()
            )
            p_b = p_law.model_source_priors[index].value()
            p_a = p_law.state_source_priors[index].value()
            q_probability *= float(q_b[b]) * float(q_a_given_b[b, a])
            p_probability *= float(p_b[b]) * float(p_a[a])
        paths.append(
            _SourcePath(
                declared.path_id,
                declared.a,
                declared.b,
                q_probability,
                p_probability,
            )
        )
    if not math.isclose(
        math.fsum(path.q_probability for path in paths),
        1.0,
        rel_tol=0.0,
        abs_tol=64.0 * _FLOAT64_EPSILON,
    ):
        raise ValueError("recognition source-path probabilities do not normalize")
    return tuple(paths)


def _recognition_joint_moments(
    recognition: H7RecognitionSnapshot,
    path: _SourcePath,
) -> _JointMoments:
    return _propagate_joint_moments(
        initial=recognition.initial_joint,
        model_selector=lambda receiver_t, source_j: _find_recognition_model(
            recognition,
            receiver_t,
            source_j,
        ),
        state_selector=lambda receiver_t, source_j, model_source_j: (
            _find_recognition_state(
                recognition,
                receiver_t,
                source_j,
                model_source_j,
            )
        ),
        path=path,
    )


def _generative_joint_moments(
    generative: H7GenerativeSnapshot,
    path: _SourcePath,
) -> _JointMoments:
    return _propagate_joint_moments(
        initial=generative.initial_joint,
        model_selector=lambda receiver_t, source_j: _find_generative_transition(
            generative,
            "model",
            receiver_t,
            source_j,
        ),
        state_selector=lambda receiver_t, source_j, _model_source_j: (
            _find_generative_transition(
                generative,
                "state",
                receiver_t,
                source_j,
            )
        ),
        path=path,
    )


def _propagate_joint_moments(
    *,
    initial: H7GaussianComponentSnapshot,
    model_selector: Callable[[int, int], H7AffineComponentSnapshot],
    state_selector: Callable[[int, int, int], H7AffineComponentSnapshot],
    path: _SourcePath,
) -> _JointMoments:
    initial_mean = initial.mean.value()
    initial_covariance = initial.covariance.value()
    dimension = initial_mean.numel() // 2
    total_dimension = 6 * dimension
    mean = torch.zeros(total_dimension, dtype=torch.float64)
    covariance = torch.zeros(
        (total_dimension, total_dimension),
        dtype=torch.float64,
    )
    mean[: 2 * dimension] = initial_mean
    covariance[: 2 * dimension, : 2 * dimension] = initial_covariance
    active = list(range(2 * dimension))
    for receiver_t in (1, 2):
        a = path.a[receiver_t - 1]
        b = path.b[receiver_t - 1]
        model = model_selector(receiver_t, b)
        model_target = _block_indices("m", receiver_t, dimension)
        model_parent = _block_indices("m", b, dimension)
        _insert_affine_moments(
            mean,
            covariance,
            active=tuple(active),
            target=model_target,
            parent_blocks=((model_parent, model.parent_map.value()),),
            offset=model.offset.value(),
            noise_covariance=model.receiver_law.covariance.value(),
        )
        active.extend(model_target)

        state = state_selector(receiver_t, a, b)
        state_target = _block_indices("z", receiver_t, dimension)
        state_parent = _block_indices("z", a, dimension)
        model_map = state.same_receiver_model_map
        if model_map is None:
            raise ValueError("state conditional lacks its same-receiver model map")
        _insert_affine_moments(
            mean,
            covariance,
            active=tuple(active),
            target=state_target,
            parent_blocks=(
                (state_parent, state.parent_map.value()),
                (model_target, model_map.value()),
            ),
            offset=state.offset.value(),
            noise_covariance=state.receiver_law.covariance.value(),
        )
        active.extend(state_target)
    return _JointMoments(mean, covariance)


def _insert_affine_moments(
    mean: Tensor,
    covariance: Tensor,
    *,
    active: tuple[int, ...],
    target: tuple[int, ...],
    parent_blocks: tuple[tuple[tuple[int, ...], Tensor], ...],
    offset: Tensor,
    noise_covariance: Tensor,
) -> None:
    output_dimension = len(target)
    linear = torch.zeros(
        (output_dimension, mean.numel()),
        dtype=torch.float64,
    )
    for indices, matrix in parent_blocks:
        linear[:, list(indices)] += matrix
    target_mean = linear @ mean + offset
    cross = linear @ covariance[:, list(active)]
    target_covariance = linear @ covariance @ linear.T + noise_covariance
    mean[list(target)] = target_mean
    covariance[list(target), :] = 0.0
    covariance[:, list(target)] = 0.0
    target_rows = torch.tensor(target, dtype=torch.long).unsqueeze(1)
    active_columns = torch.tensor(active, dtype=torch.long).unsqueeze(0)
    covariance[target_rows, active_columns] = cross
    covariance[active_columns.T, target_rows.T] = cross.T
    target_columns = torch.tensor(target, dtype=torch.long).unsqueeze(0)
    covariance[target_rows, target_columns] = target_covariance


def _block_indices(
    channel: Literal["z", "m"],
    population_label: int,
    dimension: int,
) -> tuple[int, ...]:
    start = 2 * population_label * dimension
    if channel == "m":
        start += dimension
    return tuple(range(start, start + dimension))


def _find_generative_transition(
    generative: H7GenerativeSnapshot,
    bank: Literal["model", "state"],
    receiver_t: int,
    source_j: int,
) -> H7AffineComponentSnapshot:
    matches = tuple(
        item
        for item in generative.transitions
        if item.bank == bank
        and item.receiver_t == receiver_t
        and item.source_j == source_j
    )
    if len(matches) != 1:
        raise ValueError("generative transition lookup is incomplete or ambiguous")
    return matches[0]


def _find_recognition_model(
    recognition: H7RecognitionSnapshot,
    receiver_t: int,
    source_j: int,
) -> H7AffineComponentSnapshot:
    matches = tuple(
        item
        for item in recognition.model_conditionals
        if item.receiver_t == receiver_t and item.source_j == source_j
    )
    if len(matches) != 1:
        raise ValueError("recognition model conditional is incomplete or ambiguous")
    return matches[0]


def _find_recognition_state(
    recognition: H7RecognitionSnapshot,
    receiver_t: int,
    source_j: int,
    model_source_j: int,
) -> H7AffineComponentSnapshot:
    candidates = tuple(
        item
        for item in recognition.state_conditionals
        if item.receiver_t == receiver_t
    )
    if recognition.initial_joint.mean.shape == (2,):
        marker = f".a_{source_j}.b_{model_source_j}."
        matches = tuple(item for item in candidates if marker in item.component_id)
    else:
        matches = tuple(
            item for item in candidates if item.source_j == source_j
        )
    if len(matches) != 1:
        raise ValueError("recognition state conditional is incomplete or ambiguous")
    return matches[0]


def _expected_log_emission(
    law: H7CompleteLawSnapshot,
    moments: _JointMoments,
    *,
    receiver_t: int,
    quadrature_order: int,
) -> float:
    decoder = tuple(
        item
        for item in law.generative.decoders
        if item.receiver_t == receiver_t
    )
    if len(decoder) != 1:
        raise ValueError("decoder inventory is incomplete or ambiguous")
    selected = law.recognition.context.observation_labels[receiver_t - 1]
    if law.fixture_id == "h1-v1":
        selected -= 1
    state_indices = _block_indices(
        "z",
        receiver_t,
        decoder[0].state_weight.shape[1],
    )
    model_indices = _block_indices(
        "m",
        receiver_t,
        decoder[0].model_weight.shape[1],
    )
    indices = (*state_indices, *model_indices)
    latent_mean = moments.mean[list(indices)]
    latent_covariance = moments.covariance[list(indices)][:, list(indices)]
    weight = torch.cat(
        (
            decoder[0].state_weight.value(),
            decoder[0].model_weight.value(),
        ),
        dim=1,
    )
    logits_mean = weight @ latent_mean + decoder[0].bias.value()
    logits_covariance = weight @ latent_covariance @ weight.T
    vocabulary = logits_mean.numel()
    if selected < 0 or selected >= vocabulary:
        raise ValueError("observation label is outside the decoder vocabulary")

    contrast = torch.zeros(
        (vocabulary - 1, vocabulary),
        dtype=torch.float64,
    )
    for row in range(vocabulary - 1):
        contrast[row, row] = 1.0
        contrast[row, vocabulary - 1] = -1.0
    contrast_mean = contrast @ logits_mean
    contrast_covariance = contrast @ logits_covariance @ contrast.T
    cholesky = torch.linalg.cholesky(contrast_covariance)
    nodes, weights = probabilists_gauss_hermite(
        quadrature_order,
        dtype=torch.float64,
    )
    grids = torch.meshgrid(
        *(nodes for _ in range(vocabulary - 1)),
        indexing="ij",
    )
    standards = torch.stack(
        tuple(grid.reshape(-1) for grid in grids),
        dim=1,
    )
    weight_grids = torch.meshgrid(
        *(weights for _ in range(vocabulary - 1)),
        indexing="ij",
    )
    quadrature_weights = torch.ones(
        standards.shape[0],
        dtype=torch.float64,
    )
    for grid in weight_grids:
        quadrature_weights *= grid.reshape(-1)
    contrasts = contrast_mean + standards @ cholesky.T
    augmented = torch.cat(
        (
            contrasts,
            torch.zeros(
                (contrasts.shape[0], 1),
                dtype=torch.float64,
            ),
        ),
        dim=1,
    )
    selected_values = augmented[:, selected] - torch.logsumexp(
        augmented,
        dim=1,
    )
    value = torch.sum(quadrature_weights * selected_values)
    result = float(value.item())
    if not math.isfinite(result):
        raise ValueError("emission expectation is nonfinite")
    return result


def _require_corresponding_probe_pairs(
    law_pair: H7LawPairSnapshot,
    action: H7TensorActionSnapshot,
    supplied: tuple[H7DensityProbePair, ...] | None,
) -> tuple[H7DensityProbePair, ...]:
    if law_pair.original.fixture_id == "h1-v1":
        probe_set = law_pair.original.scalar_probe_set
        transformed_probe_set = law_pair.transformed.scalar_probe_set
        if probe_set is None or transformed_probe_set is None:
            raise ValueError("scalar law pair lacks its frozen probe set")
        expected = tuple(
            pair
            for pair in probe_set.probe_pairs
            if pair.action_sha256 == action.action_sha256
        )
        if len(expected) != 4:
            raise ValueError("scalar action requires all four frozen source probes")
        if tuple(
            pair.probe_sha256 for pair in transformed_probe_set.probe_pairs
            if pair.action_sha256 == action.action_sha256
        ) != tuple(pair.probe_sha256 for pair in expected):
            raise ValueError("scalar transformed probe provenance changed")
        probes = expected if supplied is None else supplied
        if tuple(pair.probe_sha256 for pair in probes) != tuple(
            pair.probe_sha256 for pair in expected
        ):
            raise ValueError(
                "scalar evaluator accepts only its exact pre-expanded probes"
            )
    else:
        if supplied is None:
            raise ValueError(
                "matrix H7 requires parser-owned pre-expanded density probes"
            )
        probes = supplied
        _require_matrix_probe_inventory(law_pair, action, probes)
    if type(probes) is not tuple or not probes:
        raise ValueError("density probe inventory must be a nonempty tuple")
    for pair in probes:
        if (
            type(pair) is not H7DensityProbePair
            or pair.fixture_id != law_pair.original.fixture_id
            or pair.action_sha256 != action.action_sha256
        ):
            raise ValueError("density probe fixture/action provenance disagrees")
        _validate_probe_action(pair, action)
    return probes


def _require_matrix_probe_inventory(
    law_pair: H7LawPairSnapshot,
    action: H7TensorActionSnapshot,
    probes: tuple[H7DensityProbePair, ...],
) -> None:
    if type(action) is not H7GLPlus2Action:
        raise ValueError("matrix probe inventory requires GL+(2)")
    recognition = law_pair.original.recognition
    expected_components = (
        (law_pair.original.generative.initial_joint.component_id, "initial", 4),
        *tuple(
            (item.component_id, _transition_source_id(item), 2)
            for item in sorted(
                (
                    item
                    for item in law_pair.original.generative.transitions
                    if item.bank == "model"
                ),
                key=lambda item: item.receiver_t,
            )
        ),
        *tuple(
            (item.component_id, _transition_source_id(item), 2)
            for item in sorted(
                (
                    item
                    for item in law_pair.original.generative.transitions
                    if item.bank == "state"
                ),
                key=lambda item: item.receiver_t,
            )
        ),
        (recognition.initial_joint.component_id, "initial", 4),
        *tuple(
            (item.component_id, _transition_source_id(item), 2)
            for item in recognition.model_conditionals
        ),
        *tuple(
            (item.component_id, _transition_source_id(item), 2)
            for item in recognition.state_conditionals
        ),
        ("p.global", "matrix-singleton-path", 12),
        (
            f"q.{_recognition_component_prefix(recognition)}.global",
            "matrix-singleton-path",
            12,
        ),
    )
    expected_rows = tuple(
        (component_id, source_id, direction)
        for component_id, source_id, dimension in expected_components
        for direction in _density_directions(dimension)
    )
    observed_rows = tuple(
        (
            pair.component_id,
            pair.source_id,
            pair.probe_id.rsplit(":", maxsplit=1)[-1],
        )
        for pair in probes
    )
    if observed_rows != expected_rows:
        raise ValueError(
            "matrix density probes are missing, duplicated, extra, or reordered"
        )
    if any(
        pair.anchor_provenance != _MATRIX_ANCHOR_PROVENANCE
        for pair in probes
    ):
        raise ValueError("matrix density probe anchor provenance changed")


def _density_directions(dimension: int) -> tuple[str, ...]:
    return (
        "zero",
        *tuple(
            name
            for index in range(dimension)
            for name in (f"+e{index}", f"-e{index}")
        ),
    )


def _transition_source_id(item: H7AffineComponentSnapshot) -> str:
    return f"{item.bank}:{item.receiver_t}<-{item.source_j}"


def _recognition_component_prefix(
    recognition: H7RecognitionSnapshot,
) -> Literal["structured", "factorized"]:
    return (
        "structured"
        if recognition.origin_family == "structured_full_block"
        else "factorized"
    )


def _validate_probe_action(
    pair: H7DensityProbePair,
    action: H7TensorActionSnapshot,
) -> None:
    component_action = _component_action(action, pair.component_id)
    x = pair.x.value()
    x_prime = pair.x_prime.value()
    expected_prime = component_action @ x
    if not torch.equal(x_prime, expected_prime):
        raise ValueError(
            "density x_prime is not the declared action on the same anchor"
        )
    expected_shift = float(torch.linalg.slogdet(component_action)[1].item())
    observed_scope = _probe_scope_shift(pair)
    if not math.isclose(
        observed_scope,
        expected_shift,
        rel_tol=0.0,
        abs_tol=64.0 * _FLOAT64_EPSILON * max(1.0, abs(expected_shift)),
    ):
        raise ValueError("density probe Jacobian shift disagrees with its action")


def _probe_scope_shift(pair: H7DensityProbePair) -> float:
    nonzero = tuple(
        value
        for value in (
            pair.initial_log_jacobian_shift,
            pair.receiver_log_jacobian_shift,
            pair.global_log_jacobian_shift,
        )
        if value != 0.0
    )
    if len(nonzero) > 1:
        raise ValueError("density probe mixes Jacobian scopes")
    return 0.0 if not nonzero else nonzero[0]


def _component_action(
    action: H7TensorActionSnapshot,
    component_id: str,
) -> Tensor:
    elements = tuple(item.value() for item in action.elements)
    if ".initial_joint" in component_id:
        return torch.block_diag(elements[0], elements[0])
    if ".global" in component_id:
        return torch.block_diag(
            elements[0],
            elements[0],
            elements[1],
            elements[1],
            elements[2],
            elements[2],
        )
    marker = ".receiver_"
    if marker not in component_id:
        raise ValueError("density probe component has no declared action scope")
    suffix = component_id.split(marker, maxsplit=1)[1]
    try:
        receiver_t = int(suffix.split(".", maxsplit=1)[0])
    except ValueError as error:
        raise ValueError("density receiver component ID is malformed") from error
    if receiver_t not in (1, 2):
        raise ValueError("density receiver label is outside H7")
    return elements[receiver_t]


def _evaluate_density_probes(
    law_pair: H7LawPairSnapshot,
    action: H7TensorActionSnapshot,
    probes: tuple[H7DensityProbePair, ...],
    original_values: _CompleteValues,
    transformed_values: _CompleteValues,
    budgets: Mapping[str, H7BudgetRecord],
    used_budget_ids: set[str],
) -> tuple[H7DensityProbeEvaluation, ...]:
    _validate_jacobian_metadata(law_pair, action)
    evaluations: list[H7DensityProbeEvaluation] = []
    for pair in probes:
        x = pair.x.value()
        x_prime = pair.x_prime.value()
        shift = _probe_scope_shift(pair)
        observations: list[H7DensityObservationRecord] = []
        if ".global" in pair.component_id:
            original_path = _path_by_id(original_values.paths, pair.source_id)
            transformed_path = _path_by_id(
                transformed_values.paths,
                pair.source_id,
            )
            original_p = _global_log_density(
                law_pair.original,
                original_values.p_moments[pair.source_id],
                original_path,
                x,
                role="p",
            )
            transformed_p = _global_log_density(
                law_pair.transformed,
                transformed_values.p_moments[pair.source_id],
                transformed_path,
                x_prime,
                role="p",
            )
            original_q = _global_log_density(
                law_pair.original,
                original_values.q_moments[pair.source_id],
                original_path,
                x,
                role="q",
            )
            transformed_q = _global_log_density(
                law_pair.transformed,
                transformed_values.q_moments[pair.source_id],
                transformed_path,
                x_prime,
                role="q",
            )
            observations.extend(
                (
                    _make_density_observation(
                        pair=pair,
                        role="p",
                        original_value=original_p,
                        transformed_value=transformed_p,
                        expected_shift=shift,
                        budgets=budgets,
                        used_budget_ids=used_budget_ids,
                    ),
                    _make_density_observation(
                        pair=pair,
                        role="q",
                        original_value=original_q,
                        transformed_value=transformed_q,
                        expected_shift=shift,
                        budgets=budgets,
                        used_budget_ids=used_budget_ids,
                    ),
                    _make_density_observation(
                        pair=pair,
                        role="log_ratio",
                        original_value=original_p - original_q,
                        transformed_value=transformed_p - transformed_q,
                        expected_shift=0.0,
                        budgets=budgets,
                        used_budget_ids=used_budget_ids,
                    ),
                )
            )
        else:
            original_component = _density_component(
                law_pair.original,
                pair.component_id,
            )
            transformed_component = _density_component(
                law_pair.transformed,
                pair.component_id,
            )
            original_logpdf = _gaussian_logpdf(
                x,
                original_component.mean.value(),
                original_component.covariance.value(),
            )
            transformed_logpdf = _gaussian_logpdf(
                x_prime,
                transformed_component.mean.value(),
                transformed_component.covariance.value(),
            )
            if pair.component_id.startswith("p."):
                role: Literal["p", "q"] = "p"
            elif pair.component_id.startswith("q."):
                role = "q"
            else:
                raise ValueError("density component has no p/q role")
            observations.append(
                _make_density_observation(
                    pair=pair,
                    role=role,
                    original_value=original_logpdf,
                    transformed_value=transformed_logpdf,
                    expected_shift=shift,
                    budgets=budgets,
                    used_budget_ids=used_budget_ids,
                )
            )
        evaluations.append(
            H7DensityProbeEvaluation.create(
                probe=pair,
                observations=tuple(observations),
            )
        )
    if not evaluations:
        raise ValueError("density evaluator produced no comparisons")
    observed_roles = {
        observation.role
        for evaluation in evaluations
        for observation in evaluation.observations
    }
    if observed_roles != {"p", "q", "log_ratio"}:
        raise ValueError("density probes do not cover p, q, and log ratio")
    return tuple(evaluations)


def _make_density_observation(
    *,
    pair: H7DensityProbePair,
    role: Literal["p", "q", "log_ratio"],
    original_value: float,
    transformed_value: float,
    expected_shift: float,
    budgets: Mapping[str, H7BudgetRecord],
    used_budget_ids: set[str],
) -> H7DensityObservationRecord:
    invariant_id = f"density_probe.{pair.probe_sha256}.{role}"
    residual = _make_residual(
        invariant_id=invariant_id,
        category="density",
        value=abs(
            (transformed_value - original_value) + expected_shift
        ),
        expected_budget_category="density",
        budgets=budgets,
        used_budget_ids=used_budget_ids,
    )
    operand_roles = {item.role for item in residual.budget.operands}
    if not {"original", "transformed"}.issubset(operand_roles):
        raise ValueError(
            "per-probe density budget must retain original/transformed operands"
        )
    return H7DensityObservationRecord.create(
        probe_sha256=pair.probe_sha256,
        role=role,
        original_value=float(original_value),
        transformed_value=float(transformed_value),
        expected_log_jacobian_shift=float(expected_shift),
        residual=residual,
    )


def _density_component(
    law: H7CompleteLawSnapshot,
    component_id: str,
) -> H7GaussianComponentSnapshot:
    candidates = (
        law.generative.initial_joint,
        *(item.receiver_law for item in law.generative.transitions),
        law.recognition.initial_joint,
        *(
            item.receiver_law
            for item in (
                *law.recognition.model_conditionals,
                *law.recognition.state_conditionals,
            )
        ),
    )
    matches = tuple(
        item
        for item in candidates
        if item.component_id == component_id
        or item.component_id.removesuffix(".receiver") == component_id
    )
    if len(matches) != 1:
        raise ValueError("density component lookup is incomplete or ambiguous")
    return matches[0]


def _global_log_density(
    law: H7CompleteLawSnapshot,
    moments: _JointMoments,
    path: _SourcePath,
    value: Tensor,
    *,
    role: Literal["p", "q"],
) -> float:
    probability = (
        path.p_probability if role == "p" else path.q_probability
    )
    if probability <= 0.0:
        raise ValueError("global density requires positive source probability")
    result = math.log(probability) + _gaussian_logpdf(
        value,
        moments.mean,
        moments.covariance,
    )
    if role == "p":
        result += _point_log_likelihood(law, value)
    return result


def _point_log_likelihood(
    law: H7CompleteLawSnapshot,
    value: Tensor,
) -> float:
    dimension = law.generative.decoders[0].state_weight.shape[1]
    contributions: list[float] = []
    for receiver_t, decoder in enumerate(law.generative.decoders, start=1):
        z = value[list(_block_indices("z", receiver_t, dimension))]
        m = value[list(_block_indices("m", receiver_t, dimension))]
        logits = (
            decoder.state_weight.value() @ z
            + decoder.model_weight.value() @ m
            + decoder.bias.value()
        )
        selected = law.recognition.context.observation_labels[receiver_t - 1]
        if law.fixture_id == "h1-v1":
            selected -= 1
        contributions.append(float(torch.log_softmax(logits, dim=0)[selected]))
    return math.fsum(contributions)


def _gaussian_logpdf(
    value: Tensor,
    mean: Tensor,
    covariance: Tensor,
) -> float:
    if value.shape != mean.shape:
        raise ValueError("density point and Gaussian mean shapes disagree")
    cholesky = torch.linalg.cholesky(covariance)
    displacement = (value - mean).unsqueeze(1)
    quadratic = torch.sum(
        displacement * torch.cholesky_solve(displacement, cholesky)
    )
    logdet = 2.0 * torch.log(torch.diagonal(cholesky)).sum()
    result = -0.5 * (
        value.numel() * math.log(2.0 * math.pi) + logdet + quadratic
    )
    checked = float(result.item())
    if not math.isfinite(checked):
        raise ValueError("Gaussian log density is nonfinite")
    return checked


def _path_by_id(
    paths: tuple[_SourcePath, ...],
    path_id: str,
) -> _SourcePath:
    matches = tuple(path for path in paths if path.path_id == path_id)
    if len(matches) != 1:
        raise ValueError("density source path is missing or ambiguous")
    return matches[0]


def _validate_jacobian_metadata(
    law_pair: H7LawPairSnapshot,
    action: H7TensorActionSnapshot,
) -> None:
    expected_global = _global_log_jacobian(action)
    action_values = tuple(item.value() for item in action.elements)
    expected_initial = 2.0 * float(
        torch.linalg.slogdet(action_values[0])[1].item()
    )
    original = law_pair.original
    transformed = law_pair.transformed
    for scope, original_metadata, transformed_metadata in (
        (
            "generative",
            original.generative.jacobian,
            transformed.generative.jacobian,
        ),
        (
            "recognition",
            original.recognition.jacobian,
            transformed.recognition.jacobian,
        ),
    ):
        observed = float(
            transformed_metadata.global_logabsdet.value().item()
            - original_metadata.global_logabsdet.value().item()
        )
        if not math.isclose(
            observed,
            expected_global,
            rel_tol=0.0,
            abs_tol=64.0
            * _FLOAT64_EPSILON
            * max(1.0, abs(expected_global)),
        ):
            raise ValueError(f"{scope} global Jacobian metadata is inconsistent")
        initial_delta = float(
            transformed_metadata.initial_logabsdet.value().item()
            - original_metadata.initial_logabsdet.value().item()
        )
        if not math.isclose(
            initial_delta,
            expected_initial,
            rel_tol=0.0,
            abs_tol=64.0
            * _FLOAT64_EPSILON
            * max(1.0, abs(expected_initial)),
        ):
            raise ValueError(f"{scope} initial Jacobian metadata is inconsistent")
        for component_id in original_metadata.receiver_logabsdet:
            receiver_t = _receiver_from_component_id(component_id)
            expected_receiver = float(
                torch.linalg.slogdet(action_values[receiver_t])[1].item()
            )
            receiver_delta = float(
                transformed_metadata.receiver_logabsdet[
                    component_id
                ].value().item()
                - original_metadata.receiver_logabsdet[
                    component_id
                ].value().item()
            )
            if not math.isclose(
                receiver_delta,
                expected_receiver,
                rel_tol=0.0,
                abs_tol=64.0
                * _FLOAT64_EPSILON
                * max(1.0, abs(expected_receiver)),
            ):
                raise ValueError(
                    f"{scope} receiver Jacobian metadata is inconsistent"
                )
        if scope == "recognition":
            original_entropy = original_metadata.entropy_shift
            transformed_entropy = transformed_metadata.entropy_shift
            if original_entropy is None or transformed_entropy is None:
                raise ValueError("recognition Jacobian metadata lacks entropy")
            entropy_delta = float(
                transformed_entropy.value().item()
                - original_entropy.value().item()
            )
            if not math.isclose(
                entropy_delta,
                expected_global,
                rel_tol=0.0,
                abs_tol=64.0
                * _FLOAT64_EPSILON
                * max(1.0, abs(expected_global)),
            ):
                raise ValueError("recognition entropy Jacobian sign is wrong")


def _receiver_from_component_id(component_id: str) -> int:
    if ".receiver_" in component_id:
        suffix = component_id.split(".receiver_", maxsplit=1)[1]
        candidate = suffix.split(".", maxsplit=1)[0]
    elif "<-" in component_id:
        left = component_id.split("<-", maxsplit=1)[0]
        candidate = left.rsplit(".", maxsplit=1)[-1]
    else:
        fields = component_id.split(".")
        candidates = tuple(
            fields[index + 1]
            for index, field in enumerate(fields[:-1])
            if field in ("model", "state")
        )
        if len(candidates) != 1:
            raise ValueError("Jacobian receiver component ID is malformed")
        candidate = candidates[0]
    try:
        receiver_t = int(candidate)
    except ValueError as error:
        raise ValueError("Jacobian receiver label is malformed") from error
    if receiver_t not in (1, 2):
        raise ValueError("Jacobian receiver label is outside H7")
    return receiver_t


def _global_log_jacobian(action: H7TensorActionSnapshot) -> float:
    return 2.0 * math.fsum(
        float(torch.linalg.slogdet(item.value())[1].item())
        for item in action.elements
    )


def _evaluate_scorer_residuals(
    law_pair: H7LawPairSnapshot,
    action: H7TensorActionSnapshot,
    budgets: Mapping[str, H7BudgetRecord],
    used_budget_ids: set[str],
) -> tuple[H7ResidualRecord, ...]:
    if law_pair.original.fixture_id == "h1-v1":
        if (
            law_pair.original.generative.source_context is not None
            or law_pair.transformed.generative.source_context is not None
            or law_pair.original.recognition.source_rows
            or law_pair.transformed.recognition.source_rows
        ):
            raise ValueError("scalar H7 replay cannot carry matrix scorer rows")
        return ()
    if type(action) is not H7GLPlus2Action:
        raise ValueError("matrix scorer laws require a GL+(2) action")
    original_context = law_pair.original.generative.source_context
    transformed_context = law_pair.transformed.generative.source_context
    if original_context is None or transformed_context is None:
        raise ValueError("matrix law pair lacks exact source-scorer contexts")
    if (
        original_context.source_scorer_profile
        != "h7-linear-history-source-v1"
        or transformed_context.source_scorer_profile
        != "h7-linear-history-source-v1"
    ):
        raise ValueError("matrix source-scorer profile changed")
    original_rows = {
        (row.bank, row.receiver_t): row
        for row in original_context.scorer_rows
    }
    transformed_rows = {
        (row.bank, row.receiver_t): row
        for row in transformed_context.scorer_rows
    }
    expected_keys = (
        ("model", 1),
        ("model", 2),
        ("state", 1),
        ("state", 2),
    )
    if tuple(original_rows) != expected_keys or tuple(transformed_rows) != expected_keys:
        raise ValueError("matrix source-scorer row inventory changed")

    covector_values: dict[str, float] = {}
    raw_score_values: dict[str, float] = {}
    for bank, receiver_t in expected_keys:
        original_row = original_rows[(bank, receiver_t)]
        transformed_row = transformed_rows[(bank, receiver_t)]
        _validate_scorer_row_provenance(
            original_row,
            transformed_row,
            action,
        )
        source_action = action.elements[original_row.source_j].value()
        for channel, original_covector, transformed_covector in (
            ("z", original_row.z_covector, transformed_row.z_covector),
            ("m", original_row.m_covector, transformed_row.m_covector),
        ):
            expected = torch.linalg.solve(
                source_action.T,
                original_covector.value.value(),
            )
            observed = transformed_covector.value.value()
            covector_values[
                f"source_scorer.{bank}.receiver_{receiver_t}.{channel}_covector"
            ] = float(torch.max(torch.abs(observed - expected)).item())
        raw_score_values[
            f"source_scorer.{bank}.receiver_{receiver_t}.raw_score"
        ] = float(
            torch.max(
                torch.abs(
                    transformed_row.raw_scores.value()
                    - original_row.raw_scores.value()
                )
            ).item()
        )
    values = {**covector_values, **raw_score_values}
    require_h7_matrix_scorer_residual_inventory(tuple(values))
    return tuple(
        _make_residual(
            invariant_id=invariant_id,
            category="source",
            value=values[invariant_id],
            expected_budget_category="vector",
            budgets=budgets,
            used_budget_ids=used_budget_ids,
        )
        for invariant_id in H7_MATRIX_SCORER_RESIDUAL_IDS
    )


def _validate_scorer_row_provenance(
    original: H7SourceScorerRowSnapshot,
    transformed: H7SourceScorerRowSnapshot,
    action: H7GLPlus2Action,
) -> None:
    if (
        (
            original.bank,
            original.receiver_t,
            original.source_j,
            original.prefix_tokens,
            original.prefix_bytes,
            original.prefix_term,
            original.mask,
            original.support,
        )
        != (
            transformed.bank,
            transformed.receiver_t,
            transformed.source_j,
            transformed.prefix_tokens,
            transformed.prefix_bytes,
            transformed.prefix_term,
            transformed.mask,
            transformed.support,
        )
        or not torch.equal(
            original.probabilities.value(),
            transformed.probabilities.value(),
        )
    ):
        raise ValueError("source scorer discrete/prefix provenance changed")
    for original_history, transformed_history in zip(
        (*original.z_history, *original.m_history),
        (*transformed.z_history, *transformed.m_history),
    ):
        if (
            original_history.channel != transformed_history.channel
            or original_history.population_label
            != transformed_history.population_label
        ):
            raise ValueError("source scorer history provenance changed")
        expected = (
            action.elements[original_history.population_label].value()
            @ original_history.value.value()
        )
        if not torch.equal(transformed_history.value.value(), expected):
            raise ValueError("source scorer history is not the declared pushforward")


def _make_residual(
    *,
    invariant_id: str,
    category: H7ResidualCategory,
    value: float,
    expected_budget_category: str,
    budgets: Mapping[str, H7BudgetRecord],
    used_budget_ids: set[str],
) -> H7ResidualRecord:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{invariant_id} residual must be finite and nonnegative")
    budget = budgets.get(invariant_id)
    if (
        type(budget) is not H7BudgetRecord
        or budget.invariant_id != invariant_id
        or budget.category != expected_budget_category
    ):
        raise ValueError(
            f"{invariant_id} lacks its exact Task-6 operand-local budget"
        )
    if invariant_id in used_budget_ids:
        raise ValueError("one objective budget was consumed more than once")
    used_budget_ids.add(invariant_id)
    return H7ResidualRecord.create(
        invariant_id=invariant_id,
        category=category,
        value=value,
        budget=budget,
        passed=value <= budget.total_allowance,
    )


def _signed_child_ids(
    law: H7CompleteLawSnapshot,
    term_id: str,
) -> tuple[str, ...]:
    if term_id.startswith("expected_log_emission"):
        receiver_t = int(term_id[-2])
        return (f"+decoder[{receiver_t}]",)
    if term_id.startswith("model_source_kl"):
        receiver_t = int(term_id[-2])
        return (f"-q_model_source[{receiver_t}]", f"+p_model_source[{receiver_t}]")
    if term_id.startswith("state_source_kl"):
        receiver_t = int(term_id[-2])
        return (f"-q_state_source[{receiver_t}]", f"+p_state_source[{receiver_t}]")
    if term_id.startswith("model_transition_kl"):
        receiver_t = int(term_id[-2])
        return (
            f"-q_model_transition[{receiver_t}]",
            f"+p_model_transition[{receiver_t}]",
        )
    if term_id.startswith("state_transition_kl"):
        receiver_t = int(term_id[-2])
        return (
            f"-q_state_transition[{receiver_t}]",
            f"+p_state_transition[{receiver_t}]",
        )
    if term_id == "joint_recognition_entropy":
        return (
            f"+{law.recognition.initial_joint.component_id}",
            *tuple(
                f"+{item.component_id}"
                for item in (
                    *law.recognition.model_conditionals,
                    *law.recognition.state_conditionals,
                )
            ),
            "+recognition_source_entropy",
        )
    raise ValueError("unknown H7 local term ID")


def _require_gaussian_inputs(
    q_mean: Tensor,
    q_covariance: Tensor,
    p_mean: Tensor,
    p_covariance: Tensor,
) -> int:
    if (
        not isinstance(q_mean, Tensor)
        or not isinstance(q_covariance, Tensor)
        or not isinstance(p_mean, Tensor)
        or not isinstance(p_covariance, Tensor)
        or q_mean.dtype is not torch.float64
        or q_covariance.dtype is not torch.float64
        or p_mean.dtype is not torch.float64
        or p_covariance.dtype is not torch.float64
        or q_mean.ndim != 1
        or p_mean.shape != q_mean.shape
        or q_covariance.shape != (q_mean.numel(), q_mean.numel())
        or p_covariance.shape != q_covariance.shape
        or q_mean.numel() <= 0
    ):
        raise ValueError("Gaussian KL inputs must share finite float64 shapes")
    if not all(
        bool(torch.isfinite(item).all().item())
        for item in (q_mean, q_covariance, p_mean, p_covariance)
    ):
        raise ValueError("Gaussian KL inputs must be finite")
    return q_mean.numel()


__all__ = [
    "H7_COMPLETE_LOCAL_TERM_IDS",
    "H7_COMPLETE_LOCAL_INVARIANT_ID",
    "H7_COMPLETE_MONOLITHIC_INVARIANT_ID",
    "H7_ENTROPY_SHIFT_INVARIANT_ID",
    "H7_INDEPENDENT_H1_NORMALIZATION_IDENTITY_SHA256",
    "H7_INDEPENDENT_H1_PRODUCER_IDENTITY_SHA256",
    "H7_MATRIX_EVIDENCE_NOT_APPLICABLE_REASON",
    "H7_MATRIX_SCORER_RESIDUAL_IDS",
    "H7_POINTWISE_LOG_RATIO_INVARIANT_ID",
    "H7_POINTWISE_P_SHIFT_INVARIANT_ID",
    "H7_POINTWISE_Q_SHIFT_INVARIANT_ID",
    "H7_SCALAR_EVIDENCE_INVARIANT_ID",
    "H7_SCALAR_POSTERIOR_KL_INVARIANT_ID",
    "H7IndependentH1EvidenceRecord",
    "evaluate_h7_complete_covariance",
    "evaluate_h7_law_pair_covariance",
    "h7_joint_gaussian_kl",
    "require_h7_complete_term_inventory",
    "require_h7_matrix_scorer_residual_inventory",
]
