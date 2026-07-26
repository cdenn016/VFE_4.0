from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from collections.abc import Mapping
from types import MappingProxyType
from typing import get_type_hints

import pytest
import torch

from vfe4.numerics.block_layout import BlockChainLayout
from vfe4.types import (
    BackendCounterSnapshot,
    BlockFillRecord,
    BlockStorageRecord,
    BlockWorkspaceRecord,
    ElboTermAllowances,
    ElboTerms,
    CurrentH8PrerequisiteRefs,
    GateResult,
    GateStatus,
    H7PredecessorReference,
    H8AllocationRecord,
    H8ChildAttemptRecord,
    H8ChildRequest,
    H8ChildResult,
    H8ControlResult,
    H8GateResult,
    H8H1H5Reference,
    H8H1PrefixPriorReference,
    H8H6PredictionReference,
    H8H6PrefixReference,
    H8H6PrefixSemanticFamilyReference,
    H8H7Reference,
    H8InvariantRecord,
    H8ObjectiveTerm,
    H8ObjectiveTerms,
    H8ResourceRecord,
    H8_H7_PLAN_SHA256,
    H8_INTERPRETATION_SHA256,
    H8_NEGATIVE_CONTROL_IDS,
    H8_PRODUCTION_SEEDS,
    H8_REQUIRED_OPERATIONS,
    InvariantResult,
    NumericalAllowance,
    PopulationFrames,
    SourcePath,
    StructuralData,
)
from vfe4.types.h8 import (
    H8DecodedPassEvidence,
    H8LocalSPDDiagnostics,
    H8ProductionProblemEvidence,
    H8TransitionNorms,
    SparseConditionDiagnostics,
)


def _structure(**overrides: object) -> StructuralData:
    values: dict[str, object] = {
        "horizon": 2,
        "d_z": 1,
        "d_m": 1,
        "vocabulary_size": 3,
        "state_parent_sets": ((0,), (0, 1)),
        "model_parent_sets": ((0,), (0, 1)),
        "state_source_support": ((0,), (0, 1)),
        "model_source_support": ((0,), (0, 1)),
    }
    values.update(overrides)
    return StructuralData(**values)  # type: ignore[arg-type]


def test_structural_data_accepts_the_h1_shape() -> None:
    assert _structure().state_parent_sets == ((0,), (0, 1))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("state_parent_sets", ((0,),)),
        ("model_parent_sets", ((0,), (0, 2))),
        ("model_source_support", ((0,), (0, 2))),
    ],
)
def test_structural_data_rejects_malformed_or_out_of_range_sets(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError, match=field):
        _structure(**{field: value})


def test_structural_data_is_frozen() -> None:
    structure = _structure()

    with pytest.raises(dataclasses.FrozenInstanceError):
        structure.horizon = 3  # type: ignore[misc]


def test_source_path_requires_nonnegative_coordinate_pairs() -> None:
    assert SourcePath((0, 1), (2, 3)).b == (2, 3)
    with pytest.raises(ValueError, match="a"):
        SourcePath((0,), (2, 3))  # type: ignore[arg-type]


def test_population_frames_returns_scalar_ratio_and_owns_tensor() -> None:
    raw = torch.tensor([2.0, 4.0, 8.0], dtype=torch.float64)
    frames = PopulationFrames(raw)
    raw[0] = 20.0

    ratio = frames.omega(2, 1)

    assert ratio.dtype is torch.float64
    assert ratio.item() == pytest.approx(2.0)
    returned = frames.values
    returned[0] = 99.0
    assert frames.values[0].item() == pytest.approx(2.0)


@pytest.mark.parametrize(
    "value",
    [
        torch.ones((1, 3), dtype=torch.float64),
        torch.tensor([1.0, 0.0, 2.0], dtype=torch.float64),
        torch.tensor([1.0, float("nan"), 2.0], dtype=torch.float64),
        torch.ones(3, dtype=torch.float32),
    ],
)
def test_population_frames_rejects_invalid_values(value: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="values"):
        PopulationFrames(value)


@pytest.mark.parametrize("receiver, source", [(-1, 0), (0, 3)])
def test_population_frames_checks_omega_indices(receiver: int, source: int) -> None:
    frames = PopulationFrames(torch.ones(3, dtype=torch.float64))

    with pytest.raises(ValueError, match="index"):
        frames.omega(receiver, source)


def test_population_frames_rejects_an_overflowing_derived_ratio() -> None:
    frames = PopulationFrames(
        torch.tensor([sys.float_info.max, sys.float_info.min, 1.0], dtype=torch.float64)
    )

    with pytest.raises(ValueError, match="omega"):
        frames.omega(0, 1)


def _allowance() -> NumericalAllowance:
    return NumericalAllowance(convergence_estimate=0.0, rounding_allowance=1e-15)


def test_numerical_allowance_is_nonnegative_and_sums_its_components() -> None:
    allowance = NumericalAllowance(0.125, 0.25)

    assert allowance.total == pytest.approx(0.375)
    with pytest.raises(ValueError, match="convergence_estimate"):
        NumericalAllowance(-1.0, 0.0)


def test_numerical_allowance_rejects_an_overflowing_total() -> None:
    with pytest.raises(ValueError, match="total"):
        NumericalAllowance(sys.float_info.max, sys.float_info.max)


def _term_allowances() -> ElboTermAllowances:
    allowance = _allowance()
    return ElboTermAllowances(
        expected_log_emission=(allowance, allowance),
        initial_model_kl=allowance,
        initial_state_kl=allowance,
        model_source_kl=(allowance, allowance),
        model_transition_kl=(allowance, allowance),
        state_source_kl=(allowance, allowance),
        state_transition_kl=(allowance, allowance),
        joint_recognition_entropy=allowance,
        complete_elbo=allowance,
    )


def test_elbo_terms_accepts_consistent_partition_without_double_counting_entropy() -> None:
    terms = ElboTerms(
        expected_log_emission=(-2.0, -3.0),
        initial_model_kl=1.0,
        initial_state_kl=2.0,
        model_source_kl=(0.5, 0.25),
        model_transition_kl=(0.75, 0.5),
        state_source_kl=(0.25, 0.125),
        state_transition_kl=(0.5, 0.25),
        joint_recognition_entropy=1.25,
        allowances=_term_allowances(),
        complete_elbo=-11.125,
    )

    assert terms.complete_elbo == pytest.approx(-11.125)


def test_elbo_terms_rejects_an_inconsistent_complete_total() -> None:
    with pytest.raises(ValueError, match="complete_elbo"):
        ElboTerms(
            expected_log_emission=(-2.0, -3.0),
            initial_model_kl=1.0,
            initial_state_kl=2.0,
            model_source_kl=(0.5, 0.25),
            model_transition_kl=(0.75, 0.5),
            state_source_kl=(0.25, 0.125),
            state_transition_kl=(0.5, 0.25),
            joint_recognition_entropy=1.25,
            allowances=_term_allowances(),
            complete_elbo=-11.0,
        )


def test_elbo_terms_rejects_an_overflowing_derived_objective() -> None:
    with pytest.raises(ValueError, match="expected objective"):
        ElboTerms(
            expected_log_emission=(sys.float_info.max, sys.float_info.max),
            initial_model_kl=0.0,
            initial_state_kl=0.0,
            model_source_kl=(0.0, 0.0),
            model_transition_kl=(0.0, 0.0),
            state_source_kl=(0.0, 0.0),
            state_transition_kl=(0.0, 0.0),
            joint_recognition_entropy=0.0,
            allowances=_term_allowances(),
            complete_elbo=0.0,
        )


def test_gate_result_uses_an_immutable_copy_of_measurements() -> None:
    measurements = {"elbo": 2.0}
    result = GateResult(
        gate="H1",
        status=GateStatus.PASS,
        fixture_id="h1-v1",
        residual=0.0,
        calibrated_allowance=1e-12,
        measurements=measurements,
        invariants=(InvariantResult("normalization", True, 1.0, 1.0, "ok"),),
        obligations=(),
    )
    measurements["elbo"] = 3.0

    assert isinstance(result.measurements, MappingProxyType)
    assert result.measurements["elbo"] == pytest.approx(2.0)
    with pytest.raises(TypeError):
        result.measurements["new"] = 1.0  # type: ignore[index]


def test_gate_result_requires_obligation_when_inconclusive() -> None:
    with pytest.raises(ValueError, match="obligation"):
        GateResult(
            gate="H1",
            status=GateStatus.INCONCLUSIVE,
            fixture_id="h1-v1",
            residual=None,
            calibrated_allowance=None,
            measurements={"elbo": None},
            invariants=(),
            obligations=(),
        )


@pytest.mark.parametrize(
    ("residual", "calibrated_allowance"),
    [
        (float("nan"), None),
        (float("inf"), None),
        (None, float("nan")),
        (None, float("inf")),
    ],
)
def test_inconclusive_gate_rejects_nonfinite_optional_scalars(
    residual: float | None, calibrated_allowance: float | None
) -> None:
    with pytest.raises(ValueError):
        GateResult(
            gate="H1",
            status=GateStatus.INCONCLUSIVE,
            fixture_id="h1-v1",
            residual=residual,
            calibrated_allowance=calibrated_allowance,
            measurements={"elbo": None},
            invariants=(),
            obligations=("obtain the unavailable measurement",),
        )


def _current_h8_refs() -> tuple[
    CurrentH8PrerequisiteRefs,
    dict[str, H7PredecessorReference],
]:
    digest = "a" * 64
    head = "1" * 40
    h7_compatibility_refs = {
        key: H7PredecessorReference.create(
            artifact_path=f"{key}-artifact",
            git_head=head,
            dirty_digest=digest,
            junit_sha256=digest,
            junit_path=f"{key}-junit.xml",
            manifest_sha256=digest,
            payload_hashes={f"{key}.json": digest},
            ledger_path=f"{key}-ledger",
            ledger_sha256=digest,
        )
        for key in ("h1_h5", "h1_prefix_prior", "h6_prefix")
    }

    def common(key: str) -> dict[str, object]:
        transitive = h7_compatibility_refs[key]
        return {
            "artifact_path": transitive.artifact_path,
            "manifest_sha256": transitive.manifest_sha256,
            "result_path": f"{key}-result",
            "result_sha256": digest,
            "content_hashes": {f"{key}-content": digest},
            "payload_hashes": dict(transitive.payload_hashes),
            "ledger_path": transitive.ledger_path,
            "ledger_sha256": transitive.ledger_sha256,
            "producer_head": transitive.git_head,
            "producer_dirty_digest": transitive.dirty_digest,
            "candidate_junit_sha256": transitive.junit_sha256,
            "status": "pass",
        }

    h7_common: dict[str, object] = {
        "artifact_path": "h7-artifact",
        "manifest_sha256": digest,
        "result_path": "h7-result",
        "result_sha256": digest,
        "content_hashes": {"h7-content": digest},
        "payload_hashes": {"h7.json": digest},
        "ledger_path": "h7-ledger",
        "ledger_sha256": digest,
        "producer_head": head,
        "producer_dirty_digest": digest,
        "candidate_junit_sha256": digest,
        "status": "pass",
    }
    prediction_common = {
        **h7_common,
        "artifact_path": "prediction-artifact",
        "result_path": "prediction-result",
        "content_hashes": {"prediction-content": digest},
        "payload_hashes": {"prediction.json": digest},
        "ledger_path": "prediction-ledger",
        "candidate_junit_sha256": digest,
    }
    refs = CurrentH8PrerequisiteRefs(
        candidate_head=head,
        candidate_dirty_digest=digest,
        candidate_junit_sha256=digest,
        h7_compatibility_refs=h7_compatibility_refs,
        h1_h5=H8H1H5Reference(
            kind="h1_h5", **common("h1_h5")  # type: ignore[arg-type]
        ),
        h1_prefix_prior=H8H1PrefixPriorReference(
            kind="h1_prefix_prior",
            **common("h1_prefix_prior"),  # type: ignore[arg-type]
        ),
        h6_prefix=H8H6PrefixReference(
            kind="h6_prefix",
            config_schema="h6-prefix-config-v3",
            validation_schema="h6-prefix-validation-set-v2",
            certificate_set_schema="h6-prefix-certificate-set-v2",
            config_sha256=digest,
            workload_plan_sha256=digest,
            validation_payload_sha256=digest,
            prefix_certificate_set_sha256=digest,
            semantic_families=(
                H8H6PrefixSemanticFamilyReference(
                    semantic_family_index=0,
                    semantic_family_sha256="b" * 64,
                    validation_payload_sha256="c" * 64,
                    certificate_sha256="d" * 64,
                ),
                H8H6PrefixSemanticFamilyReference(
                    semantic_family_index=1,
                    semantic_family_sha256="e" * 64,
                    validation_payload_sha256="f" * 64,
                    certificate_sha256="0" * 64,
                ),
            ),
            **common("h6_prefix"),  # type: ignore[arg-type]
        ),
        h7=H8H7Reference(
            kind="h7",
            result_pointer_path="h7-result-pointer",
            result_pointer_sha256=digest,
            fixture_set_sha256=digest,
            **h7_common,  # type: ignore[arg-type]
        ),
        h6_prediction=H8H6PredictionReference(
            kind="h6_prediction",
            prediction_schema="h6-prediction-amended-v2",
            config_schema="h6-prediction-config-v2",
            readiness_schema="h6-prediction-readiness-v2",
            metrics_schema="h6-prediction-metrics-v2",
            result_schema="h6-prediction-result-v2",
            experiment_sha256=digest,
            config_sha256=digest,
            readiness_artifact_path="prediction-readiness",
            readiness_manifest_sha256=digest,
            readiness_sha256=digest,
            correctness_artifact_paths={
                gate: f"prediction-{gate.lower()}-correctness"
                for gate in ("H1", "H2", "H3", "H5")
            },
            h1_prefix_prior_artifact_path="prediction-h1-prefix-prior",
            smc_accuracy_artifact_path="prediction-smc-accuracy",
            smc_accuracy_manifest_sha256=digest,
            h6_prefix_artifact_path="prediction-h6-prefix",
            h6_prefix_manifest_sha256=digest,
            blinded_data_artifact_path="prediction-blinded-data",
            blinded_data_manifest_sha256=digest,
            matching_artifact_path="prediction-matching",
            matching_manifest_sha256=digest,
            matching_set_sha256=digest,
            h1_prefix_prior_generative_factor_schema_sha256=digest,
            smc_bias_semantics_sha256=digest,
            objective_gate_spec_sha256=digest,
            metrics_sha256=digest,
            **prediction_common,  # type: ignore[arg-type]
        ),
        registry_sha256=digest,
    )
    return refs, h7_compatibility_refs


def test_current_h8_h7_references_are_exact_lossless_and_immutable() -> None:
    refs, source = _current_h8_refs()
    original = tuple(source.items())

    assert (
        get_type_hints(CurrentH8PrerequisiteRefs)["h7_compatibility_refs"]
        == Mapping[str, H7PredecessorReference]
    )
    assert tuple(refs.h7_compatibility_refs.items()) == original
    assert all(
        retained is supplied
        and retained.reference_sha256 == supplied.reference_sha256
        and retained.payload_hashes == supplied.payload_hashes
        for (_, retained), (_, supplied) in zip(
            refs.h7_compatibility_refs.items(),
            original,
            strict=True,
        )
    )

    source.clear()
    assert tuple(refs.h7_compatibility_refs.items()) == original
    with pytest.raises(TypeError):
        refs.h7_compatibility_refs["h1_h5"] = original[0][1]  # type: ignore[index]
    with pytest.raises(TypeError):
        original[0][1].payload_hashes["replacement"] = "b" * 64  # type: ignore[index]


def test_current_h8_rejects_a_lossy_untyped_h7_reference() -> None:
    refs, source = _current_h8_refs()
    source["h6_prefix"] = {"reference_sha256": "a" * 64}  # type: ignore[assignment]

    with pytest.raises(ValueError, match="exact types"):
        dataclasses.replace(refs, h7_compatibility_refs=source)


def test_current_h8_rejects_direct_reference_drift_from_h7_transitive_bytes() -> None:
    refs, _source = _current_h8_refs()
    changed = dataclasses.replace(
        refs.h1_prefix_prior,
        result_path="an-independent-result-path-is-allowed",
    )
    assert dataclasses.replace(refs, h1_prefix_prior=changed)

    changed_payload = dataclasses.replace(
        refs.h1_prefix_prior,
        payload_hashes={"different.json": "b" * 64},
    )
    with pytest.raises(ValueError, match="H7 transitive"):
        dataclasses.replace(refs, h1_prefix_prior=changed_payload)


def test_current_h8_preserves_amended_prediction_from_its_frozen_candidate() -> None:
    refs, _source = _current_h8_refs()
    changed_prediction = dataclasses.replace(
        refs.h6_prediction,
        producer_head="2" * 40,
        producer_dirty_digest="b" * 64,
        candidate_junit_sha256="c" * 64,
    )

    preserved = dataclasses.replace(refs, h6_prediction=changed_prediction)

    assert preserved.h6_prediction == changed_prediction
    assert preserved.prerequisite_obligations == ()
    with pytest.raises(ValueError, match="candidate_junit_sha256"):
        dataclasses.replace(
            changed_prediction,
            candidate_junit_sha256=None,
        )


def _h8_request_sha256(request: H8ChildRequest) -> str:
    payload = {
        field.name: getattr(request, field.name)
        for field in dataclasses.fields(request)
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _h8_request(
    *,
    mode: str = "production",
    seed: int = H8_PRODUCTION_SEEDS[0],
    repetition: int | None = 0,
    control_id: str | None = None,
) -> H8ChildRequest:
    return H8ChildRequest(
        mode=mode,  # type: ignore[arg-type]
        seed=seed,
        repetition=repetition,
        config_sha256="a" * 64,
        protocol_sha256="b" * 64,
        control_id=control_id,
    )


def _h8_child_result(
    *,
    mode: str = "production",
    seed: int = H8_PRODUCTION_SEEDS[0],
    repetition: int | None = 0,
    resource_parent_elapsed_ns: int = 0,
) -> H8ChildResult:
    layout = BlockChainLayout(horizon=1, d_z=1, d_m=1)

    def term(role: str, receiver_t: int | None) -> H8ObjectiveTerm:
        factor_id = (
            "initial_joint"
            if receiver_t is None
            else f"{role}:{receiver_t:04d}"
        )
        return H8ObjectiveTerm(
            factor_id=factor_id,
            role=role,  # type: ignore[arg-type]
            receiver_t=receiver_t,
            value=0.0,
            absolute_sum_bound=0.0,
        )

    objective = H8ObjectiveTerms(
        horizon=1,
        initial_joint=term("initial_joint", None),
        model_transitions=(term("model_transition", 1),),
        state_transitions=(term("state_transition", 1),),
        emissions_order21=(term("emission_order21", 1),),
        emissions_order17=(term("emission_order17", 1),),
        recognition_entropy=0.0,
        log_normalizer=0.0,
        model_source_kl=0.0,
        state_source_kl=0.0,
        source_entropy=0.0,
        quadrature_absolute_difference=0.0,
        complete_order21=0.0,
        absolute_term_sum=0.0,
    )
    storage = BlockStorageRecord(
        layout=layout,
        precision_scalar_count=layout.band_storage_scalar_count,
        factor_scalar_count=layout.band_storage_scalar_count,
        selected_inverse_scalar_count=layout.band_storage_scalar_count,
        information_scalar_count=layout.information_scalar_count,
        upper_block_scalar_count=0,
    )
    fill = BlockFillRecord(
        layout=layout,
        stored_block_ids=layout.stored_block_ids,
        observed_offband_blocks=0,
        duplicated_upper_blocks=0,
    )
    workspace = BlockWorkspaceRecord(
        maximum_shape=(layout.block_size, layout.block_size),
        maximum_scalar_count=layout.block_size**2,
        maximum_rhs_width=layout.block_size,
    )
    counters = BackendCounterSnapshot(
        layout=layout,
        factorization_calls=1,
        forward_substitution_calls=2,
        backward_substitution_calls=3,
        solve_calls=1,
        logdet_calls=1,
        selected_inverse_calls=2,
        sample_calls=1,
        quadratic_calls=1,
        trace_calls=1,
        sparse_matvec_calls=1,
        maximum_rhs_width=layout.block_size,
        maximum_sample_rhs_width=1,
        selected_block_ids=layout.stored_block_ids,
        selected_block_count=len(layout.stored_block_ids),
        attempted_forbidden_selected_blocks=0,
    )
    allocation = H8AllocationRecord(
        dispatch_trace_sha256="c" * 64,
        dispatch_event_count=1,
        dispatch_forbidden_attempt_count=0,
        dispatch_live_peak_bytes=0,
        torch_population_peak_bytes=0,
        profiler_trace_sha256=None,
        profiler_events=(),
        profiler_lossy_rows=(),
        preexisting_storage_count=None,
        preexisting_bytes=None,
        baseline_live_bytes=None,
        profiler_reconstructed_live_peak_bytes=None,
        profiler_all_joined_and_liveness_reconciled=None,
        numpy_guard_event_count=1,
        backend_forbidden_attempt_count=0,
        observed_channels=("dispatch",),
    )
    resources = H8ResourceRecord(
        adapter="test",
        adapter_sha256="d" * 64,
        pre_current_rss_bytes=0,
        pre_lifetime_peak_bytes=0,
        pre_private_bytes=0,
        post_current_rss_bytes=0,
        post_lifetime_peak_bytes=0,
        post_private_bytes=0,
        conservative_incremental_hwm_bytes=0,
        peak_to_peak_diagnostic_bytes=0,
        parent_elapsed_ns=resource_parent_elapsed_ns,
        child_elapsed_ns=1,
    )
    invariant = H8InvariantRecord(
        invariant_id="test_pass",
        status=GateStatus.PASS,
        value=1,
        limit=1,
        detail="test_pass=True",
        obligations=(),
    )
    return H8ChildResult(
        mode=mode,  # type: ignore[arg-type]
        seed=seed,
        repetition=repetition,
        input_sha256="e" * 64,
        objective=objective,
        storage=storage,
        fill=fill,
        workspace=workspace,
        counters=counters,
        allocation=allocation,
        resources=resources,
        invariants=(invariant,),
    )


def _h8_control_result(
    control_id: str = H8_NEGATIVE_CONTROL_IDS[0],
) -> H8ControlResult:
    return H8ControlResult(
        control_id=control_id,
        requested_operation=f"exercise:{control_id}",
        logical_shapes=((1, 1),),
        assigned_channels=("dispatch",),
        observed_channels=("dispatch",),
        execution_witnessed=True,
        event_sha256="f" * 64,
        assignment_complete=True,
        detected=True,
        status=GateStatus.PASS,
        obligations=(),
    )


def _h8_private_pass_evidence() -> H8DecodedPassEvidence:
    local = H8LocalSPDDiagnostics(
        schema_version="h8-local-spd-diagnostics-v1",
        horizon=1,
        generative_initial_min_pivot=1.0,
        model_transition_min_pivots=(1.0,),
        state_transition_min_pivots=(1.0,),
        recognition_initial_min_pivot=1.0,
        recognition_transition_min_pivots=(1.0,),
        global_min_pivot=1.0,
    )
    norms = H8TransitionNorms(
        schema_version="h8-transition-norms-v1",
        horizon=1,
        norm="operator_2",
        model_transition_norms=(0.1,),
        state_transition_norms=(0.1,),
        state_model_coupling_norms=(0.1,),
        recognition_transition_norms=(0.1,),
        max_model_transition_norm=0.1,
        max_state_transition_norm=0.1,
        max_state_model_coupling_norm=0.1,
        max_recognition_transition_norm=0.1,
    )
    return H8DecodedPassEvidence(
        sample_noise_sha256="0" * 64,
        problem_evidence=H8ProductionProblemEvidence(
            generative_sha256="1" * 64,
            recognition_sha256="2" * 64,
            local_spd_diagnostics=local,
            transition_norms=norms,
            observation_sha256="3" * 64,
        ),
        condition_diagnostics=SparseConditionDiagnostics(
            estimator="HagerHigham1NormEstimate-v1",
            kappa_1_estimate=1.0,
            iterations=1,
            convergence_reason="test",
            index_sha256="4" * 64,
            sign_sha256="5" * 64,
            per_block_min_pivots=(1.0, 1.0),
            global_min_pivot=1.0,
            per_block_pivot_margins=(1.0 - 1e-8, 1.0 - 1e-8),
            global_pivot_margin=1.0 - 1e-8,
        ),
        allocation={"fixture": True},
        child_identities={
            name: {"kind": name}
            for name in ("hardware", "affinity", "thread", "blas")
        },
    )


def _h8_attempt(
    request: H8ChildRequest,
    *,
    status: GateStatus,
    reasons: tuple[str, ...],
    result: H8ChildResult | H8ControlResult | None,
    timed_out: bool = False,
    exit_code: int | None = 0,
    operation_reachability: Mapping[str, bool] | None = None,
    residuals: Mapping[str, float] | None = None,
    resource_decisions: Mapping[str, object] | None = None,
    nonpass_envelope: Mapping[str, object] | None = None,
) -> H8ChildAttemptRecord:
    return H8ChildAttemptRecord(
        request=request,
        status=status,
        reasons=reasons,
        result=result,
        pass_evidence=(
            _h8_private_pass_evidence()
            if (
                status is GateStatus.PASS
                and request.mode in ("production", "profiler")
            )
            else None
        ),
        timed_out=timed_out,
        exit_code=exit_code,
        parent_elapsed_ns=1,
        request_sha256=_h8_request_sha256(request),
        identities_sha256="1" * 64,
        stdout_sha256="2" * 64,
        stderr_sha256="3" * 64,
        operation_reachability=operation_reachability,
        residuals=residuals,
        resource_decisions=resource_decisions,
        nonpass_envelope=nonpass_envelope,
    )


def _h8_pass_child_attempt(
    *,
    seed: int = H8_PRODUCTION_SEEDS[0],
    repetition: int = 0,
) -> tuple[H8ChildAttemptRecord, H8ChildResult]:
    request = _h8_request(seed=seed, repetition=repetition)
    result = _h8_child_result(seed=seed, repetition=repetition)
    attempt = _h8_attempt(
        request,
        status=GateStatus.PASS,
        reasons=(),
        result=result,
        operation_reachability={
            operation: True for operation in H8_REQUIRED_OPERATIONS
        },
        residuals={
            "factor_reconstruction": 0.0,
            "solve": 0.0,
            "backward_substitution": 0.0,
            "selected_diagonal_symmetry": 0.0,
        },
        resource_decisions={"time_pass": True},
    )
    return attempt, result


def test_h8_child_attempt_owns_partial_parent_evidence_without_erasing_it() -> None:
    request = _h8_request()
    reachability = {"factorization": False}
    residuals = {"factor_reconstruction": 1.0}
    decisions: dict[str, object] = {
        "time_pass": False,
        "witnesses": {"channels": ["dispatch"]},
    }
    envelope: dict[str, object] = {
        "status": "FAIL",
        "obligations": ["witnessed_forbidden_operation"],
    }

    attempt = _h8_attempt(
        request,
        status=GateStatus.FAIL,
        reasons=("witnessed_forbidden_operation",),
        result=None,
        exit_code=1,
        operation_reachability=reachability,
        residuals=residuals,
        resource_decisions=decisions,
        nonpass_envelope=envelope,
    )
    reachability["factorization"] = True
    residuals["factor_reconstruction"] = 0.0
    nested = decisions["witnesses"]
    assert isinstance(nested, dict)
    nested["channels"].append("backend")  # type: ignore[union-attr]
    envelope["obligations"] = []

    assert isinstance(attempt.operation_reachability, MappingProxyType)
    assert attempt.operation_reachability == {"factorization": False}
    assert isinstance(attempt.residuals, MappingProxyType)
    assert attempt.residuals == {"factor_reconstruction": 1.0}
    assert isinstance(attempt.resource_decisions, MappingProxyType)
    retained_witnesses = attempt.resource_decisions["witnesses"]
    assert isinstance(retained_witnesses, MappingProxyType)
    assert retained_witnesses["channels"] == ("dispatch",)
    with pytest.raises(TypeError):
        attempt.resource_decisions["replacement"] = True  # type: ignore[index]
    assert isinstance(attempt.nonpass_envelope, MappingProxyType)
    assert attempt.nonpass_envelope["obligations"] == (
        "witnessed_forbidden_operation",
    )


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"status": GateStatus.PASS, "reasons": ()}, "typed result"),
        ({"status": GateStatus.FAIL, "reasons": ()}, "reason"),
        ({"status": GateStatus.INCONCLUSIVE, "reasons": ()}, "reason"),
        (
            {
                "status": GateStatus.INCONCLUSIVE,
                "reasons": ("wait for evidence",),
                "timed_out": True,
                "exit_code": None,
            },
            "timed out",
        ),
        ({"request_sha256": "0" * 64}, "request_sha256"),
    ],
)
def test_h8_child_attempt_rejects_status_and_request_hash_drift(
    changes: dict[str, object],
    match: str,
) -> None:
    request = _h8_request()
    attempt = _h8_attempt(
        request,
        status=GateStatus.FAIL,
        reasons=("nonzero_child_exit",),
        result=None,
        exit_code=1,
    )

    with pytest.raises(ValueError, match=match):
        dataclasses.replace(attempt, **changes)


@pytest.mark.parametrize(
    "changes",
    (
        {"exit_code": 9},
        {"parent_elapsed_ns": 60_000_000_001},
        {"operation_reachability": {"factorization": False}},
        {"resource_decisions": {"time_pass": False}},
        {
            "reasons": (
                "child_request_or_environment_identity_mismatch",
            ),
        },
        {"nonpass_envelope": {"status": "fail"}},
        {
            "nonpass_envelope": {
                "status": "inconclusive",
                "error": {
                    "kind": "forbidden_operation",
                    "message": "operation executed",
                    "witnessed_violation": True,
                },
            },
        },
    ),
)
def test_h8_child_attempt_cannot_mask_retained_failure_as_inconclusive(
    changes: dict[str, object],
) -> None:
    request = _h8_request()
    inconclusive = _h8_attempt(
        request,
        status=GateStatus.INCONCLUSIVE,
        reasons=("partial_child_evidence",),
        result=None,
    )

    with pytest.raises(ValueError, match="witnessed.*INCONCLUSIVE"):
        dataclasses.replace(inconclusive, **changes)


def test_h8_child_attempt_binds_request_to_result_and_parent_timing() -> None:
    attempt, result = _h8_pass_child_attempt()

    assert attempt.result is result
    assert result.resources.parent_elapsed_ns == 0
    with pytest.raises(ValueError, match="request identity"):
        dataclasses.replace(
            attempt,
            result=_h8_child_result(seed=H8_PRODUCTION_SEEDS[1]),
        )
    with pytest.raises(ValueError, match="parent-owned"):
        dataclasses.replace(
            attempt,
            result=_h8_child_result(resource_parent_elapsed_ns=1),
        )
    with pytest.raises(ValueError, match="operation_reachability"):
        dataclasses.replace(
            attempt,
            operation_reachability={"factorization": True},
        )


def test_h8_child_attempt_pass_control_has_control_only_endpoints() -> None:
    control_id = H8_NEGATIVE_CONTROL_IDS[0]
    request = _h8_request(
        mode="negative_control",
        repetition=None,
        control_id=control_id,
    )
    result = _h8_control_result(control_id)
    attempt = _h8_attempt(
        request,
        status=GateStatus.PASS,
        reasons=(),
        result=result,
    )

    assert attempt.result is result
    with pytest.raises(ValueError, match="control endpoints"):
        dataclasses.replace(attempt, residuals={"unexpected": 0.0})
    with pytest.raises(ValueError, match="request identity"):
        dataclasses.replace(
            attempt,
            result=_h8_control_result(H8_NEGATIVE_CONTROL_IDS[1]),
        )


def test_h8_gate_cross_binds_attempts_to_result_inventories() -> None:
    attempt, child = _h8_pass_child_attempt()
    result = H8GateResult(
        gate="H8",
        status=GateStatus.INCONCLUSIVE,
        config_sha256="a" * 64,
        candidate_junit_sha256=None,
        current_refs_registry_sha256=None,
        h7_manifest_sha256=None,
        h6_prediction_manifest_sha256=None,
        correctness=(),
        child_attempts=(attempt,),
        production_runs=(child,),
        profiler_runs=(),
        controls=(),
        obligations=("complete the remaining H8 attempts",),
    )

    assert result.production_runs == (child,)
    with pytest.raises(ValueError, match="result-bearing attempts"):
        dataclasses.replace(result, production_runs=())

    skipped_attempt, skipped_child = _h8_pass_child_attempt(repetition=1)
    with pytest.raises(ValueError, match="frozen order"):
        dataclasses.replace(
            result,
            child_attempts=(skipped_attempt,),
            production_runs=(skipped_child,),
        )


def test_h8_gate_attempt_failure_dominates_inconclusive_evidence() -> None:
    request = _h8_request()
    failure = _h8_attempt(
        request,
        status=GateStatus.FAIL,
        reasons=("nonzero_child_exit",),
        result=None,
        exit_code=1,
    )
    result = H8GateResult(
        gate="H8",
        status=GateStatus.FAIL,
        config_sha256="a" * 64,
        candidate_junit_sha256="4" * 64,
        current_refs_registry_sha256="5" * 64,
        h7_manifest_sha256="6" * 64,
        h6_prediction_manifest_sha256="7" * 64,
        correctness=(),
        child_attempts=(failure,),
        production_runs=(),
        profiler_runs=(),
        controls=(),
        obligations=(),
    )

    assert result.status is GateStatus.FAIL
    with pytest.raises(ValueError, match="cannot be masked"):
        dataclasses.replace(
            result,
            status=GateStatus.INCONCLUSIVE,
            obligations=("later evidence is unavailable",),
        )


def test_h8_task7_result_and_pins_are_fail_closed() -> None:
    digest = "a" * 64
    result = H8GateResult(
        gate="H8",
        status=GateStatus.INCONCLUSIVE,
        config_sha256=digest,
        candidate_junit_sha256=None,
        current_refs_registry_sha256=None,
        h7_manifest_sha256=None,
        h6_prediction_manifest_sha256=None,
        correctness=(),
        child_attempts=(),
        production_runs=(),
        profiler_runs=(),
        controls=(),
        obligations=("produce current prerequisite evidence",),
    )
    assert dataclasses.is_dataclass(result)
    assert H8GateResult.__dataclass_params__.frozen
    assert result.status is GateStatus.INCONCLUSIVE
    assert (
        H8_H7_PLAN_SHA256,
        H8_INTERPRETATION_SHA256,
    ) == (
        "3549153ac123b26f1d2372c59e80db93a78ed451fd4724781280dd7f413f1242",
        "e3fd048126c8133384e026826cf00bbea08280f4e248bc4cd5689e8f9f26e865",
    )

    with pytest.raises(ValueError, match="witnessed-failure evidence"):
        dataclasses.replace(
            result,
            status=GateStatus.FAIL,
            candidate_junit_sha256=digest,
            current_refs_registry_sha256=digest,
            h7_manifest_sha256=digest,
            h6_prediction_manifest_sha256=digest,
            obligations=(),
        )
