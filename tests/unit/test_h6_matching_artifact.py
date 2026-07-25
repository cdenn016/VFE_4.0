from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from vfe4.artifacts.atomic import canonical_json_bytes as artifact_json_bytes
from vfe4.artifacts.h6_matching import (
    H6_MATCHING_ENDPOINT_LAYOUT,
    H6MatrixMatchingReportRecord,
    H6MatchingPublicationBlocked,
    derive_h6_matrix_reports,
    publish_h6_matching_set,
    read_h6_matching_set,
)
from vfe4.artifacts.provenance import source_candidate_sha256
from vfe4.config import (
    H6PrimaryMatchingResolvedConfig,
    resolve_h6_primary_matching_config,
)
from vfe4.training.h6_readiness import (
    _validate_matching_artifact,
)
from vfe4.training.matching import (
    H6FormulaSelection,
    H6PrimaryJointSelection,
    H6TrainingWorkload,
    select_outcome_blind_allocation,
    select_parent_specific_primary_allocation,
)
from vfe4.training.parameter_counts import (
    outcome_blind_feasibility_assessments,
)
from vfe4.types.h6 import (
    ArmConfig,
    ArmId,
    CapacityAllocation,
    VocabularyIdentity,
)


_SHA_A = "a" * 64
_GIT_HEAD = "1" * 40
_DIRTY_DIGEST = "2" * 64
_REFERENCE_ID = H6_MATCHING_ENDPOINT_LAYOUT[5][0]
_PRIMARY_BOUND_IDS = frozenset(
    (
        H6_MATCHING_ENDPOINT_LAYOUT[0][0],
        H6_MATCHING_ENDPOINT_LAYOUT[7][0],
        H6_MATCHING_ENDPOINT_LAYOUT[9][0],
    )
)
_SEMANTICS = {
    "h6-a0-transformer-v2": (
        ArmId.A0,
        False,
        False,
        False,
        "absent",
        "absent",
        "absent",
        "absent",
        "absent",
        "absent",
        "cross_entropy",
    ),
    "h6-a1-ordinary-latent-v1": (
        ArmId.A1,
        True,
        True,
        False,
        "absent",
        "absent",
        "structured",
        "smoothing",
        "absent",
        "absent",
        "complete_elbo",
    ),
    "h6-a2-generic-map-v1": (
        ArmId.A2,
        True,
        True,
        True,
        "categorical",
        "generic_fixed_frame_non_coboundary",
        "structured",
        "smoothing",
        "fixed",
        "exact",
        "complete_elbo",
    ),
    "h6-a3-immediate-predecessor-v1": (
        ArmId.A3,
        True,
        True,
        True,
        "immediate_predecessor",
        "shared_vertex_coboundary",
        "structured",
        "smoothing",
        "absent",
        "absent",
        "complete_elbo",
    ),
    "h6-a4-state-only-v1": (
        ArmId.A4,
        True,
        True,
        False,
        "categorical",
        "shared_vertex_coboundary",
        "structured",
        "smoothing",
        "fixed",
        "exact",
        "complete_elbo",
    ),
    "h6-a5-structured-fixed-exact-complete-latent-smoothing-v1": (
        ArmId.A5,
        True,
        True,
        True,
        "categorical",
        "shared_vertex_coboundary",
        "structured",
        "smoothing",
        "fixed",
        "exact",
        "complete_elbo",
    ),
    "h6-a5-factorized-fixed-exact-complete-latent-smoothing-v1": (
        ArmId.A5,
        True,
        True,
        True,
        "categorical",
        "shared_vertex_coboundary",
        "factorized",
        "smoothing",
        "fixed",
        "exact",
        "complete_elbo",
    ),
    (
        "h6-a5-structured-parent-specific-prefix-exact-complete-"
        "latent-smoothing-v2"
    ): (
        ArmId.A5,
        True,
        True,
        True,
        "categorical",
        "shared_vertex_coboundary",
        "structured",
        "smoothing",
        "parent_specific_pooled_prefix",
        "exact",
        "complete_elbo",
    ),
    "h6-a5-structured-fixed-projection-complete-latent-smoothing-v1": (
        ArmId.A5,
        True,
        True,
        True,
        "categorical",
        "shared_vertex_coboundary",
        "structured",
        "smoothing",
        "fixed",
        "moment_projection",
        "complete_elbo",
    ),
    (
        "h6-a5-structured-parent-specific-prefix-exact-emission-"
        "latent-smoothing-v2"
    ): (
        ArmId.A5,
        True,
        True,
        True,
        "categorical",
        "shared_vertex_coboundary",
        "structured",
        "smoothing",
        "parent_specific_pooled_prefix",
        "exact",
        "emission_only_ablation_non_elbo",
    ),
    (
        "h6-a5-structured-fixed-exact-complete-"
        "nolatent-norecognition-v1"
    ): (
        ArmId.A5,
        False,
        False,
        False,
        "absent",
        "absent",
        "absent",
        "absent",
        "absent",
        "absent",
        "complete_elbo",
    ),
    "h6-a5-structured-fixed-exact-complete-latent-filtering-v1": (
        ArmId.A5,
        True,
        True,
        True,
        "categorical",
        "shared_vertex_coboundary",
        "structured",
        "filtering",
        "fixed",
        "exact",
        "complete_elbo",
    ),
}


def _config(
    config_id: str,
    allocation: CapacityAllocation,
) -> ArmConfig:
    (
        arm,
        latent_enabled,
        state_channel_enabled,
        model_channel_enabled,
        source_mode,
        map_mode,
        recognition_family,
        recognition_conditioning,
        prior_variant,
        mixture_mode,
        objective_kind,
    ) = _SEMANTICS[config_id]
    return ArmConfig.create(
        arm=arm,
        config_id=config_id,
        vocabulary=VocabularyIdentity("wikitext-2-byte-v1", 258, _SHA_A),
        horizon=32,
        latent_enabled=latent_enabled,
        state_channel_enabled=state_channel_enabled,
        model_channel_enabled=model_channel_enabled,
        source_mode=source_mode,
        map_mode=map_mode,
        recognition_family=recognition_family,
        recognition_conditioning=recognition_conditioning,
        prior_variant=prior_variant,
        mixture_mode=mixture_mode,
        objective_kind=objective_kind,
        capacity_allocation=allocation,
    )


def _evidence_fixture() -> tuple[
    H6PrimaryMatchingResolvedConfig,
    H6TrainingWorkload,
    H6PrimaryJointSelection,
    tuple[H6FormulaSelection, ...],
]:
    assessments = {
        item.config_id: item
        for item in outcome_blind_feasibility_assessments()
    }
    workload = H6TrainingWorkload.from_train_tokens(
        train_token_count=34,
        train_token_sha256="b" * 64,
    )
    allocations = {
        config_id: CapacityAllocation.create(
            emission_width=assessment.emission_width,
            latent_width=assessment.latent_width,
            recognition_width=assessment.recognition_width,
            prior_context_width=assessment.prior_context_width,
        )
        for config_id, assessment in assessments.items()
    }
    configs = {
        config_id: _config(config_id, allocations[config_id])
        for config_id, _ in H6_MATCHING_ENDPOINT_LAYOUT
    }
    reference = configs[_REFERENCE_ID]
    primary_matching_config = resolve_h6_primary_matching_config(
        {
            "schema_version": "h6-primary-matching-config-v1",
            "operation": "H6-Primary-Matching",
            "a0_config": configs[H6_MATCHING_ENDPOINT_LAYOUT[0][0]],
            "a5_template": configs[H6_MATCHING_ENDPOINT_LAYOUT[7][0]],
            "latent_width_candidates": (2, 4, 8),
            "prior_context_width_candidates": (4, 6, 8),
            "emission_width_candidates": (84, 85, 86, 87, 88, 89),
            "recognition_width_candidates": (
                113,
                114,
                115,
                116,
                117,
                118,
            ),
            "parameter_relative_tolerance": 0.01,
            "flop_relative_tolerance": 0.05,
        },
        repo_root=Path.cwd(),
    )
    primary_selection = select_parent_specific_primary_allocation(
        matching_config=primary_matching_config,
        a0_config=primary_matching_config.a0_config,
        a5_template=primary_matching_config.a5_template,
        workload=workload,
    )
    component_selections = tuple(
        select_outcome_blind_allocation(
            endpoint_template=configs[config_id],
            reference_config=reference,
            workload=workload,
        )
        for config_id, _ in H6_MATCHING_ENDPOINT_LAYOUT
        if config_id not in _PRIMARY_BOUND_IDS
    )
    return (
        primary_matching_config,
        workload,
        primary_selection,
        component_selections,
    )


def _publish_fixture(root: Path, run_name: str) -> Path:
    (
        primary_matching_config,
        workload,
        primary_selection,
        component_selections,
    ) = _evidence_fixture()
    _, run_directory = publish_h6_matching_set(
        artifact_root=root,
        run_name=run_name,
        git_head=_GIT_HEAD,
        dirty_digest=_DIRTY_DIGEST,
        source_sha256=source_candidate_sha256(
            git_head_value=_GIT_HEAD,
            dirty_digest_value=_DIRTY_DIGEST,
        ),
        primary_matching_config=primary_matching_config,
        workload=workload,
        primary_selection=primary_selection,
        component_selections=component_selections,
    )
    return run_directory


def _rewrite_manifest(root: Path) -> None:
    paths = (
        "matching/endpoints.json",
        "matching/matrix_reports.json",
        "validation/h6_matching_set.json",
    )
    (root / "manifest.sha256").write_bytes(
        "".join(
            f"{hashlib.sha256((root / path).read_bytes()).hexdigest()}  {path}\n"
            for path in paths
        ).encode("ascii")
    )


def _selected_config(config_id: str) -> ArmConfig:
    assessments = {
        item.config_id: item
        for item in outcome_blind_feasibility_assessments()
    }
    assessment = assessments[config_id]
    return _config(
        config_id,
        CapacityAllocation.create(
            emission_width=assessment.emission_width,
            latent_width=assessment.latent_width,
            recognition_width=assessment.recognition_width,
            prior_context_width=assessment.prior_context_width,
        ),
    )


def test_inference_compute_claim_has_only_validated_public_constructor(
) -> None:
    import vfe4.evaluation as evaluation
    import vfe4.evaluation.compute_ledger as compute_ledger
    from vfe4.artifacts.h6_matching import (
        derive_h6_inference_inclusive_compute_report,
    )

    for name in (
        "InferenceInclusiveComputeReport",
        "InferenceInclusiveComputeRow",
        "build_inference_inclusive_compute_report",
    ):
        assert not hasattr(evaluation, name)
        assert not hasattr(compute_ledger, name)
    with pytest.raises(
        ValueError,
        match="matching_set must be an exact H6MatchingSetRecord",
    ):
        derive_h6_inference_inclusive_compute_report(
            matching_set=SimpleNamespace(
                status="ELIGIBLE",
                obligations=(),
                matching_set_sha256="d" * 64,
            ),
            inference_records=(),
        )


def test_inference_scorer_authorization_comes_from_typed_configs() -> None:
    import vfe4.artifacts.h6_matching as matching_artifact

    config_ids = (
        "h6-a0-transformer-v2",
        "h6-a1-ordinary-latent-v1",
        "h6-a2-generic-map-v1",
        "h6-a3-immediate-predecessor-v1",
        "h6-a4-state-only-v1",
        "h6-a5-factorized-fixed-exact-complete-latent-smoothing-v1",
        (
            "h6-a5-structured-fixed-exact-complete-"
            "nolatent-norecognition-v1"
        ),
    )
    configs = tuple(_selected_config(config_id) for config_id in config_ids)

    assert matching_artifact._inference_scorer_authorization(configs) == (
        ("h6-a0-transformer-v2", "exact_autoregressive"),
        ("h6-a1-ordinary-latent-v1", "weighted_smc"),
        ("h6-a2-generic-map-v1", "weighted_smc"),
        ("h6-a3-immediate-predecessor-v1", "weighted_smc"),
        ("h6-a4-state-only-v1", "weighted_smc"),
        (
            "h6-a5-factorized-fixed-exact-complete-latent-smoothing-v1",
            "weighted_smc",
        ),
        (
            "h6-a5-structured-fixed-exact-complete-"
            "nolatent-norecognition-v1",
            "exact_autoregressive",
        ),
    )


@pytest.mark.parametrize(
    "config_id",
    (
        "h6-a1-ordinary-latent-v1",
        "h6-a2-generic-map-v1",
        "h6-a3-immediate-predecessor-v1",
        "h6-a4-state-only-v1",
    ),
)
def test_latent_endpoint_cannot_masquerade_as_exact(
    config_id: str,
) -> None:
    import vfe4.artifacts.h6_matching as matching_artifact
    from vfe4.evaluation.compute_ledger import (
        _build_inference_inclusive_compute_report,
    )
    from vfe4.types.h6 import InferenceComputeRecord

    config = _selected_config(config_id)
    exact_record = InferenceComputeRecord.create(
        endpoint_id=config_id,
        scorer_kind="exact_autoregressive",
        particle_count=None,
        replicate_count=1,
        prefix_cache_mode="causal_kv_cache",
        checkpoint_load_flops=10,
        cache_build_flops=20,
        scoring_flops=30,
        total_flops=60,
        wall_time_seconds=0.25,
    )

    with pytest.raises(
        ValueError,
        match="inference scorer differs from typed endpoint authorization",
    ):
        _build_inference_inclusive_compute_report(
            training_matching_set_sha256="d" * 64,
            training_flops_by_endpoint=((config_id, 1_000),),
            scorer_authorization=(
                matching_artifact._inference_scorer_authorization((config,))
            ),
            inference_records=(exact_record,),
        )


def test_h6_matching_v2_round_trip_reconstructs_exact_preimages(
    tmp_path: Path,
) -> None:
    run_directory = _publish_fixture(tmp_path, "valid")

    record = read_h6_matching_set(run_directory)

    assert record.schema_version == "h6-matching-set-v2"
    assert record.primary_selection.status == "ELIGIBLE"
    assert record.status == "ELIGIBLE"
    assert record.obligations == ()
    assert len(record.primary_selection.candidates) == 324
    assert record.primary_selection.candidate_inventory_sha256
    assert record.primary_selection.workload_sha256 == (
        record.workload.workload_sha256
    )
    assert len(record.component_selections) == 9
    assert len(record.ownership_inventories) == len(
        H6_MATCHING_ENDPOINT_LAYOUT
    )
    assert len(record.matrix_reports) == 8
    assert record.authorizing_matching_report_ids == ("PRIMARY",)
    assert {"MAP", "LATENT"} <= set(record.unmatched_report_ids)
    assert all(
        item.report.obligations
        or item.selection_obligations
        or item.row.row_id == "OBJECTIVE"
        for item in record.matrix_reports
        if item.row.row_id in record.unmatched_report_ids
    )
    assert all(
        not item.matched_claim_authorized
        and item.selection_obligations
        for item in record.matrix_reports
        if item.row.row_id in ("MAP", "LATENT")
    )
    assert derive_h6_matrix_reports(
        matching_config=record.primary_matching_config,
        workload=record.workload,
        primary_selection=record.primary_selection,
        component_selections=record.component_selections,
        ownership_inventories=record.ownership_inventories,
    ) == tuple(
        (item.row, item.report) for item in record.matrix_reports
    )

    inventories = {
        item.config.config_id: item.config
        for item in record.ownership_inventories
    }
    primary = next(
        item for item in record.matrix_reports if item.row.row_id == "PRIMARY"
    )
    objective = next(
        item
        for item in record.matrix_reports
        if item.row.row_id == "OBJECTIVE"
    )
    selected_a5 = inventories[primary.row.right_config_id]
    objective_complete = inventories[objective.row.left_config_id]
    objective_emission = inventories[objective.row.right_config_id]
    assert selected_a5 == objective_complete
    assert (
        selected_a5.capacity_allocation
        == objective_emission.capacity_allocation
    )
    assert objective_emission.objective_kind == (
        "emission_only_ablation_non_elbo"
    )
    assert objective.row.row_id not in (
        record.authorizing_matching_report_ids
    )

    selection_blocked = H6MatrixMatchingReportRecord.create(
        row=primary.row,
        report=primary.report,
        selection_obligations=("component formula selection is unmatched",),
    )
    assert primary.report.eligible
    assert not selection_blocked.matched_claim_authorized
    assert selection_blocked.selection_obligations == (
        "component formula selection is unmatched",
    )


def test_readiness_matching_validator_binds_current_source_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vfe4.artifacts.h6_matching as matching_artifact

    run_directory = _publish_fixture(tmp_path, "readiness")
    current_record = read_h6_matching_set(run_directory)
    expected_set = current_record.matching_set_sha256
    observed: dict[str, object] = {}

    def read_matching(root: Path, **kwargs: object) -> object:
        observed["root"] = root
        observed.update(kwargs)
        return current_record

    monkeypatch.setattr(
        matching_artifact, "read_h6_matching_set", read_matching
    )
    assert _validate_matching_artifact(
        tmp_path,
        expected_set_sha256=expected_set,
        expected_git_head=_GIT_HEAD,
        expected_dirty_digest=_DIRTY_DIGEST,
    ) == current_record
    assert observed == {
        "root": tmp_path,
        "expected_set_sha256": expected_set,
        "expected_git_head": _GIT_HEAD,
        "expected_dirty_digest": _DIRTY_DIGEST,
    }

    for current_record in (
        SimpleNamespace(
            matching_set_sha256=expected_set,
            git_head="3" * 40,
            dirty_digest=_DIRTY_DIGEST,
            status="ELIGIBLE",
            obligations=(),
        ),
        SimpleNamespace(
            matching_set_sha256=expected_set,
            git_head=_GIT_HEAD,
            dirty_digest="4" * 64,
            status="ELIGIBLE",
            obligations=(),
        ),
    ):
        with pytest.raises(ValueError, match="source differs"):
            _validate_matching_artifact(
                tmp_path,
                expected_set_sha256=expected_set,
                expected_git_head=_GIT_HEAD,
                expected_dirty_digest=_DIRTY_DIGEST,
            )


def test_h6_matching_v2_rejects_mutated_and_reordered_preimages(
    tmp_path: Path,
) -> None:
    original = _publish_fixture(tmp_path, "original")
    mutated = tmp_path / "mutated"
    reordered = tmp_path / "reordered"
    authorization = tmp_path / "authorization"
    shutil.copytree(original, mutated)
    shutil.copytree(original, reordered)
    shutil.copytree(original, authorization)

    mutated_path = mutated / "matching" / "endpoints.json"
    mutated_payload = json.loads(mutated_path.read_bytes())
    mutated_payload["primary_joint_selection"]["candidates"][0][
        "a0_parameter_count"
    ] += 1
    mutated_path.write_bytes(artifact_json_bytes(mutated_payload))
    _rewrite_manifest(mutated)
    with pytest.raises(ValueError, match="stale"):
        read_h6_matching_set(mutated)

    reordered_path = reordered / "matching" / "endpoints.json"
    reordered_payload = json.loads(reordered_path.read_bytes())
    reordered_payload["primary_joint_selection"]["candidates"][0:2] = (
        reversed(
            reordered_payload["primary_joint_selection"]["candidates"][0:2]
        )
    )
    reordered_path.write_bytes(artifact_json_bytes(reordered_payload))
    _rewrite_manifest(reordered)
    with pytest.raises(ValueError, match="ordered product"):
        read_h6_matching_set(reordered)

    authorization_path = (
        authorization / "matching" / "matrix_reports.json"
    )
    authorization_payload = json.loads(authorization_path.read_bytes())
    authorization_payload["reports"][1]["matched_claim_authorized"] = True
    authorization_path.write_bytes(
        artifact_json_bytes(authorization_payload)
    )
    _rewrite_manifest(authorization)
    with pytest.raises(ValueError, match="digest or fields"):
        read_h6_matching_set(authorization)


def test_h6_matching_v2_rejects_legacy_primary_formula_evidence(
    tmp_path: Path,
) -> None:
    original = _publish_fixture(tmp_path, "current")
    legacy = tmp_path / "legacy"
    shutil.copytree(original, legacy)
    endpoints_path = legacy / "matching" / "endpoints.json"
    payload = json.loads(endpoints_path.read_bytes())
    payload["formula_selections"] = payload.pop(
        "component_formula_selections"
    )
    endpoints_path.write_bytes(artifact_json_bytes(payload))
    _rewrite_manifest(legacy)

    with pytest.raises(
        H6MatchingPublicationBlocked,
        match="legacy H6 PRIMARY evidence",
    ):
        read_h6_matching_set(legacy)
