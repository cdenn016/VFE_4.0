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
    H6MatchingOwnershipRecord,
    derive_h6_matrix_reports,
    publish_h6_matching_set,
    read_h6_matching_set,
)
from vfe4.artifacts.provenance import source_candidate_sha256
from vfe4.training.arms import build_arm
from vfe4.training.h6_readiness import (
    _validate_matching_artifact,
)
from vfe4.training.matching import (
    H6FormulaSelection,
    H6TrainingWorkload,
    select_outcome_blind_allocation,
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
    "h6-a5-structured-prefix-exact-complete-latent-smoothing-v1": (
        ArmId.A5,
        True,
        True,
        True,
        "categorical",
        "shared_vertex_coboundary",
        "structured",
        "smoothing",
        "prefix_conditioned",
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
    "h6-a5-structured-fixed-exact-emission-latent-smoothing-v1": (
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
        vocabulary=VocabularyIdentity("h6-artifact-fixture-v1", 258, _SHA_A),
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


def _ownership(
    selection: H6FormulaSelection,
) -> H6MatchingOwnershipRecord:
    config = selection.selected_endpoint_config
    assert config is not None
    arm = build_arm(config.arm, config)
    return H6MatchingOwnershipRecord.create(
        config=config,
        parameter_roles=arm.parameter_roles,
        optimizer_bindings=arm.optimizer_bindings,
    )


def _evidence_fixture() -> tuple[
    tuple[H6FormulaSelection, ...],
    tuple[H6MatchingOwnershipRecord, ...],
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
    selections = tuple(
        select_outcome_blind_allocation(
            endpoint_template=configs[config_id],
            reference_config=reference,
            workload=workload,
        )
        for config_id, _ in H6_MATCHING_ENDPOINT_LAYOUT
    )
    return selections, tuple(_ownership(item) for item in selections)


def _publish_fixture(root: Path, run_name: str) -> Path:
    selections, ownership = _evidence_fixture()
    matching_config_sha256 = "c" * 64
    reports = derive_h6_matrix_reports(
        matching_config_sha256=matching_config_sha256,
        selections=selections,
        ownership_inventories=ownership,
    )
    _, run_directory = publish_h6_matching_set(
        artifact_root=root,
        run_name=run_name,
        git_head=_GIT_HEAD,
        dirty_digest=_DIRTY_DIGEST,
        source_sha256=source_candidate_sha256(
            git_head_value=_GIT_HEAD,
            dirty_digest_value=_DIRTY_DIGEST,
        ),
        matching_config_sha256=matching_config_sha256,
        selections=selections,
        ownership_inventories=ownership,
        matrix_reports=reports,
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


def test_h6_matching_v2_round_trip_reconstructs_exact_preimages(
    tmp_path: Path,
) -> None:
    run_directory = _publish_fixture(tmp_path, "valid")

    record = read_h6_matching_set(run_directory)

    assert record.schema_version == "h6-matching-set-v2"
    assert record.status == "ELIGIBLE"
    assert len(record.selections) == 12
    assert len(record.matrix_reports) == 8


def test_readiness_matching_validator_binds_current_source_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vfe4.artifacts.h6_matching as matching_artifact

    expected_set = "d" * 64
    current_record = SimpleNamespace(
        matching_set_sha256=expected_set,
        git_head=_GIT_HEAD,
        dirty_digest=_DIRTY_DIGEST,
    )
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
    ) == expected_set
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
        ),
        SimpleNamespace(
            matching_set_sha256=expected_set,
            git_head=_GIT_HEAD,
            dirty_digest="4" * 64,
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
    shutil.copytree(original, mutated)
    shutil.copytree(original, reordered)

    mutated_path = mutated / "matching" / "endpoints.json"
    mutated_payload = json.loads(mutated_path.read_bytes())
    mutated_payload["formula_selections"][0]["ledger"]["terms"][0][
        "operation"
    ] = "mutated_operation"
    mutated_path.write_bytes(artifact_json_bytes(mutated_payload))
    _rewrite_manifest(mutated)
    with pytest.raises(ValueError, match="stale"):
        read_h6_matching_set(mutated)

    reordered_path = reordered / "matching" / "endpoints.json"
    reordered_payload = json.loads(reordered_path.read_bytes())
    reordered_payload["formula_selections"][0:2] = reversed(
        reordered_payload["formula_selections"][0:2]
    )
    reordered_path.write_bytes(artifact_json_bytes(reordered_payload))
    _rewrite_manifest(reordered)
    with pytest.raises(ValueError, match="reordered"):
        read_h6_matching_set(reordered)
