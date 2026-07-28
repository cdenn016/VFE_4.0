from __future__ import annotations

import dataclasses
import hashlib
import io
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import verification.h8_h6_prediction_v3 as h8_adapter
import vfe4.training.h6_test_transaction_v3 as transaction_v3
import vfe4.training.h6_training_attempt_v3 as attempt_v3
import vfe4.training.h6_validation_v3 as validation_v3
from vfe4.artifacts.atomic import canonical_json_bytes, publish_run_directory
from vfe4.artifacts.h6_matching import H6MatchingSetRecord
from vfe4.artifacts.h6_prediction_v3 import (
    H6PredictionResultV3,
    H6PredictionV3Authorities,
    publish_h6_prediction_v3_authorities,
    read_h6_prediction_v3_authorities,
)
from vfe4.config import (
    H6ArchiveMemberExpectation,
    H6DataConfig,
    H6ObservedArchive,
    H6PredictionV2ResolvedConfig,
    H6PredictionV3ResolvedConfig,
    resolve_h6_prediction_v3_config,
)
from vfe4.data.access import (
    issue_h6_train_capability_v3,
    issue_h6_validation_capability_v3,
    open_train_for_training_v3,
)
from vfe4.data.byte_tokenizer import ByteTokenizerV1
from vfe4.data.h6_sealed_store_v3 import (
    AUTHENTICATED_BLINDED_STORE_MANIFEST_V3_FILENAME,
    reopen_authenticated_blinded_store_v3,
)
from vfe4.data.windows import CausalPrefix, build_causal_windows
from vfe4.data.wikitext2 import (
    WIKITEXT2_RAW_URL,
    BlindedCorpusStore,
    H6DataAcquisitionRequest,
    _acquire_wikitext2_blinded,
)
from vfe4.evaluation.smc_uncertainty import SMC_BIAS_SEMANTICS
from vfe4.numerics.critical_values import CRITICAL_VALUES_PROTOCOL_SHA256
from vfe4.training.checkpoint import H6CheckpointManifest
from vfe4.training.checkpoint_v3 import hydrate_h6_checkpoint_v3
from vfe4.training.h6_execution_v3 import (
    H6ExecutableAttemptV3,
    bind_h6_executable_attempt_v3,
)
from vfe4.training.h6_experiment_v3 import (
    H6ExperimentPlanV3,
    H6PlannedAttemptV3,
    plan_h6_experiment_v3,
    run_h6_experiment_v3,
)
from vfe4.training.h6_matching_v3 import (
    H6_MATCHING_POLICY_V3,
    H6MatchingSetV3,
    build_h6_matching_set_v3,
)
from vfe4.training.h6_readiness import (
    _derive_h6_prediction_readiness_v3,
)
from vfe4.training.h6_runtime_v3 import (
    H6SyntheticCpuRuntimeV3,
    bounded_synthetic_cpu_runtime_v3,
)
from vfe4.training.h6_test_transaction_v3 import (
    H6PredictionPointerV3,
    H6TestReservationV3,
    execute_h6_test_transaction_v3,
    finalize_h6_test_transaction_v3,
    read_h6_prediction_pointer_v3,
    read_h6_test_reservation_v3,
    read_h6_test_terminal_v3,
)
from vfe4.training.matching import (
    ARM_MATRIX_ROWS,
    H6_ADAMW_POLICY,
    arm_matrix_sha256,
)
from vfe4.types import H1_PREFIX_PRIOR_V2_GENERATIVE_FACTOR_SCHEMA_SHA256
from vfe4.types.h6 import (
    EndpointSmcProtocol,
    H6ArmPhaseSchedule,
    H6OuterSchedule,
    ObjectiveGateSpec,
    TrainingPhase,
)
from vfe4.types.h6_prediction_v3 import (
    H6_CHECKPOINT_CODEC_SHA256,
    H6_COUNTER_MAPPING_SHA256,
    H6_PHASE_OWNERSHIP_SHA256,
    H6_PREDICTION_METRICS_SCHEMA,
    H6_RAW_ENDPOINT_INVENTORY_SCHEMA,
    H6_SCORING_INVENTORY_SHA256,
    H6PredictionRuntimeIdentity,
    H6PredictionV3ReadinessToken,
    H6RecognitionEstimatorSpec,
    H6TrainingScheduleV3,
)
from vfe4.types.h8 import (
    H8H6PredictionReference,
    H8H6PredictionV3Reference,
)


_GIT_HEAD = "1" * 40
_DIRTY_DIGEST = "2" * 64
_DIRECT_CERTIFICATE_SHA256 = "3" * 64
_MEMBERS = (
    "wikitext-2-raw/wiki.train.raw",
    "wikitext-2-raw/wiki.valid.raw",
    "wikitext-2-raw/wiki.test.raw",
)
_MAXIMUM_CHECKPOINT_BYTES = 256 * 1024 * 1024
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _semantic_hash(domain: str, payload: object) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\0" + canonical_json_bytes(payload)
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_unsafe_mutation(record: object, **changes: object) -> None:
    mutated = object.__new__(type(record))
    for field in dataclasses.fields(record):
        object.__setattr__(
            mutated,
            field.name,
            changes.get(field.name, getattr(record, field.name)),
        )
    mutated.__post_init__()  # type: ignore[attr-defined]


def _archive_bytes() -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        directory = zipfile.ZipInfo("wikitext-2-raw/")
        directory.external_attr = (0o40755 << 16) | 0x10
        archive.writestr(directory, b"")
        archive.writestr(_MEMBERS[0], b"t")
        # Exactly 4,096 stride-32 validation windows after BOS/EOS framing.
        archive.writestr(_MEMBERS[1], b"v" * 131_041)
        archive.writestr(_MEMBERS[2], b"x")
    return stream.getvalue()


def _acquisition_request(
    archive_bytes: bytes,
    artifact_root: Path,
) -> H6DataAcquisitionRequest:
    with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as archive:
        members = tuple(
            H6ArchiveMemberExpectation(
                info.filename,  # type: ignore[arg-type]
                info.compress_size,
                info.file_size,
                info.compress_type,  # type: ignore[arg-type]
                info.CRC,
                hashlib.sha256(archive.read(info)).hexdigest(),
            )
            for info in archive.infolist()
            if not info.is_dir()
        )
    return H6DataAcquisitionRequest(
        data=H6DataConfig(
            "h6-data-config-v1",
            WIKITEXT2_RAW_URL,
            16_777_216,
            ("wikitext-2-raw/", *_MEMBERS),
            (0, 8),
            16_777_216,
            33_554_432,
            100,
            H6ObservedArchive(
                len(archive_bytes),
                hashlib.sha256(archive_bytes).hexdigest(),
                members,
            ),
        ),
        artifact_root=artifact_root,
    )


def _runtime_identity() -> H6PredictionRuntimeIdentity:
    return H6PredictionRuntimeIdentity.create(
        python_version="3.13.5",
        torch_full_version="2.10.0.dev20251210+cu128",
        cuda_runtime_version="12.8",
        cuda_device_name="NVIDIA GeForce RTX 5090",
        cuda_compute_capability=(12, 0),
    )


def _schedule(
    matching_set: H6MatchingSetV3,
    runtime: H6PredictionRuntimeIdentity,
    estimator: H6RecognitionEstimatorSpec,
) -> H6TrainingScheduleV3:
    return H6TrainingScheduleV3.create(
        outer=H6OuterSchedule.create(
            optimizer_policy_sha256=H6_ADAMW_POLICY.optimizer_policy_sha256,
        ),
        endpoint_phases=tuple(
            H6ArmPhaseSchedule.create(
                endpoint_config_sha256=config.config_sha256,
                latent_enabled=config.latent_enabled,
                phases=(
                    (
                        TrainingPhase.RECOGNITION_ADAMW,
                        TrainingPhase.IMMUTABLE_DETACHED_SNAPSHOT,
                        TrainingPhase.MODEL_ADAMW,
                    )
                    if config.latent_enabled
                    else (TrainingPhase.MODEL_CE_ADAMW,)
                ),
            )
            for config in matching_set.endpoint_configs
        ),
        estimator=estimator,
        runtime=runtime,
    )


def _resolved_config(
    *,
    matching_set: H6MatchingSetV3,
    store: BlindedCorpusStore,
    artifact_root: Path,
    runtime: H6PredictionRuntimeIdentity,
) -> H6PredictionV3ResolvedConfig:
    estimator = H6RecognitionEstimatorSpec.create()
    schedule = _schedule(matching_set, runtime, estimator)
    protocol = EndpointSmcProtocol.create(
        particle_counts=(128, 256, 512, 1024),
        replicate_count=64,
        registry_root_seed=2026072198,
        common_stream_domain="h6-wt2-endpoint-mc-v1",
        simultaneous_interval_count=352,
        familywise_alpha=0.01,
        critical_value_df63=4.5144904535377144,
        remainder_contraction=0.75,
    )
    objective_gate = ObjectiveGateSpec.create()
    estimator_payload = estimator.canonical_payload()
    estimator_payload["estimator_sha256"] = estimator.estimator_sha256
    runtime_payload = runtime.canonical_payload()
    runtime_payload["cuda_compute_capability"] = list(runtime.cuda_compute_capability)
    runtime_payload["runtime_identity_sha256"] = runtime.runtime_identity_sha256
    raw = {
        "schema_version": "h6-prediction-config-v3",
        "operation": "H6-Prediction",
        "source": {
            "git_head": _GIT_HEAD,
            "dirty_digest": _DIRTY_DIGEST,
            "source_sha256": _digest("source"),
        },
        "data": {
            "schema_version": "h6-data-config-v1",
            "source_url": WIKITEXT2_RAW_URL,
            "max_archive_bytes": 16_777_216,
            "member_paths": ["wikitext-2-raw/", *_MEMBERS],
            "allowed_compression_methods": [0, 8],
            "max_member_bytes": 16_777_216,
            "max_total_uncompressed_bytes": 33_554_432,
            "max_compression_ratio": 100,
            "observed_archive": None,
        },
        "prerequisites": {
            "correctness_manifests": {
                "H1": _digest("H1"),
                "H2": _digest("H2"),
                "H3": _digest("H3"),
                "H5": _digest("H5"),
            },
            "h1_prefix_prior_manifest_sha256": _digest("h1-prefix"),
            "h1_prefix_prior_generative_factor_schema_sha256": (
                H1_PREFIX_PRIOR_V2_GENERATIVE_FACTOR_SCHEMA_SHA256
            ),
            "smc_validation_manifest_sha256": _digest("smc-validation"),
            "prefix_certificate_set_sha256": _digest("prefix-certificates"),
            "a0_direct_exact_prefix_certificate_sha256": (_DIRECT_CERTIFICATE_SHA256),
        },
        "h5_update_binding_sha256": _digest("h5-update"),
        "training_schedule": {
            "schedule_schema": schedule.schedule_schema,
            "outer": {
                "schedule_schema": schedule.outer.schedule_schema,
                "optimizer_class": schedule.outer.optimizer_class,
                "optimizer_policy_sha256": (schedule.outer.optimizer_policy_sha256),
                "model_updates_per_batch": (schedule.outer.model_updates_per_batch),
                "validation_twentieths_per_pass": (
                    schedule.outer.validation_twentieths_per_pass
                ),
                "full_passes": schedule.outer.full_passes,
            },
            "endpoint_phases": [
                {
                    "endpoint_config_sha256": phase.endpoint_config_sha256,
                    "latent_enabled": phase.latent_enabled,
                    "phases": [item.value for item in phase.phases],
                    "recognition_updates_per_batch": (
                        phase.recognition_updates_per_batch
                    ),
                    "model_updates_per_batch": phase.model_updates_per_batch,
                    "no_op_phases": phase.no_op_phases,
                }
                for phase in schedule.endpoint_phases
            ],
            "recognition_estimator_sha256": estimator.estimator_sha256,
            "runtime_identity_sha256": runtime.runtime_identity_sha256,
            "training_noise_domain": schedule.training_noise_domain,
            "counter_mapping_sha256": H6_COUNTER_MAPPING_SHA256,
            "phase_ownership_sha256": H6_PHASE_OWNERSHIP_SHA256,
            "checkpoint_codec_sha256": H6_CHECKPOINT_CODEC_SHA256,
        },
        "critical_values_sha256": CRITICAL_VALUES_PROTOCOL_SHA256,
        "endpoint_smc_protocol": {
            "protocol_schema": protocol.protocol_schema,
            "particle_counts": list(protocol.particle_counts),
            "replicate_count": protocol.replicate_count,
            "registry_root_seed": protocol.registry_root_seed,
            "common_stream_domain": protocol.common_stream_domain,
            "simultaneous_interval_count": protocol.simultaneous_interval_count,
            "familywise_alpha": protocol.familywise_alpha,
            "critical_value_df63": protocol.critical_value_df63,
            "remainder_contraction": protocol.remainder_contraction,
        },
        "smc_bias_semantics_sha256": SMC_BIAS_SEMANTICS.semantics_sha256,
        "attribution_matrix_sha256": arm_matrix_sha256(ARM_MATRIX_ROWS),
        "matching_policy_schema": "h6-amended-matching-policy-v3",
        "matching_policy_sha256": H6_MATCHING_POLICY_V3.policy_sha256,
        "matching_set_schema": "h6-amended-matching-set-v3",
        "matching_set_sha256": matching_set.matching_set_sha256,
        "objective_gate": {
            "schema_version": objective_gate.schema_version,
            "complete_arm_id": objective_gate.complete_arm_id,
            "emission_arm_id": objective_gate.emission_arm_id,
            "orientation": objective_gate.orientation,
            "delta_obj": objective_gate.delta_obj,
            "opening_policy": objective_gate.opening_policy,
            "evaluation_order": objective_gate.evaluation_order,
            "spec_sha256": objective_gate.spec_sha256,
        },
        "data_identity_sha256": store.data_identity_sha256,
        "access_policy_sha256": store.data_identity.access_policy_sha256,
        "recognition_contract": {
            "trajectory_schema": "h6-language-recognition-trajectory-v3",
            "categorical_posterior_schema": ("h6-categorical-source-posterior-v3"),
            "terminal_mixture_schema": "h6-terminal-source-mixture-v1",
            "estimator": estimator_payload,
        },
        "runtime": runtime_payload,
        "counter_mapping_sha256": H6_COUNTER_MAPPING_SHA256,
        "phase_ownership_sha256": H6_PHASE_OWNERSHIP_SHA256,
        "checkpoint_codec_sha256": H6_CHECKPOINT_CODEC_SHA256,
        "scoring_inventory_sha256": H6_SCORING_INVENTORY_SHA256,
        "expected_test_row_count": 4104,
        "artifact_root": str(artifact_root.resolve()),
    }
    return resolve_h6_prediction_v3_config(raw, repo_root=_REPO_ROOT)


@dataclass(frozen=True)
class _CpuFixtureCase:
    root: Path
    store: BlindedCorpusStore
    matching_set: H6MatchingSetV3
    config: H6PredictionV3ResolvedConfig
    readiness: H6PredictionV3ReadinessToken
    plan: H6ExperimentPlanV3
    authorities_path: Path
    authorities: H6PredictionV3Authorities
    planned_attempt: H6PlannedAttemptV3
    executable: H6ExecutableAttemptV3
    runtime: H6SyntheticCpuRuntimeV3
    training_data: object
    checkpoint_path: Path
    training_result: object


@pytest.fixture(scope="module")
def _cpu_fixture_case(
    tmp_path_factory: pytest.TempPathFactory,
) -> _CpuFixtureCase:
    root = tmp_path_factory.mktemp("h6-v3-integration").resolve()
    store_root = (root / "sealed-store").resolve()
    archive_bytes = _archive_bytes()
    _acquire_wikitext2_blinded(
        _acquisition_request(archive_bytes, store_root),
        lambda _url: io.BytesIO(archive_bytes),
    )
    store = reopen_authenticated_blinded_store_v3(
        store_root / AUTHENTICATED_BLINDED_STORE_MANIFEST_V3_FILENAME,
        store_root,
    )
    identity = store.data_identity
    matching_set = build_h6_matching_set_v3(
        git_head=_GIT_HEAD,
        dirty_digest=_DIRTY_DIGEST,
        train_token_count=identity.train_tokens.token_count,
        train_token_sha256=identity.train_tokens.encoded_token_sha256,
        vocabulary=ByteTokenizerV1().vocabulary_identity,
        horizon=32,
    )
    runtime_identity = _runtime_identity()
    config = _resolved_config(
        matching_set=matching_set,
        store=store,
        artifact_root=(root / "declared-artifacts").resolve(),
        runtime=runtime_identity,
    )
    readiness = _derive_h6_prediction_readiness_v3(
        config=config,
        matching_set=matching_set,
        git_head=_GIT_HEAD,
        dirty_digest=_DIRTY_DIGEST,
    )
    plan = plan_h6_experiment_v3(
        readiness=readiness,
        matching_set=matching_set,
        training_schedule=config.training_schedule,
        runtime_identity=config.runtime,
    )
    authorities_path = publish_h6_prediction_v3_authorities(
        run_root=root,
        run_name="authorities",
        config=config,
        matching_set=matching_set,
        readiness=readiness,
        plan=plan,
    )
    authorities = read_h6_prediction_v3_authorities(authorities_path)
    training_data = open_train_for_training_v3(
        issue_h6_train_capability_v3(
            store,
            authorities.readiness,
            authorities.plan,
        ),
        plan=authorities.plan,
    )
    planned_attempt = next(
        attempt
        for attempt in authorities.plan.tuning_attempts
        if attempt.endpoint_config_id == "h6-a0-transformer-v2"
    )
    executable = bind_h6_executable_attempt_v3(
        authorities=authorities,
        planned_attempt=planned_attempt,
    )
    runtime = bounded_synthetic_cpu_runtime_v3(
        fixture_id="h6-v3-bounded-integration",
    )
    checkpoint_path = (root / "a0-terminal.h6v3").resolve()
    training_result = attempt_v3.execute_h6_training_attempt_v3(
        executable=executable,
        training_data=training_data,
        runtime=runtime,
        checkpoint_path=checkpoint_path,
        maximum_checkpoint_bytes=_MAXIMUM_CHECKPOINT_BYTES,
    )
    return _CpuFixtureCase(
        root=root,
        store=store,
        matching_set=matching_set,
        config=config,
        readiness=readiness,
        plan=plan,
        authorities_path=authorities_path,
        authorities=authorities,
        planned_attempt=planned_attempt,
        executable=executable,
        runtime=runtime,
        training_data=training_data,
        checkpoint_path=checkpoint_path,
        training_result=training_result,
    )


def _score_one_authorized_validation_target(
    case: _CpuFixtureCase,
    monkeypatch: pytest.MonkeyPatch,
) -> object:
    capability = issue_h6_validation_capability_v3(
        case.store,
        case.readiness,
        case.plan,
    )
    real_consume = validation_v3._consume_h6_validation_capability_v3

    def bounded_consume(received: object, *, plan: object) -> object:
        authorized = real_consume(received, plan=plan)
        assert authorized.windows.counted_target_total >= 4_096
        return SimpleNamespace(
            experiment_config_sha256=authorized.experiment_config_sha256,
            readiness_sha256=authorized.readiness_sha256,
            plan_sha256=authorized.plan_sha256,
            matching_set_sha256=authorized.matching_set_sha256,
            data_identity_sha256=authorized.data_identity_sha256,
            runtime_identity_sha256=authorized.runtime_identity_sha256,
            windows=build_causal_windows((2, 3), split="validation"),
            vocabulary=authorized.vocabulary,
        )

    monkeypatch.setattr(
        validation_v3,
        "_consume_h6_validation_capability_v3",
        bounded_consume,
    )
    return validation_v3.score_h6_validation_checkpoint_v3(
        capability=capability,
        checkpoint=case.training_result.terminal_checkpoint,
        planned_attempt=case.planned_attempt,
        plan=case.plan,
    )


def _fake_result(
    reservation: H6TestReservationV3,
) -> tuple[H6PredictionResultV3, object, object]:
    raw_inventory_sha256 = _digest("arithmetic-inventory-4104")
    metrics_sha256 = _digest("arithmetic-metrics-4104")
    payload = {
        "result_schema": "h6-prediction-result-v3",
        "reservation_sha256": reservation.reservation_sha256,
        "opening_proof_sha256": reservation.opening_proof_sha256,
        "raw_inventory_sha256": raw_inventory_sha256,
        "metrics_sha256": metrics_sha256,
        "logical_row_count": 4104,
    }
    result = H6PredictionResultV3(
        **payload,  # type: ignore[arg-type]
        result_sha256=_semantic_hash(
            "vfe4.h6.prediction-result.v3",
            payload,
        ),
    )
    inventory = SimpleNamespace(
        inventory_schema=H6_RAW_ENDPOINT_INVENTORY_SCHEMA,
        inventory_sha256=raw_inventory_sha256,
        opening_proof_sha256=reservation.opening_proof_sha256,
        logical_row_count=4104,
    )
    metrics = SimpleNamespace(
        metrics_schema=H6_PREDICTION_METRICS_SCHEMA,
        raw_inventory_sha256=raw_inventory_sha256,
        metrics_sha256=metrics_sha256,
    )
    return result, inventory, metrics


def _reference_shell(root: Path) -> H8H6PredictionV3Reference:
    digest = _digest("reference-shell")
    payload_hashes = {
        "metrics.json": digest,
        "raw_inventory.json": digest,
        "result.json": digest,
    }
    return H8H6PredictionV3Reference(
        kind="h6_prediction",
        config_schema="h6-prediction-config-v3",
        readiness_schema="h6-prediction-readiness-v3",
        raw_inventory_schema=H6_RAW_ENDPOINT_INVENTORY_SCHEMA,
        metrics_schema=H6_PREDICTION_METRICS_SCHEMA,
        result_schema="h6-prediction-result-v3",
        artifact_path=str(root / "RESULT"),
        manifest_sha256=digest,
        result_path=str(root / "RESULT" / "result.json"),
        result_sha256=digest,
        content_hashes={"result.json": digest},
        payload_hashes=payload_hashes,
        authorities_path=str(root / "authorities"),
        authorities_manifest_sha256=digest,
        authorities_sha256=digest,
        config_sha256=digest,
        readiness_sha256=digest,
        plan_sha256=digest,
        matching_set_sha256=digest,
        validation_bundle_path=str(root / "validation"),
        validation_bundle_manifest_sha256=digest,
        validation_bundle_sha256=digest,
        checkpoint_selection_sha256=digest,
        reservation_path=str(root / "RESERVED"),
        reservation_sha256=digest,
        reservation_file_sha256=digest,
        terminal_path=str(root / "STATE" / "TERMINAL"),
        terminal_sha256=digest,
        terminal_manifest_sha256=digest,
        finalized_path=str(root / "STATE" / "FINALIZED"),
        finalized_manifest_sha256=digest,
        pointer_path=str(root / "pointer"),
        pointer_sha256=digest,
        pointer_manifest_sha256=digest,
        experiment_identity_sha256=digest,
        opening_proof_sha256=digest,
        raw_inventory_sha256=digest,
        metrics_sha256=digest,
        result_record_sha256=digest,
        ledger_path=str(root / "ledger.json"),
        ledger_sha256=digest,
        ledger_validator_sha256=(h8_adapter.H8_PREDICTION_V3_LEDGER_VALIDATOR_SHA256),
        artifact_revision=(f"git:{_GIT_HEAD}:sha256:{_DIRTY_DIGEST}"),
        producer_head=_GIT_HEAD,
        producer_dirty_digest=_DIRTY_DIGEST,
        candidate_junit_path=str(root / "candidate.xml"),
        candidate_junit_sha256=digest,
        status="pass",
    )


def test_h6_v3_click_fixture_reaches_closed_result_and_h8_adapter(
    _cpu_fixture_case: _CpuFixtureCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _cpu_fixture_case
    reopened = read_h6_prediction_v3_authorities(
        case.authorities_path,
        expected_authority_sha256=case.authorities.authority_sha256,
    )
    assert reopened == case.authorities
    assert case.training_result.checkpoint_path == case.checkpoint_path

    recovered = attempt_v3.recover_h6_training_attempt_v3(
        executable=case.executable,
        runtime=case.runtime,
        checkpoint_path=case.checkpoint_path,
        maximum_checkpoint_bytes=_MAXIMUM_CHECKPOINT_BYTES,
    )
    assert recovered is not None
    history = attempt_v3.read_h6_training_attempt_history_v3(
        checkpoint_path=case.checkpoint_path,
        maximum_bytes=_MAXIMUM_CHECKPOINT_BYTES,
    )
    assert recovered.history == history == case.training_result.terminal_history
    assert recovered.checkpoint == case.training_result.terminal_checkpoint
    prefix = CausalPrefix.create(
        receiver_t=1,
        vocabulary=case.executable.endpoint_config.vocabulary,
        token_ids=torch.empty(0, dtype=torch.int64),
    )
    next_prediction = recovered.model.prefix_log_probs(prefix)
    assert next_prediction.shape == (case.executable.endpoint_config.vocabulary.size,)
    assert bool(torch.isfinite(next_prediction).all())

    validation = _score_one_authorized_validation_target(case, monkeypatch)
    assert validation.counted_target_total == 1
    assert validation.scoring_method == "target-blind-prefix-prior-nll"

    checkpoint_selection_sha256 = _digest("fixture-checkpoint-selection")
    validation_bundle_sha256 = _digest("fixture-validation-bundle")
    validation_path = publish_run_directory(
        tmp_path.resolve(),
        "validation-bundle",
        {
            "validation.json": {
                "checkpoint_sha256": (
                    case.training_result.terminal_checkpoint.checkpoint_sha256
                ),
                "validation_record_sha256": (validation.validation_record_sha256),
                "arithmetic_record_count": 1,
            }
        },
    )
    validation_selection = SimpleNamespace(
        checkpoint_selection_sha256=checkpoint_selection_sha256,
        experiment_config_sha256=case.config.config_sha256,
        readiness_sha256=case.readiness.readiness_sha256,
        plan_sha256=case.plan.plan_sha256,
        matching_set_sha256=case.matching_set.matching_set_sha256,
    )
    validation_bundle = SimpleNamespace(
        experiment_config_sha256=case.config.config_sha256,
        plan_sha256=case.plan.plan_sha256,
        validation_bundle_sha256=validation_bundle_sha256,
        checkpoint_selection=validation_selection,
        validation_records=(validation,),
    )

    result_root = (tmp_path / "transaction").resolve()
    state_root = (result_root / "STATE").resolve()
    pointer_root = (tmp_path / "pointers").resolve()
    reservation = H6TestReservationV3.create(
        experiment_config_sha256=case.config.config_sha256,
        readiness_sha256=case.readiness.readiness_sha256,
        plan_sha256=case.plan.plan_sha256,
        experiment_identity_sha256=_digest("fixture-experiment"),
        data_identity_sha256=case.store.data_identity_sha256,
        sealed_test_sha256=(case.store.sealed_test_handle.sealed_content_sha256),
        test_inventory_sha256=case.store.sealed_test_handle.handle_sha256,
        access_policy_sha256=case.store.data_identity.access_policy_sha256,
        tuning_selection_sha256=_digest("fixture-tuning-selection"),
        checkpoint_selection_sha256=checkpoint_selection_sha256,
        validation_bundle_sha256=validation_bundle_sha256,
        scoring_inventory_sha256=H6_SCORING_INVENTORY_SHA256,
        expected_row_count=4104,
        result_root=result_root,
        state_root=state_root,
        pointer_root=pointer_root,
        pointer_name="current",
    )
    reservation_path = (tmp_path / "RESERVED.h6v3").resolve()
    reservation_path.write_bytes(reservation.canonical_bytes())
    assert read_h6_test_reservation_v3(reservation_path) == reservation
    result, inventory, metrics = _fake_result(reservation)

    def publish_arithmetic_result(
        run_root: Path,
        run_name: str,
        *,
        result: object,
        inventory: object,
        metrics: object,
    ) -> Path:
        assert result is result_record
        assert inventory is arithmetic_inventory
        assert metrics is arithmetic_metrics
        return publish_run_directory(
            run_root,
            run_name,
            {
                "metrics.json": {
                    "metrics_schema": metrics.metrics_schema,
                    "raw_inventory_sha256": metrics.raw_inventory_sha256,
                    "metrics_sha256": metrics.metrics_sha256,
                    "arithmetic_row_count": 4104,
                },
                "raw_inventory.json": {
                    "inventory_schema": inventory.inventory_schema,
                    "inventory_sha256": inventory.inventory_sha256,
                    "opening_proof_sha256": (inventory.opening_proof_sha256),
                    "logical_row_count": inventory.logical_row_count,
                    "materialized_row_count": 0,
                },
                "result.json": result.artifact_payload(),
            },
        )

    result_record = result
    arithmetic_inventory = inventory
    arithmetic_metrics = metrics
    monkeypatch.setattr(
        transaction_v3,
        "publish_h6_prediction_result_v3",
        publish_arithmetic_result,
    )
    finalized = finalize_h6_test_transaction_v3(
        reservation_path=reservation_path,
        result=result,
        inventory=inventory,  # type: ignore[arg-type]
        metrics=metrics,  # type: ignore[arg-type]
    )
    assert finalized.terminal.state == "FINALIZED"
    assert finalized.terminal.result_sha256 == result.result_sha256
    assert read_h6_test_terminal_v3(finalized.terminal_directory) == finalized.terminal
    pointer = read_h6_prediction_pointer_v3(finalized.pointer_directory)
    assert pointer.result_sha256 == result.result_sha256
    assert pointer.reservation_sha256 == reservation.reservation_sha256

    ledger_path = (tmp_path / "fixture-ledger.json").resolve()
    junit_path = (tmp_path / "fixture-candidate.xml").resolve()
    ledger_path.write_bytes(
        canonical_json_bytes(
            {
                "mode": "bounded-fixture",
                "claim": "arithmetic-only-result-closure",
            }
        )
    )
    junit_path.write_bytes(
        b'<testsuite tests="1" failures="0" errors="0" skipped="0"/>'
    )
    result_payload_hashes = {
        name: _file_sha256(finalized.result_directory / name)
        for name in ("metrics.json", "raw_inventory.json", "result.json")
    }
    reference = H8H6PredictionV3Reference(
        kind="h6_prediction",
        config_schema="h6-prediction-config-v3",
        readiness_schema="h6-prediction-readiness-v3",
        raw_inventory_schema=H6_RAW_ENDPOINT_INVENTORY_SCHEMA,
        metrics_schema=H6_PREDICTION_METRICS_SCHEMA,
        result_schema="h6-prediction-result-v3",
        artifact_path=str(finalized.result_directory),
        manifest_sha256=_file_sha256(finalized.result_directory / "manifest.sha256"),
        result_path=str(finalized.result_directory / "result.json"),
        result_sha256=result_payload_hashes["result.json"],
        content_hashes={
            "result.json": result_payload_hashes["result.json"],
        },
        payload_hashes=result_payload_hashes,
        authorities_path=str(case.authorities_path),
        authorities_manifest_sha256=_file_sha256(
            case.authorities_path / "manifest.sha256"
        ),
        authorities_sha256=case.authorities.authority_sha256,
        config_sha256=case.config.config_sha256,
        readiness_sha256=case.readiness.readiness_sha256,
        plan_sha256=case.plan.plan_sha256,
        matching_set_sha256=case.matching_set.matching_set_sha256,
        validation_bundle_path=str(validation_path),
        validation_bundle_manifest_sha256=_file_sha256(
            validation_path / "manifest.sha256"
        ),
        validation_bundle_sha256=validation_bundle_sha256,
        checkpoint_selection_sha256=checkpoint_selection_sha256,
        reservation_path=str(reservation_path),
        reservation_sha256=reservation.reservation_sha256,
        reservation_file_sha256=_file_sha256(reservation_path),
        terminal_path=str(finalized.terminal_directory),
        terminal_sha256=finalized.terminal.terminal_sha256,
        terminal_manifest_sha256=_file_sha256(
            finalized.terminal_directory / "manifest.sha256"
        ),
        finalized_path=str(finalized.state_alias_directory),
        finalized_manifest_sha256=_file_sha256(
            finalized.state_alias_directory / "manifest.sha256"
        ),
        pointer_path=str(finalized.pointer_directory),
        pointer_sha256=pointer.pointer_sha256,
        pointer_manifest_sha256=_file_sha256(
            finalized.pointer_directory / "manifest.sha256"
        ),
        experiment_identity_sha256=reservation.experiment_identity_sha256,
        opening_proof_sha256=reservation.opening_proof_sha256,
        raw_inventory_sha256=inventory.inventory_sha256,
        metrics_sha256=metrics.metrics_sha256,
        result_record_sha256=result.result_sha256,
        ledger_path=str(ledger_path),
        ledger_sha256=_file_sha256(ledger_path),
        ledger_validator_sha256=(h8_adapter.H8_PREDICTION_V3_LEDGER_VALIDATOR_SHA256),
        artifact_revision=(f"git:{_GIT_HEAD}:sha256:{_DIRTY_DIGEST}"),
        producer_head=_GIT_HEAD,
        producer_dirty_digest=_DIRTY_DIGEST,
        candidate_junit_path=str(junit_path),
        candidate_junit_sha256=_file_sha256(junit_path),
        status="pass",
    )
    injected_calls: list[str] = []

    def reopen_validation(
        path: Path,
        *,
        expected_plan_sha256: str,
        expected_experiment_config_sha256: str,
        expected_validation_bundle_sha256: str,
    ) -> object:
        injected_calls.append("validation")
        assert path == validation_path
        assert _file_sha256(path / "manifest.sha256") == (
            reference.validation_bundle_manifest_sha256
        )
        assert expected_plan_sha256 == case.plan.plan_sha256
        assert expected_experiment_config_sha256 == case.config.config_sha256
        assert expected_validation_bundle_sha256 == validation_bundle_sha256
        return validation_bundle

    def reopen_result(
        path: Path,
        *,
        expected_result_sha256: str,
    ) -> tuple[object, object, object]:
        injected_calls.append("result")
        assert path == finalized.result_directory
        assert expected_result_sha256 == result.result_sha256
        return result, inventory, metrics

    def validate_fixture_ledger(received: object) -> None:
        injected_calls.append("ledger")
        assert received is reference
        assert _file_sha256(ledger_path) == reference.ledger_sha256
        assert _file_sha256(junit_path) == reference.candidate_junit_sha256

    monkeypatch.setattr(
        h8_adapter,
        "read_h6_validation_bundle_v3",
        reopen_validation,
    )
    monkeypatch.setattr(
        h8_adapter,
        "read_h6_prediction_result_v3",
        reopen_result,
    )
    monkeypatch.setattr(h8_adapter, "_validate_ledger", validate_fixture_ledger)
    h8_adapter.validate_h8_h6_prediction_v3_reference(
        reference,
        expected_a0_direct_exact_prefix_certificate_sha256=(_DIRECT_CERTIFICATE_SHA256),
    )
    assert injected_calls == ["validation", "result", "ledger"]


def test_h6_v3_fixture_refuses_identity_drift_at_each_boundary(
    _cpu_fixture_case: _CpuFixtureCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _cpu_fixture_case
    recovered = attempt_v3.recover_h6_training_attempt_v3(
        executable=case.executable,
        runtime=case.runtime,
        checkpoint_path=case.checkpoint_path,
        maximum_checkpoint_bytes=_MAXIMUM_CHECKPOINT_BYTES,
    )
    assert recovered is not None
    validation = _score_one_authorized_validation_target(case, monkeypatch)
    reservation = H6TestReservationV3.create(
        experiment_config_sha256=case.config.config_sha256,
        readiness_sha256=case.readiness.readiness_sha256,
        plan_sha256=case.plan.plan_sha256,
        experiment_identity_sha256=_digest("drift-experiment"),
        data_identity_sha256=case.store.data_identity_sha256,
        sealed_test_sha256=case.store.sealed_test_handle.sealed_content_sha256,
        test_inventory_sha256=case.store.sealed_test_handle.handle_sha256,
        access_policy_sha256=case.store.data_identity.access_policy_sha256,
        tuning_selection_sha256=_digest("drift-tuning"),
        checkpoint_selection_sha256=_digest("drift-checkpoint-selection"),
        validation_bundle_sha256=_digest("drift-validation-bundle"),
        scoring_inventory_sha256=H6_SCORING_INVENTORY_SHA256,
        expected_row_count=4104,
        result_root=(tmp_path / "result").resolve(),
        state_root=(tmp_path / "state").resolve(),
        pointer_root=(tmp_path / "pointer-root").resolve(),
        pointer_name="current",
    )
    result, _inventory, _metrics = _fake_result(reservation)
    pointer = H6PredictionPointerV3.create(
        reservation_sha256=reservation.reservation_sha256,
        terminal_sha256=_digest("terminal"),
        result_sha256=result.result_sha256,
        publication_atomicity=(
            "individual-no-replace-directories-no-cross-directory-atomicity"
        ),
    )
    h8_reference = _reference_shell(tmp_path.resolve())
    sentinel_effects: list[str] = []

    drift_checks = (
        lambda: _derive_h6_prediction_readiness_v3(
            config=dataclasses.replace(
                case.config,
                config_sha256="0" * 64,
            ),
            matching_set=case.matching_set,
            git_head=_GIT_HEAD,
            dirty_digest=_DIRTY_DIGEST,
        ),
        lambda: _validate_unsafe_mutation(
            case.readiness,
            readiness_sha256="0" * 64,
        ),
        lambda: _validate_unsafe_mutation(
            case.executable,
            executable_attempt_sha256="0" * 64,
        ),
        lambda: _validate_unsafe_mutation(
            recovered.boundary,
            checkpoint_sha256="0" * 64,
        ),
        lambda: _validate_unsafe_mutation(
            case.training_result.terminal_checkpoint,
            checkpoint_sha256="0" * 64,
        ),
        lambda: _validate_unsafe_mutation(
            validation,
            validation_record_sha256="0" * 64,
        ),
        lambda: _validate_unsafe_mutation(
            result,
            result_sha256="0" * 64,
        ),
        lambda: _validate_unsafe_mutation(
            pointer,
            pointer_sha256="0" * 64,
        ),
        lambda: _validate_unsafe_mutation(
            h8_reference,
            artifact_revision="git:drift",
        ),
    )
    for check in drift_checks:
        with pytest.raises(ValueError):
            check()
        assert sentinel_effects == []


class _ExplodingMapping(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        raise AssertionError(f"operation config read before v3 type check: {key}")

    def __iter__(self):
        raise AssertionError("operation config iterated before v3 type check")

    def __len__(self) -> int:
        raise AssertionError("operation config sized before v3 type check")


def test_every_v3_dispatcher_rejects_v1_v2_types_before_effects(
    _cpu_fixture_case: _CpuFixtureCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _cpu_fixture_case
    legacy_config = object.__new__(H6PredictionV2ResolvedConfig)
    legacy_checkpoint = object.__new__(H6CheckpointManifest)
    legacy_h8 = object.__new__(H8H6PredictionReference)
    relabeled_matching = object.__new__(H6MatchingSetRecord)
    object.__setattr__(
        relabeled_matching,
        "schema_version",
        "h6-amended-matching-set-v3",
    )
    effects: list[str] = []
    absent_checkpoint = (tmp_path / "must-not-be-read.h6v3").resolve()
    absent_publication = (tmp_path / "must-not-be-published").resolve()

    with pytest.raises(
        ValueError,
        match="exact H6PredictionV3ResolvedConfig|exact.*v3",
    ):
        _derive_h6_prediction_readiness_v3(
            config=legacy_config,  # type: ignore[arg-type]
            matching_set=case.matching_set,
            git_head=_GIT_HEAD,
            dirty_digest=_DIRTY_DIGEST,
        )
    with pytest.raises(ValueError, match="exact H6MatchingSetV3|v2"):
        _derive_h6_prediction_readiness_v3(
            config=case.config,
            matching_set=relabeled_matching,  # type: ignore[arg-type]
            git_head=_GIT_HEAD,
            dirty_digest=_DIRTY_DIGEST,
        )
    with pytest.raises(ValueError, match="exact.*readiness|v3"):
        plan_h6_experiment_v3(
            readiness=object(),  # type: ignore[arg-type]
            matching_set=case.matching_set,
            training_schedule=case.config.training_schedule,
            runtime_identity=case.config.runtime,
        )
    with pytest.raises(ValueError, match="exact H6-Prediction v3"):
        bind_h6_executable_attempt_v3(
            authorities=legacy_config,
            planned_attempt=case.planned_attempt,
        )
    with pytest.raises(ValueError, match="exact executable"):
        attempt_v3.execute_h6_training_attempt_v3(
            executable=legacy_config,  # type: ignore[arg-type]
            training_data=object(),  # type: ignore[arg-type]
            runtime=object(),  # type: ignore[arg-type]
            checkpoint_path=absent_checkpoint,
            maximum_checkpoint_bytes=_MAXIMUM_CHECKPOINT_BYTES,
        )

    original_lexists = attempt_v3.os.path.lexists

    def forbidden_lexists(path: object) -> bool:
        effects.append(f"filesystem:{path}")
        return original_lexists(path)

    monkeypatch.setattr(attempt_v3.os.path, "lexists", forbidden_lexists)
    with pytest.raises(ValueError, match="exact executable"):
        attempt_v3.recover_h6_training_attempt_v3(
            executable=legacy_config,  # type: ignore[arg-type]
            runtime=case.runtime,
            checkpoint_path=absent_checkpoint,
            maximum_checkpoint_bytes=_MAXIMUM_CHECKPOINT_BYTES,
        )
    assert effects == []

    with pytest.raises(ValueError, match="exact H6CheckpointV3"):
        hydrate_h6_checkpoint_v3(
            legacy_checkpoint,  # type: ignore[arg-type]
            expected_attempt_spec=case.planned_attempt.attempt_spec,
            expected_runtime_identity=case.config.runtime,
            live_deterministic_policy_sha256=(
                case.training_result.terminal_checkpoint.deterministic_policy_sha256
            ),
            factory_authority=object(),  # type: ignore[arg-type]
            authorized_device="cpu",
            allow_synthetic_cpu=True,
        )
    assert effects == []

    def forbidden_validation_consume(
        _capability: object,
        *,
        plan: object,
    ) -> object:
        effects.append(f"validation:{plan!r}")
        raise AssertionError("legacy checkpoint consumed validation access")

    monkeypatch.setattr(
        validation_v3,
        "_consume_h6_validation_capability_v3",
        forbidden_validation_consume,
    )
    with pytest.raises(ValueError, match="exact.*checkpoint|H6CheckpointV3"):
        validation_v3.score_h6_validation_checkpoint_v3(
            capability=object(),
            checkpoint=legacy_checkpoint,  # type: ignore[arg-type]
            planned_attempt=case.planned_attempt,
            plan=case.plan,
        )
    assert effects == []

    with pytest.raises(ValueError, match="exact H6-Prediction v3"):
        execute_h6_test_transaction_v3(
            config=legacy_config,
            readiness=object(),
            plan=object(),
            validation_bundle=object(),
            store=object(),
            journal_root=absent_publication,
            score_inventory=lambda *_args: effects.append("score"),
        )
    assert effects == []
    assert not absent_publication.exists()

    with pytest.raises(ValueError, match="exact resolved v3 config"):
        run_h6_experiment_v3(
            operation="plan",
            config=legacy_config,
            runtime=None,
            operation_config=_ExplodingMapping(),
            authorization_sha256=_digest("irrelevant"),
        )
    assert effects == []

    for name in (
        "read_h6_prediction_v3_authorities",
        "read_h6_validation_bundle_v3",
        "read_h6_test_reservation_v3",
        "read_h6_prediction_result_v3",
        "read_h6_test_terminal_v3",
        "read_h6_prediction_pointer_v3",
    ):
        monkeypatch.setattr(
            h8_adapter,
            name,
            lambda *_args, _name=name, **_kwargs: effects.append(_name),
        )
    monkeypatch.setattr(
        h8_adapter,
        "_validate_manifest_digest",
        lambda *_args, **_kwargs: effects.append("manifest"),
    )
    with pytest.raises(ValueError, match="exact H6-Prediction v3 reference"):
        h8_adapter.validate_h8_h6_prediction_v3_reference(
            legacy_h8,  # type: ignore[arg-type]
            expected_a0_direct_exact_prefix_certificate_sha256=(
                _DIRECT_CERTIFICATE_SHA256
            ),
        )
    assert effects == []
