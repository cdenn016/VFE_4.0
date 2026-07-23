from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from vfe4.data.access import (
    OpeningCapabilityError,
    materialize_prediction_train,
    materialize_validation_safety_fixture,
    open_test_for_scoring,
    reserve_and_issue_durable_test_opening_capability,
    validate_durable_test_opening_capability,
)
from vfe4.data.wikitext2 import (
    ACCESS_POLICY_SHA256,
    WIKITEXT2_RAW_URL,
    H6DataAcquisitionRequest,
    _acquire_wikitext2_blinded,
)
from vfe4.config import H6ArchiveMemberExpectation, H6DataConfig, H6ObservedArchive
from vfe4.types import GateStatus
from vfe4.types.h6 import (
    H6_PREFIX_REQUIRED_CHECKS,
    ArmId,
    DataIdentity,
    DurableTestOpeningCapability,
    EndpointSmcProtocol,
    EvidenceStatus,
    ExperimentIdentity,
    H5UpdateBinding,
    H6ArmPhaseSchedule,
    H6OuterSchedule,
    H6TrainingSchedule,
    H1PrefixPriorArtifactRef,
    PredictionCorrectnessArtifactRef,
    PrefixCaseKey,
    PrefixCertificate,
    SmcAccuracyArtifactRef,
    TrainingPhase,
    canonical_json_bytes,
    issue_prediction_readiness,
)


GIT_HEAD = "1" * 40
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
MEMBERS = (
    "wikitext-2-raw/wiki.train.raw",
    "wikitext-2-raw/wiki.valid.raw",
    "wikitext-2-raw/wiki.test.raw",
)


def _archive_bytes(*, test_payload: bytes = b"test bytes") -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        directory = zipfile.ZipInfo("wikitext-2-raw/")
        directory.external_attr = (0o40755 << 16) | 0x10
        archive.writestr(directory, b"")
        archive.writestr(MEMBERS[0], b"train bytes" * 10)
        archive.writestr(MEMBERS[1], bytes(range(256)) * 513)
        archive.writestr(MEMBERS[2], test_payload)
    return stream.getvalue()


def _config(archive_bytes: bytes, artifact_root: Path) -> H6DataAcquisitionRequest:
    with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as archive:
        expected_members = tuple(
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
            ("wikitext-2-raw/", *MEMBERS),
            (0, 8),
            16_777_216,
            33_554_432,
            100,
            H6ObservedArchive(
                len(archive_bytes),
                hashlib.sha256(archive_bytes).hexdigest(),
                expected_members,
            ),
        ),
        artifact_root=artifact_root,
    )


def _store(tmp_path: Path, *, test_payload: bytes = b"test bytes"):
    archive_bytes = _archive_bytes(test_payload=test_payload)
    return _acquire_wikitext2_blinded(
        _config(archive_bytes, tmp_path),
        lambda _: io.BytesIO(archive_bytes),
    )


def _manifest(entries: dict[str, bytes]) -> bytes:
    return "".join(
        f"{hashlib.sha256(entries[path]).hexdigest()}  {path}\n"
        for path in sorted(entries)
    ).encode("ascii")


def _producer_payload(
    *, gate: str, config: bytes, extra: dict[str, object] | None = None
) -> bytes:
    return canonical_json_bytes(
        {
            "gate": gate,
            "status": GateStatus.PASS.value,
            "obligations": (),
            "git_head": GIT_HEAD,
            "dirty_digest": SHA_A,
            "config_sha256": hashlib.sha256(config).hexdigest(),
            **({} if extra is None else extra),
        }
    )


def _correctness(gate: str) -> PredictionCorrectnessArtifactRef:
    config = f"config-{gate}".encode()
    path = Path(f"validation/{gate.lower()}.json")
    payload = _producer_payload(gate=gate, config=config)
    return PredictionCorrectnessArtifactRef.from_bytes(
        gate=gate,  # type: ignore[arg-type]
        artifact_path=path,
        manifest_bytes=_manifest({"config.json": config, path.as_posix(): payload}),
        git_head=GIT_HEAD,
        dirty_digest=SHA_A,
        config_bytes=config,
        validation_payload_bytes=payload,
    )


def _h1_prefix() -> H1PrefixPriorArtifactRef:
    config = b"h1-prefix-config"
    schema = b"generative-schema"
    path = Path("validation/h1_prefix_prior.json")
    payload = _producer_payload(
        gate="H1-Prefix-Prior",
        config=config,
        extra={"generative_factor_schema_sha256": hashlib.sha256(schema).hexdigest()},
    )
    return H1PrefixPriorArtifactRef.from_bytes(
        artifact_path=path,
        manifest_bytes=_manifest(
            {
                "config.json": config,
                "schemas/generative_factor.json": schema,
                path.as_posix(): payload,
            }
        ),
        git_head=GIT_HEAD,
        dirty_digest=SHA_A,
        generative_factor_schema_bytes=schema,
        config_bytes=config,
        validation_payload_bytes=payload,
    )


def _smc() -> SmcAccuracyArtifactRef:
    config = b"smc-config"
    estimator = b"estimator"
    fixtures = b"fixture-set"
    path = Path("validation/h6_smc_accuracy.json")
    payload = _producer_payload(
        gate="H6-SMC-Accuracy",
        config=config,
        extra={
            "estimator_sha256": hashlib.sha256(estimator).hexdigest(),
            "fixture_set_sha256": hashlib.sha256(fixtures).hexdigest(),
        },
    )
    return SmcAccuracyArtifactRef.from_bytes(
        artifact_path=path,
        manifest_bytes=_manifest(
            {
                "config.json": config,
                "protocol/estimator.json": estimator,
                "fixtures/finite_smc.json": fixtures,
                path.as_posix(): payload,
            }
        ),
        git_head=GIT_HEAD,
        dirty_digest=SHA_A,
        estimator_preimage_bytes=estimator,
        fixture_set_bytes=fixtures,
        config_bytes=config,
        validation_payload_bytes=payload,
    )


def _readiness(data: DataIdentity):
    key = PrefixCaseKey(
        arm=ArmId.A5,
        predictor_config_sha256=SHA_A,
        estimator_sha256=SHA_B,
        model_family_sha256=SHA_C,
        vocabulary_sha256=SHA_D,
        data_safety_sha256="e" * 64,
        git_head=GIT_HEAD,
        dirty_digest=SHA_A,
    )
    certificate = PrefixCertificate.create(
        key=key,
        status=EvidenceStatus.PASS,
        checks={name: True for name in H6_PREFIX_REQUIRED_CHECKS},
        obligations=(),
    )
    h5_binding = H5UpdateBinding.from_producer_preimages(
        producer_preimages={
            "h5_manifest_sha256": b"manifest",
            "h5_payload_sha256": b"payload",
            "update_spec_raw_sha256": b"raw-update",
            "update_spec_canonical_sha256": b"canonical-update",
            "objective_schema_sha256": b"objective",
            "factor_input_schema_sha256": b"factor-input",
            "reference_sha256": b"reference",
            "recognition_state_sha256": b"recognition",
            "model_state_sha256": b"model",
            "validation_payload_sha256": b"validation",
        },
        enabled_update_labels=(
            "exact_coordinate",
            "generalized_em",
            "natural_gradient_proposal",
        ),
    )
    outer = H6OuterSchedule.create(optimizer_policy_sha256=SHA_A)
    phase = H6ArmPhaseSchedule.create(
        endpoint_config_sha256=SHA_B,
        latent_enabled=False,
        phases=(TrainingPhase.MODEL_CE_ADAMW,),
    )
    schedule = H6TrainingSchedule.create(outer=outer, endpoint_phases=(phase,))
    endpoint = EndpointSmcProtocol.create(
        particle_counts=(128, 256, 512, 1024),
        replicate_count=64,
        registry_root_seed=2026072198,
        common_stream_domain="h6-wt2-endpoint-mc-v1",
        simultaneous_interval_count=352,
        familywise_alpha=0.01,
        critical_value_df63=4.5144904535377144,
        remainder_contraction=0.75,
    )
    return issue_prediction_readiness(
        git_head=GIT_HEAD,
        dirty_digest=SHA_A,
        experiment_config_sha256=SHA_B,
        correctness_artifacts=tuple(_correctness(g) for g in ("H1", "H2", "H3", "H5")),
        h1_prefix_prior_artifact=_h1_prefix(),
        h5_update_binding=h5_binding,
        h6_training_schedule=schedule,
        smc_accuracy_artifact=_smc(),
        critical_values_sha256="0" * 64,
        endpoint_smc_protocol=endpoint,
        attribution_matrix_sha256="2" * 64,
        matching_set_sha256="3" * 64,
        prefix_certificates={key: certificate},
        data_identity=data,
    )


def _experiment(store) -> ExperimentIdentity:
    return ExperimentIdentity.create(
        checkpoint_set_sha256=SHA_A,
        current_candidate_sha256=SHA_B,
        sealed_data_sha256=store.data_identity_sha256,
        access_policy_sha256=store.data_identity.access_policy_sha256,
        analysis_sha256=SHA_C,
        stream_protocol_sha256=SHA_D,
    )


def _synthetic_reservation(marker_anchor: Path, store) -> Path:
    marker_sha256 = hashlib.sha256(
        b"VFE4-H6-TEST-OPENING-MARKER-V1\x00"
        + bytes.fromhex(store.data_identity_sha256)
        + bytes.fromhex(store.data_identity.access_policy_sha256)
    ).hexdigest()
    return (
        marker_anchor
        / ".vfe4-h6-synthetic-opening-reservations"
        / (marker_sha256 + ".reservation.bin")
    )


def test_pre_readiness_surface_exposes_only_frozen_validation_fixture(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)

    fixture = materialize_validation_safety_fixture(store)

    assert fixture is store.frozen_validation_fixture
    assert not hasattr(fixture, "train")
    for forbidden in (
        "_directory",
        "_split_paths",
        "_read_split",
        "_opening_validator",
    ):
        assert not hasattr(store, forbidden)
    import vfe4.data.wikitext2 as wikitext2

    for forbidden in (
        "_TRAIN_MAPPING_TOKEN",
        "_TEST_MAPPING_TOKEN",
        "_read_prediction_splits",
        "_read_test_split",
    ):
        assert not hasattr(wikitext2, forbidden)
    import vfe4.data.access as access

    for forbidden in (
        "_STORE_ACCESS_REGISTRY",
        "_TRAIN_READ_AUTHORITY",
        "_TEST_READ_AUTHORITY",
        "_ISSUER_TOKEN",
        "_REGISTRATION_TOKEN",
        "_StoreAccessState",
        "_RegisteredOpeningProof",
        "_OpeningProofValidator",
        "_state_for_store",
        "_build_access_api",
    ):
        assert not hasattr(access, forbidden)
    with pytest.raises((TypeError, ValueError, OpeningCapabilityError)):
        materialize_prediction_train(store, object())  # type: ignore[arg-type]
    with pytest.raises((TypeError, ValueError, OpeningCapabilityError)):
        open_test_for_scoring(store, "forged")  # type: ignore[arg-type]


def test_pass_readiness_materializes_only_train_and_validation(tmp_path: Path) -> None:
    store = _store(tmp_path)
    readiness = _readiness(store.data_identity)

    materialized = materialize_prediction_train(store, readiness)

    assert materialized.train.split == "train"
    assert materialized.validation.split == "validation"
    assert not hasattr(materialized, "test")
    assert materialized.schedule_for_pass(0).window_count == len(materialized.train)
    other = _store(tmp_path / "other")
    with pytest.raises(OpeningCapabilityError, match="data identity"):
        materialize_prediction_train(other, readiness)


def test_durable_opening_proof_is_exact_opaque_and_one_shot(tmp_path: Path) -> None:
    test_payload = b"held-out\r\ntest"
    store = _store(tmp_path / "store", test_payload=test_payload)
    readiness = _readiness(store.data_identity)
    experiment = _experiment(store)
    reservation = _synthetic_reservation(tmp_path, store)

    opening = reserve_and_issue_durable_test_opening_capability(
        store=store,
        readiness=readiness,
        experiment_identity=experiment,
    )

    normalized = str(reservation.resolve(strict=False)).replace("\\", "/").encode("utf-8")
    expected = (
        b"VFE4-H6-DURABLE-TEST-OPENING-PROOF-V1\x00"
        + len(normalized).to_bytes(4, "little")
        + normalized
        + bytes.fromhex(readiness.readiness_sha256)
        + bytes.fromhex(experiment.experiment_identity_sha256)
        + bytes.fromhex(store.data_identity_sha256)
        + bytes.fromhex(store.sealed_test_handle.sealed_content_sha256)
        + bytes.fromhex(store.data_identity.access_policy_sha256)
        + b"RESERVED\x00"
    )
    assert reservation.read_bytes() == expected
    assert opening.proof_identity_sha256 == hashlib.sha256(expected).hexdigest()
    with pytest.raises(TypeError):
        DurableTestOpeningCapability()  # type: ignore[misc]
    with pytest.raises(TypeError):
        reserve_and_issue_durable_test_opening_capability(  # type: ignore[call-arg]
            store=store,
            readiness=readiness,
            experiment_identity=experiment,
            reservation_path=tmp_path / "alternate.bin",
        )
    alternate_experiment = ExperimentIdentity.create(
        checkpoint_set_sha256=SHA_A,
        current_candidate_sha256=SHA_B,
        sealed_data_sha256=store.data_identity_sha256,
        access_policy_sha256=store.data_identity.access_policy_sha256,
        analysis_sha256="e" * 64,
        stream_protocol_sha256=SHA_D,
    )
    with pytest.raises(FileExistsError):
        reserve_and_issue_durable_test_opening_capability(
            store=store,
            readiness=readiness,
            experiment_identity=alternate_experiment,
        )

    test_windows = open_test_for_scoring(store, opening)
    assert test_windows.split == "test"
    assert test_windows.counted_target_total == len(test_payload) + 1
    with pytest.raises(OpeningCapabilityError, match="consumed"):
        validate_durable_test_opening_capability(store, opening)


def test_opening_rejects_reservation_or_identity_mutation_before_test_mapping(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "store")
    readiness = _readiness(store.data_identity)
    experiment = _experiment(store)
    reservation = _synthetic_reservation(tmp_path, store)
    opening = reserve_and_issue_durable_test_opening_capability(
        store=store,
        readiness=readiness,
        experiment_identity=experiment,
    )
    proof = reservation.read_bytes()
    for offset in (0, len(proof) // 2, len(proof) - 1):
        changed = bytearray(proof)
        changed[offset] ^= 1
        reservation.write_bytes(changed)
        with pytest.raises(OpeningCapabilityError):
            open_test_for_scoring(store, opening)
        reservation.write_bytes(proof)

    object.__setattr__(opening, "_proof_identity_sha256", "0" * 64)
    with pytest.raises(OpeningCapabilityError):
        open_test_for_scoring(store, opening)


def test_issuer_rejects_forged_readiness_experiment_and_existing_reservation(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "store")
    readiness = _readiness(store.data_identity)
    wrong_store = _store(tmp_path / "wrong", test_payload=b"different test bytes")
    experiment = _experiment(store)
    reservation = _synthetic_reservation(tmp_path, store)
    reservation.parent.mkdir(parents=True)
    reservation.write_bytes(b"do not replace")

    with pytest.raises(OpeningCapabilityError):
        reserve_and_issue_durable_test_opening_capability(
            store=store,
            readiness=_readiness(wrong_store.data_identity),
            experiment_identity=_experiment(store),
        )
    wrong_experiment = ExperimentIdentity.create(
        checkpoint_set_sha256=SHA_A,
        current_candidate_sha256=SHA_B,
        sealed_data_sha256=wrong_store.data_identity_sha256,
        access_policy_sha256=store.data_identity.access_policy_sha256,
        analysis_sha256=SHA_C,
        stream_protocol_sha256=SHA_D,
    )
    with pytest.raises(OpeningCapabilityError):
        reserve_and_issue_durable_test_opening_capability(
            store=store,
            readiness=readiness,
            experiment_identity=wrong_experiment,
        )
    with pytest.raises(FileExistsError):
        reserve_and_issue_durable_test_opening_capability(
            store=store,
            readiness=readiness,
            experiment_identity=experiment,
        )
    assert reservation.read_bytes() == b"do not replace"


def test_synthetic_sibling_artifact_roots_share_one_data_policy_marker(
    tmp_path: Path,
) -> None:
    first = _store(tmp_path / "first")
    second = _store(tmp_path / "second")
    first_readiness = _readiness(first.data_identity)
    second_readiness = _readiness(second.data_identity)
    first_experiment = _experiment(first)
    second_experiment = ExperimentIdentity.create(
        checkpoint_set_sha256=SHA_A,
        current_candidate_sha256=SHA_B,
        sealed_data_sha256=second.data_identity_sha256,
        access_policy_sha256=second.data_identity.access_policy_sha256,
        analysis_sha256="e" * 64,
        stream_protocol_sha256=SHA_D,
    )

    reserve_and_issue_durable_test_opening_capability(
        store=first,
        readiness=first_readiness,
        experiment_identity=first_experiment,
    )

    assert first.data_identity_sha256 == second.data_identity_sha256
    with pytest.raises(FileExistsError):
        reserve_and_issue_durable_test_opening_capability(
            store=second,
            readiness=second_readiness,
            experiment_identity=second_experiment,
        )
