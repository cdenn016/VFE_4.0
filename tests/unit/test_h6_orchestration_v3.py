from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import test_h6_readiness_v3 as readiness_fixtures
from vfe4.artifacts.h6_prediction_v3 import H6PredictionV3Authorities
from vfe4.training.h6_experiment_v3 import plan_h6_experiment_v3
from vfe4.training.h6_readiness import (
    _derive_h6_prediction_readiness_v3 as validate_h6_prediction_readiness_v3,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _operation_config(tmp_path: Path) -> dict[str, object]:
    return {
        "scientific_config": {"complete": True},
        "correctness_artifact_roots": {
            gate: (tmp_path / gate.lower()).as_posix()
            for gate in ("H1", "H2", "H3", "H5")
        },
        "h1_prefix_prior_artifact_root": (
            tmp_path / "h1-prefix-prior"
        ).as_posix(),
        "smc_accuracy_artifact_root": (
            tmp_path / "smc-accuracy"
        ).as_posix(),
        "h6_prefix_artifact_root": (tmp_path / "prefix").as_posix(),
        "h6_prefix_manifest_sha256": _sha("prefix-manifest"),
        "h6_prefix_junit_sha256": _sha("prefix-junit"),
        "blinded_store_manifest_path": (
            tmp_path / "store" / "authenticated_blinded_store_v3.json"
        ).as_posix(),
        "blinded_store_artifact_root": (tmp_path / "store").as_posix(),
        "authorities_run_root": (tmp_path / "runs").as_posix(),
        "authorities_run_name": "AUTHORITIES",
        "authorities_directory": (
            tmp_path / "runs" / "AUTHORITIES"
        ).as_posix(),
        "planned_attempt_sha256": "0" * 64,
        "checkpoint_path": (tmp_path / "checkpoints" / "attempt.h6v3").as_posix(),
        "maximum_checkpoint_bytes": 1_073_741_824,
        "validation_bundle_directory": (
            tmp_path / "runs" / "VALIDATION"
        ).as_posix(),
        "transaction_pointer_root": (tmp_path / "runs" / "POINTERS").as_posix(),
        "transaction_pointer_name": "current",
    }


@pytest.fixture(scope="module")
def authorities(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[H6PredictionV3Authorities]:
    matching_set = readiness_fixtures._matching_set()
    config = readiness_fixtures._config(
        matching_set=matching_set,
        artifact_root=tmp_path_factory.mktemp("h6-orchestration-authorities"),
    )
    readiness = validate_h6_prediction_readiness_v3(
        config=config,
        matching_set=matching_set,
        git_head=readiness_fixtures._GIT_HEAD,
        dirty_digest=readiness_fixtures._DIRTY_DIGEST,
    )
    plan = plan_h6_experiment_v3(
        readiness=readiness,
        matching_set=matching_set,
        training_schedule=config.training_schedule,
        runtime_identity=config.runtime,
    )
    yield H6PredictionV3Authorities.create(
        config=config,
        matching_set=matching_set,
        readiness=readiness,
        plan=plan,
    )


def test_operation_paths_are_explicit_absolute_and_callback_free(
    tmp_path: Path,
) -> None:
    from vfe4.training.h6_orchestration_v3 import H6OperationPathsV3

    raw = _operation_config(tmp_path)
    parsed = H6OperationPathsV3.from_mapping(raw)

    assert parsed.authorities_directory == tmp_path / "runs" / "AUTHORITIES"
    assert parsed.checkpoint_path == tmp_path / "checkpoints" / "attempt.h6v3"
    assert parsed.checkpoint_catalog_root == tmp_path / "checkpoints" / "CATALOG"
    assert parsed.planned_attempt_sha256 == "0" * 64

    with pytest.raises(ValueError, match="unknown|inventory"):
        H6OperationPathsV3.from_mapping(
            raw | {"score_inventory": lambda: None}
        )
    relative = dict(raw)
    relative["checkpoint_path"] = "checkpoints/attempt.h6v3"
    with pytest.raises(ValueError, match="absolute"):
        H6OperationPathsV3.from_mapping(relative)


def test_readiness_operation_reopens_store_publishes_and_reopens_authorities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authorities: H6PredictionV3Authorities,
) -> None:
    import vfe4.training.h6_orchestration_v3 as orchestration

    events: list[str] = []
    store = SimpleNamespace(data_identity_sha256=authorities.config.data_identity_sha256)
    paths = _operation_config(tmp_path)

    monkeypatch.setattr(
        orchestration,
        "_reopen_store_for_config",
        lambda **kwargs: events.append("store") or store,
    )
    monkeypatch.setattr(
        orchestration,
        "_matching_from_store",
        lambda **kwargs: events.append("matching") or authorities.matching_set,
    )
    prerequisite_evidence = object()
    monkeypatch.setattr(
        orchestration,
        "reopen_h6_prediction_v3_prerequisite_evidence",
        lambda **kwargs: (
            events.append("prerequisites") or prerequisite_evidence
        ),
    )
    monkeypatch.setattr(
        orchestration,
        "validate_h6_prediction_readiness_v3",
        lambda **kwargs: (
            events.append("readiness")
            or (
                authorities.readiness
                if kwargs["prerequisite_evidence"] is prerequisite_evidence
                else (_ for _ in ()).throw(AssertionError("evidence drift"))
            )
        ),
    )
    monkeypatch.setattr(
        orchestration,
        "plan_h6_experiment_v3",
        lambda **kwargs: events.append("plan") or authorities.plan,
    )
    monkeypatch.setattr(
        orchestration,
        "publish_h6_prediction_v3_authorities",
        lambda **kwargs: (
            events.append("publish")
            or Path(paths["authorities_directory"])
        ),
    )
    monkeypatch.setattr(
        orchestration,
        "read_h6_prediction_v3_authorities",
        lambda *args, **kwargs: events.append("reopen") or authorities,
    )

    result = orchestration.run_h6_experiment_v3(
        operation="prediction_readiness",
        config=authorities.config,
        runtime=None,
        operation_config=paths,
        authorization_sha256=_sha(
            "AUTHORIZE_VFE4_H6_PREDICTION_READINESS_V1"
        ),
    )

    assert result is authorities
    assert events == [
        "prerequisites",
        "store",
        "matching",
        "readiness",
        "plan",
        "publish",
        "reopen",
    ]


def test_readiness_refuses_bad_prefix_before_authority_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authorities: H6PredictionV3Authorities,
) -> None:
    import vfe4.training.h6_orchestration_v3 as orchestration

    published: list[object] = []
    monkeypatch.setattr(
        orchestration,
        "_reopen_store_for_config",
        lambda **kwargs: SimpleNamespace(
            data_identity_sha256=authorities.config.data_identity_sha256
        ),
    )
    monkeypatch.setattr(
        orchestration,
        "_matching_from_store",
        lambda **kwargs: authorities.matching_set,
    )
    monkeypatch.setattr(
        orchestration,
        "reopen_h6_prediction_v3_prerequisite_evidence",
        lambda **kwargs: (_ for _ in ()).throw(
            FileNotFoundError("bounded Prefix artifact is absent")
        ),
    )
    monkeypatch.setattr(
        orchestration,
        "publish_h6_prediction_v3_authorities",
        lambda **kwargs: published.append(kwargs),
    )

    with pytest.raises(FileNotFoundError, match="Prefix artifact is absent"):
        orchestration.run_h6_experiment_v3(
            operation="prediction_readiness",
            config=authorities.config,
            runtime=None,
            operation_config=_operation_config(tmp_path),
            authorization_sha256=_sha(
                "AUTHORIZE_VFE4_H6_PREDICTION_READINESS_V1"
            ),
        )
    assert published == []


def test_train_publishes_and_reopens_exact_catalog_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authorities: H6PredictionV3Authorities,
) -> None:
    import vfe4.training.h6_checkpoint_catalog_v3 as catalog_module
    import vfe4.training.h6_orchestration_v3 as orchestration
    import vfe4.training.h6_training_attempt_v3 as training_module

    attempt = authorities.plan.tuning_attempts[0]
    paths = _operation_config(tmp_path)
    paths["planned_attempt_sha256"] = attempt.planned_attempt_sha256
    executable = SimpleNamespace(
        planned_attempt=attempt,
        executable_attempt_sha256=_sha("executable"),
    )
    checkpoint = SimpleNamespace(
        checkpoint_sha256=_sha("checkpoint"),
        to_bytes=lambda: b"checkpoint",
    )
    training_result = SimpleNamespace(
        stage="tuning",
        planned_attempt_sha256=attempt.planned_attempt_sha256,
        executable_attempt=executable,
        terminal_checkpoint=checkpoint,
        checkpoint_path=Path(paths["checkpoint_path"]),
    )
    catalog_item = SimpleNamespace(
        executable_attempt=executable,
        checkpoint=checkpoint,
        entry=SimpleNamespace(
            planned_attempt_sha256=attempt.planned_attempt_sha256,
        ),
    )
    events: list[tuple[str, object]] = []

    monkeypatch.setattr(
        orchestration,
        "_reopen_store_for_config",
        lambda **kwargs: SimpleNamespace(
            data_identity_sha256=authorities.readiness.data_identity_sha256
        ),
    )
    monkeypatch.setattr(
        orchestration,
        "_reopen_authorities",
        lambda **kwargs: authorities,
    )
    prefix_source = SimpleNamespace(
        source_sha256=authorities.config.source.source_sha256
    )
    monkeypatch.setattr(
        orchestration,
        "read_h6_prefix_authorities_for_scoring_v3",
        lambda *args, **kwargs: (prefix_source, prefix_source),
    )
    monkeypatch.setattr(
        training_module,
        "run_h6_training_attempt_v3",
        lambda **kwargs: events.append(("train", kwargs)) or training_result,
    )
    monkeypatch.setattr(
        catalog_module,
        "publish_h6_checkpoint_catalog_entry_v3",
        lambda **kwargs: (
            events.append(("publish", kwargs))
            or Path(paths["checkpoint_path"]).parent
            / "CATALOG"
            / attempt.planned_attempt_sha256
        ),
    )
    monkeypatch.setattr(
        catalog_module,
        "read_h6_checkpoint_catalog_v3",
        lambda *args, **kwargs: (
            events.append(("reopen_catalog", (args, kwargs)))
            or SimpleNamespace(items=(catalog_item,))
        ),
    )

    result = orchestration.run_h6_experiment_v3(
        operation="train",
        config=authorities.config,
        runtime=object(),
        operation_config=paths,
        authorization_sha256=_sha("AUTHORIZE_VFE4_H6_TRAINING_V1"),
    )

    assert result is training_result
    publish = dict(events)["publish"]
    assert publish["catalog_root"] == tmp_path / "checkpoints" / "CATALOG"
    assert publish["executable_attempt"] is executable
    assert publish["checkpoint"] is checkpoint
    reopen_args, reopen_kwargs = dict(events)["reopen_catalog"]
    assert reopen_args == (tmp_path / "checkpoints" / "CATALOG",)
    assert reopen_kwargs["required_inventory"] == "partial"
    assert reopen_kwargs["tuning_selection"] is None


def test_validation_consumes_deterministic_checkpoint_catalog_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authorities: H6PredictionV3Authorities,
) -> None:
    import vfe4.training.h6_orchestration_v3 as orchestration
    import vfe4.training.h6_validation_campaign_v3 as campaign

    store = SimpleNamespace(
        data_identity_sha256=authorities.readiness.data_identity_sha256
    )
    paths = _operation_config(tmp_path)
    observed: dict[str, object] = {}
    expected = object()
    monkeypatch.setattr(
        orchestration,
        "_reopen_store_for_config",
        lambda **kwargs: store,
    )
    monkeypatch.setattr(
        orchestration,
        "_reopen_authorities",
        lambda **kwargs: authorities,
    )
    prefix_source = SimpleNamespace(
        source_sha256=authorities.config.source.source_sha256
    )
    monkeypatch.setattr(
        orchestration,
        "read_h6_prefix_authorities_for_scoring_v3",
        lambda *args, **kwargs: (prefix_source, prefix_source),
    )
    monkeypatch.setattr(
        campaign,
        "run_h6_validation_campaign_v3",
        lambda **kwargs: observed.update(kwargs) or expected,
    )

    result = orchestration.run_h6_experiment_v3(
        operation="score_validation",
        config=authorities.config,
        runtime=None,
        operation_config=paths,
        authorization_sha256=_sha(
            "AUTHORIZE_VFE4_H6_VALIDATION_SCORING_V1"
        ),
    )

    assert result is expected
    assert observed["checkpoint_catalog_root"] == (
        tmp_path / "checkpoints" / "CATALOG"
    )


def test_prepare_test_transaction_binds_complete_catalog_and_canonical_scorer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authorities: H6PredictionV3Authorities,
) -> None:
    import vfe4.artifacts.h6_prediction_v3 as prediction_artifacts
    import vfe4.training.h6_checkpoint_catalog_v3 as catalog_module
    import vfe4.training.h6_heldout_scoring_v3 as heldout
    import vfe4.training.h6_orchestration_v3 as orchestration
    import vfe4.training.h6_readiness as readiness_module
    import vfe4.training.h6_validation_campaign_v3 as campaign
    import vfe4.training.h6_validation_v3 as validation
    from vfe4.training.h6_experiment_v3 import (
        H6_CONFIRMATORY_SEEDS_V3,
        prepare_h6_test_transaction_v3,
    )
    from vfe4.training.h6_matching_v3 import (
        H6_MATCHING_V3_ENDPOINT_CONFIG_IDS,
    )

    paths = _operation_config(tmp_path)
    store = SimpleNamespace(
        data_identity_sha256=authorities.readiness.data_identity_sha256
    )
    tuning_selection = SimpleNamespace(
        tuning_selection_sha256=_sha("tuning-selection")
    )
    items = tuple(
        SimpleNamespace(
            executable_attempt=SimpleNamespace(planned_attempt=attempt),
            checkpoint=SimpleNamespace(
                checkpoint_sha256=_sha(
                    f"checkpoint:{attempt.planned_attempt_sha256}"
                )
            ),
        )
        for attempt in authorities.plan.confirmatory_attempts
    )
    candidates = tuple(
        SimpleNamespace(
            endpoint_config_id=attempt.endpoint_config_id,
            training_seed=attempt.training_seed,
            checkpoint_sha256=item.checkpoint.checkpoint_sha256,
        )
        for attempt, item in zip(
            authorities.plan.confirmatory_attempts,
            items,
            strict=True,
        )
    )
    checkpoint_selection = SimpleNamespace(
        checkpoints=candidates,
        checkpoint_selection_sha256=_sha("checkpoint-selection"),
    )
    expected_bundle = SimpleNamespace(
        checkpoint_selection=checkpoint_selection,
        validation_bundle_sha256=_sha("validation-bundle"),
    )
    catalog_calls: list[tuple[object, object]] = []
    built: list[tuple[str, int, str]] = []
    scorer_calls: list[dict[str, object]] = []
    prefix_certificate_set = SimpleNamespace(
        source_sha256=authorities.config.source.source_sha256
    )
    direct_certificate = SimpleNamespace(
        source_sha256=authorities.config.source.source_sha256
    )

    monkeypatch.setattr(
        orchestration,
        "_reopen_store_for_config",
        lambda **kwargs: store,
    )
    monkeypatch.setattr(
        orchestration,
        "_reopen_authorities",
        lambda **kwargs: authorities,
    )
    monkeypatch.setattr(
        campaign,
        "read_h6_tuning_selection_v3",
        lambda *args, **kwargs: tuning_selection,
    )
    monkeypatch.setattr(
        catalog_module,
        "read_h6_checkpoint_catalog_v3",
        lambda *args, **kwargs: (
            catalog_calls.append((args, kwargs))
            or SimpleNamespace(confirmatory_items=items)
        ),
    )
    monkeypatch.setattr(
        prediction_artifacts,
        "bind_h6_checkpoint_selection_v3",
        lambda *args, **kwargs: checkpoint_selection,
    )
    monkeypatch.setattr(
        prediction_artifacts,
        "H6ValidationBundleV3",
        SimpleNamespace(
            create=lambda **kwargs: expected_bundle,
        ),
    )
    monkeypatch.setattr(
        prediction_artifacts,
        "read_h6_validation_bundle_v3",
        lambda *args, **kwargs: expected_bundle,
    )
    monkeypatch.setattr(
        validation,
        "build_h6_evaluation_arm_v3",
        lambda checkpoint, *, plan, planned_attempt, evaluation_role: (
            built.append(
                (
                    planned_attempt.endpoint_config_id,
                    planned_attempt.training_seed,
                    evaluation_role,
                )
            )
            or SimpleNamespace(
                endpoint_config_id=planned_attempt.endpoint_config_id,
                training_seed=planned_attempt.training_seed,
            )
        ),
    )
    monkeypatch.setattr(
        heldout,
        "H6HeldoutCheckpointArmV3",
        lambda *, candidate, evaluation: SimpleNamespace(
            candidate=candidate,
            evaluation=evaluation,
        ),
    )
    inventory = object()
    monkeypatch.setattr(
        heldout,
        "score_h6_heldout_inventory_v3",
        lambda **kwargs: scorer_calls.append(kwargs) or inventory,
    )
    prefix_calls: list[tuple[object, object]] = []
    monkeypatch.setattr(
        readiness_module,
        "read_h6_prefix_authorities_for_scoring_v3",
        lambda root, **kwargs: (
            prefix_calls.append((root, kwargs))
            or (prefix_certificate_set, direct_certificate)
        ),
    )

    prepared = prepare_h6_test_transaction_v3(
        config=authorities.config,
        operation_config=paths,
        authorization_sha256=_sha(
            "AUTHORIZE_VFE4_H6_ONE_TIME_TEST_TRANSACTION_V1"
        ),
    )

    assert frozenset(prepared) == {
        "config",
        "readiness",
        "plan",
        "validation_bundle",
        "store",
        "journal_root",
        "score_inventory",
        "experiment_identity",
        "journal_name",
        "pointer_root",
        "pointer_name",
    }
    assert prepared["experiment_identity"].analysis_sha256 == (
        authorities.config.source.source_sha256
    )
    assert prepared["experiment_identity"].analysis_sha256 != (
        authorities.config.scoring_inventory_sha256
    )
    with pytest.raises(ValueError, match="experiment_identity_sha256"):
        replace(
            prepared["experiment_identity"],
            analysis_sha256=authorities.config.scoring_inventory_sha256,
        )
    assert catalog_calls[0][0] == (
        tmp_path / "checkpoints" / "CATALOG",
    )
    assert catalog_calls[0][1]["required_inventory"] == "complete"
    assert prefix_calls == [
        (
            tmp_path / "prefix",
            {
                "expected_manifest_sha256": _sha("prefix-manifest"),
                "expected_junit_sha256": _sha("prefix-junit"),
                "readiness": authorities.readiness,
            },
        )
    ]
    expected_keys = tuple(
        (endpoint_id, seed, "heldout")
        for endpoint_id in (
            H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[0],
            H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[5],
            H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[9],
        )
        for seed in H6_CONFIRMATORY_SEEDS_V3
    )
    assert tuple(built) == expected_keys
    windows = object()
    opening_proof_sha256 = _sha("opening-proof")
    assert prepared["score_inventory"](
        windows,
        opening_proof_sha256,
    ) is inventory
    assert scorer_calls == [
        {
            "windows": windows,
            "opening_proof_sha256": opening_proof_sha256,
            "checkpoint_arms": tuple(
                SimpleNamespace(
                    candidate=candidate,
                    evaluation=SimpleNamespace(
                        endpoint_config_id=endpoint_id,
                        training_seed=seed,
                    ),
                )
                for endpoint_id in (
                    H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[0],
                    H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[5],
                    H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[9],
                )
                for seed in H6_CONFIRMATORY_SEEDS_V3
                for candidate in (
                    next(
                        item
                        for item in candidates
                        if (
                            item.endpoint_config_id,
                            item.training_seed,
                        )
                        == (endpoint_id, seed)
                    ),
                )
            ),
            "prefix_certificate_set": prefix_certificate_set,
            "a0_direct_exact_prefix_certificate": direct_certificate,
            "readiness": authorities.readiness,
        }
    ]


def test_test_transaction_refuses_uncertified_direct_a0_before_reopening_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authorities: H6PredictionV3Authorities,
) -> None:
    import vfe4.training.h6_orchestration_v3 as orchestration
    import vfe4.training.h6_readiness as readiness_module
    from vfe4.training.h6_experiment_v3 import prepare_h6_test_transaction_v3

    reopened: list[object] = []
    monkeypatch.setattr(
        orchestration,
        "_reopen_authorities",
        lambda **kwargs: authorities,
    )
    monkeypatch.setattr(
        orchestration,
        "_reopen_store_for_config",
        lambda **kwargs: reopened.append(kwargs),
    )
    monkeypatch.setattr(
        readiness_module,
        "read_h6_prefix_authorities_for_scoring_v3",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("direct-A0 Prefix certificate is not exact current PASS")
        ),
    )
    with pytest.raises(
        ValueError,
        match="direct-A0 Prefix certificate.*PASS",
    ):
        prepare_h6_test_transaction_v3(
            config=authorities.config,
            operation_config=_operation_config(tmp_path),
            authorization_sha256=_sha(
                "AUTHORIZE_VFE4_H6_ONE_TIME_TEST_TRANSACTION_V1"
            ),
        )
    assert reopened == []


def test_paired_prefix_reader_uses_explicit_digests_and_readiness_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authorities: H6PredictionV3Authorities,
) -> None:
    import vfe4.training.h6_readiness as readiness_module
    observed: dict[str, object] = {}
    certificate_set = object()
    direct_certificate = object()

    def reopen(root: Path, **kwargs: object) -> tuple[object, object]:
        observed["root"] = root
        observed.update(kwargs)
        return certificate_set, direct_certificate

    monkeypatch.setattr(
        readiness_module,
        "_reopen_h6_prefix_authorities_v3",
        reopen,
    )
    root = (tmp_path / "prefix").resolve()
    readiness = authorities.readiness

    assert (
        readiness_module.read_h6_prefix_authorities_for_scoring_v3(
            root,
            expected_manifest_sha256=_sha("manifest"),
            expected_junit_sha256=_sha("junit"),
            readiness=readiness,
        )
        == (certificate_set, direct_certificate)
    )
    assert observed == {
        "root": root,
        "expected_manifest_sha256": _sha("manifest"),
        "expected_junit_sha256": _sha("junit"),
        "expected_set_sha256": readiness.prefix_certificate_set_sha256,
        "expected_direct_certificate_sha256": (
            readiness.a0_direct_exact_prefix_certificate_sha256
        ),
        "expected_git_head": readiness.git_head,
        "expected_dirty_digest": readiness.dirty_digest,
        "expected_source_sha256": hashlib.sha256(
            b"VFE4-H6-SOURCE-CANDIDATE-V1\x00"
            + bytes.fromhex(readiness.git_head)
            + bytes.fromhex(readiness.dirty_digest)
        ).hexdigest(),
    }
