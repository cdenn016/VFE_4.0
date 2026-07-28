from __future__ import annotations

import hashlib
import inspect
import os
import shutil
from functools import lru_cache
from pathlib import Path

import pytest

from vfe4.artifacts.atomic import ArtifactPublicationError
from vfe4.types.h6_prediction_v3 import H6_SCORING_INVENTORY_SHA256
from vfe4.training.h6_experiment_v3 import H6_CONFIRMATORY_SEEDS_V3
from vfe4.training.h6_matching_v3 import H6_MATCHING_V3_ENDPOINT_CONFIG_IDS


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _reservation(
    module: object,
    root: Path,
    *,
    pointer_root: Path | None = None,
    pointer_name: str = "current",
) -> object:
    return module.H6TestReservationV3.create(
        experiment_config_sha256=_sha("config"),
        readiness_sha256=_sha("readiness"),
        plan_sha256=_sha("plan"),
        experiment_identity_sha256=_sha("experiment"),
        data_identity_sha256=_sha("data"),
        sealed_test_sha256=_sha("sealed-test"),
        test_inventory_sha256=_sha("test-inventory"),
        access_policy_sha256=_sha("access-policy"),
        tuning_selection_sha256=_sha("tuning-selection"),
        checkpoint_selection_sha256=_sha("checkpoint-selection"),
        validation_bundle_sha256=_sha("validation-bundle"),
        scoring_inventory_sha256=H6_SCORING_INVENTORY_SHA256,
        expected_row_count=4104,
        result_root=root / "result",
        state_root=root / "state",
        pointer_root=pointer_root or root / "pointer",
        pointer_name=pointer_name,
    )


@lru_cache(maxsize=8)
def _result_records(
    reservation_sha256: str,
    opening_proof_sha256: str,
) -> tuple[object, object, object]:
    import vfe4.artifacts.h6_prediction_v3 as artifacts

    particle_counts = (128, 256, 512, 1024)
    exact = tuple(
        artifacts.H6ExactA0CorpusTotalV3.create(
            endpoint_config_id=H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[0],
            training_seed=seed,
            checkpoint_sha256=_sha(f"a0:{seed}"),
            counted_test_targets=37,
            exact_total_nll=1.0,
            opening_proof_sha256=opening_proof_sha256,
        )
        for seed in H6_CONFIRMATORY_SEEDS_V3
    )
    complete = tuple(
        artifacts.H6WeightedA5CorpusTotalV3.create(
            endpoint_role="complete_a5",
            endpoint_config_id=H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[5],
            training_seed=seed,
            checkpoint_sha256=_sha(f"complete:{seed}"),
            particle_count=particle_count,
            replicate_id=replicate_id,
            counted_test_targets=37,
            weighted_total_nll=1.0,
            monte_carlo_half_width=0.1,
            smc_bias_bound=0.05,
            opening_proof_sha256=opening_proof_sha256,
        )
        for seed in H6_CONFIRMATORY_SEEDS_V3
        for replicate_id in range(64)
        for particle_count in particle_counts
    )
    emission = tuple(
        artifacts.H6WeightedA5CorpusTotalV3.create(
            endpoint_role="emission_a5",
            endpoint_config_id=H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[9],
            training_seed=seed,
            checkpoint_sha256=_sha(f"emission:{seed}"),
            particle_count=particle_count,
            replicate_id=replicate_id,
            counted_test_targets=37,
            weighted_total_nll=1.0,
            monte_carlo_half_width=0.1,
            smc_bias_bound=0.05,
            opening_proof_sha256=opening_proof_sha256,
        )
        for seed in H6_CONFIRMATORY_SEEDS_V3
        for replicate_id in range(64)
        for particle_count in particle_counts
    )
    inventory = artifacts.H6RawEndpointInventoryV4.create(
        exact_a0_rows=exact,
        complete_a5_rows=complete,
        emission_a5_rows=emission,
    )
    metrics = artifacts.H6PredictionMetricsV3.from_raw_inventory(inventory)
    result = artifacts.H6PredictionResultV3.create(
        reservation_sha256=reservation_sha256,
        opening_proof_sha256=opening_proof_sha256,
        inventory=inventory,
        metrics=metrics,
    )
    return result, inventory, metrics


def test_transaction_validates_eligibility_before_reservation(
    tmp_path: Path,
) -> None:
    from vfe4.training.h6_test_transaction_v3 import (
        execute_h6_test_transaction_v3,
    )

    with pytest.raises(ValueError, match="exact H6-Prediction v3"):
        execute_h6_test_transaction_v3(
            config=object(),
            readiness=object(),
            plan=object(),
            validation_bundle=object(),
            store=object(),
            journal_root=tmp_path,
            score_inventory=lambda _opening: pytest.fail(
                "scoring must not begin before eligibility validation"
            ),
        )

    assert tuple(tmp_path.iterdir()) == ()


def test_reservation_binds_config_data_checkpoints_and_inventory(
    tmp_path: Path,
) -> None:
    import vfe4.training.h6_test_transaction_v3 as transaction

    reservation = _reservation(transaction, tmp_path)
    payload = reservation.artifact_payload()

    assert payload["state"] == "RESERVED"
    assert payload["experiment_config_sha256"] == _sha("config")
    assert payload["readiness_sha256"] == _sha("readiness")
    assert payload["plan_sha256"] == _sha("plan")
    assert payload["data_identity_sha256"] == _sha("data")
    assert payload["sealed_test_sha256"] == _sha("sealed-test")
    assert payload["test_inventory_sha256"] == _sha("test-inventory")
    assert payload["tuning_selection_sha256"] == _sha("tuning-selection")
    assert payload["checkpoint_selection_sha256"] == _sha("checkpoint-selection")
    assert payload["validation_bundle_sha256"] == _sha("validation-bundle")
    assert payload["scoring_inventory_sha256"] == H6_SCORING_INVENTORY_SHA256
    assert payload["expected_row_count"] == 4104
    assert payload["experiment_identity_sha256"] == _sha("experiment")
    assert payload["access_policy_sha256"] == _sha("access-policy")
    assert payload["opening_proof_sha256"] == reservation.opening_proof_sha256
    assert payload["reservation_sha256"] == reservation.reservation_sha256


def test_crash_after_reservation_is_terminal_inconclusive(
    tmp_path: Path,
) -> None:
    import vfe4.training.h6_test_transaction_v3 as transaction

    reservation = _reservation(transaction, tmp_path)
    marker = tmp_path / "opening.reservation.json"
    marker.write_bytes(reservation.canonical_bytes())
    issued: list[object] = []

    terminal = transaction.recover_h6_test_transaction_v3(
        marker,
        capability_issuer=lambda: issued.append(object()),
    )

    assert terminal.state == "INCONCLUSIVE"
    assert terminal.reservation_sha256 == reservation.reservation_sha256
    assert issued == []
    assert (Path(reservation.state_root_path) / "INCONCLUSIVE").is_dir()
    repeated = transaction.recover_h6_test_transaction_v3(
        marker,
        capability_issuer=lambda: issued.append(object()),
    )
    assert repeated == terminal
    assert issued == []


def test_final_result_and_pointer_are_no_replace_published(
    tmp_path: Path,
) -> None:
    import vfe4.training.h6_test_transaction_v3 as transaction

    reservation = _reservation(transaction, tmp_path)
    marker = tmp_path / "opening.reservation.json"
    marker.write_bytes(reservation.canonical_bytes())
    result, inventory, metrics = _result_records(
        reservation.reservation_sha256,
        reservation.opening_proof_sha256,
    )
    finalized = transaction.finalize_h6_test_transaction_v3(
        reservation_path=marker,
        result=result,
        inventory=inventory,
        metrics=metrics,
    )
    repeated = transaction.finalize_h6_test_transaction_v3(
        reservation_path=marker,
        result=result,
        inventory=inventory,
        metrics=metrics,
    )

    assert finalized.terminal_directory.name == "TERMINAL"
    assert finalized.pointer_directory.name == "current"
    assert repeated == finalized


def test_public_transaction_readers_authenticate_exact_published_state(
    tmp_path: Path,
) -> None:
    import vfe4.training.h6_test_transaction_v3 as transaction

    reservation = _reservation(transaction, tmp_path)
    marker = tmp_path / "opening.reservation.json"
    marker.write_bytes(reservation.canonical_bytes())
    result, inventory, metrics = _result_records(
        reservation.reservation_sha256,
        reservation.opening_proof_sha256,
    )
    finalized = transaction.finalize_h6_test_transaction_v3(
        reservation_path=marker,
        result=result,
        inventory=inventory,
        metrics=metrics,
    )

    assert transaction.read_h6_test_reservation_v3(marker) == reservation
    assert (
        transaction.read_h6_test_terminal_v3(finalized.terminal_directory)
        == finalized.terminal
    )
    pointer = transaction.read_h6_prediction_pointer_v3(
        finalized.pointer_directory
    )
    assert pointer.reservation_sha256 == reservation.reservation_sha256
    assert pointer.terminal_sha256 == finalized.terminal.terminal_sha256
    assert pointer.result_sha256 == result.result_sha256

    corrupt_marker = tmp_path / "corrupt.reservation.json"
    corrupt_marker.write_bytes(reservation.canonical_bytes() + b"\n")
    with pytest.raises(ArtifactPublicationError, match="authenticate|canonical"):
        transaction.read_h6_test_reservation_v3(corrupt_marker)

    corrupt_terminal = tmp_path / "corrupt-terminal"
    shutil.copytree(finalized.terminal_directory, corrupt_terminal)
    (corrupt_terminal / "terminal.json").write_bytes(b"{}")
    with pytest.raises(ArtifactPublicationError, match="manifest|journal"):
        transaction.read_h6_test_terminal_v3(corrupt_terminal)

    corrupt_pointer = tmp_path / "corrupt-pointer"
    shutil.copytree(finalized.pointer_directory, corrupt_pointer)
    (corrupt_pointer / "pointer.json").write_bytes(b"{}")
    with pytest.raises(ArtifactPublicationError, match="manifest|pointer"):
        transaction.read_h6_prediction_pointer_v3(corrupt_pointer)


def test_authoritative_marker_requires_complete_reservation_bytes_and_recovers_gap(
    tmp_path: Path,
) -> None:
    import vfe4.data.access as access
    import vfe4.training.h6_test_transaction_v3 as transaction

    reservation = _reservation(transaction, tmp_path)
    assert (
        "reservation"
        in inspect.signature(
            access.reserve_and_issue_durable_test_opening_capability_v3
        ).parameters
    )
    marker = tmp_path / "opening.reservation.json"
    marker.write_bytes(reservation.canonical_bytes())

    terminal = transaction.recover_h6_test_transaction_v3(
        marker,
        capability_issuer=lambda: pytest.fail(
            "recovery after authoritative reservation cannot issue a capability"
        ),
    )

    assert terminal.state == "INCONCLUSIVE"
    assert terminal.reservation_sha256 == reservation.reservation_sha256
    assert marker.read_bytes() == reservation.canonical_bytes()


def test_terminal_commit_follows_bound_result_pointer_and_is_idempotent(
    tmp_path: Path,
) -> None:
    import vfe4.training.h6_test_transaction_v3 as transaction

    reservation = _reservation(transaction, tmp_path)
    marker = tmp_path / "opening.reservation.json"
    marker.write_bytes(reservation.canonical_bytes())
    result, inventory, metrics = _result_records(
        reservation.reservation_sha256,
        reservation.opening_proof_sha256,
    )
    paths = transaction.finalize_h6_test_transaction_v3(
        reservation_path=marker,
        result=result,
        inventory=inventory,
        metrics=metrics,
    )

    assert paths.result_directory.is_dir()
    assert paths.pointer_directory.is_dir()
    assert paths.state_alias_directory.is_dir()
    assert paths.terminal_directory.is_dir()
    assert paths.terminal_directory.stat().st_mtime_ns >= max(
        paths.result_directory.stat().st_mtime_ns,
        paths.pointer_directory.stat().st_mtime_ns,
        paths.state_alias_directory.stat().st_mtime_ns,
    )
    repaired = transaction.recover_h6_test_transaction_v3(
        marker,
        capability_issuer=lambda: pytest.fail(
            "terminal recovery cannot issue a capability"
        ),
    )
    assert repaired.state == "FINALIZED"
    assert repaired.terminal_sha256 == paths.terminal.terminal_sha256


def test_v3_access_rechecks_vocabulary_before_exclusive_reservation() -> None:
    import vfe4.data.access as access

    parameters = inspect.signature(
        access.reserve_and_issue_durable_test_opening_capability_v3
    ).parameters
    assert "reservation" in parameters
    assert access.H6_V3_RESERVATION_AUTHORITY_CHECKS == (
        "authenticated_store",
        "destination_absent",
        "readiness_plan_cross_binding",
        "experiment_identity",
        "vocabulary_equality",
        "complete_reservation_bytes",
    )


def test_reservation_durability_claim_is_explicitly_scoped() -> None:
    import vfe4.data.access as access

    assert access.H6_TEST_RESERVATION_DURABILITY in {
        "power-loss-durable-parent-directory-fsync",
        "process-crash-durable-no-power-loss-claim",
    }


def test_authoritative_marker_install_never_exposes_partial_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vfe4.data.access as access

    destination = tmp_path / "opening.reservation.json"
    canonical = b'{"complete":"reservation"}'

    def fail_install(_source: Path, _destination: Path) -> None:
        raise OSError("injected install crash")

    monkeypatch.setattr(
        access,
        "_install_staged_file_no_replace",
        fail_install,
    )
    with pytest.raises(Exception, match="install|reservation"):
        access._install_complete_reservation_file_no_replace(
            destination,
            canonical,
        )

    assert not destination.exists()
    assert tuple(tmp_path.glob(".reservation-stage-*")) == ()


def test_reservation_binds_output_namespace_and_conflict_recovers_inconclusive(
    tmp_path: Path,
) -> None:
    import vfe4.training.h6_test_transaction_v3 as transaction

    reservation = transaction.H6TestReservationV3.create(
        experiment_config_sha256=_sha("config"),
        readiness_sha256=_sha("readiness"),
        plan_sha256=_sha("plan"),
        experiment_identity_sha256=_sha("experiment"),
        data_identity_sha256=_sha("data"),
        sealed_test_sha256=_sha("sealed-test"),
        test_inventory_sha256=_sha("test-inventory"),
        access_policy_sha256=_sha("access-policy"),
        tuning_selection_sha256=_sha("tuning-selection"),
        checkpoint_selection_sha256=_sha("checkpoint-selection"),
        validation_bundle_sha256=_sha("validation-bundle"),
        scoring_inventory_sha256=H6_SCORING_INVENTORY_SHA256,
        expected_row_count=4104,
        result_root=tmp_path / "result",
        state_root=tmp_path / "state",
        pointer_root=tmp_path / "pointer",
        pointer_name="current",
    )
    assert reservation.result_root_path == (tmp_path / "result").as_posix()
    assert reservation.state_root_path == (tmp_path / "state").as_posix()
    assert reservation.pointer_root_path == (tmp_path / "pointer").as_posix()
    assert reservation.pointer_name == "current"

    (tmp_path / "result" / "RESULT").mkdir(parents=True)
    with pytest.raises(ValueError, match="output namespace|RESULT"):
        transaction.preflight_h6_test_output_namespace_v3(reservation)

    marker = tmp_path / "opening.reservation.json"
    marker.write_bytes(reservation.canonical_bytes())
    terminal = transaction.recover_h6_test_transaction_v3(
        marker,
        capability_issuer=lambda: pytest.fail(
            "conflict recovery cannot issue a capability"
        ),
    )
    assert terminal.state == "INCONCLUSIVE"
    assert (tmp_path / "state" / "TERMINAL").is_dir()


def test_only_safe_marker_state_machine_is_public() -> None:
    import vfe4.training.h6_test_transaction_v3 as transaction

    assert "finalize_h6_test_transaction_v3" in transaction.__all__
    assert "publish_h6_test_reservation_v3" not in transaction.__all__
    assert "publish_h6_test_terminal_and_pointer_v3" not in transaction.__all__
    assert set(transaction.__all__).isdisjoint(
        {
            "publish_h6_test_reservation_v3",
            "publish_h6_test_terminal_and_pointer_v3",
        }
    )


@pytest.mark.parametrize(
    ("pointer_root_name", "reserved_name"),
    (
        ("result", "RESULT"),
        ("state", "FINALIZED"),
        ("state", "INCONCLUSIVE"),
        ("state", "TERMINAL"),
    ),
)
def test_preflight_rejects_pointer_aliases_before_existence_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pointer_root_name: str,
    reserved_name: str,
) -> None:
    import vfe4.training.h6_test_transaction_v3 as transaction

    case_alias = reserved_name.lower()
    pointer_name = (
        case_alias
        if os.path.normcase(case_alias) == os.path.normcase(reserved_name)
        else reserved_name
    )
    reservation = _reservation(
        transaction,
        tmp_path,
        pointer_root=tmp_path / pointer_root_name,
        pointer_name=pointer_name,
    )
    targets = (
        Path(reservation.result_root_path) / "RESULT",
        Path(reservation.state_root_path) / "FINALIZED",
        Path(reservation.state_root_path) / "INCONCLUSIVE",
        Path(reservation.state_root_path) / "TERMINAL",
        Path(reservation.pointer_root_path) / reservation.pointer_name,
    )
    normalized_identities = tuple(
        os.path.normcase(os.path.normpath(os.fspath(target.resolve(strict=False))))
        for target in targets
    )
    assert len(set(normalized_identities)) == 4

    def unexpected_existence_check(_path: object) -> bool:
        pytest.fail("alias rejection must precede existence checks")

    monkeypatch.setattr(
        transaction.os.path,
        "lexists",
        unexpected_existence_check,
    )
    with pytest.raises(ValueError, match="pairwise distinct"):
        transaction.preflight_h6_test_output_namespace_v3(reservation)


def test_task10_transaction_exports_are_safe_and_lazy() -> None:
    import vfe4.training as public_training
    import vfe4.training.h6_test_transaction_v3 as transaction

    safe_names = {
        "execute_h6_test_transaction_v3",
        "finalize_h6_test_transaction_v3",
        "read_h6_prediction_pointer_v3",
        "read_h6_test_reservation_v3",
        "read_h6_test_terminal_v3",
        "recover_h6_test_transaction_v3",
    }
    removed_names = {
        "publish_h6_test_reservation_v3",
        "publish_h6_test_terminal_and_pointer_v3",
    }
    assert safe_names <= set(public_training.__all__)
    assert removed_names.isdisjoint(public_training.__all__)
    for name in safe_names:
        assert getattr(public_training, name) is getattr(transaction, name)
    for name in removed_names:
        assert not hasattr(public_training, name)
