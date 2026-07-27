from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from vfe4.artifacts import ArtifactPublicationError
from vfe4.types import GateResult, GateStatus, InvariantResult


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "verify_vfe4.py"


def _load_launcher():
    spec = importlib.util.spec_from_file_location("verify_vfe4_import_test", LAUNCHER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _h3_compatibility_config(module: object) -> dict[str, object]:
    raw = copy.deepcopy(module.CONFIG["operations"]["h1_h5"]["config"])
    raw["validation"]["gates"] = ["H1", "H2", "H3"]
    raw.pop("h4")
    raw.pop("h5")
    return raw


def _h1_h5_scientific_config(module: object) -> dict[str, object]:
    return copy.deepcopy(module.CONFIG["operations"]["h1_h5"]["config"])


def _run_h1_h5_scientific(module: object, scientific: dict[str, object]):
    launcher = copy.deepcopy(module.CONFIG)
    operation = launcher["operations"]["h1_h5"]
    operation["enabled"] = True
    operation["authorization"] = module._VERIFY_AUTHORIZATIONS["h1_h5"]
    operation["config"] = scientific
    return module.main(launcher)


def _h8_current_refs(
    head: str,
    dirty: str,
    junit: str,
    *,
    prediction_head: str | None = None,
    prediction_dirty: str | None = None,
    prediction_junit: str | None = None,
):
    import verification.h8_gate as h8_gate
    from vfe4.types.h7 import H7PredecessorReference
    from vfe4.types.h8 import (
        CurrentH8PrerequisiteRefs,
        H8H1H5Reference,
        H8H1PrefixPriorReference,
        H8H6PredictionReference,
        H8H6PrefixReference,
        H8H6PrefixSemanticFamilyReference,
        H8H7Reference,
    )

    digest = "a" * 64
    compatibility = {
        key: H7PredecessorReference.create(
            artifact_path=f"C:/immutable/{key}-artifact",
            git_head=head,
            dirty_digest=dirty,
            junit_sha256=junit,
            junit_path=f"C:/immutable/{key}-junit.xml",
            manifest_sha256=digest,
            payload_hashes={f"{key}.json": digest},
            ledger_path=f"C:/immutable/{key}-ledger",
            ledger_sha256=digest,
        )
        for key in ("h1_h5", "h1_prefix_prior", "h6_prefix")
    }

    def common(key: str) -> dict[str, object]:
        transitive = compatibility[key]
        return {
            "artifact_path": transitive.artifact_path,
            "manifest_sha256": transitive.manifest_sha256,
            "result_path": f"C:/immutable/{key}-result",
            "result_sha256": digest,
            "content_hashes": {f"{key}-content.json": digest},
            "payload_hashes": dict(transitive.payload_hashes),
            "ledger_path": transitive.ledger_path,
            "ledger_sha256": transitive.ledger_sha256,
            "producer_head": transitive.git_head,
            "producer_dirty_digest": transitive.dirty_digest,
            "candidate_junit_sha256": transitive.junit_sha256,
            "status": "pass",
        }

    h7_common = {
        "artifact_path": "C:/immutable/h7-artifact",
        "manifest_sha256": digest,
        "result_path": "C:/immutable/h7-result",
        "result_sha256": digest,
        "content_hashes": {"h7-content.json": digest},
        "payload_hashes": {"h7.json": digest},
        "ledger_path": "C:/immutable/h7-ledger",
        "ledger_sha256": digest,
        "producer_head": head,
        "producer_dirty_digest": dirty,
        "candidate_junit_sha256": junit,
        "status": "pass",
    }
    prediction_common = {
        **h7_common,
        "artifact_path": "C:/immutable/prediction-artifact",
        "result_path": "C:/immutable/prediction-result",
        "content_hashes": {"prediction-content.json": digest},
        "payload_hashes": {"prediction.json": digest},
        "ledger_path": "C:/immutable/prediction-ledger",
        "producer_head": prediction_head or head,
        "producer_dirty_digest": prediction_dirty or dirty,
        "candidate_junit_sha256": prediction_junit or junit,
    }
    base = CurrentH8PrerequisiteRefs(
        candidate_head=head,
        candidate_dirty_digest=dirty,
        candidate_junit_sha256=junit,
        h7_compatibility_refs=compatibility,
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
            result_pointer_path="h7-pointer",
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
            readiness_artifact_path="C:/immutable/prediction-readiness",
            readiness_manifest_sha256=digest,
            readiness_sha256=digest,
            correctness_artifact_paths={
                gate: f"C:/immutable/prediction-{gate.lower()}-correctness"
                for gate in ("H1", "H2", "H3", "H5")
            },
            h1_prefix_prior_artifact_path=(
                "C:/immutable/prediction-h1-prefix-prior"
            ),
            smc_accuracy_artifact_path="C:/immutable/prediction-smc-accuracy",
            smc_accuracy_manifest_sha256=digest,
            h6_prefix_artifact_path="C:/immutable/prediction-h6-prefix",
            h6_prefix_manifest_sha256=digest,
            blinded_data_artifact_path="C:/immutable/prediction-blinded-data",
            blinded_data_manifest_sha256=digest,
            matching_artifact_path="C:/immutable/prediction-matching",
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
    registry_bytes = h8_gate.canonical_h8_json_bytes(
        h8_gate.h8_current_refs_registry_payload(base)
    )
    return (
        replace(
            base,
            registry_sha256=hashlib.sha256(registry_bytes).hexdigest(),
        ),
        registry_bytes,
    )


def test_h8_registry_v3_preserves_separate_h6_prediction_candidate() -> None:
    import verification.h8_gate as h8_gate
    import verification.run_gates as gates

    prediction_head = "7" * 40
    prediction_dirty = "8" * 64
    prediction_junit = "9" * 64
    refs, registry_bytes = _h8_current_refs(
        "1" * 40,
        "2" * 64,
        "3" * 64,
        prediction_head=prediction_head,
        prediction_dirty=prediction_dirty,
        prediction_junit=prediction_junit,
    )

    assert refs.registry_schema_version == "h8-current-candidate-refs-v3"
    assert refs.prerequisite_obligations == ()
    assert refs.h6_prediction.producer_head == prediction_head
    assert refs.h6_prediction.producer_dirty_digest == prediction_dirty
    assert refs.h6_prediction.candidate_junit_sha256 == prediction_junit
    assert (
        refs.h6_prediction.readiness_sha256
        == refs.h6_prediction.config_sha256
        == "a" * 64
    )
    parsed = gates.parse_h8_reference_registry_bytes(registry_bytes)
    assert parsed.h6_prediction == refs.h6_prediction
    assert (
        h8_gate.canonical_h8_json_bytes(
            h8_gate.h8_current_refs_registry_payload(parsed)
        )
        == registry_bytes
    )
    assert (
        "h6_prediction_frozen_scientific_dependency_closure"
        in h8_gate._prerequisite_payload(parsed)["compatibility_checks"]
    )
    with pytest.raises(ValueError, match="candidate_junit_sha256"):
        replace(parsed.h6_prediction, candidate_junit_sha256=None)


def _legacy_h8_prefix_payload(
    bounded: dict[str, object],
) -> dict[str, object]:
    common_fields = {
        "kind",
        "artifact_path",
        "manifest_sha256",
        "result_path",
        "result_sha256",
        "content_hashes",
        "payload_hashes",
        "ledger_path",
        "ledger_sha256",
        "producer_head",
        "producer_dirty_digest",
        "candidate_junit_sha256",
        "status",
    }
    return {
        **{key: value for key, value in bounded.items() if key in common_fields},
        "certificate_set_sha256": bounded["prefix_certificate_set_sha256"],
        "certificate_hashes": {
            "certificate-0.json": "d" * 64,
            "certificate-1.json": "0" * 64,
        },
    }


def test_h8_registry_v1_is_readable_but_never_authorizing() -> None:
    import verification.h8_gate as h8_gate
    import verification.run_gates as gates

    _refs, registry_bytes = _h8_current_refs("1" * 40, "2" * 64, "3" * 64)
    payload = json.loads(registry_bytes)
    payload["schema_version"] = "h8-current-candidate-refs-v1"
    h6_prefix = payload["references"]["h6_prefix"]
    payload["references"]["h6_prefix"] = _legacy_h8_prefix_payload(h6_prefix)
    h6_prediction = payload["references"]["h6_prediction"]
    legacy_fields = {
        "kind",
        "artifact_path",
        "manifest_sha256",
        "result_path",
        "result_sha256",
        "content_hashes",
        "payload_hashes",
        "experiment_sha256",
        "ledger_path",
        "ledger_sha256",
        "producer_head",
        "producer_dirty_digest",
        "candidate_junit_sha256",
        "status",
    }
    payload["references"]["h6_prediction"] = {
        key: value
        for key, value in h6_prediction.items()
        if key in legacy_fields
    }
    legacy_bytes = h8_gate.canonical_h8_json_bytes(payload)

    parsed = gates.parse_h8_reference_registry_bytes(legacy_bytes)

    assert type(parsed.h6_prefix).__name__ == "H8LegacyH6PrefixReference"
    assert type(parsed.h6_prediction).__name__ == "H8LegacyH6PredictionReference"
    assert parsed.registry_schema_version == "h8-current-candidate-refs-v1"
    assert parsed.prerequisite_obligations == (
        "h8_prerequisite_legacy_registry_requires_bounded_h6_prefix_v3",
        "h8_prerequisite_registry_v1_requires_amended_h6_prediction_v2",
    )
    assert (
        h8_gate.canonical_h8_json_bytes(
            h8_gate.h8_current_refs_registry_payload(parsed)
        )
        == legacy_bytes
    )
    evaluation = h8_gate.assemble_h8_source_only_evaluation(
        config_sha256="4" * 64,
        current_refs=parsed,
        correctness=(),
        production_runs=(),
        profiler_runs=(),
        controls=(),
        dependency_closure_sha256="5" * 64,
        preregistration_sha256="6" * 64,
    )
    assert evaluation.result.status is GateStatus.INCONCLUSIVE
    assert parsed.prerequisite_obligations[0] in evaluation.result.obligations


def test_h8_registry_v2_is_readable_but_never_authorizing() -> None:
    import verification.h8_gate as h8_gate
    import verification.run_gates as gates

    _refs, registry_bytes = _h8_current_refs("1" * 40, "2" * 64, "3" * 64)
    payload = json.loads(registry_bytes)
    payload["schema_version"] = "h8-current-candidate-refs-v2"
    h6_prefix = payload["references"]["h6_prefix"]
    payload["references"]["h6_prefix"] = _legacy_h8_prefix_payload(h6_prefix)
    legacy_bytes = h8_gate.canonical_h8_json_bytes(payload)

    parsed = gates.parse_h8_reference_registry_bytes(legacy_bytes)

    assert type(parsed.h6_prefix).__name__ == "H8LegacyH6PrefixReference"
    assert type(parsed.h6_prediction).__name__ == "H8H6PredictionReference"
    assert parsed.registry_schema_version == "h8-current-candidate-refs-v2"
    assert parsed.prerequisite_obligations == (
        "h8_prerequisite_legacy_registry_requires_bounded_h6_prefix_v3",
    )
    assert (
        h8_gate.canonical_h8_json_bytes(
            h8_gate.h8_current_refs_registry_payload(parsed)
        )
        == legacy_bytes
    )
    evaluation = h8_gate.assemble_h8_source_only_evaluation(
        config_sha256="4" * 64,
        current_refs=parsed,
        correctness=(),
        production_runs=(),
        profiler_runs=(),
        controls=(),
        dependency_closure_sha256="5" * 64,
        preregistration_sha256="6" * 64,
    )
    assert evaluation.result.status is GateStatus.INCONCLUSIVE
    assert parsed.prerequisite_obligations[0] in evaluation.result.obligations


def test_launcher_import_is_safe_and_has_no_cli_framework(monkeypatch: pytest.MonkeyPatch) -> None:
    import verification.run_gates as gates

    monkeypatch.setattr(
        gates, "run_verification", lambda config: pytest.fail("ran on import")
    )
    module = _load_launcher()
    assert type(module.CONFIG) is dict
    assert tuple(module.CONFIG["operations"]) == (
        "h1_h5",
        "h1_prefix_prior",
        "h1_prefix_prior_v2",
        "h6_prefix",
        "h6_smc_accuracy",
        "h7",
        "h8_preflight",
        "h8",
    )
    h7_config = module.CONFIG["operations"]["h7"]["config"]
    assert h7_config["validation"]["gates"] == [
        "H1", "H2", "H3", "H4", "H5", "H6-Prefix", "H7"
    ]
    assert "h8" not in h7_config
    h8_config = module.CONFIG["operations"]["h8"]["config"]
    assert h8_config["validation"]["gates"] == [
        "H1", "H2", "H3", "H4", "H5", "H6-Prefix", "H7", "H8"
    ]
    h3 = module.CONFIG["operations"]["h1_h5"]["config"]["h3"]
    assert h3["recognition_families"] == [
        "structured_full_spd",
        "fine_factorized_diagonal",
    ]
    assert h3["common_initialization"] == {
        "mean": [0.0, 0.0, 0.0, 0.0],
        "precision": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }
    assert h3["optimization_operation"] == "maximize_direct_h3_elbo_lbfgs"
    assert h3["expected_autograd_scope"] == "h3_recognition_only"
    assert h3["threshold_decision_rule"] == "signed_margin_three_way"
    tree = ast.parse(LAUNCHER.read_text(encoding="utf-8"))
    imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    assert imported.isdisjoint({"argparse", "click", "typer", "hydra"})


def test_h7_main_publishes_only_reference_records_and_h7_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import verification.run_gates as gates
    from vfe4.artifacts.h6 import CandidateArtifactReference
    from vfe4.types.h7 import H7InconclusiveOutcome
    from vfe4.types.results import H7GateResult

    module = _load_launcher()
    launcher = copy.deepcopy(module.CONFIG)
    h7_entry = launcher["operations"]["h7"]
    h7_entry["enabled"] = True
    h7_entry["authorization"] = module._VERIFY_AUTHORIZATIONS["h7"]
    h7_entry["config"]["artifacts"]["run_root"] = str(tmp_path / "runs")

    head, dirty, junit = "a" * 40, "b" * 64, "c" * 64
    references = {}
    for key in ("h1_h5", "h1_prefix_prior", "h6_prefix"):
        digest = hashlib.sha256(key.encode("ascii")).hexdigest()
        candidate = CandidateArtifactReference(
            tmp_path / "predecessors" / key,
            head,
            dirty,
            digest,
            {f"validation/{key}.json": digest},
        )
        references[key] = gates.candidate_artifact_reference_to_h7_reference(
            candidate,
            junit_sha256=junit,
            junit_path=tmp_path / "candidate-junit.xml",
            ledger_path=tmp_path / ".verification" / f"{key}-ledger.json",
            ledger_sha256=digest,
        )
    registry = gates.h7_reference_registry_bytes(references)
    registry_root = tmp_path / ".verification"
    registry_root.mkdir()
    (registry_root / f"h7-current-candidate-{head}-refs.json").write_bytes(registry)
    h1_path = tmp_path / "h1.json"
    h7_path = tmp_path / "h7.json"
    h1_path.write_bytes(b"h1")
    h7_path.write_bytes(b"h7")

    obligations = ("source-only H7 runtime evidence is unavailable",)
    result = H7GateResult.create(
        gate="H7",
        status=GateStatus.INCONCLUSIVE,
        fixture_hashes={},
        predecessor_references=references,
        trials=(),
        controls=(),
        outcome=H7InconclusiveOutcome.create(
            kind="INCONCLUSIVE",
            obligations=obligations,
        ),
        obligations=obligations,
    )
    evaluation = SimpleNamespace(result=result)
    monkeypatch.setattr(gates, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(gates, "FIXTURE_PATH", h1_path)
    monkeypatch.setattr(gates, "H7_FIXTURE_PATH", h7_path)
    monkeypatch.setattr(
        gates,
        "current_source_identity",
        lambda *_args: (head, dirty, "d" * 64),
    )
    monkeypatch.setattr(
        gates,
        "assemble_h7_gate_evaluation",
        lambda **_kwargs: evaluation,
    )
    monkeypatch.setattr(
        gates,
        "build_h7_provenance",
        lambda **_kwargs: {"schema_version": "vfe4-h7-provenance-v1"},
    )
    monkeypatch.setattr(gates, "build_environment", lambda _config: {"device": "cpu"})
    monkeypatch.setattr(
        gates,
        "h7_validation_payload",
        lambda _evaluation: {
            "gate": "H7",
            "status": "inconclusive",
            "obligations": list(obligations),
        },
    )

    published = module.main(launcher)

    assert tuple(item.gate for item in published.gate_results) == ("H7",)
    files = sorted(
        path.relative_to(published.run_directory).as_posix()
        for path in published.run_directory.rglob("*")
        if path.is_file()
    )
    assert files == [
        "config.json",
        "environment.json",
        "manifest.sha256",
        "provenance.json",
        "references/h1_h5.json",
        "references/h1_prefix_prior.json",
        "references/h6_prefix.json",
        "validation/h7.json",
    ]
    assert not any("validation/h1" in name or "h8" in name for name in files)


def test_editable_mapping_runs_three_gates_once_from_one_fixture_snapshot_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import verification.run_gates as gates

    module = _load_launcher()
    raw = _h3_compatibility_config(module)
    raw["artifacts"]["run_root"] = str(tmp_path / "absolute runs")
    before = copy.deepcopy(raw)
    fixture_paths = {
        "h1": gates.FIXTURE_PATH,
        "coupled": gates.H3_COUPLED_FIXTURE_PATH,
        "zero_control": gates.H3_ZERO_CONTROL_FIXTURE_PATH,
    }
    reads = {name: 0 for name in fixture_paths}

    class CountingPath:
        def __init__(self, name: str, path: Path) -> None:
            self.name = name
            self.path = path

        def read_bytes(self) -> bytes:
            reads[self.name] += 1
            return self.path.read_bytes()

    monkeypatch.setattr(gates, "FIXTURE_PATH", CountingPath("h1", fixture_paths["h1"]))
    monkeypatch.setattr(
        gates,
        "H3_COUPLED_FIXTURE_PATH",
        CountingPath("coupled", fixture_paths["coupled"]),
    )
    monkeypatch.setattr(
        gates,
        "H3_ZERO_CONTROL_FIXTURE_PATH",
        CountingPath("zero_control", fixture_paths["zero_control"]),
    )
    original_h1 = gates.evaluate_h1
    original_h2 = gates.evaluate_h2
    original_h3 = gates.evaluate_h3
    evaluations: list[tuple[str, tuple[bytes | None, ...]]] = []

    def evaluate_h1(config: object, *, fixture_bytes: bytes | None = None):
        evaluations.append(("H1", (fixture_bytes,)))
        return original_h1(config, fixture_bytes=fixture_bytes)

    def evaluate_h2(config: object, *, fixture_bytes: bytes | None = None):
        evaluations.append(("H2", (fixture_bytes,)))
        return original_h2(config, fixture_bytes=fixture_bytes)

    def evaluate_h3(
        config: object,
        *,
        coupled_fixture_bytes: bytes | None = None,
        zero_control_fixture_bytes: bytes | None = None,
    ):
        evaluations.append(
            ("H3", (coupled_fixture_bytes, zero_control_fixture_bytes))
        )
        return original_h3(
            config,
            coupled_fixture_bytes=coupled_fixture_bytes,
            zero_control_fixture_bytes=zero_control_fixture_bytes,
        )

    monkeypatch.setattr(gates, "evaluate_h1", evaluate_h1)
    monkeypatch.setattr(gates, "evaluate_h2", evaluate_h2)
    monkeypatch.setattr(gates, "evaluate_h3", evaluate_h3)

    result = _run_h1_h5_scientific(module, raw)

    assert tuple(item.gate for item in result.gate_results) == ("H1", "H2", "H3")
    assert all(item.status is GateStatus.PASS for item in result.gate_results)
    assert [name for name, _ in evaluations] == ["H1", "H2", "H3"]
    assert evaluations[0][1][0] is not None
    assert evaluations[0][1][0] is evaluations[1][1][0]
    assert evaluations[2][1][0] is not None
    assert evaluations[2][1][1] is not None
    assert reads == {"h1": 1, "coupled": 1, "zero_control": 1}
    assert raw == before
    runs = list((tmp_path / "absolute runs").iterdir())
    assert len(runs) == 1
    assert sorted(path.relative_to(runs[0]).as_posix() for path in runs[0].rglob("*.*")) == [
        "config.json", "environment.json", "manifest.sha256", "provenance.json",
        "validation/h1.json", "validation/h2.json", "validation/h3.json",
    ]
    for line in (runs[0] / "manifest.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((runs[0] / name).read_bytes()).hexdigest() == digest
    provenance = json.loads((runs[0] / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["gate_states"] == {
        "H1": "pass",
        "H2": "pass",
        "H3": "pass",
    }
    assert provenance["fixture_consumers"] == ["H1", "H2"]
    assert provenance["gate_fixture_consumers"] == {
        "H1": ["h1-v1"],
        "H2": ["h1-v1"],
        "H3": ["h3-coupled-v1", "h3-zero-control-v1"],
    }
    assert tuple(provenance["fixture_hashes"]) == (
        "h1-v1",
        "h3-coupled-v1",
        "h3-zero-control-v1",
    )
    assert provenance["h3_profile"] == raw["h3"]
    assert provenance["fixture_observed_sha256"] == json.loads(
        (runs[0] / "validation" / "h1.json").read_text(encoding="utf-8")
    )["fixture_observed_sha256"]
    assert provenance["fixture_observed_sha256"] == json.loads(
        (runs[0] / "validation" / "h2.json").read_text(encoding="utf-8")
    )["fixture_observed_sha256"]
    h3_payload = json.loads(
        (runs[0] / "validation" / "h3.json").read_text(encoding="utf-8")
    )
    assert provenance["fixture_hashes"]["h3-coupled-v1"][
        "observed_sha256"
    ] == h3_payload["fixtures"]["coupled"]["observed_sha256"]
    assert provenance["fixture_hashes"]["h3-zero-control-v1"][
        "observed_sha256"
    ] == h3_payload["fixtures"]["zero_control"]["observed_sha256"]


def test_coupled_click_run_captures_once_orders_five_gates_and_publishes_eight_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import verification.run_gates as gates
    from verification.h5_gate import H5GateResult
    from vfe4.types import H3GateResult, H4GateResult

    module = _load_launcher()
    raw = _h1_h5_scientific_config(module)
    raw["artifacts"]["run_root"] = str(tmp_path / "runs")
    fixture_paths = {
        "h1": gates.FIXTURE_PATH,
        "coupled": gates.H3_COUPLED_FIXTURE_PATH,
        "zero": gates.H3_ZERO_CONTROL_FIXTURE_PATH,
        "h5": gates.H5_UPDATE_SPEC_FIXTURE_PATH,
    }
    reads = {name: 0 for name in fixture_paths}

    class CountingPath:
        def __init__(self, name: str, path: Path) -> None:
            self.name = name
            self.path = path

        def read_bytes(self) -> bytes:
            reads[self.name] += 1
            return self.path.read_bytes()

    for attribute, name in (
        ("FIXTURE_PATH", "h1"),
        ("H3_COUPLED_FIXTURE_PATH", "coupled"),
        ("H3_ZERO_CONTROL_FIXTURE_PATH", "zero"),
        ("H5_UPDATE_SPEC_FIXTURE_PATH", "h5"),
    ):
        monkeypatch.setattr(gates, attribute, CountingPath(name, fixture_paths[name]))

    def forged(cls: type, gate: str, status: GateStatus):
        value = object.__new__(cls)
        object.__setattr__(value, "gate", gate)
        object.__setattr__(value, "status", status)
        return value

    passed_invariant = (InvariantResult("mock", True, 0.0, 0.0, "mock"),)
    h1_result = GateResult(
        "H1", GateStatus.PASS, "h1-v1", 0.0, 0.0,
        {"x": 0.0}, passed_invariant, (),
    )
    h2_result = GateResult(
        "H2", GateStatus.PASS, "h1-v1", 0.0, 0.0,
        {"x": 0.0}, passed_invariant, (),
    )
    h3_result = forged(H3GateResult, "H3", GateStatus.PASS)
    h4_result = forged(H4GateResult, "H4", GateStatus.INCONCLUSIVE)
    h5_result = forged(H5GateResult, "H5", GateStatus.PASS)
    object.__setattr__(
        h5_result,
        "update_spec_raw_sha256",
        raw["h5"]["update_spec_raw_sha256"],
    )
    order: list[str] = []
    captures: dict[str, tuple[bytes, ...]] = {}

    def evaluation(name: str, result: object, captured: tuple[bytes, ...]):
        order.append(name)
        captures[name] = captured
        return SimpleNamespace(result=result, validation_payload={"gate": name})

    monkeypatch.setattr(
        gates, "evaluate_h1",
        lambda _config, *, fixture_bytes: evaluation("H1", h1_result, (fixture_bytes,)),
    )
    monkeypatch.setattr(
        gates, "evaluate_h2",
        lambda _config, *, fixture_bytes: evaluation("H2", h2_result, (fixture_bytes,)),
    )
    monkeypatch.setattr(
        gates, "evaluate_h3",
        lambda _config, *, coupled_fixture_bytes, zero_control_fixture_bytes: evaluation(
            "H3", h3_result, (coupled_fixture_bytes, zero_control_fixture_bytes)
        ),
    )
    monkeypatch.setattr(
        gates, "evaluate_h4",
        lambda _config, *, h3_coupled_bytes, h3_zero_bytes: evaluation(
            "H4", h4_result, (h3_coupled_bytes, h3_zero_bytes)
        ),
    )
    monkeypatch.setattr(
        gates, "evaluate_h5",
        lambda _config, *, h1_fixture_bytes, h5_update_spec_bytes: evaluation(
            "H5", h5_result, (h1_fixture_bytes, h5_update_spec_bytes)
        ),
    )
    monkeypatch.setattr(gates, "h2_validation_payload", lambda value: value.validation_payload)
    monkeypatch.setattr(gates, "h3_validation_payload", lambda value: value.validation_payload)
    monkeypatch.setattr(gates, "h4_validation_artifact", lambda value: value)
    monkeypatch.setattr(gates, "h4_validation_payload", lambda value: value.validation_payload)
    monkeypatch.setattr(gates, "h5_validation_payload", lambda value: value.validation_payload)
    monkeypatch.setattr(
        gates,
        "_combined_provenance",
        lambda *_args: {
            "gate_states": {
                "H1": "pass", "H2": "pass", "H3": "pass",
                "H4": "inconclusive", "H5": "pass",
            }
        },
    )

    result = _run_h1_h5_scientific(module, raw)

    assert order == ["H1", "H2", "H3", "H4", "H5"]
    assert reads == {"h1": 1, "coupled": 1, "zero": 1, "h5": 1}
    assert captures["H1"][0] is captures["H2"][0] is captures["H5"][0]
    assert captures["H3"][0] is captures["H4"][0]
    assert captures["H3"][1] is captures["H4"][1]
    assert tuple(item.status for item in result.gate_results[-2:]) == (
        GateStatus.INCONCLUSIVE, GateStatus.PASS,
    )
    manifest_paths = [
        line.split("  ", 1)[1]
        for line in (result.run_directory / "manifest.sha256")
        .read_text(encoding="utf-8").splitlines()
    ]
    assert manifest_paths == [
        "config.json", "environment.json", "provenance.json",
        "validation/h1.json", "validation/h2.json", "validation/h3.json",
        "validation/h4.json", "validation/h5.json",
    ]


def test_coupled_runner_prechecks_h5_digest_then_publishes_typed_inconclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import verification.h5_gate as h5_gate
    import verification.run_gates as gates
    from verification.h5_gate import (
        H5PreflightErrorKind,
        H5PreflightPhase,
        H5UnavailableField,
    )
    from vfe4.types import H3GateResult, H4GateResult, H5_NONCLAIM_IDS

    module = _load_launcher()
    raw = _h1_h5_scientific_config(module)
    raw["artifacts"]["run_root"] = str(tmp_path / "runs")
    wrong_bytes = bytes(bytearray(b'{"wrong":"must-not-decode"}\n'))
    original_sha256 = hashlib.sha256
    wrong_digest = original_sha256(wrong_bytes).hexdigest()
    expected_digest = raw["h5"]["update_spec_raw_sha256"]
    assert wrong_digest != expected_digest

    reads = 0

    class WrongBytes:
        def read_bytes(self) -> bytes:
            nonlocal reads
            reads += 1
            return wrong_bytes

    order: list[str] = []
    wrong_hash_orders: list[tuple[str, ...]] = []
    comparison_orders: list[tuple[str, ...]] = []

    class ObservedDigest(str):
        def __eq__(self, other: object) -> bool:
            comparison_orders.append(tuple(order))
            return super().__eq__(other)

        def __ne__(self, other: object) -> bool:
            comparison_orders.append(tuple(order))
            return super().__ne__(other)

    class DigestProbe:
        def __init__(self, digest: str) -> None:
            self.digest = digest

        def hexdigest(self) -> str:
            return ObservedDigest(self.digest)

    def recording_sha256(value: bytes = b""):
        if value is wrong_bytes:
            wrong_hash_orders.append(tuple(order))
            digest = original_sha256(value)
            if len(wrong_hash_orders) == 1:
                return DigestProbe(digest.hexdigest())
            return digest
        return original_sha256(value)

    def begin_evaluation(name: str) -> None:
        assert wrong_hash_orders[0] == ()
        assert comparison_orders[0] == ()
        order.append(name)

    def forged(cls: type, gate: str):
        value = object.__new__(cls)
        object.__setattr__(value, "gate", gate)
        object.__setattr__(value, "status", GateStatus.PASS)
        return value

    passed_invariant = (InvariantResult("mock", True, 0.0, 0.0, "mock"),)
    h1_result = GateResult(
        "H1", GateStatus.PASS, "h1-v1", 0.0, 0.0,
        {"x": 0.0}, passed_invariant, (),
    )
    h2_result = GateResult(
        "H2", GateStatus.PASS, "h1-v1", 0.0, 0.0,
        {"x": 0.0}, passed_invariant, (),
    )
    h3_result = forged(H3GateResult, "H3")
    h4_result = forged(H4GateResult, "H4")
    h1_capture: list[bytes] = []
    h5_capture: list[bytes] = []

    def evaluate_h1(_config: object, *, fixture_bytes: bytes):
        begin_evaluation("H1")
        h1_capture.append(fixture_bytes)
        return SimpleNamespace(
            result=h1_result,
            validation_payload={"gate": "H1", "status": "pass"},
            fixture_observed_sha256=original_sha256(fixture_bytes).hexdigest(),
        )

    def evaluate_h2(_config: object, *, fixture_bytes: bytes):
        begin_evaluation("H2")
        assert fixture_bytes is h1_capture[0]
        return SimpleNamespace(
            result=h2_result,
            validation_payload={"gate": "H2", "status": "pass"},
            fixture_observed_sha256=original_sha256(fixture_bytes).hexdigest(),
        )

    def evaluate_h3(
        _config: object,
        *,
        coupled_fixture_bytes: bytes,
        zero_control_fixture_bytes: bytes,
    ):
        begin_evaluation("H3")
        return SimpleNamespace(
            result=h3_result,
            validation_payload={"gate": "H3", "status": "pass"},
            fixture_hashes=SimpleNamespace(
                coupled_expected_sha256=raw["h3"]["coupled_expected_sha256"],
                coupled_observed_sha256=original_sha256(
                    coupled_fixture_bytes
                ).hexdigest(),
                zero_control_expected_sha256=raw["h3"][
                    "zero_control_expected_sha256"
                ],
                zero_control_observed_sha256=original_sha256(
                    zero_control_fixture_bytes
                ).hexdigest(),
            ),
        )

    def evaluate_h4(
        config: object,
        *,
        h3_coupled_bytes: bytes,
        h3_zero_bytes: bytes,
    ):
        begin_evaluation("H4")
        assert type(h3_coupled_bytes) is bytes
        assert type(h3_zero_bytes) is bytes
        return SimpleNamespace(
            result=h4_result,
            validation_payload={"gate": "H4", "status": "pass"},
            h4_config_sha256=config.config_sha256,
            bounded_claim="mock H4 bounded claim",
            nonclaims=("mock H4 nonclaim",),
        )

    original_evaluate_h5 = gates.evaluate_h5

    def evaluate_h5(
        config: object,
        *,
        h1_fixture_bytes: bytes,
        h5_update_spec_bytes: bytes,
    ):
        begin_evaluation("H5")
        assert h1_fixture_bytes is h1_capture[0]
        assert h5_update_spec_bytes is wrong_bytes
        h5_capture.append(h5_update_spec_bytes)
        return original_evaluate_h5(
            config,
            h1_fixture_bytes=h1_fixture_bytes,
            h5_update_spec_bytes=h5_update_spec_bytes,
        )

    monkeypatch.setattr(gates, "H5_UPDATE_SPEC_FIXTURE_PATH", WrongBytes())
    monkeypatch.setattr(gates.hashlib, "sha256", recording_sha256)
    monkeypatch.setattr(
        h5_gate,
        "parse_h5_update_spec_bytes",
        lambda _value: pytest.fail("digest-mismatched H5 bytes were decoded"),
    )
    monkeypatch.setattr(gates, "evaluate_h1", evaluate_h1)
    monkeypatch.setattr(gates, "evaluate_h2", evaluate_h2)
    monkeypatch.setattr(gates, "evaluate_h3", evaluate_h3)
    monkeypatch.setattr(gates, "evaluate_h4", evaluate_h4)
    monkeypatch.setattr(gates, "evaluate_h5", evaluate_h5)
    monkeypatch.setattr(
        gates, "h2_validation_payload", lambda value: value.validation_payload
    )
    monkeypatch.setattr(
        gates, "h3_validation_payload", lambda value: value.validation_payload
    )
    monkeypatch.setattr(gates, "h4_validation_artifact", lambda value: value)
    monkeypatch.setattr(
        gates, "h4_validation_payload", lambda value: value.validation_payload
    )

    result = _run_h1_h5_scientific(module, raw)

    assert reads == 1
    assert h5_capture == [wrong_bytes]
    assert h5_capture[0] is wrong_bytes
    assert wrong_hash_orders == [(), ("H1", "H2", "H3", "H4", "H5")]
    assert comparison_orders[0] == ()
    assert order == ["H1", "H2", "H3", "H4", "H5"]
    assert tuple(item.gate for item in result.gate_results) == (
        "H1", "H2", "H3", "H4", "H5",
    )
    assert tuple(item.status for item in result.gate_results) == (
        GateStatus.PASS,
        GateStatus.PASS,
        GateStatus.PASS,
        GateStatus.PASS,
        GateStatus.INCONCLUSIVE,
    )

    runs = list((tmp_path / "runs").iterdir())
    assert runs == [result.run_directory]
    manifest_paths = [
        line.split("  ", 1)[1]
        for line in (result.run_directory / "manifest.sha256")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert manifest_paths == [
        "config.json", "environment.json", "provenance.json",
        "validation/h1.json", "validation/h2.json", "validation/h3.json",
        "validation/h4.json", "validation/h5.json",
    ]
    assert len(manifest_paths) == 8
    assert all(name.endswith(".json") for name in manifest_paths)
    for name in manifest_paths:
        json.loads((result.run_directory / name).read_text(encoding="utf-8"))

    h5_payload = json.loads(
        (result.run_directory / "validation" / "h5.json").read_text(
            encoding="utf-8"
        )
    )
    h5_result = h5_payload["result"]
    preflight = h5_result["preflight"]
    assert h5_result["status"] == "inconclusive"
    assert h5_result["update_spec_raw_sha256"] == wrong_digest
    assert preflight["phase"] == H5PreflightPhase.UPDATE_SPEC_VALIDATION.value
    assert preflight["update_spec_raw_sha256"] == wrong_digest
    assert preflight["errors"][0]["kind"] == (
        H5PreflightErrorKind.UPDATE_SPEC_RAW_DIGEST_MISMATCH.value
    )
    assert expected_digest in preflight["errors"][0]["detail"]
    assert wrong_digest in preflight["errors"][0]["detail"]
    assert preflight["unavailable_fields"] == [
        item.value for item in H5UnavailableField
    ]
    for name in (
        "update_spec_canonical_sha256",
        "objective_schema_sha256",
        "factor_input_schema_version",
        "factor_input_schema_sha256",
        "reference_sha256",
        "positive_cases",
        "controls",
    ):
        assert h5_result[name] is None
    for name in (
        "reference_sha256",
        "factor_universe",
        "recognition_coordinate_universe",
        "model_block_universe",
        "variable_dependency_rows",
        "parameter_dependency_rows",
        "positive_attempts",
        "controls",
        "oracle_results",
    ):
        assert h5_payload[name] is None
    assert h5_payload["nonclaims"] == list(H5_NONCLAIM_IDS)

    provenance = json.loads(
        (result.run_directory / "provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["gate_state"] == "inconclusive"
    assert provenance["gate_states"] == {
        "H1": "pass",
        "H2": "pass",
        "H3": "pass",
        "H4": "pass",
        "H5": "inconclusive",
    }
    assert provenance["fixture_hashes"]["h5-conditional-update-v1"] == {
        "expected_sha256": expected_digest,
        "observed_sha256": wrong_digest,
        "hash_domain": "raw_fixture_bytes",
    }
    assert provenance["h5_config"]["update_spec_raw_sha256"] == expected_digest
    for name in (
        "update_spec_canonical_sha256",
        "objective_schema_sha256",
        "factor_input_schema_version",
        "factor_input_schema_sha256",
    ):
        assert provenance["h5_config"][name] is None
    assert provenance["h5_state_hashes"] == {
        "reference_sha256": None,
        "recognition_sha256": None,
        "model_sha256": None,
        "validation_payload_sha256": h5_payload["payload_sha256"],
    }
    assert provenance["h5_update_hash_records"] == {
        "positive": [],
        "controls": [],
    }
    assert provenance["h5_nonclaims"] == list(H5_NONCLAIM_IDS)

    def assert_full_sha256_fields(value: object) -> None:
        if isinstance(value, dict):
            for name, item in value.items():
                if name.endswith("sha256") and item is not None:
                    assert type(item) is str
                    assert len(item) == 64
                assert_full_sha256_fields(item)
        elif isinstance(value, list):
            for item in value:
                assert_full_sha256_fields(item)

    assert_full_sha256_fields(h5_payload)
    assert_full_sha256_fields(provenance)


@pytest.mark.parametrize("gates", (("H1",), ("H1", "H2")))
def test_compatibility_prefixes_never_touch_or_publish_h3(
    gates: tuple[str, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import verification.run_gates as run_gates

    module = _load_launcher()
    raw = _h1_h5_scientific_config(module)
    raw["validation"]["gates"] = list(gates)
    del raw["h3"]
    del raw["h4"]
    del raw["h5"]
    raw["artifacts"]["run_root"] = str(tmp_path / "runs")

    class ForbiddenH3Path:
        def read_bytes(self) -> bytes:
            pytest.fail("compatibility prefix touched an H3 fixture")

    monkeypatch.setattr(run_gates, "H3_COUPLED_FIXTURE_PATH", ForbiddenH3Path())
    monkeypatch.setattr(run_gates, "H3_ZERO_CONTROL_FIXTURE_PATH", ForbiddenH3Path())
    monkeypatch.setattr(
        run_gates,
        "evaluate_h3",
        lambda *_args, **_kwargs: pytest.fail("compatibility prefix evaluated H3"),
    )
    monkeypatch.setattr(run_gates, "H5_UPDATE_SPEC_FIXTURE_PATH", ForbiddenH3Path())
    monkeypatch.setattr(
        run_gates, "evaluate_h4",
        lambda *_args, **_kwargs: pytest.fail("compatibility prefix evaluated H4"),
    )
    monkeypatch.setattr(
        run_gates, "evaluate_h5",
        lambda *_args, **_kwargs: pytest.fail("compatibility prefix evaluated H5"),
    )

    result = _run_h1_h5_scientific(module, raw)

    assert tuple(item.gate for item in result.gate_results) == gates
    run_dir = result.run_directory
    assert not (run_dir / "validation" / "h3.json").exists()
    provenance_bytes = (run_dir / "provenance.json").read_text(encoding="utf-8")
    provenance = json.loads(provenance_bytes)
    assert provenance["gate_states"] == {gate: "pass" for gate in gates}
    assert provenance["fixture_consumers"] == list(gates)
    assert "fixture_hashes" not in provenance
    assert "gate_fixture_consumers" not in provenance
    assert "h3_profile" not in provenance
    assert "h3-coupled-v1" not in provenance_bytes
    assert "h3-zero-control-v1" not in provenance_bytes
    assert "H4" not in provenance_bytes and "H5" not in provenance_bytes
    assert not (run_dir / "validation" / "h4.json").exists()
    assert not (run_dir / "validation" / "h5.json").exists()


def test_invalid_raw_mapping_fails_resolution_and_creates_no_run(tmp_path: Path) -> None:
    module = _load_launcher()
    raw = _h1_h5_scientific_config(module)
    raw["artifacts"]["run_root"] = str(tmp_path / "runs")
    raw["model"]["horizon"] = 3

    with pytest.raises(ValueError):
        _run_h1_h5_scientific(module, raw)

    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize(
    "control",
    [
        ".verification",
        ".VeRiFiCaTiOn",
        ".verification...",
        ".verification....",
        ".VeRiFiCaTiOn... ",
        ".git",
        ".GIT",
        ".git...",
        ".git....",
        ".GiT... ",
    ],
)
def test_launcher_rejects_control_tree_run_root_before_evaluation(
    monkeypatch: pytest.MonkeyPatch, control: str
) -> None:
    import verification.run_gates as gates

    module = _load_launcher()
    raw = _h1_h5_scientific_config(module)
    forbidden = REPO_ROOT / control / "task6-forbidden-publication"
    raw["artifacts"]["run_root"] = str(forbidden)
    monkeypatch.setattr(gates, "run_verification", lambda config: pytest.fail("evaluated"))

    assert not forbidden.exists()
    with pytest.raises(ValueError, match="control tree"):
        _run_h1_h5_scientific(module, raw)
    assert not forbidden.exists()


@pytest.mark.parametrize("control", [".verification", ".git"])
def test_launcher_rejects_explicit_extended_control_path_before_evaluation(
    monkeypatch: pytest.MonkeyPatch, control: str
) -> None:
    import verification.run_gates as gates

    module = _load_launcher()
    raw = _h1_h5_scientific_config(module)
    forbidden = REPO_ROOT / control / "task6-extended-forbidden"
    raw["artifacts"]["run_root"] = "\\\\?\\" + str(forbidden)
    monkeypatch.setattr(gates, "run_verification", lambda config: pytest.fail("evaluated"))

    assert not forbidden.exists()
    with pytest.raises(ValueError, match="control tree"):
        _run_h1_h5_scientific(module, raw)
    assert not forbidden.exists()


def test_launcher_rejects_extended_repository_and_parent_before_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import verification.run_gates as gates

    module = _load_launcher()
    monkeypatch.setattr(gates, "run_verification", lambda config: pytest.fail("evaluated"))

    for unsafe in (REPO_ROOT, REPO_ROOT.parent):
        raw = _h1_h5_scientific_config(module)
        raw["artifacts"]["run_root"] = "\\\\?\\" + str(unsafe)
        with pytest.raises(ValueError, match="contain the repository"):
            _run_h1_h5_scientific(module, raw)


def test_launcher_resolves_repo_paths_from_file_not_cwd_with_spaces(tmp_path: Path) -> None:
    working = tmp_path / "cwd with spaces"
    working.mkdir()
    run_root = tmp_path / "subprocess runs"
    code = (
        f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r}); "
        "import verify_vfe4; "
        "op=verify_vfe4.CONFIG['operations']['h1_h5']; "
        "op['enabled']=True; "
        "op['authorization']=verify_vfe4._VERIFY_AUTHORIZATIONS['h1_h5']; "
        "op['config']['validation']['gates']=['H1','H2','H3']; "
        "op['config'].pop('h4'); op['config'].pop('h5'); "
        f"op['config']['artifacts']['run_root']={str(run_root)!r}; "
        "raise SystemExit(verify_vfe4._script_main())"
    )
    completed = subprocess.run([sys.executable, "-c", code], cwd=working, text=True, capture_output=True, timeout=120)
    assert completed.returncode == 0, completed.stderr
    assert "H1: pass" in completed.stdout
    assert "H2: pass" in completed.stdout
    assert "H3: pass" in completed.stdout
    assert "artifact:" in completed.stdout


def test_publication_error_prints_artifact_unavailable_and_returns_nonzero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import verification.run_gates as gates

    module = _load_launcher()
    monkeypatch.setattr(
        gates,
        "run_verification",
        lambda config, **_kwargs: (_ for _ in ()).throw(
            ArtifactPublicationError("disk")
        ),
    )
    h1_h5 = module.CONFIG["operations"]["h1_h5"]
    monkeypatch.setitem(h1_h5, "enabled", True)
    monkeypatch.setitem(
        h1_h5,
        "authorization",
        module._VERIFY_AUTHORIZATIONS["h1_h5"],
    )
    assert module._script_main() != 0
    assert "artifact unavailable" in capsys.readouterr().err


@pytest.mark.parametrize("status", [GateStatus.FAIL, GateStatus.INCONCLUSIVE])
def test_script_exit_is_nonzero_for_fail_and_inconclusive(
    monkeypatch: pytest.MonkeyPatch, status: GateStatus
) -> None:
    module = _load_launcher()
    if status is GateStatus.FAIL:
        result = GateResult(
            gate="H1",
            status=status,
            fixture_id="h1-v1",
            residual=1.0,
            calibrated_allowance=0.0,
            measurements={"m": 0.0},
            invariants=(InvariantResult("i", False, 1.0, 0.0, "failed"),),
            obligations=(),
        )
    else:
        result = GateResult(
            gate="H1",
            status=status,
            fixture_id="h1-v1",
            residual=None,
            calibrated_allowance=None,
            measurements={},
            invariants=(),
            obligations=("unavailable",),
        )
    passing = GateResult(
        gate="H2",
        status=GateStatus.PASS,
        fixture_id="h1-v1",
        residual=0.0,
        calibrated_allowance=0.0,
        measurements={"m": 0.0},
        invariants=(InvariantResult("i", True, 0.0, 0.0, "passed"),),
        obligations=(),
    )
    monkeypatch.setattr(
        module,
        "main",
        lambda: SimpleNamespace(gate_results=(result, passing), run_directory=Path("run")),
    )
    assert module._script_main() == 1


def test_repeated_frozen_clock_collision_preserves_first_run_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import verification.run_gates as gates

    module = _load_launcher()
    raw = _h3_compatibility_config(module)
    raw["artifacts"]["run_root"] = str(tmp_path / "runs")
    monkeypatch.setattr(gates, "_utc_now", lambda: "2026-07-21T23-59-58.000000Z")
    first = _run_h1_h5_scientific(module, raw)
    assert all(item.status is GateStatus.PASS for item in first.gate_results)
    run_dir = next((tmp_path / "runs").iterdir())
    before = {path.relative_to(run_dir): path.read_bytes() for path in run_dir.rglob("*") if path.is_file()}

    with pytest.raises(ArtifactPublicationError, match="already exists"):
        _run_h1_h5_scientific(module, raw)

    after = {path.relative_to(run_dir): path.read_bytes() for path in run_dir.rglob("*") if path.is_file()}
    assert after == before


def test_provenance_schema_is_frozen_and_content_hashes_recompute(tmp_path: Path) -> None:
    module = _load_launcher()
    raw = _h3_compatibility_config(module)
    raw["artifacts"]["run_root"] = str(tmp_path / "runs")
    _run_h1_h5_scientific(module, raw)
    run_dir = next((tmp_path / "runs").iterdir())
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    assert set(provenance) == {
        "git_head", "dirty_digest", "dirty_content_digest", "config_sha256", "objective_schema_input",
        "objective_schema_sha256", "fixture_expected_sha256",
        "fixture_observed_sha256", "fixture_available", "python_version", "pytorch_version",
        "numpy_version", "device", "dtype", "seed", "deterministic",
        "stochastic_policy", "started_utc", "ended_utc", "gate_state",
        "gate_states", "fixture_consumers", "fixture_hashes",
        "gate_fixture_consumers", "h3_profile",
    }
    assert len(provenance["git_head"]) == 40 and provenance["git_head"] != "unknown"
    assert provenance["dirty_content_digest"] == provenance["dirty_digest"]
    assert provenance["config_sha256"] == json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["config_sha256"]
    fixture = REPO_ROOT / "vfe4" / "validation" / "fixtures" / "h1_v1.json"
    fixture_sha = hashlib.sha256(fixture.read_bytes()).hexdigest()
    assert provenance["fixture_expected_sha256"] == fixture_sha
    assert provenance["fixture_observed_sha256"] == fixture_sha
    assert provenance["fixture_available"] is True
    assert provenance["objective_schema_sha256"] == hashlib.sha256(provenance["objective_schema_input"].encode("utf-8")).hexdigest()
    assert provenance["device"] == "cpu" and provenance["dtype"] == "float64"
    assert provenance["stochastic_policy"] == "no-stochastic-operations"
    assert provenance["gate_state"] == "pass"
    assert provenance["gate_states"] == {
        "H1": "pass",
        "H2": "pass",
        "H3": "pass",
    }
    assert provenance["fixture_consumers"] == ["H1", "H2"]
    assert provenance["gate_fixture_consumers"] == {
        "H1": ["h1-v1"],
        "H2": ["h1-v1"],
        "H3": ["h3-coupled-v1", "h3-zero-control-v1"],
    }
    assert provenance["h3_profile"] == raw["h3"]


def test_h8_click_run_executes_fake_selected_runner_and_publishes_runtime_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import verification.h8_gate as h8_gate
    import verification.h8_orchestrator as h8_orchestrator
    import verification.run_gates as gates
    import vfe4.artifacts as artifact_api
    import vfe4.config as config_api
    from test_support.h8_runtime_fakes import (
        make_fake_h8_process_record,
        make_pass_correctness_cells,
        make_test_parent_identities,
    )
    from vfe4.artifacts import source_candidate_sha256
    from vfe4.config import H8ValidationConfig
    from vfe4.types.results import H8GateResult

    module = _load_launcher()
    head, dirty, junit = "1" * 40, "2" * 64, "3" * 64
    refs, registry_bytes = _h8_current_refs(head, dirty, junit)
    refs_root = tmp_path / ".verification"
    refs_root.mkdir()
    registry_path = refs_root / f"h8-current-candidate-{head}-refs.json"
    registry_path.write_bytes(registry_bytes)
    preregistration = tmp_path / "h8-preregistration.md"
    preregistration.write_bytes(b"selected runtime H8 preregistration\n")
    run_root = tmp_path / "runs"

    launcher = copy.deepcopy(module.CONFIG)
    h8_entry = launcher["operations"]["h8"]
    h8_entry["enabled"] = True
    h8_entry["authorization"] = module._VERIFY_AUTHORIZATIONS["h8"]
    h8_entry["config"]["artifacts"]["run_root"] = str(run_root)

    counts = {
        "read": 0,
        "parse": 0,
        "bind": 0,
        "prerequisites": 0,
        "correctness": 0,
        "parent": 0,
        "identities": 0,
        "child": 0,
        "pointer": 0,
        "subprocess": 0,
    }
    pointer_payloads: list[dict[str, object]] = []
    original_read_bytes = Path.read_bytes
    original_parse = gates.parse_h8_reference_registry_bytes
    original_bind = config_api.bind_h8_current_refs
    original_pointer_builder = gates.h8_current_candidate_result_payload

    def counted_read_bytes(path: Path) -> bytes:
        if path.resolve(strict=False) == registry_path.resolve(strict=False):
            counts["read"] += 1
        return original_read_bytes(path)

    def counted_parse(value: bytes):
        counts["parse"] += 1
        assert value == registry_bytes
        return original_parse(value)

    def counted_bind(scientific: object, current_refs: object):
        counts["bind"] += 1
        assert current_refs == refs
        return original_bind(scientific, current_refs)  # type: ignore[arg-type]

    def validate_prerequisites(current_refs: object):
        counts["prerequisites"] += 1
        assert current_refs == refs
        return h8_gate.H8PrerequisiteArtifactValidation.create(
            registry_sha256=refs.registry_sha256,
            revalidated_reference_names=h8_gate.H8_POINTER_PREDECESSOR_KEYS,
            obligations=(),
        )

    correctness = make_pass_correctness_cells()

    def produce_correctness():
        counts["correctness"] += 1
        return correctness

    def fake_parent_attempt(
        *,
        authorization: object,
        repository_root: str | Path,
        base_environment: object = None,
    ):
        counts["parent"] += 1
        assert authorization.valid_start is True
        counts["identities"] += 1
        identities = make_test_parent_identities()

        def fake_child(invocation: object):
            counts["child"] += 1
            return make_fake_h8_process_record(invocation)  # type: ignore[arg-type]

        parent_run = h8_orchestrator._run_h8_parent_attempt_for_test(
            authorization=authorization,
            repository_root=repository_root,
            identities=identities,
            base_environment=base_environment,
            child_runner=fake_child,
        )
        return h8_orchestrator._mint_h8_parent_attempt_authority(parent_run)

    def counted_pointer(*args: object, **kwargs: object) -> dict[str, object]:
        counts["pointer"] += 1
        payload = original_pointer_builder(*args, **kwargs)
        pointer_payloads.append(payload)
        return payload

    def forbidden_subprocess(*_args: object, **_kwargs: object) -> object:
        counts["subprocess"] += 1
        raise AssertionError("fake selected H8 runner launched a subprocess")

    source_identity = (
        head,
        dirty,
        source_candidate_sha256(
            git_head_value=head,
            dirty_digest_value=dirty,
        ),
    )
    timestamps = iter(
        ("2026-07-23T00:00:00.000000Z", "2026-07-23T00:00:00.000001Z")
    )
    monkeypatch.setattr(module, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(gates, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(gates, "H8_PREREGISTRATION_PATH", preregistration)
    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)
    monkeypatch.setattr(gates, "parse_h8_reference_registry_bytes", counted_parse)
    monkeypatch.setattr(config_api, "bind_h8_current_refs", counted_bind)
    monkeypatch.setattr(
        gates,
        "validate_h8_prerequisite_artifacts",
        validate_prerequisites,
    )
    monkeypatch.setattr(gates, "produce_h8_correctness_grid", produce_correctness)
    monkeypatch.setattr(gates, "run_h8_parent_attempt", fake_parent_attempt)
    monkeypatch.setattr(
        gates,
        "h8_current_candidate_result_payload",
        counted_pointer,
    )
    monkeypatch.setattr(
        artifact_api, "current_source_identity", lambda *_args: source_identity
    )
    monkeypatch.setattr(gates, "current_source_identity", lambda *_args: source_identity)
    monkeypatch.setattr(gates, "_utc_now", lambda: next(timestamps))
    monkeypatch.setattr(gates, "run_verification", forbidden_subprocess)
    monkeypatch.setattr(gates, "run_h7_verification", forbidden_subprocess)
    monkeypatch.setattr(subprocess, "run", forbidden_subprocess)
    monkeypatch.setattr(subprocess, "Popen", forbidden_subprocess)

    published = module.main(launcher)

    assert counts == {
        "read": 1,
        "parse": 1,
        "bind": 1,
        "prerequisites": 1,
        "correctness": 1,
        "parent": 1,
        "identities": 1,
        "child": 30,
        "pointer": 1,
        "subprocess": 0,
    }
    assert len(published.gate_results) == 1
    result = published.gate_results[0]
    assert type(result) is H8GateResult
    assert result.gate == "H8"
    assert result.status is GateStatus.PASS
    assert result.obligations == ()
    assert len(result.correctness) == 12
    assert len(result.child_attempts) == 30
    assert len(result.production_runs) == 15
    assert len(result.profiler_runs) == 3
    assert len(result.controls) == 12

    class DerivedH8GateResult(H8GateResult):
        pass

    with pytest.raises(ValueError, match="exact result type"):
        DerivedH8GateResult(**vars(result))

    config_bytes = (published.run_directory / "config.json").read_bytes()
    validation = json.loads(
        (published.run_directory / "validation" / "h8.json").read_bytes()
    )
    environment = json.loads(
        (published.run_directory / "environment.json").read_bytes()
    )
    provenance = json.loads(
        (published.run_directory / "provenance.json").read_bytes()
    )
    h8_config_sha256 = H8ValidationConfig.create().config_sha256
    assert hashlib.sha256(config_bytes).hexdigest() == result.config_sha256
    assert result.config_sha256 != h8_config_sha256
    assert validation["config"]["config_sha256"] == h8_config_sha256
    assert validation["config"]["canonical_json_sha256"] == h8_config_sha256
    assert validation["status"] == "pass"
    assert validation["correctness"]["cell_count"] == 12
    assert validation["correctness"]["all_pass"] is True
    assert len(validation["child_attempts"]) == 30
    assert len(validation["production_runs"]) == 15
    assert len(validation["profiler_runs"]) == 3
    assert len(validation["controls"]) == 12
    assert validation["invariants"]["all_pass"] is True
    assert environment == validation["environment"]
    assert environment["platform"] == "test"
    assert environment["hardware_identity_sha256"]
    assert environment["thread_identity_sha256"]
    assert environment["blas_identity_sha256"]
    assert provenance["config_sha256"] == result.config_sha256
    assert provenance["validation_sha256"] == hashlib.sha256(
        (published.run_directory / "validation" / "h8.json").read_bytes()
    ).hexdigest()
    assert provenance["status"] == "pass"
    assert provenance["execution_scope"] == "h8-parent-orchestrated-runtime-v1"
    assert provenance["external_pointer_in_artifact"] is False

    files = sorted(
        path.relative_to(published.run_directory).as_posix()
        for path in published.run_directory.rglob("*")
        if path.is_file()
    )
    assert files == [
        "config.json",
        "environment.json",
        "manifest.sha256",
        "provenance.json",
        "references/h6_prediction.json",
        "references/h7.json",
        "validation/h8.json",
    ]
    manifest_names = [
        line.split("  ", 1)[1]
        for line in (published.run_directory / "manifest.sha256")
        .read_text(encoding="ascii")
        .splitlines()
    ]
    assert manifest_names == sorted(name for name in files if name != "manifest.sha256")
    assert len(pointer_payloads) == 1
    pointer = pointer_payloads[0]
    assert tuple(pointer) == (
        "schema_version",
        "candidate",
        "artifact",
        "current_refs",
        "predecessors",
    )
    assert pointer["schema_version"] == "h8-current-candidate-result-v2"
    assert pointer["candidate"] == {
        "git_head": head,
        "dirty_digest": dirty,
        "junit_sha256": junit,
    }
    manifest_sha256 = hashlib.sha256(
        (published.run_directory / "manifest.sha256").read_bytes()
    ).hexdigest()
    assert pointer["artifact"] == {
        "path": published.run_directory.resolve(strict=True).as_posix(),
        "manifest_sha256": manifest_sha256,
        "config_sha256": result.config_sha256,
        "validation_sha256": provenance["validation_sha256"],
    }
    assert pointer["current_refs"] == {
        "path": registry_path.resolve(strict=False).as_posix(),
        "sha256": refs.registry_sha256,
    }
    assert set(pointer["predecessors"]) == set(
        h8_gate.H8_POINTER_PREDECESSOR_KEYS
    )
    assert pointer["predecessors"] == {
        name: json.loads(h8_gate.canonical_h8_json_bytes(getattr(refs, name)))
        for name in h8_gate.H8_POINTER_PREDECESSOR_KEYS
    }
    assert sorted(path.name for path in refs_root.iterdir()) == [registry_path.name]


def test_h8_selected_runner_rejects_invalid_start_without_parent_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import verification.h8_gate as h8_gate
    import verification.run_gates as gates
    import vfe4.config as config_api
    from test_support.h8_runtime_fakes import make_pass_correctness_cells
    from vfe4.artifacts import source_candidate_sha256

    module = _load_launcher()
    head, dirty, junit = "1" * 40, "2" * 64, "3" * 64
    refs, registry_bytes = _h8_current_refs(head, dirty, junit)
    refs_root = tmp_path / ".verification"
    refs_root.mkdir()
    registry_path = refs_root / f"h8-current-candidate-{head}-refs.json"
    registry_path.write_bytes(registry_bytes)
    preregistration = tmp_path / "h8-preregistration.md"
    preregistration.write_bytes(b"selected runtime H8 preregistration\n")
    scientific = copy.deepcopy(module.CONFIG["operations"]["h8"]["config"])
    scientific["artifacts"]["run_root"] = str(tmp_path / "runs")
    canonical = config_api.bind_h8_current_refs(scientific, refs)
    source_identity = (
        head,
        dirty,
        source_candidate_sha256(
            git_head_value=head,
            dirty_digest_value=dirty,
        ),
    )
    prerequisite_validation = h8_gate.H8PrerequisiteArtifactValidation.create(
        registry_sha256=refs.registry_sha256,
        revalidated_reference_names=(),
        obligations=("h8_test_prerequisite_unavailable",),
    )
    parent_calls = 0

    def forbidden_parent(*_args: object, **_kwargs: object) -> object:
        nonlocal parent_calls
        parent_calls += 1
        raise AssertionError("invalid H8 authorization reached the parent runner")

    timestamps = iter(
        ("2026-07-23T00:00:00.000000Z", "2026-07-23T00:00:00.000001Z")
    )
    monkeypatch.setattr(gates, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(gates, "H8_PREREGISTRATION_PATH", preregistration)
    monkeypatch.setattr(
        gates,
        "current_source_identity",
        lambda *_args: source_identity,
    )
    monkeypatch.setattr(gates, "_utc_now", lambda: next(timestamps))
    monkeypatch.setattr(
        gates,
        "validate_h8_prerequisite_artifacts",
        lambda current_refs: (
            prerequisite_validation
            if current_refs == refs
            else pytest.fail("validated another current-reference object")
        ),
    )
    monkeypatch.setattr(
        gates,
        "produce_h8_correctness_grid",
        make_pass_correctness_cells,
    )
    monkeypatch.setattr(gates, "run_h8_parent_attempt", forbidden_parent)

    published = gates.run_h8_verification(
        canonical,
        registry_path=registry_path,
        registry_bytes=registry_bytes,
    )

    assert parent_calls == 0
    result = published.gate_results[0]
    assert result.status is GateStatus.INCONCLUSIVE
    assert "h8_test_prerequisite_unavailable" in result.obligations
    assert result.child_attempts == ()


def test_h8_preflight_click_run_is_advisory_and_writes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import verification.h8_preflight as preflight_api

    module = _load_launcher()
    launcher = copy.deepcopy(module.CONFIG)
    entry = launcher["operations"]["h8_preflight"]
    entry["enabled"] = True
    entry["authorization"] = module._VERIFY_AUTHORIZATIONS["h8_preflight"]
    candidate = {
        "git_head": "1" * 40,
        "dirty_digest": "2" * 64,
        "source_sha256": "3" * 64,
    }

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("H8 preflight launched scientific work")

    monkeypatch.setattr(module, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        preflight_api,
        "capture_current_candidate",
        lambda **_kwargs: candidate,
    )
    monkeypatch.setattr(module, "_run_h1_h5", forbidden)
    monkeypatch.setattr(module, "_run_projected", forbidden)
    monkeypatch.setattr(module, "_run_h6_smc_accuracy", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    before = tuple(sorted(path.as_posix() for path in tmp_path.rglob("*")))

    result = module.main(launcher)

    after = tuple(sorted(path.as_posix() for path in tmp_path.rglob("*")))
    assert after == before
    assert result.operation == "H8-Preflight"
    assert result.disposition == "blocked"
    assert result.scientific_status == "not_evaluated"
    assert result.execution_policy["scientific_children_launched"] == 0
    assert result.execution_policy["artifact_writes"] == 0
    output = capsys.readouterr().out
    assert json.loads(output) == result.as_dict()
    assert result.execution_policy == {
        "inspection_policy": "metadata_only",
        "tests_launched": 0,
        "training_runs_launched": 0,
        "scientific_evaluations_launched": 0,
        "profiler_runs_launched": 0,
        "scientific_children_launched": 0,
        "artifact_writes": 0,
        "result_delivery": "stdout_and_return_value_only",
        "scientific_status_authority": "none",
    }
