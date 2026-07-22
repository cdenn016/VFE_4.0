from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
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


def test_launcher_import_is_safe_and_has_no_cli_framework(monkeypatch: pytest.MonkeyPatch) -> None:
    import verification.run_gates as gates

    monkeypatch.setattr(
        gates, "run_verification", lambda config: pytest.fail("ran on import")
    )
    module = _load_launcher()
    assert type(module.CONFIG) is dict
    assert module.CONFIG["validation"]["gates"] == ["H1", "H2", "H3"]
    h3 = module.CONFIG["h3"]
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


def test_editable_mapping_runs_three_gates_once_from_one_fixture_snapshot_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import verification.run_gates as gates

    module = _load_launcher()
    raw = copy.deepcopy(module.CONFIG)
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

    result = module.main(raw)

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


@pytest.mark.parametrize("gates", (("H1",), ("H1", "H2")))
def test_compatibility_prefixes_never_touch_or_publish_h3(
    gates: tuple[str, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import verification.run_gates as run_gates

    module = _load_launcher()
    raw = copy.deepcopy(module.CONFIG)
    raw["validation"]["gates"] = list(gates)
    del raw["h3"]
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

    result = module.main(raw)

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


def test_invalid_raw_mapping_fails_resolution_and_creates_no_run(tmp_path: Path) -> None:
    module = _load_launcher()
    raw = copy.deepcopy(module.CONFIG)
    raw["artifacts"]["run_root"] = str(tmp_path / "runs")
    raw["model"]["horizon"] = 3

    with pytest.raises(ValueError):
        module.main(raw)

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
    module = _load_launcher()
    raw = copy.deepcopy(module.CONFIG)
    forbidden = REPO_ROOT / control / "task6-forbidden-publication"
    raw["artifacts"]["run_root"] = str(forbidden)
    monkeypatch.setattr(module, "run_verification", lambda config: pytest.fail("evaluated"))

    assert not forbidden.exists()
    with pytest.raises(ValueError, match="control tree"):
        module.main(raw)
    assert not forbidden.exists()


@pytest.mark.parametrize("control", [".verification", ".git"])
def test_launcher_rejects_explicit_extended_control_path_before_evaluation(
    monkeypatch: pytest.MonkeyPatch, control: str
) -> None:
    module = _load_launcher()
    raw = copy.deepcopy(module.CONFIG)
    forbidden = REPO_ROOT / control / "task6-extended-forbidden"
    raw["artifacts"]["run_root"] = "\\\\?\\" + str(forbidden)
    monkeypatch.setattr(module, "run_verification", lambda config: pytest.fail("evaluated"))

    assert not forbidden.exists()
    with pytest.raises(ValueError, match="control tree"):
        module.main(raw)
    assert not forbidden.exists()


def test_launcher_rejects_extended_repository_and_parent_before_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_launcher()
    monkeypatch.setattr(module, "run_verification", lambda config: pytest.fail("evaluated"))

    for unsafe in (REPO_ROOT, REPO_ROOT.parent):
        raw = copy.deepcopy(module.CONFIG)
        raw["artifacts"]["run_root"] = "\\\\?\\" + str(unsafe)
        with pytest.raises(ValueError, match="contain the repository"):
            module.main(raw)


def test_launcher_resolves_repo_paths_from_file_not_cwd_with_spaces(tmp_path: Path) -> None:
    working = tmp_path / "cwd with spaces"
    working.mkdir()
    run_root = tmp_path / "subprocess runs"
    code = (
        f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r}); "
        "import verify_vfe4; "
        f"verify_vfe4.CONFIG['artifacts']['run_root'] = {str(run_root)!r}; "
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
    module = _load_launcher()
    monkeypatch.setattr(module, "run_verification", lambda config: (_ for _ in ()).throw(ArtifactPublicationError("disk")))

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
    raw = copy.deepcopy(module.CONFIG)
    raw["artifacts"]["run_root"] = str(tmp_path / "runs")
    monkeypatch.setattr(gates, "_utc_now", lambda: "2026-07-21T23-59-58.000000Z")
    first = module.main(raw)
    assert all(item.status is GateStatus.PASS for item in first.gate_results)
    run_dir = next((tmp_path / "runs").iterdir())
    before = {path.relative_to(run_dir): path.read_bytes() for path in run_dir.rglob("*") if path.is_file()}

    with pytest.raises(ArtifactPublicationError, match="already exists"):
        module.main(raw)

    after = {path.relative_to(run_dir): path.read_bytes() for path in run_dir.rglob("*") if path.is_file()}
    assert after == before


def test_provenance_schema_is_frozen_and_content_hashes_recompute(tmp_path: Path) -> None:
    module = _load_launcher()
    raw = copy.deepcopy(module.CONFIG)
    raw["artifacts"]["run_root"] = str(tmp_path / "runs")
    module.main(raw)
    run_dir = next((tmp_path / "runs").iterdir())
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    assert set(provenance) == {
        "git_head", "dirty_digest", "config_sha256", "objective_schema_input",
        "objective_schema_sha256", "fixture_expected_sha256",
        "fixture_observed_sha256", "fixture_available", "python_version", "pytorch_version",
        "numpy_version", "device", "dtype", "seed", "deterministic",
        "stochastic_policy", "started_utc", "ended_utc", "gate_state",
        "gate_states", "fixture_consumers", "fixture_hashes",
        "gate_fixture_consumers", "h3_profile",
    }
    assert len(provenance["git_head"]) == 40 and provenance["git_head"] != "unknown"
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
