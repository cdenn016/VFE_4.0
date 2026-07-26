from __future__ import annotations

import ast
import builtins
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def test_click_builder_is_idle_authorized_and_lazy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    launcher_path = (
        Path(__file__).parents[2] / "build_h6_validation_perturbations.py"
    )
    tree = ast.parse(launcher_path.read_text(encoding="utf-8"))
    assert [
        node.targets[0].id
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "CONFIG"
    ] == ["CONFIG"]
    assert [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ] == ["main"]
    assert sum(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        for node in tree.body
    ) == 1

    module_name = "_vfe4_test_h6_validation_candidate_launcher"
    spec = importlib.util.spec_from_file_location(module_name, launcher_path)
    assert spec is not None and spec.loader is not None
    launcher = importlib.util.module_from_spec(spec)
    oracle_name = "verification.numpy_oracles.h6_prefix"
    candidate_name = "verification.h6_validation_candidate"
    before_oracle = sys.modules.get(oracle_name)
    before_candidate = sys.modules.get(candidate_name)
    spec.loader.exec_module(launcher)
    assert sys.modules.get(oracle_name) is before_oracle
    assert sys.modules.get(candidate_name) is before_candidate
    assert launcher.CONFIG == {
        "enabled": False,
        "authorization": None,
        "config": {},
    }

    imported: list[str] = []
    touched: list[object] = []
    fake_candidate = ModuleType(candidate_name)

    def fake_run(config: object) -> dict[str, object]:
        touched.append(config)
        return {
            "status": "CANDIDATE",
            "artifact_reference": {"reference_sha256": "a" * 64},
        }

    fake_candidate.run_h6_validation_perturbation_build = fake_run
    original_import = builtins.__import__

    def controlled_import(
        name: str,
        globals: object = None,
        locals: object = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> ModuleType:
        imported.append(name)
        if name == candidate_name:
            return fake_candidate
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", controlled_import)
    idle = launcher.main()
    artifact_root = tmp_path / "artifacts"
    assert idle == {
        "operation": "H6-Validation-Perturbations",
        "status": "IDLE",
    }
    assert '"status":"IDLE"' in capsys.readouterr().out
    assert candidate_name not in imported
    assert oracle_name not in imported
    assert touched == []
    assert not artifact_root.exists()

    for invalid in (None, "wrong", object()):
        launcher.CONFIG = {
            "enabled": True,
            "authorization": invalid,
            "config": {"artifact_root": artifact_root},
        }
        with pytest.raises(PermissionError, match="authorization"):
            launcher.main()
        assert candidate_name not in imported
        assert touched == []
        assert not artifact_root.exists()

    configured = {"artifact_root": artifact_root}
    launcher.CONFIG = {
        "enabled": True,
        "authorization": (
            "AUTHORIZE_VFE4_H6_VALIDATION_PERTURBATIONS_4096_V1"
        ),
        "config": configured,
    }
    result = launcher.main()
    assert result == {
        "status": "CANDIDATE",
        "artifact_reference": {"reference_sha256": "a" * 64},
    }
    assert touched == [configured]
    assert imported.count(candidate_name) == 1
    assert oracle_name not in imported
    assert '"status":"CANDIDATE"' in capsys.readouterr().out
