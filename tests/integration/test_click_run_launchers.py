from __future__ import annotations

import ast
import copy
import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path, module_name: str) -> object:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def test_click_run_launchers_are_idle_authorized_and_cli_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for filename, module_name in (
        ("verify_vfe4.py", "verify_vfe4_click_contract"),
        ("train_vfe4.py", "train_vfe4_click_contract"),
    ):
        path = REPO_ROOT / filename
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=filename)
        config_assignments = tuple(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "CONFIG"
        )
        mains = tuple(
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "main"
        )
        guards = tuple(
            node
            for node in tree.body
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
        )
        imports = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert len(config_assignments) == len(mains) == len(guards) == 1
        assert imports.isdisjoint({"argparse", "click", "typer", "hydra"})

        module = _load(path, module_name)
        config = copy.deepcopy(module.CONFIG)
        operations = config["operations"]
        assert operations and all(
            entry["enabled"] is False for entry in operations.values()
        )
        idle = module.main(config)
        if filename == "verify_vfe4.py":
            assert idle is None
        else:
            assert idle.status == "IDLE" and idle.operation is None
            with pytest.raises(ValueError, match="IDLE or COMPLETED"):
                module.TrainLauncherResult(
                    "vfe4-train-click-run-v1",
                    next(iter(operations)),
                    "BROKEN",
                    object(),
                )

        first_name = next(iter(operations))
        operations[first_name]["enabled"] = True
        operations[first_name]["authorization"] = "wrong"
        with pytest.raises(PermissionError, match="explicit phrase"):
            module.main(config)

        class EqualitySpoof:
            def __eq__(self, other: object) -> bool:
                return True

        operations[first_name]["authorization"] = EqualitySpoof()
        with pytest.raises(PermissionError, match="explicit phrase"):
            module.main(config)

        second_name = tuple(operations)[1]
        operations[first_name]["authorization"] = (
            module._VERIFY_AUTHORIZATIONS[first_name]
            if filename == "verify_vfe4.py"
            else module._AUTHORIZATION_PHRASES[first_name]
        )
        operations[second_name]["enabled"] = True
        with pytest.raises(ValueError, match="exactly one"):
            module.main(config)

        dispatched: list[tuple[str, object]] = []
        marker = object()
        if filename == "verify_vfe4.py":
            monkeypatch.setattr(
                module,
                "_run_h1_h5",
                lambda raw: dispatched.append(("h1_h5", raw)) or marker,
            )
            monkeypatch.setattr(
                module,
                "_run_projected",
                lambda operation, raw: (
                    dispatched.append((operation, raw)) or marker
                ),
            )
            authorizations = module._VERIFY_AUTHORIZATIONS
        else:
            monkeypatch.setattr(
                module,
                "_run_readiness",
                lambda raw: dispatched.append(
                    ("prediction_readiness", raw)
                )
                or marker,
            )
            monkeypatch.setattr(
                module,
                "_run_experiment",
                lambda operation, raw, authorization: (
                    dispatched.append((operation, raw)) or marker
                ),
            )
            authorizations = module._AUTHORIZATION_PHRASES

        for operation in operations:
            authorized = copy.deepcopy(module.CONFIG)
            entry = authorized["operations"][operation]
            entry["enabled"] = True
            entry["authorization"] = authorizations[operation]
            result = module.main(authorized)
            if filename == "verify_vfe4.py":
                assert result is marker
            else:
                assert result.status == "COMPLETED"
                assert result.operation == operation
                assert result._payload is marker
        assert tuple(name for name, _ in dispatched) == tuple(operations)
