from __future__ import annotations

import ast
import builtins
import copy
import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_LAUNCHER_PATH = _REPO_ROOT / "train_vfe4.py"
_OPERATIONS = (
    "prediction_readiness",
    "plan",
    "train",
    "score_validation",
    "score_test_transaction",
)
_AUTHORIZATIONS = {
    "prediction_readiness": "AUTHORIZE_VFE4_H6_PREDICTION_READINESS_V1",
    "plan": "AUTHORIZE_VFE4_H6_EXPERIMENT_PLAN_V1",
    "train": "AUTHORIZE_VFE4_H6_TRAINING_V1",
    "score_validation": "AUTHORIZE_VFE4_H6_VALIDATION_SCORING_V1",
    "score_test_transaction": (
        "AUTHORIZE_VFE4_H6_ONE_TIME_TEST_TRANSACTION_V1"
    ),
}


def _load_launcher(module_name: str) -> object:
    spec = importlib.util.spec_from_file_location(module_name, _LAUNCHER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _enabled_config(launcher: object, operation: str, authorization: str) -> dict:
    config = copy.deepcopy(launcher.CONFIG)
    entry = config["operations"][operation]
    entry["enabled"] = True
    entry["authorization"] = authorization
    return config


def _contains_empty_mapping(value: object) -> bool:
    if isinstance(value, dict):
        return not value or any(
            _contains_empty_mapping(item) for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_empty_mapping(item) for item in value)
    return False


def test_click_launcher_accepts_only_v3_operation_inventory() -> None:
    launcher = _load_launcher("train_vfe4_task11_inventory")

    assert launcher.CONFIG["launcher_schema"] == "vfe4-train-click-run-v3"
    assert tuple(launcher.CONFIG["operations"]) == _OPERATIONS
    assert all(
        entry["enabled"] is False
        for entry in launcher.CONFIG["operations"].values()
    )
    assert not _contains_empty_mapping(launcher.CONFIG)

    from vfe4.config import (
        H6PredictionV3ResolvedConfig,
        resolve_h6_prediction_v3_config,
    )

    for entry in launcher.CONFIG["operations"].values():
        operation_config = entry["config"]
        scientific = operation_config["scientific_config"]
        assert scientific["schema_version"] == "h6-prediction-config-v3"
        assert len(
            scientific["prerequisites"][
                "a0_direct_exact_prefix_certificate_sha256"
            ]
        ) == 64
        assert tuple(operation_config["correctness_artifact_roots"]) == (
            "H1",
            "H2",
            "H3",
            "H5",
        )
        assert operation_config["h1_prefix_prior_artifact_root"]
        assert operation_config["smc_accuracy_artifact_root"]
        assert operation_config["h6_prefix_artifact_root"]
        for name in (
            "h6_prefix_manifest_sha256",
            "h6_prefix_junit_sha256",
        ):
            digest = operation_config[name]
            assert type(digest) is str
            assert len(digest) == 64
            assert set(digest) <= set("0123456789abcdef")
        resolved = resolve_h6_prediction_v3_config(
            scientific,
            repo_root=_REPO_ROOT,
        )
        assert type(resolved) is H6PredictionV3ResolvedConfig

    idle = launcher.main(copy.deepcopy(launcher.CONFIG))
    assert idle.launcher_schema == "vfe4-train-click-run-v3"
    assert idle.operation is None
    assert idle.status == "IDLE"


def test_click_launcher_retains_prediction_readiness_authorization() -> None:
    launcher = _load_launcher("train_vfe4_task11_authorizations")

    for operation in _OPERATIONS[:4]:
        selected = launcher._selected_operation(
            _enabled_config(
                launcher,
                operation,
                _AUTHORIZATIONS[operation],
            )
        )
        assert selected is not None
        assert selected[0] == operation
        assert selected[2] == _AUTHORIZATIONS[operation]


def test_click_launcher_rejects_both_legacy_split_test_operations() -> None:
    launcher = _load_launcher("train_vfe4_task11_legacy_rejection")

    assert "reserve_test_opening" not in launcher.CONFIG["operations"]
    assert "score_test" not in launcher.CONFIG["operations"]
    assert "reserve_test_opening" not in launcher._AUTHORIZATION_PHRASES
    assert "score_test" not in launcher._AUTHORIZATION_PHRASES

    for legacy_name in ("reserve_test_opening", "score_test"):
        config = copy.deepcopy(launcher.CONFIG)
        config["operations"][legacy_name] = {
            "enabled": False,
            "authorization": None,
            "config": {"forbidden": True},
        }
        with pytest.raises(ValueError, match="incomplete|reordered|unknown"):
            launcher.main(config)


def test_test_transaction_requires_exact_new_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _load_launcher("train_vfe4_task11_test_authorization")
    compared: list[tuple[object, object]] = []
    compare_digest = launcher.hmac.compare_digest

    def recording_compare_digest(left: object, right: object) -> bool:
        compared.append((left, right))
        return compare_digest(left, right)

    monkeypatch.setattr(
        launcher.hmac,
        "compare_digest",
        recording_compare_digest,
    )
    for rejected in (
        "AUTHORIZE_VFE4_H6_TEST_RESERVATION_V1",
        "AUTHORIZE_VFE4_H6_ONE_TIME_TEST_SCORING_V1",
        "AUTHORIZE_VFE4_H6_ONE_TIME_TEST_TRANSACTION_V1 ",
    ):
        with pytest.raises(PermissionError, match="explicit phrase"):
            launcher._selected_operation(
                _enabled_config(
                    launcher,
                    "score_test_transaction",
                    rejected,
                )
            )

    selected = launcher._selected_operation(
        _enabled_config(
            launcher,
            "score_test_transaction",
            _AUTHORIZATIONS["score_test_transaction"],
        )
    )
    assert selected is not None
    assert selected[0] == "score_test_transaction"
    assert compared[-1] == (
        _AUTHORIZATIONS["score_test_transaction"],
        _AUTHORIZATIONS["score_test_transaction"],
    )

    source = _LAUNCHER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=_LAUNCHER_PATH.name)
    imported_task10_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "vfe4.training.h6_test_transaction_v3"
        for alias in node.names
    }
    assert "execute_h6_test_transaction_v3" in imported_task10_names
    assert imported_task10_names <= {
        "execute_h6_test_transaction_v3",
        "finalize_h6_test_transaction_v3",
        "recover_h6_test_transaction_v3",
    }


def test_click_launcher_requires_no_cli_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported_vfe4: list[str] = []
    real_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: object = None,
        locals: object = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "vfe4" or name.startswith("vfe4."):
            imported_vfe4.append(name)
            raise AssertionError("launcher import performed scientific work")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    launcher = _load_launcher("train_vfe4_task11_idle_import")
    monkeypatch.setattr(builtins, "__import__", real_import)
    assert imported_vfe4 == []

    source = _LAUNCHER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=_LAUNCHER_PATH.name)
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imports.isdisjoint({"argparse", "click", "typer", "hydra"})
    assert not any(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
        and node.attr == "argv"
        for node in ast.walk(tree)
    )
    assert not any(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
        and node.attr in {"environ", "getenv"}
        for node in ast.walk(tree)
    )
    config_assignments = tuple(
        node
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "CONFIG"
    )
    assert len(config_assignments) == 1
    assert tuple(inspect.signature(launcher.main).parameters) == ("config",)

    events: list[str] = []
    resolved = SimpleNamespace(runtime=object())
    runtime = object()
    payload = object()

    fake_config = ModuleType("vfe4.config")

    def resolve_v3(raw: object, *, repo_root: Path) -> object:
        assert raw
        assert repo_root == _REPO_ROOT
        events.append("resolve_v3")
        return resolved

    def reject_legacy(*_args: object, **_kwargs: object) -> object:
        pytest.fail("launcher reached a legacy resolver or runner")

    fake_config.resolve_h6_prediction_v3_config = resolve_v3
    fake_config.resolve_h6_prediction_config = reject_legacy

    fake_runtime = ModuleType("vfe4.training.h6_runtime_v3")

    def configure_runtime(*, expected_identity: object) -> object:
        assert expected_identity is resolved.runtime
        events.append("configure_runtime")
        return runtime

    fake_runtime.configure_installed_runtime_v3 = configure_runtime

    fake_experiment = ModuleType("vfe4.training.h6_experiment_v3")

    def run_v3(**kwargs: object) -> object:
        assert kwargs["operation"] == "train"
        assert kwargs["config"] is resolved
        assert kwargs["runtime"] is runtime
        events.append("run_v3")
        return payload

    fake_experiment.run_h6_experiment_v3 = run_v3
    fake_legacy_experiment = ModuleType("vfe4.training.h6_experiment")
    fake_legacy_experiment.run_h6_experiment = reject_legacy

    monkeypatch.setitem(sys.modules, "vfe4.config", fake_config)
    monkeypatch.setitem(
        sys.modules,
        "vfe4.training.h6_runtime_v3",
        fake_runtime,
    )
    monkeypatch.setitem(
        sys.modules,
        "vfe4.training.h6_experiment_v3",
        fake_experiment,
    )
    monkeypatch.setitem(
        sys.modules,
        "vfe4.training.h6_experiment",
        fake_legacy_experiment,
    )

    result = launcher.main(
        _enabled_config(
            launcher,
            "train",
            _AUTHORIZATIONS["train"],
        )
    )
    assert events == ["resolve_v3", "configure_runtime", "run_v3"]
    assert result.status == "COMPLETED"
    assert result.operation == "train"
    assert result._payload is payload
