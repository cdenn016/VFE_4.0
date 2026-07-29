from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TYPES_ROOT = REPOSITORY_ROOT / "vfe4" / "types"
_H7_SNAPSHOT_ADAPTER_OWNERS = {
    "_pushforward_h7_generative_snapshot": (
        REPOSITORY_ROOT / "vfe4" / "generative" / "pushforward.py"
    ),
    "_pushforward_h7_recognition_snapshot": (
        REPOSITORY_ROOT / "vfe4" / "recognition" / "pushforward.py"
    ),
}


def _h7_consumer_python_paths() -> tuple[Path, ...]:
    roots = (
        REPOSITORY_ROOT / "vfe4",
        REPOSITORY_ROOT / "verification",
        REPOSITORY_ROOT / "tests",
        REPOSITORY_ROOT / "test_support",
    )
    paths = sorted(REPOSITORY_ROOT.glob("*.py"))
    for root in roots:
        paths.extend(sorted(root.rglob("*.py")))
    return tuple(paths)


def _is_numerics_module(module_name: str) -> bool:
    return module_name == "vfe4.numerics" or module_name.startswith("vfe4.numerics.")


def _resolved_imports(path: Path, node: ast.Import | ast.ImportFrom) -> tuple[str, ...]:
    if isinstance(node, ast.Import):
        return tuple(alias.name for alias in node.names)

    package = ".".join(path.relative_to(REPOSITORY_ROOT).parts[:-1])
    if node.level:
        base = importlib.util.resolve_name(
            "." * node.level + (node.module or ""),
            package,
        )
    else:
        base = node.module or ""
    qualified_aliases = tuple(f"{base}.{alias.name}" for alias in node.names)
    if node.module is not None:
        return (base, *qualified_aliases)
    return qualified_aliases


def test_resolved_imports_qualifies_from_import_aliases() -> None:
    node = ast.parse("from vfe4 import numerics").body[0]

    assert isinstance(node, ast.ImportFrom)
    assert "vfe4.numerics" in _resolved_imports(TYPES_ROOT / "synthetic.py", node)


def test_type_modules_do_not_import_numerics() -> None:
    violations: list[str] = []
    for path in sorted(TYPES_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            for module_name in _resolved_imports(path, node):
                if _is_numerics_module(module_name):
                    location = path.relative_to(REPOSITORY_ROOT).as_posix()
                    violations.append(f"{location}:{node.lineno}: imports {module_name}")

    assert not violations, "found foundational type-to-numerics imports:\n" + "\n".join(
        violations
    )


def test_importing_type_modules_does_not_load_numerics() -> None:
    module_names: set[str] = set()
    for path in TYPES_ROOT.rglob("*.py"):
        module_parts = list(path.relative_to(REPOSITORY_ROOT).with_suffix("").parts)
        if module_parts[-1] == "__init__":
            module_parts.pop()
        module_names.add(".".join(module_parts))

    program = f"""
import importlib
import json
import sys

for module_name in {sorted(module_names)!r}:
    importlib.import_module(module_name)
loaded = sorted(
    name
    for name in sys.modules
    if name == "vfe4.numerics" or name.startswith("vfe4.numerics.")
)
print(json.dumps(loaded))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []


def test_numerics_package_exports_validator_wrappers() -> None:
    from vfe4 import numerics
    from vfe4.numerics import categorical, gaussian

    assert numerics.require_probability_vector is categorical.require_probability_vector
    assert numerics.require_spd is gaussian.require_spd
    assert categorical.require_probability_vector.__module__ == "vfe4.numerics.categorical"
    assert gaussian.require_spd.__module__ == "vfe4.numerics.gaussian"


def test_h7_snapshot_pushforward_wrappers_are_public_package_exports() -> None:
    import vfe4.generative as public_generative
    import vfe4.generative.pushforward as generative_pushforward
    import vfe4.recognition as public_recognition
    import vfe4.recognition.pushforward as recognition_pushforward

    assert (
        public_generative.pushforward_h7_generative_snapshot
        is generative_pushforward.pushforward_h7_generative_snapshot
    )
    assert (
        public_recognition.pushforward_h7_recognition_snapshot
        is recognition_pushforward.pushforward_h7_recognition_snapshot
    )
    assert "pushforward_h7_generative_snapshot" in public_generative.__all__
    assert "pushforward_h7_generative_snapshot" in generative_pushforward.__all__
    assert "pushforward_h7_recognition_snapshot" in public_recognition.__all__
    assert "pushforward_h7_recognition_snapshot" in recognition_pushforward.__all__


def test_h7_private_snapshot_adapter_scan_covers_test_support() -> None:
    assert (
        REPOSITORY_ROOT / "test_support" / "wt103_figure_fakes.py"
        in _h7_consumer_python_paths()
    )


def test_h7_consumers_do_not_reach_into_private_snapshot_adapters() -> None:
    violations: list[str] = []
    for path in _h7_consumer_python_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            private_names: tuple[str, ...] = ()
            if isinstance(node, ast.ImportFrom):
                private_names = tuple(
                    alias.name
                    for alias in node.names
                    if alias.name in _H7_SNAPSHOT_ADAPTER_OWNERS
                )
            elif isinstance(node, ast.Attribute):
                if node.attr in _H7_SNAPSHOT_ADAPTER_OWNERS:
                    private_names = (node.attr,)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value in _H7_SNAPSHOT_ADAPTER_OWNERS
            ):
                private_names = (node.args[1].value,)

            for private_name in private_names:
                if path != _H7_SNAPSHOT_ADAPTER_OWNERS[private_name]:
                    location = path.relative_to(REPOSITORY_ROOT).as_posix()
                    violations.append(
                        f"{location}:{node.lineno}: accesses {private_name}"
                    )

    assert not violations, "found external H7 fixture-snapshot adapter access:\n" + (
        "\n".join(violations)
    )
