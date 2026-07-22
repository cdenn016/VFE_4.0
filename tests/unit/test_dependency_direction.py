from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TYPES_ROOT = REPOSITORY_ROOT / "vfe4" / "types"


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
