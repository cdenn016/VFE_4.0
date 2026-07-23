"""Deterministic, source-only safety audit for the H6 predictor boundary.

The auditor parses Python source with :mod:`ast`; it never imports or executes
the audited modules.  Its job is deliberately narrow: witness obvious causal
leaks and fail closed when reflection prevents a static conclusion.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from vfe4.types.h6 import (
    EvidenceStatus,
    PrefixCaseKey,
    canonical_json_bytes,
)


_SCHEMA_VERSION = "h6-static-audit-v1"
_CHECK_NAMES = (
    "import_signature_access",
    "taint_cache_capability",
    "mask_normalization_support",
    "inventory_identity",
)
_RULE_GROUP = {
    "recognition_import": _CHECK_NAMES[0],
    "predictor_signature": _CHECK_NAMES[0],
    "private_unsealer": _CHECK_NAMES[0],
    "pre_readiness_access": _CHECK_NAMES[0],
    "durable_opening": _CHECK_NAMES[0],
    "reflection": _CHECK_NAMES[0],
    "syntax": _CHECK_NAMES[0],
    "target_dataflow": _CHECK_NAMES[1],
    "cache_target_data": _CHECK_NAMES[1],
    "split_dataflow": _CHECK_NAMES[1],
    "preprocessing_escape": _CHECK_NAMES[1],
    "taint_resolution": _CHECK_NAMES[1],
    "post_softmax_mask": _CHECK_NAMES[2],
    "duplicate_normalizer": _CHECK_NAMES[2],
    "direct_source_softmax": _CHECK_NAMES[2],
    "noncausal_parent": _CHECK_NAMES[2],
    "all_invalid_fallback": _CHECK_NAMES[2],
    "base_count": _CHECK_NAMES[3],
    "sink_inventory": _CHECK_NAMES[3],
    "source_inventory": _CHECK_NAMES[3],
}
_RULE_DESCRIPTIONS = {
    "recognition_import": "predictor modules cannot import recognition state",
    "predictor_signature": "predictor input is prefix, estimator stream, and cache only",
    "private_unsealer": "private split authority remains inside vfe4.data.access",
    "pre_readiness_access": "train materialization requires a readiness token",
    "durable_opening": "test mapping requires the durable validated opening path",
    "reflection": "dynamic dispatch needs a separate resolution proof",
    "syntax": "every audited source file must parse",
    "target_dataflow": "target, suffix, and recognition data cannot reach predictor sinks",
    "cache_target_data": "cache keys and values contain causal filter state only",
    "split_dataflow": "sealed split bytes cannot flow directly to empirical inputs",
    "preprocessing_escape": "blinded preprocessing cannot return sealed contents",
    "taint_resolution": "unresolved helper returns cannot certify sink safety",
    "post_softmax_mask": "source support is applied before normalization",
    "duplicate_normalizer": "one source normalization helper owns normalization",
    "direct_source_softmax": "source banks use the shared masked helper",
    "noncausal_parent": "declared parents are strictly before the receiver",
    "all_invalid_fallback": "an empty source row raises instead of fabricating mass",
    "base_count": "frozen base-mask counts are 168 and 16384",
    "sink_inventory": "the static sink inventory is complete",
    "source_inventory": "the production audit module set is complete",
}
_REFLECTIVE_CALLS = frozenset(
    {
        "getattr",
        "setattr",
        "eval",
        "exec",
        "__import__",
        "globals",
        "locals",
        "importlib.import_module",
    }
)
_PRIVATE_ACCESS_FRAGMENTS = (
    "unseal",
    "issuer",
    "validatedopening",
    "durableopening",
    "proof_validator",
    "read_split",
    "_issue",
    "_validate",
    "_register",
)
_CACHE_FORBIDDEN_CATEGORIES = frozenset(
    {"target", "suffix", "recognition", "posterior", "full_window"}
)
_PREDICTOR_SINK_FRAGMENTS = (
    "source_prior",
    "prior_logits",
    "transition",
    "emission_logits",
    "proposal",
    "weight",
    "predictor_output",
    "prediction_output",
    "next_token_log_probs",
)
_STATIC_SAFE_CALLS = frozenset(
    {
        "all",
        "any",
        "bool",
        "bytes",
        "dict",
        "enumerate",
        "float",
        "frozenset",
        "int",
        "len",
        "list",
        "max",
        "min",
        "range",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
    }
)
_REQUIRED_SINKS = frozenset(
    {
        "source_prior_logits",
        "transition_parameters",
        "emission_logits",
        "estimator_proposals",
        "estimator_weights",
        "predictor_outputs",
        "cache_keys",
        "cache_values",
        "training_inputs",
        "tuning_inputs",
        "analysis_inputs",
        "public_preprocessing_returns",
    }
)
_PRODUCTION_MARKERS = (
    "vfe4/data/access.py",
    "vfe4/predictive/prior.py",
    "vfe4/predictive/cache.py",
    "vfe4/generative/source_priors.py",
    "vfe4/numerics/categorical.py",
)
_CACHE_KEY_IDENTITIES = {
    "source_sha256": ("source_sha256", "vocabulary_sha256", "data_safety_sha256"),
    "predictor_config_sha256": ("predictor_config_sha256",),
    "model_state_sha256": ("model_state_sha256",),
    "estimator_sha256": (
        "estimator_sha256",
        "estimator_semantic_sha256",
        "estimator_artifact_bytes_sha256",
        "estimator_stream_sha256",
    ),
    "prefix_sha256": ("prefix_sha256", "prefix_tokens", "prefix"),
}
_CACHE_VALUE_ALLOWED_NAMES = frozenset(
    {
        "causal_filter_state",
        "counter_position",
        "filtered_population",
        "filtered_log_weights",
        "cumulative_log_normalizer",
        "pending",
        "assimilations",
        "counter_consumption",
    }
)
_PREFIX_CACHE_KEY_FIELDS = frozenset(
    {
        "prefix_tokens",
        "prefix_sha256",
        "vocabulary_sha256",
        "predictor_config_sha256",
        "model_family_sha256",
        "model_state_sha256",
        "proposal_identity_sha256",
        "estimator_semantic_sha256",
        "estimator_artifact_bytes_sha256",
        "estimator_stream_sha256",
        "data_safety_sha256",
        "key_sha256",
    }
)
_PREFIX_CACHE_FIELDS = frozenset(
    {
        "key",
        "filtered_population",
        "filtered_log_weights",
        "cumulative_log_normalizer",
        "pending",
        "assimilations",
        "counter_consumption",
        "cache_sha256",
    }
)


def _owned_hash(domain: str, payload: object) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + canonical_json_bytes(payload)
    ).hexdigest()


def _require_text(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256 hex")
    return value


@dataclass(frozen=True, slots=True)
class StaticAuditFinding:
    """One source-bound failure witness or unresolved static obligation."""

    rule_id: str
    status: EvidenceStatus
    path: str
    line: int
    message: str
    witness_sha256: str
    finding_sha256: str

    def __post_init__(self) -> None:
        _require_text(self.rule_id, "rule_id")
        if self.rule_id not in _RULE_GROUP:
            raise ValueError("finding has an unknown rule")
        if self.status not in (EvidenceStatus.FAIL, EvidenceStatus.INCONCLUSIVE):
            raise ValueError("findings are FAIL or INCONCLUSIVE witnesses")
        _require_text(self.path, "path")
        if type(self.line) is not int or self.line <= 0:
            raise ValueError("finding line must be positive")
        _require_text(self.message, "message")
        _require_sha256(self.witness_sha256, "witness_sha256")
        expected = _owned_hash(
            "vfe4.h6.static-audit-finding.v1",
            {
                "rule_id": self.rule_id,
                "status": self.status.value,
                "path": self.path,
                "line": self.line,
                "message": self.message,
                "witness_sha256": self.witness_sha256,
            },
        )
        if self.finding_sha256 != expected:
            raise ValueError("finding_sha256 is stale")

    @classmethod
    def create(
        cls,
        *,
        rule_id: str,
        status: EvidenceStatus,
        path: str,
        line: int,
        message: str,
        witness: str,
    ) -> "StaticAuditFinding":
        witness_sha256 = hashlib.sha256(witness.encode("utf-8")).hexdigest()
        values = {
            "rule_id": rule_id,
            "status": status,
            "path": path,
            "line": line,
            "message": message,
            "witness_sha256": witness_sha256,
        }
        return cls(
            **values,
            finding_sha256=_owned_hash(
                "vfe4.h6.static-audit-finding.v1",
                {
                    **values,
                    "status": status.value,
                },
            ),
        )


@dataclass(frozen=True, slots=True)
class StaticAuditCheck:
    name: str
    status: EvidenceStatus
    finding_sha256s: tuple[str, ...]
    obligations: tuple[str, ...]
    check_sha256: str

    def __post_init__(self) -> None:
        if self.name not in _CHECK_NAMES:
            raise ValueError("unknown static-audit check")
        if not isinstance(self.status, EvidenceStatus):
            raise ValueError("check status must be EvidenceStatus")
        if type(self.finding_sha256s) is not tuple:
            raise ValueError("finding_sha256s must be a tuple")
        for digest in self.finding_sha256s:
            _require_sha256(digest, "finding_sha256")
        if type(self.obligations) is not tuple or any(
            type(item) is not str or not item for item in self.obligations
        ):
            raise ValueError("check obligations must be nonempty strings")
        if self.status is EvidenceStatus.PASS and (
            self.finding_sha256s or self.obligations
        ):
            raise ValueError("PASS checks cannot retain findings")
        if self.status is EvidenceStatus.FAIL and self.obligations:
            raise ValueError("FAIL checks cannot retain obligations")
        if self.status is EvidenceStatus.INCONCLUSIVE and not self.obligations:
            raise ValueError("INCONCLUSIVE checks require obligations")
        expected = _owned_hash(
            "vfe4.h6.static-audit-check.v1",
            {
                "name": self.name,
                "status": self.status.value,
                "finding_sha256s": self.finding_sha256s,
                "obligations": self.obligations,
            },
        )
        if self.check_sha256 != expected:
            raise ValueError("check_sha256 is stale")


@dataclass(frozen=True, slots=True)
class StaticAuditReport:
    schema_version: str
    source_manifest_sha256: str
    rules_sha256: str
    case_key_manifest_sha256: str
    checks: tuple[StaticAuditCheck, ...]
    findings: tuple[StaticAuditFinding, ...]
    status: EvidenceStatus
    obligations: tuple[str, ...]
    report_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("unsupported static-audit schema")
        for name in (
            "source_manifest_sha256",
            "rules_sha256",
            "case_key_manifest_sha256",
            "report_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if (
            type(self.checks) is not tuple
            or tuple(check.name for check in self.checks) != _CHECK_NAMES
            or any(type(check) is not StaticAuditCheck for check in self.checks)
        ):
            raise ValueError("report checks are incomplete or out of order")
        if type(self.findings) is not tuple or any(
            type(item) is not StaticAuditFinding for item in self.findings
        ):
            raise ValueError("report findings must be exact immutable records")
        for finding in self.findings:
            finding.__post_init__()
        for check in self.checks:
            check.__post_init__()
            owned = tuple(
                finding
                for finding in self.findings
                if _RULE_GROUP[finding.rule_id] == check.name
            )
            expected_check_status = (
                EvidenceStatus.FAIL
                if any(item.status is EvidenceStatus.FAIL for item in owned)
                else EvidenceStatus.INCONCLUSIVE
                if owned
                else EvidenceStatus.PASS
            )
            expected_check_obligations = (
                tuple(
                    f"{item.rule_id}: {item.message} ({item.path}:{item.line})"
                    for item in owned
                    if item.status is EvidenceStatus.INCONCLUSIVE
                )
                if expected_check_status is EvidenceStatus.INCONCLUSIVE
                else ()
            )
            if (
                check.status is not expected_check_status
                or check.finding_sha256s
                != tuple(item.finding_sha256 for item in owned)
                or check.obligations != expected_check_obligations
            ):
                raise ValueError("report check does not match its owned findings")
        if type(self.obligations) is not tuple or any(
            type(item) is not str or not item for item in self.obligations
        ):
            raise ValueError("report obligations must be nonempty strings")
        expected_status = (
            EvidenceStatus.FAIL
            if any(item.status is EvidenceStatus.FAIL for item in self.findings)
            else EvidenceStatus.INCONCLUSIVE
            if self.findings
            else EvidenceStatus.PASS
        )
        if self.status is not expected_status:
            raise ValueError("report status does not follow FAIL precedence")
        if self.status is EvidenceStatus.FAIL and self.obligations:
            raise ValueError("FAIL reports cannot retain top-level obligations")
        if self.status is EvidenceStatus.INCONCLUSIVE and not self.obligations:
            raise ValueError("INCONCLUSIVE reports require obligations")
        if self.status is EvidenceStatus.PASS and (
            self.findings or self.obligations
        ):
            raise ValueError("PASS reports cannot retain findings")
        expected = _owned_hash(
            "vfe4.h6.static-audit-report.v1",
            {
                "schema_version": self.schema_version,
                "source_manifest_sha256": self.source_manifest_sha256,
                "rules_sha256": self.rules_sha256,
                "case_key_manifest_sha256": self.case_key_manifest_sha256,
                "checks": tuple(check.check_sha256 for check in self.checks),
                "findings": tuple(
                    finding.finding_sha256 for finding in self.findings
                ),
                "status": self.status.value,
                "obligations": self.obligations,
            },
        )
        if self.report_sha256 != expected:
            raise ValueError("report_sha256 is stale")


@dataclass(frozen=True, slots=True)
class _SourceUnit:
    path: Path
    relative_path: str
    raw: bytes
    text: str | None
    tree: ast.Module | None
    parse_error: str | None


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _bound_names(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(
            name for item in node.elts for name in _bound_names(item)
        )
    if isinstance(node, ast.Attribute):
        return (_dotted_name(node),)
    return ()


def _function_arguments(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, ...]:
    positional = tuple(
        argument.arg
        for argument in (*node.args.posonlyargs, *node.args.args)
    )
    keyword_only = tuple(argument.arg for argument in node.args.kwonlyargs)
    variadic = tuple(
        argument.arg
        for argument in (node.args.vararg, node.args.kwarg)
        if argument is not None
    )
    return (*positional, *keyword_only, *variadic)


def _category_for_name(name: str) -> frozenset[str]:
    normalized = name.lower()
    categories: set[str] = set()
    if "sha256" in normalized or normalized.endswith("_digest"):
        return frozenset()
    if "target" in normalized and "sha256" not in normalized:
        categories.add("target")
    if "suffix" in normalized:
        categories.add("suffix")
    if "recognition" in normalized:
        categories.add("recognition")
    if "posterior" in normalized:
        categories.add("posterior")
    if "full_window" in normalized or "complete_window" in normalized:
        categories.add("full_window")
    if "sealed" in normalized and "train" in normalized:
        categories.add("sealed_train")
    if "sealed" in normalized and "test" in normalized:
        categories.add("sealed_test")
    if (
        "raw_bytes" in normalized
        or "raw_tensor" in normalized
        or "token_tensor" in normalized
    ):
        categories.add("raw")
    return frozenset(categories)


def _expression_categories(
    node: ast.AST | None,
    taints: dict[str, frozenset[str]],
    call_summaries: dict[str, frozenset[str]] | None = None,
) -> frozenset[str]:
    if node is None:
        return frozenset()
    categories: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            categories.update(taints.get(child.id, ()))
            categories.update(_category_for_name(child.id))
        elif isinstance(child, ast.Attribute):
            dotted = _dotted_name(child)
            categories.update(taints.get(dotted, ()))
            categories.update(_category_for_name(dotted))
        elif isinstance(child, ast.Constant) and type(child.value) is str:
            categories.update(_category_for_name(child.value))
        elif isinstance(child, ast.Call) and call_summaries is not None:
            leaf = _call_name(child).split(".")[-1]
            if leaf in call_summaries:
                categories.update(call_summaries[leaf])
            elif leaf not in _STATIC_SAFE_CALLS:
                categories.add("unresolved")
    return frozenset(categories)


def _source_segment(unit: _SourceUnit, node: ast.AST) -> str:
    if unit.text is not None:
        segment = ast.get_source_segment(unit.text, node)
        if segment:
            return segment
    return ast.dump(node, include_attributes=False)


def _literal_strings(node: ast.AST) -> frozenset[str]:
    return frozenset(
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and type(child.value) is str
    )


def _literal_integers(node: ast.AST) -> tuple[int, ...]:
    return tuple(
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and type(child.value) is int
    )


def _call_name(node: ast.Call) -> str:
    return _dotted_name(node.func)


def _annotation_is_exact(node: ast.AST | None, leaf: str) -> bool:
    return (
        isinstance(node, (ast.Name, ast.Attribute))
        and _dotted_name(node).split(".")[-1] == leaf
    )


def _annotation_is_optional(node: ast.AST | None, leaf: str) -> bool:
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.BitOr):
        return False
    sides = (node.left, node.right)
    return any(_annotation_is_exact(side, leaf) for side in sides) and any(
        isinstance(side, ast.Constant) and side.value is None
        or isinstance(side, ast.Name) and side.id == "None"
        for side in sides
    )


def _softmax_aliases(tree: ast.Module) -> frozenset[str]:
    aliases = {"softmax", "log_softmax"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            aliases.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name in {"softmax", "log_softmax"}
            )
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Attribute):
            if node.value.attr in {"softmax", "log_softmax"}:
                aliases.update(
                    name
                    for target in node.targets
                    for name in _bound_names(target)
                )
    return frozenset(aliases)


def _is_softmax_call(node: ast.AST, aliases: frozenset[str]) -> bool:
    return (
        isinstance(node, ast.Call)
        and _call_name(node).split(".")[-1] in aliases
    )


def _softmax_input_is_premasked(call: ast.Call) -> bool:
    if not call.args:
        return False
    argument = call.args[0]
    source = ast.dump(argument, include_attributes=False).lower()
    explicitly_masked_name = (
        "masked" in source and "unmasked" not in source
    )
    masking_call = (
        isinstance(argument, ast.Call)
        and _call_name(argument).split(".")[-1] in {"masked_fill", "where"}
    )
    return explicitly_masked_name or masking_call


def _if_tests_empty_support(node: ast.If) -> bool:
    test = node.test
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        text = ast.dump(test.operand, include_attributes=False).lower()
        return "declared_parents" in text or "support_mask" in text
    if isinstance(test, ast.Compare) and any(
        isinstance(operator, (ast.Eq, ast.LtE)) for operator in test.ops
    ):
        text = ast.dump(test, include_attributes=False).lower()
        has_zero = any(
            isinstance(child, ast.Constant)
            and type(child.value) is int
            and child.value == 0
            for child in ast.walk(test)
        )
        has_empty_container = any(
            isinstance(child, (ast.Tuple, ast.List, ast.Set))
            and not child.elts
            for child in ast.walk(test)
        )
        return (has_zero or has_empty_container) and (
            "declared_parents" in text or "support_mask" in text
        )
    return False


def _receiver_plus_positive(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Add)
        and any(
            isinstance(child, ast.Name) and child.id == "receiver_t"
            for child in ast.walk(node)
        )
        and any(
            isinstance(child, ast.Constant)
            and type(child.value) is int
            and child.value > 0
            for child in ast.walk(node)
        )
    )


def _parent_expression_is_noncausal(node: ast.AST) -> bool:
    if isinstance(node, ast.Call) and _call_name(node).split(".")[-1] == "range":
        if not node.args:
            return False
        if any(_receiver_plus_positive(argument) for argument in node.args):
            return True
        return (
            len(node.args) > 1
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "receiver_t"
        )
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return any(
            isinstance(element, ast.Name) and element.id == "receiver_t"
            or _receiver_plus_positive(element)
            or _parent_expression_is_noncausal(element)
            for element in node.elts
        )
    if isinstance(node, ast.BinOp):
        return (
            _receiver_plus_positive(node)
            or _parent_expression_is_noncausal(node.left)
            or _parent_expression_is_noncausal(node.right)
        )
    return any(
        _parent_expression_is_noncausal(child)
        for child in ast.iter_child_nodes(node)
    )


def _production_source_path(relative_path: str) -> bool:
    if relative_path in _PRODUCTION_MARKERS:
        return True
    return relative_path.startswith(
        (
            "vfe4/predictive/",
            "vfe4/data/",
            "vfe4/training/",
            "vfe4/tuning/",
            "vfe4/analysis/",
            "vfe4/evaluation/",
        )
    ) or relative_path in {
        "vfe4/generative/language.py",
        "vfe4/generative/source_priors.py",
        "vfe4/objective/language_elbo.py",
        "vfe4/validation/h6_static_audit.py",
        "vfe4/validation/h6_static_inventory.py",
    }


def _load_sources(root: Path) -> tuple[tuple[_SourceUnit, ...], bool]:
    marker_count = sum((root / marker).is_file() for marker in _PRODUCTION_MARKERS)
    production_mode = (
        (root / "pyproject.toml").is_file() and (root / "vfe4").is_dir()
    ) or marker_count >= len(_PRODUCTION_MARKERS) - 1
    units: list[_SourceUnit] = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        if any(
            part in {".git", ".worktrees", ".verification", "__pycache__"}
            for part in path.relative_to(root).parts
        ):
            continue
        if production_mode and not _production_source_path(relative):
            continue
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8")
            tree = ast.parse(text, filename=relative)
        except (UnicodeDecodeError, SyntaxError) as exc:
            text = None
            tree = None
            parse_error = str(exc)
        else:
            parse_error = None
        units.append(_SourceUnit(path, relative, raw, text, tree, parse_error))
    if not units:
        raise ValueError("repo_root contains no auditable Python source")
    return tuple(units), production_mode


def _case_key_manifest(exact_case_keys: object) -> str:
    if (
        type(exact_case_keys) is not tuple
        or not exact_case_keys
        or any(type(key) is not PrefixCaseKey for key in exact_case_keys)
    ):
        raise ValueError("exact_case_keys must be a nonempty exact PrefixCaseKey tuple")
    payloads: list[dict[str, object]] = []
    encoded: list[bytes] = []
    for key in exact_case_keys:
        key.__post_init__()
        payload = key.canonical_payload()
        canonical = canonical_json_bytes(payload)
        if canonical in encoded:
            raise ValueError("exact_case_keys cannot contain duplicates")
        encoded.append(canonical)
        payloads.append(payload)
    ordered = tuple(
        payload
        for _, payload in sorted(
            zip(encoded, payloads, strict=True), key=lambda item: item[0]
        )
    )
    return _owned_hash("vfe4.h6.static-audit-case-keys.v1", ordered)


class _FindingCollector:
    def __init__(self) -> None:
        self._items: dict[
            tuple[str, EvidenceStatus, str, int, str], StaticAuditFinding
        ] = {}

    def add(
        self,
        unit: _SourceUnit,
        node: ast.AST | None,
        *,
        rule_id: str,
        status: EvidenceStatus,
        message: str,
        witness: str | None = None,
    ) -> None:
        line = max(1, int(getattr(node, "lineno", 1)))
        source_witness = (
            witness
            if witness is not None
            else _source_segment(unit, node) if node is not None else unit.parse_error or "source"
        )
        key = (rule_id, status, unit.relative_path, line, message)
        self._items.setdefault(
            key,
            StaticAuditFinding.create(
                rule_id=rule_id,
                status=status,
                path=unit.relative_path,
                line=line,
                message=message,
                witness=source_witness,
            ),
        )

    def values(self) -> tuple[StaticAuditFinding, ...]:
        return tuple(
            sorted(
                self._items.values(),
                key=lambda item: (
                    item.path,
                    item.line,
                    item.rule_id,
                    item.message,
                ),
            )
        )


def _function_calls(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ast.Call, ...]:
    return tuple(
        node for node in ast.walk(function) if isinstance(node, ast.Call)
    )


def _direct_statement_call(statement: ast.stmt) -> ast.Call | None:
    value: ast.AST | None = None
    if isinstance(statement, ast.Expr):
        value = statement.value
    elif isinstance(statement, (ast.Assign, ast.AnnAssign)):
        value = statement.value
    return value if isinstance(value, ast.Call) else None


def _exact_name_arguments(call: ast.Call, expected: tuple[str, ...]) -> bool:
    return (
        not call.keywords
        and len(call.args) == len(expected)
        and all(
            isinstance(argument, ast.Name) and argument.id == name
            for argument, name in zip(call.args, expected, strict=True)
        )
    )


def _is_rejecting_validated_guard(statement: ast.stmt) -> bool:
    if (
        not isinstance(statement, ast.If)
        or statement.orelse
        or not any(isinstance(child, ast.Raise) for child in statement.body)
    ):
        return False
    test = statement.test
    if (
        isinstance(test, ast.Compare)
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.IsNot)
        and len(test.comparators) == 1
        and _annotation_is_exact(test.comparators[0], "ValidatedOpening")
        and isinstance(test.left, ast.Call)
        and _call_name(test.left).split(".")[-1] == "type"
        and _exact_name_arguments(test.left, ("validated",))
    ):
        return True
    return (
        isinstance(test, ast.UnaryOp)
        and isinstance(test.op, ast.Not)
        and isinstance(test.operand, ast.Call)
        and _call_name(test.operand).split(".")[-1] == "isinstance"
        and len(test.operand.args) == 2
        and isinstance(test.operand.args[0], ast.Name)
        and test.operand.args[0].id == "validated"
        and _annotation_is_exact(test.operand.args[1], "ValidatedOpening")
        and not test.operand.keywords
    )


def _audit_access_closure(
    unit: _SourceUnit,
    collector: _FindingCollector,
) -> None:
    assert unit.tree is not None
    functions = {
        node.name: node
        for node in ast.walk(unit.tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    materialize = functions.get("_materialize_train")
    if materialize is not None:
        readiness_steps = tuple(
            (index, call)
            for index, statement in enumerate(materialize.body)
            if (call := _direct_statement_call(statement)) is not None
            and _call_name(call).split(".")[-1] == "_require_readiness"
            and _exact_name_arguments(call, ("store", "readiness"))
        )
        read_steps = tuple(
            index
            for index, statement in enumerate(materialize.body)
            if any(
                _call_name(call).split(".")[-1] == "_read_split"
                for call in ast.walk(statement)
                if isinstance(call, ast.Call)
            )
        )
        if (
            len(readiness_steps) != 1
            or read_steps and readiness_steps[0][0] >= min(read_steps)
        ):
            collector.add(
                unit,
                materialize,
                rule_id="pre_readiness_access",
                status=EvidenceStatus.FAIL,
                message=(
                    "closure-private _materialize_train must call "
                    "_require_readiness exactly once before reading a split"
                ),
            )
    opening = functions.get("_open_test")
    if opening is not None:
        validate_steps = tuple(
            (index, statement, call)
            for index, statement in enumerate(opening.body)
            if (call := _direct_statement_call(statement)) is not None
            and _call_name(call).split(".")[-1] == "_validate"
            and _exact_name_arguments(call, ("store", "opening"))
            and isinstance(statement, (ast.Assign, ast.AnnAssign))
            and any(
                name == "validated"
                for target in (
                    tuple(statement.targets)
                    if isinstance(statement, ast.Assign)
                    else (statement.target,)
                )
                for name in _bound_names(target)
            )
        )
        read_steps = tuple(
            index
            for index, statement in enumerate(opening.body)
            if any(
                _call_name(call).split(".")[-1] == "_read_split"
                for call in ast.walk(statement)
                if isinstance(call, ast.Call)
            )
        )
        validated_guards = tuple(
            index
            for index, statement in enumerate(opening.body)
            if _is_rejecting_validated_guard(statement)
        )
        if (
            len(validate_steps) != 1
            or read_steps and validate_steps[0][0] >= min(read_steps)
            or read_steps
            and not any(
                validate_steps[0][0] < guard < min(read_steps)
                for guard in validated_guards
            )
        ):
            collector.add(
                unit,
                opening,
                rule_id="durable_opening",
                status=EvidenceStatus.FAIL,
                message=(
                    "closure-private _open_test must validate the durable "
                    "opening before reading test data"
                ),
            )


def _audit_imports_signatures_and_access(
    unit: _SourceUnit, collector: _FindingCollector
) -> None:
    assert unit.tree is not None
    path_lower = unit.relative_path.lower()
    getattr_aliases = {"getattr"}
    for imported in ast.walk(unit.tree):
        if isinstance(imported, ast.ImportFrom) and imported.module == "builtins":
            getattr_aliases.update(
                alias.asname or alias.name
                for alias in imported.names
                if alias.name == "getattr"
            )
    changed = True
    while changed:
        changed = False
        for assignment in ast.walk(unit.tree):
            if (
                isinstance(assignment, ast.Assign)
                and isinstance(assignment.value, ast.Name)
                and assignment.value.id in getattr_aliases
            ):
                before = len(getattr_aliases)
                getattr_aliases.update(
                    name
                    for target in assignment.targets
                    for name in _bound_names(target)
                )
                changed = changed or len(getattr_aliases) != before
    _audit_access_closure(unit, collector)
    for node in ast.walk(unit.tree):
        if isinstance(node, ast.ImportFrom):
            module = (node.module or "").lower()
            imported = tuple(alias.name for alias in node.names)
            if path_lower.startswith(
                ("vfe4/predictive/", "vfe4/generative/")
            ) and (
                "recognition" in module
                or any(
                    "recognition" in name.lower()
                    or "posterior" in name.lower()
                    for name in imported
                )
            ):
                collector.add(
                    unit,
                    node,
                    rule_id="recognition_import",
                    status=EvidenceStatus.FAIL,
                    message="generative/predictive source imports recognition or posterior state",
                )
            if unit.relative_path != "vfe4/data/access.py" and (
                module == "vfe4.data.access" or module.endswith(".data.access")
            ):
                private = tuple(
                    name
                    for name in imported
                    if name.startswith("_")
                    and any(
                        fragment in name.lower()
                        for fragment in _PRIVATE_ACCESS_FRAGMENTS
                    )
                )
                if private:
                    collector.add(
                        unit,
                        node,
                        rule_id="private_unsealer",
                        status=EvidenceStatus.FAIL,
                        message=(
                            "private unsealer/issuer authority imported outside "
                            f"vfe4.data.access: {', '.join(private)}"
                        ),
                    )
        elif isinstance(node, ast.Import):
            if path_lower.startswith(
                ("vfe4/predictive/", "vfe4/generative/")
            ) and any(
                "recognition" in alias.name.lower()
                or "posterior" in alias.name.lower()
                for alias in node.names
            ):
                collector.add(
                    unit,
                    node,
                    rule_id="recognition_import",
                    status=EvidenceStatus.FAIL,
                    message="generative/predictive source imports recognition or posterior state",
                )
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "next_token_log_probs":
                actual = _function_arguments(node)
                required = ("self", "prefix_tokens", "estimator_rng", "cache")
                variadic = tuple(
                    name
                    for name, value in (
                        ("varargs", node.args.vararg),
                        ("kwargs", node.args.kwarg),
                    )
                    if value is not None
                )
                cache_default_is_none = (
                    len(node.args.defaults) == 1
                    and isinstance(node.args.defaults[0], ast.Constant)
                    and node.args.defaults[0].value is None
                )
                annotations = {
                    argument.arg: argument.annotation
                    for argument in (*node.args.posonlyargs, *node.args.args)
                }
                annotation_valid = (
                    not node.args.posonlyargs
                    and _annotation_is_exact(
                        annotations.get("prefix_tokens"), "CausalPrefix"
                    )
                    and _annotation_is_exact(
                        annotations.get("estimator_rng"), "EstimatorStream"
                    )
                    and _annotation_is_optional(
                        annotations.get("cache"), "PrefixCache"
                    )
                    and _annotation_is_exact(node.returns, "PriorPrediction")
                )
                if (
                    actual != required
                    or node.args.kwonlyargs
                    or variadic
                    or not cache_default_is_none
                    or not annotation_valid
                ):
                    extras = tuple(name for name in actual if name not in required)
                    collector.add(
                        unit,
                        node,
                        rule_id="predictor_signature",
                        status=EvidenceStatus.FAIL,
                        message=(
                            "next_token_log_probs must accept only self, prefix_tokens, "
                            "estimator_rng, cache=None with exact causal annotations; "
                            f"extra target/data parameters={extras}, variadic={variadic}, "
                            f"cache_none={cache_default_is_none}, annotations={annotation_valid}"
                        ),
                    )
            if node.name == "open_test_for_scoring":
                arguments = tuple(name.lower() for name in _function_arguments(node))
                calls = tuple(
                    _call_name(call).split(".")[-1]
                    for call in _function_calls(node)
                )
                if (
                    not any("opening" in name or "capability" in name for name in arguments)
                    or not any("validat" in name for name in calls)
                ):
                    collector.add(
                        unit,
                        node,
                        rule_id="durable_opening",
                        status=EvidenceStatus.FAIL,
                        message="open_test_for_scoring lacks a validated durable opening capability",
                    )
        if isinstance(node, ast.Call):
            call_name = _call_name(node)
            unresolved_reflection = (
                call_name in (_REFLECTIVE_CALLS - {"getattr", "setattr"})
                or call_name.split(".")[-1] in getattr_aliases
                and (
                    len(node.args) < 2
                    or not (
                        isinstance(node.args[1], ast.Constant)
                        and type(node.args[1].value) is str
                    )
                )
                or call_name == "setattr"
                and (
                    len(node.args) < 2
                    or not (
                        isinstance(node.args[1], ast.Constant)
                        and type(node.args[1].value) is str
                    )
                )
            )
            if unresolved_reflection:
                collector.add(
                    unit,
                    node,
                    rule_id="reflection",
                    status=EvidenceStatus.INCONCLUSIVE,
                    message=f"unresolved reflection call {call_name}",
                )
            if (
                call_name.split(".")[-1] == "materialize_prediction_train"
                and unit.relative_path != "vfe4/data/access.py"
            ):
                readiness_values = list(node.args[1:2]) + [
                    keyword.value
                    for keyword in node.keywords
                    if keyword.arg == "readiness"
                ]
                missing = not readiness_values
                invalid = any(
                    isinstance(value, ast.Constant) and value.value is None
                    for value in readiness_values
                )
                visibly_not_readiness = any(
                    isinstance(value, (ast.Name, ast.Attribute))
                    and "readiness" not in _dotted_name(value).lower()
                    for value in readiness_values
                )
                if missing or invalid:
                    collector.add(
                        unit,
                        node,
                        rule_id="pre_readiness_access",
                        status=EvidenceStatus.FAIL,
                        message="materialize_prediction_train is called before valid readiness",
                    )
                elif visibly_not_readiness:
                    collector.add(
                        unit,
                        node,
                        rule_id="pre_readiness_access",
                        status=EvidenceStatus.INCONCLUSIVE,
                        message=(
                            "materialize_prediction_train readiness argument "
                            "cannot be resolved statically"
                        ),
                    )


def _audit_counts_and_inventory(
    unit: _SourceUnit,
    collector: _FindingCollector,
    *,
    production_mode: bool,
) -> tuple[frozenset[int], bool]:
    assert unit.tree is not None
    saw_inventory = False
    declared_counts: set[int] = set()
    for node in ast.walk(unit.tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = (
            tuple(node.targets) if isinstance(node, ast.Assign) else (node.target,)
        )
        value = node.value
        if value is None:
            continue
        names = tuple(
            name.lower()
            for target in targets
            for name in _bound_names(target)
        )
        if any("base" in name and "count" in name for name in names):
            observed = _literal_integers(value)
            canonical_count_location = (
                not production_mode
                or unit.relative_path == "vfe4/generative/source_priors.py"
            )
            expected: set[int] = set()
            if any("small" in name for name in names):
                expected.add(168)
            if any("wikitext" in name for name in names):
                expected.add(16384)
            if not expected:
                expected.update((168, 16384))
            if observed and expected.issubset(observed) and canonical_count_location:
                declared_counts.update(expected)
            elif observed and canonical_count_location:
                collector.add(
                    unit,
                    node,
                    rule_id="base_count",
                    status=EvidenceStatus.FAIL,
                    message=(
                        "H6 base-mask count declaration must bind expected "
                        f"{tuple(sorted(expected))}; global expected values are "
                        f"168 and 16384; observed {observed}"
                    ),
                )
            elif observed:
                collector.add(
                    unit,
                    node,
                    rule_id="base_count",
                    status=EvidenceStatus.INCONCLUSIVE,
                    message="base-count declaration is outside its canonical source-prior module",
                )
        if any("sink_inventory" in name for name in names):
            canonical_location = (
                not production_mode
                or unit.relative_path == "vfe4/validation/h6_static_inventory.py"
            )
            observed_sinks = _literal_strings(value)
            missing = tuple(sorted(_REQUIRED_SINKS - observed_sinks))
            if not canonical_location:
                collector.add(
                    unit,
                    node,
                    rule_id="sink_inventory",
                    status=EvidenceStatus.INCONCLUSIVE,
                    message="sink inventory is outside its canonical production module",
                )
            elif missing:
                collector.add(
                    unit,
                    node,
                    rule_id="sink_inventory",
                    status=EvidenceStatus.FAIL,
                    message=f"static sink inventory is incomplete; missing {missing}",
                )
            else:
                saw_inventory = True
    return frozenset(declared_counts), saw_inventory


def _function_taints(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    call_summaries: dict[str, frozenset[str]] | None = None,
) -> dict[str, frozenset[str]]:
    taints = {
        name: _category_for_name(name)
        for name in _function_arguments(node)
        if _category_for_name(name)
    }
    bindings: list[tuple[tuple[ast.AST, ...], ast.AST]] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Assign):
            bindings.append((tuple(child.targets), child.value))
        elif isinstance(child, (ast.AnnAssign, ast.NamedExpr)):
            bindings.append(((child.target,), child.value))
        elif isinstance(child, ast.AugAssign):
            bindings.append(((child.target,), child.value))
        elif isinstance(child, (ast.For, ast.AsyncFor)):
            bindings.append(((child.target,), child.iter))
        elif isinstance(child, (ast.With, ast.AsyncWith)):
            bindings.extend(
                ((item.optional_vars,), item.context_expr)
                for item in child.items
                if item.optional_vars is not None
            )
    changed = True
    while changed:
        changed = False
        for targets, value in bindings:
            categories = _expression_categories(value, taints, call_summaries)
            if not categories:
                continue
            for target in targets:
                for name in _bound_names(target):
                    merged = frozenset((*taints.get(name, ()), *categories))
                    if taints.get(name) != merged:
                        taints[name] = merged
                        changed = True
    return taints


def _return_category_summaries(
    units: Iterable[_SourceUnit],
) -> dict[str, frozenset[str]]:
    unit_tuple = tuple(units)
    named_nodes = tuple(
        node
        for unit in unit_tuple
        if unit.tree is not None
        for node in ast.walk(unit.tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )
    counts: dict[str, int] = {}
    for node in named_nodes:
        counts[node.name] = counts.get(node.name, 0) + 1
    unique = tuple(node for node in named_nodes if counts[node.name] == 1)
    summaries: dict[str, frozenset[str]] = {
        node.name: frozenset()
        for node in unique
        if isinstance(node, ast.ClassDef)
    }
    pending = {
        node.name: node
        for node in unique
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    changed = True
    while changed:
        changed = False
        for name, function in tuple(pending.items()):
            returned_calls = tuple(
                call
                for returned in ast.walk(function)
                if isinstance(returned, ast.Return) and returned.value is not None
                for call in ast.walk(returned.value)
                if isinstance(call, ast.Call)
            )
            unresolved = tuple(
                _call_name(call).split(".")[-1]
                for call in returned_calls
                if _call_name(call).split(".")[-1] not in summaries
                and _call_name(call).split(".")[-1] not in _STATIC_SAFE_CALLS
                and _call_name(call).split(".")[-1] not in {"getattr", "setattr"}
            )
            if unresolved:
                continue
            taints = _function_taints(function, summaries)
            parameter_categories = frozenset(
                category
                for parameter in _function_arguments(function)
                for category in _category_for_name(parameter)
            )
            categories = frozenset(
                category
                for returned in ast.walk(function)
                if isinstance(returned, ast.Return)
                for category in _expression_categories(
                    returned.value, taints, summaries
                )
                if category != "unresolved"
            ) - parameter_categories
            summaries[name] = categories
            pending.pop(name)
            changed = True
    return summaries


def _identifier_inventory(node: ast.AST) -> frozenset[str]:
    values: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            values.add(child.id.lower())
        elif isinstance(child, ast.Attribute):
            values.add(child.attr.lower())
            values.add(_dotted_name(child).lower())
        elif isinstance(child, ast.keyword) and child.arg is not None:
            values.add(child.arg.lower())
        elif isinstance(child, ast.Constant) and type(child.value) is str:
            values.add(child.value.lower())
    return frozenset(values)


def _missing_cache_identities(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        relevant = tuple(
            returned.value
            for returned in ast.walk(node)
            if isinstance(returned, ast.Return) and returned.value is not None
        )
        inventory = frozenset(
            value
            for returned in relevant
            for value in _identifier_inventory(returned)
        )
    elif isinstance(node, ast.ClassDef):
        relevant = tuple(
            child
            for child in node.body
            if isinstance(child, (ast.Assign, ast.AnnAssign))
        )
        inventory = frozenset(
            value
            for child in relevant
            for value in _identifier_inventory(child)
        )
    else:
        inventory = _identifier_inventory(node)
    flattened = "\n".join(sorted(inventory))
    return tuple(
        canonical
        for canonical, alternatives in _CACHE_KEY_IDENTITIES.items()
        if not any(alternative in flattened for alternative in alternatives)
    )


def _unknown_cache_value_names(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, ...]:
    candidates = set(_function_arguments(function))
    for returned in (
        node for node in ast.walk(function) if isinstance(node, ast.Return)
    ):
        candidates.update(_literal_strings(returned))
    candidates.difference_update({"self", "cls"})
    return tuple(sorted(name for name in candidates if name not in _CACHE_VALUE_ALLOWED_NAMES))


def _class_field_names(class_node: ast.ClassDef) -> frozenset[str]:
    return frozenset(
        name
        for child in class_node.body
        if isinstance(child, (ast.Assign, ast.AnnAssign))
        for target in (
            tuple(child.targets)
            if isinstance(child, ast.Assign)
            else (child.target,)
        )
        for name in _bound_names(target)
        if not name.startswith("_")
    )


def _audit_taint(
    unit: _SourceUnit,
    collector: _FindingCollector,
    call_summaries: dict[str, frozenset[str]],
) -> None:
    assert unit.tree is not None
    role = unit.relative_path.lower()
    empirical_role = next(
        (
            name
            for name in ("training", "tuning", "analysis")
            if f"/{name}/" in f"/{role}"
        ),
        None,
    )
    for class_node in (
        node for node in ast.walk(unit.tree) if isinstance(node, ast.ClassDef)
    ):
        if (
            unit.relative_path.endswith("vfe4/predictive/cache.py")
            and class_node.name in {"PrefixCacheKey", "PrefixCache"}
        ):
            observed_fields = _class_field_names(class_node)
            allowed_fields = (
                _PREFIX_CACHE_KEY_FIELDS
                if class_node.name == "PrefixCacheKey"
                else _PREFIX_CACHE_FIELDS
            )
            unknown_fields = tuple(sorted(observed_fields - allowed_fields))
            if unknown_fields:
                collector.add(
                    unit,
                    class_node,
                    rule_id="cache_target_data",
                    status=EvidenceStatus.FAIL,
                    message=f"cache record has unknown payload fields {unknown_fields}",
                )
        if class_node.name == "PrefixCacheKey":
            missing = _missing_cache_identities(class_node)
            if missing:
                collector.add(
                    unit,
                    class_node,
                    rule_id="cache_target_data",
                    status=EvidenceStatus.FAIL,
                    message=f"cache key schema is missing required identities {missing}",
                )
        if "cache" in class_node.name.lower():
            forbidden_fields = tuple(
                field
                for child in class_node.body
                if isinstance(child, ast.AnnAssign)
                for field in _bound_names(child.target)
                if _category_for_name(field) & _CACHE_FORBIDDEN_CATEGORIES
            )
            if forbidden_fields:
                collector.add(
                    unit,
                    class_node,
                    rule_id="cache_target_data",
                    status=EvidenceStatus.FAIL,
                    message=f"cache record has forbidden fields {forbidden_fields}",
                )
    for function in (
        node
        for node in ast.walk(unit.tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ):
        taints = _function_taints(function, call_summaries)
        parameters = _function_arguments(function)
        parameter_categories = frozenset(
            category
            for name in parameters
            for category in _category_for_name(name)
        )
        sealed_categories = parameter_categories & {"sealed_train", "sealed_test"}
        if empirical_role is not None and sealed_categories:
            collector.add(
                unit,
                function,
                rule_id="split_dataflow",
                status=EvidenceStatus.FAIL,
                message=(
                    f"{empirical_role} input directly receives "
                    f"{', '.join(sorted(sealed_categories))} contents"
                ),
            )
        function_lower = function.name.lower()
        for child in ast.walk(function):
            if isinstance(child, ast.Return):
                categories = _expression_categories(
                    child.value, taints, call_summaries
                )
                forbidden = categories & _CACHE_FORBIDDEN_CATEGORIES
                if forbidden and any(
                    fragment in function_lower
                    for fragment in _PREDICTOR_SINK_FRAGMENTS
                ):
                    collector.add(
                        unit,
                        child,
                        rule_id="target_dataflow",
                        status=EvidenceStatus.FAIL,
                        message=(
                            f"predictor sink {function.name} returns tainted "
                            f"{', '.join(sorted(forbidden))} data"
                        ),
                    )
                if "unresolved" in categories and (
                    "cache" in function_lower
                    or any(
                        fragment in function_lower
                        for fragment in _PREDICTOR_SINK_FRAGMENTS
                    )
                ):
                    collector.add(
                        unit,
                        child,
                        rule_id="taint_resolution",
                        status=EvidenceStatus.INCONCLUSIVE,
                        message=f"sink {function.name} returns an unresolved helper result",
                    )
                if "cache" in function_lower and forbidden:
                    collector.add(
                        unit,
                        child,
                        rule_id="cache_target_data",
                        status=EvidenceStatus.FAIL,
                        message=(
                            f"cache {function.name} contains forbidden "
                            f"{', '.join(sorted(forbidden))} data"
                        ),
                    )
                escaped = categories & {"sealed_train", "sealed_test", "raw"}
                if empirical_role is not None and (
                    categories & {"sealed_train", "sealed_test"}
                ):
                    collector.add(
                        unit,
                        child,
                        rule_id="split_dataflow",
                        status=EvidenceStatus.FAIL,
                        message=(
                            f"{empirical_role} return carries "
                            f"{', '.join(sorted(categories & {'sealed_train', 'sealed_test'}))} contents"
                        ),
                    )
                if (
                    escaped
                    and (
                        "preprocess" in function_lower
                        or "blinded" in function_lower
                    )
                ):
                    collector.add(
                        unit,
                        child,
                        rule_id="preprocessing_escape",
                        status=EvidenceStatus.FAIL,
                        message=(
                            "public preprocessing return exposes "
                            f"{', '.join(sorted(escaped))} contents"
                        ),
                    )
            elif isinstance(child, (ast.Assign, ast.AnnAssign)):
                targets = (
                    tuple(child.targets)
                    if isinstance(child, ast.Assign)
                    else (child.target,)
                )
                value = child.value
                categories = _expression_categories(
                    value, taints, call_summaries
                )
                forbidden = categories & _CACHE_FORBIDDEN_CATEGORIES
                target_names = tuple(
                    name.lower()
                    for target in targets
                    for name in _bound_names(target)
                )
                if forbidden and any("cache" in name for name in target_names):
                    collector.add(
                        unit,
                        child,
                        rule_id="cache_target_data",
                        status=EvidenceStatus.FAIL,
                        message=(
                            "cache assignment contains forbidden "
                            f"{', '.join(sorted(forbidden))} data"
                        ),
                    )
            elif isinstance(child, ast.Call):
                categories = frozenset(
                    category
                    for argument in (
                        *child.args,
                        *(keyword.value for keyword in child.keywords),
                    )
                    for category in _expression_categories(
                        argument, taints, call_summaries
                    )
                )
                forbidden = categories & _CACHE_FORBIDDEN_CATEGORIES
                call_lower = _call_name(child).lower()
                sealed = categories & {"sealed_train", "sealed_test"}
                if empirical_role is not None and sealed:
                    collector.add(
                        unit,
                        child,
                        rule_id="split_dataflow",
                        status=EvidenceStatus.FAIL,
                        message=(
                            f"{empirical_role} call {_call_name(child)} receives "
                            f"{', '.join(sorted(sealed))} contents"
                        ),
                    )
                if empirical_role is not None and "unresolved" in categories:
                    collector.add(
                        unit,
                        child,
                        rule_id="taint_resolution",
                        status=EvidenceStatus.INCONCLUSIVE,
                        message=(
                            f"{empirical_role} call {_call_name(child)} receives "
                            "an unresolved helper result"
                        ),
                    )
                if forbidden and any(
                    fragment in call_lower for fragment in _PREDICTOR_SINK_FRAGMENTS
                ):
                    collector.add(
                        unit,
                        child,
                        rule_id="target_dataflow",
                        status=EvidenceStatus.FAIL,
                        message=(
                            f"predictor sink {_call_name(child)} receives "
                            f"{', '.join(sorted(forbidden))} data"
                        ),
                    )
                if forbidden and "cache" in call_lower:
                    collector.add(
                        unit,
                        child,
                        rule_id="cache_target_data",
                        status=EvidenceStatus.FAIL,
                        message=(
                            f"cache call {_call_name(child)} receives forbidden "
                            f"{', '.join(sorted(forbidden))} data"
                        ),
                    )
        if "cache" in function_lower:
            forbidden_parameters = parameter_categories & _CACHE_FORBIDDEN_CATEGORIES
            if forbidden_parameters:
                collector.add(
                    unit,
                    function,
                    rule_id="cache_target_data",
                    status=EvidenceStatus.FAIL,
                    message=(
                        f"cache schema accepts forbidden {', '.join(sorted(forbidden_parameters))} data"
                    ),
                )
        if function_lower in {"cache_key", "_cache_key"}:
            missing = _missing_cache_identities(function)
            if missing:
                collector.add(
                    unit,
                    function,
                    rule_id="cache_target_data",
                    status=EvidenceStatus.FAIL,
                    message=(
                        "cache key is missing required identities "
                        f"{missing}"
                    ),
                )
        if function_lower in {"cache_value", "_cache_value"}:
            unknown = _unknown_cache_value_names(function)
            if unknown:
                collector.add(
                    unit,
                    function,
                    rule_id="cache_target_data",
                    status=EvidenceStatus.FAIL,
                    message=f"cache value has noncausal payload fields {unknown}",
                )


def _audit_masking(
    unit: _SourceUnit,
    collector: _FindingCollector,
    normalizers: list[tuple[_SourceUnit, ast.FunctionDef | ast.AsyncFunctionDef]],
) -> None:
    assert unit.tree is not None
    aliases = _softmax_aliases(unit.tree)
    class_by_function: dict[int, str] = {}
    for class_node in (
        node for node in ast.walk(unit.tree) if isinstance(node, ast.ClassDef)
    ):
        for child in class_node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                class_by_function[id(child)] = class_node.name
    for function in (
        node
        for node in ast.walk(unit.tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ):
        function_lower = function.name.lower()
        calls = tuple(
            child for child in ast.walk(function) if isinstance(child, ast.Call)
        )
        identifiers = _identifier_inventory(function)
        semantic_normalizer = (
            any(_is_softmax_call(call, aliases) for call in calls)
            and (
                any(
                    _call_name(call).split(".")[-1] in {"masked_fill", "where"}
                    for call in calls
                )
                or any("support_mask" in name for name in identifiers)
            )
        )
        if (
            "masked_log_softmax" in function_lower
            or "renormalize_source" in function_lower
            or (
                "normalize" in function_lower
                and "source" in function_lower
                and "row" in function_lower
            )
            or semantic_normalizer
        ):
            normalizers.append((unit, function))
        class_lower = class_by_function.get(id(function), "").lower()
        mask_scope = (
            "source" in function_lower
            or "mask" in function_lower
            or "source_priors.py" in unit.relative_path.lower()
            or "sourcebank" in class_lower
            or "source_bank" in class_lower
        )
        if function.name != "masked_log_softmax_from_parents" and (
            "source" in function_lower
            or "source_priors.py" in unit.relative_path.lower()
            or "sourcebank" in class_lower
            or "source_bank" in class_lower
        ):
            for call in calls:
                if _is_softmax_call(call, aliases):
                    collector.add(
                        unit,
                        call,
                        rule_id="direct_source_softmax",
                        status=EvidenceStatus.FAIL,
                        message=(
                            f"source function {function.name} calls "
                            f"{_call_name(call).split('.')[-1]} outside the shared masked helper"
                        ),
                )
        softmax_names: dict[str, ast.AST] = {}
        for child in ast.walk(function):
            if isinstance(child, (ast.Assign, ast.AnnAssign)):
                targets = (
                    tuple(child.targets)
                    if isinstance(child, ast.Assign)
                    else (child.target,)
                )
                value = child.value
                if value is not None and _is_softmax_call(value, aliases):
                    for target in targets:
                        for name in _bound_names(target):
                            softmax_names[name] = child
        for child in ast.walk(function):
            if not isinstance(child, (ast.Assign, ast.AnnAssign, ast.Return, ast.Expr)):
                continue
            value = getattr(child, "value", None)
            if value is None or _is_softmax_call(value, aliases):
                continue
            referenced = {
                name.id
                for name in ast.walk(value)
                if isinstance(name, ast.Name) and name.id in softmax_names
            }
            if mask_scope and referenced:
                value_dump = ast.dump(value, include_attributes=False).lower()
                if (
                    "mask" in value_dump
                    or "where" in value_dump
                    or "renormal" in value_dump
                    or isinstance(value, (ast.BinOp, ast.Subscript))
                ):
                    collector.add(
                        unit,
                        child,
                        rule_id="post_softmax_mask",
                        status=EvidenceStatus.FAIL,
                        message=(
                            "source probabilities are masked or renormalized after softmax: "
                            f"{', '.join(sorted(referenced))}"
                        ),
                    )
            inline_softmax = tuple(
                call
                for call in ast.walk(value)
                if _is_softmax_call(call, aliases)
                and not _softmax_input_is_premasked(call)
            )
            if mask_scope and inline_softmax:
                value_dump = ast.dump(value, include_attributes=False).lower()
                outer_call = _call_name(value).lower() if isinstance(value, ast.Call) else ""
                if (
                    isinstance(value, (ast.BinOp, ast.Subscript))
                    or any(
                        marker in value_dump
                        for marker in ("support_mask", "post_softmax", "renormal")
                    )
                    or any(
                        marker in outer_call
                        for marker in ("masked_fill", "where", "renormal")
                    )
                ):
                    collector.add(
                        unit,
                        child,
                        rule_id="post_softmax_mask",
                        status=EvidenceStatus.FAIL,
                        message="source support is applied inline after softmax",
                    )
        if function.name == "masked_log_softmax_from_parents":
            for conditional in (
                child for child in ast.walk(function) if isinstance(child, ast.If)
            ):
                if _if_tests_empty_support(conditional) and any(
                    isinstance(descendant, ast.Return)
                    for statement in conditional.body
                    for descendant in ast.walk(statement)
                ):
                    collector.add(
                        unit,
                        conditional,
                        rule_id="all_invalid_fallback",
                        status=EvidenceStatus.FAIL,
                        message="empty source support returns a fabricated fallback",
                    )
        for child in ast.walk(function):
            if isinstance(child, ast.Assign):
                target_names = {
                    name.lower()
                    for target in child.targets
                    for name in _bound_names(target)
                }
                if (
                    any("declared_parents" in name or name.endswith("parents") for name in target_names)
                    and _parent_expression_is_noncausal(child.value)
                ):
                    collector.add(
                        unit,
                        child,
                        rule_id="noncausal_parent",
                        status=EvidenceStatus.FAIL,
                        message="declared parent construction includes receiver_t or a future node",
                    )
            elif isinstance(child, ast.Compare):
                comparison = ast.dump(child, include_attributes=False)
                admits_receiver = (
                    any(isinstance(operator, ast.LtE) for operator in child.ops)
                    and "receiver_t" in comparison
                ) or (
                    any(isinstance(operator, ast.Lt) for operator in child.ops)
                    and any(
                        _receiver_plus_positive(comparator)
                        for comparator in child.comparators
                    )
                )
                if (
                    admits_receiver
                    and "parent" in comparison.lower()
                ):
                    collector.add(
                        unit,
                        child,
                        rule_id="noncausal_parent",
                        status=EvidenceStatus.FAIL,
                        message="source support admits a parent at or beyond receiver_t",
                    )
            elif (
                isinstance(child, ast.Call)
                and _call_name(child).split(".")[-1]
                == "masked_log_softmax_from_parents"
                and len(child.args) >= 2
                and _parent_expression_is_noncausal(child.args[1])
            ):
                collector.add(
                    unit,
                    child,
                    rule_id="noncausal_parent",
                    status=EvidenceStatus.FAIL,
                    message="source parent range includes self or future nodes",
                )


def _build_checks(
    findings: tuple[StaticAuditFinding, ...]
) -> tuple[StaticAuditCheck, ...]:
    checks: list[StaticAuditCheck] = []
    for name in _CHECK_NAMES:
        owned = tuple(
            finding
            for finding in findings
            if _RULE_GROUP[finding.rule_id] == name
        )
        status = (
            EvidenceStatus.FAIL
            if any(item.status is EvidenceStatus.FAIL for item in owned)
            else EvidenceStatus.INCONCLUSIVE
            if owned
            else EvidenceStatus.PASS
        )
        obligations = (
            tuple(
                f"{item.rule_id}: {item.message} ({item.path}:{item.line})"
                for item in owned
                if item.status is EvidenceStatus.INCONCLUSIVE
            )
            if status is EvidenceStatus.INCONCLUSIVE
            else ()
        )
        payload = {
            "name": name,
            "status": status.value,
            "finding_sha256s": tuple(item.finding_sha256 for item in owned),
            "obligations": obligations,
        }
        checks.append(
            StaticAuditCheck(
                name,
                status,
                tuple(item.finding_sha256 for item in owned),
                obligations,
                _owned_hash("vfe4.h6.static-audit-check.v1", payload),
            )
        )
    return tuple(checks)


def audit_h6_static_source(
    repo_root: Path,
    exact_case_keys: tuple[PrefixCaseKey, ...],
) -> StaticAuditReport:
    """Audit H6 source without importing or executing the audited tree."""

    if not isinstance(repo_root, Path):
        raise ValueError("repo_root must be a pathlib.Path")
    root = repo_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("repo_root must resolve to a directory")
    case_key_manifest_sha256 = _case_key_manifest(exact_case_keys)
    units, production_mode = _load_sources(root)
    source_manifest_sha256 = _owned_hash(
        "vfe4.h6.static-audit-source-manifest.v1",
        tuple(
            {
                "path": unit.relative_path,
                "sha256": hashlib.sha256(unit.raw).hexdigest(),
            }
            for unit in units
        ),
    )
    rules_sha256 = _owned_hash(
        "vfe4.h6.static-audit-rules.v1",
        {
            "groups": _RULE_GROUP,
            "descriptions": _RULE_DESCRIPTIONS,
            "required_sinks": tuple(sorted(_REQUIRED_SINKS)),
            "production_markers": _PRODUCTION_MARKERS,
            "cache_key_identities": _CACHE_KEY_IDENTITIES,
            "cache_value_allowed_names": tuple(sorted(_CACHE_VALUE_ALLOWED_NAMES)),
            "prefix_cache_key_fields": tuple(sorted(_PREFIX_CACHE_KEY_FIELDS)),
            "prefix_cache_fields": tuple(sorted(_PREFIX_CACHE_FIELDS)),
        },
    )
    collector = _FindingCollector()
    normalizers: list[
        tuple[_SourceUnit, ast.FunctionDef | ast.AsyncFunctionDef]
    ] = []
    call_summaries = _return_category_summaries(units)
    saw_inventory = False
    declared_counts: set[int] = set()
    if production_mode:
        missing_markers = tuple(
            marker for marker in _PRODUCTION_MARKERS if not (root / marker).is_file()
        )
        if missing_markers:
            marker_unit = units[0]
            collector.add(
                marker_unit,
                marker_unit.tree,
                rule_id="source_inventory",
                status=EvidenceStatus.FAIL,
                message=f"production audit module inventory is missing {missing_markers}",
                witness="\n".join(missing_markers),
            )
    for unit in units:
        if unit.tree is None:
            collector.add(
                unit,
                None,
                rule_id="syntax",
                status=EvidenceStatus.INCONCLUSIVE,
                message=f"audited source could not be parsed: {unit.parse_error}",
            )
            continue
        _audit_imports_signatures_and_access(unit, collector)
        unit_counts, unit_inventory = _audit_counts_and_inventory(
            unit,
            collector,
            production_mode=production_mode,
        )
        declared_counts.update(unit_counts)
        saw_inventory = unit_inventory or saw_inventory
        _audit_taint(unit, collector, call_summaries)
        _audit_masking(unit, collector, normalizers)
    if len(normalizers) > 1:
        for unit, function in normalizers[1:]:
            collector.add(
                unit,
                function,
                rule_id="duplicate_normalizer",
                status=EvidenceStatus.FAIL,
                message=(
                    "second source normalization helper duplicates "
                    f"{normalizers[0][1].name}: {function.name}"
                ),
            )
    if production_mode and not saw_inventory:
        marker_unit = next(
            (
                unit
                for unit in units
                if unit.relative_path == "vfe4/predictive/prior.py"
            ),
            units[0],
        )
        collector.add(
            marker_unit,
            marker_unit.tree,
            rule_id="sink_inventory",
            status=EvidenceStatus.INCONCLUSIVE,
            message="production source has no explicit complete H6 static sink inventory",
            witness="missing H6_STATIC_SINK_INVENTORY",
        )
    if production_mode and not {168, 16384}.issubset(declared_counts):
        marker_unit = next(
            (
                unit
                for unit in units
                if unit.relative_path == "vfe4/generative/source_priors.py"
            ),
            units[0],
        )
        collector.add(
            marker_unit,
            marker_unit.tree,
            rule_id="base_count",
            status=EvidenceStatus.INCONCLUSIVE,
            message=(
                "production source lacks canonical base-count declarations "
                f"for {tuple(sorted({168, 16384} - declared_counts))}"
            ),
            witness="missing canonical H6 base counts",
        )
    findings = collector.values()
    checks = _build_checks(findings)
    status = (
        EvidenceStatus.FAIL
        if any(item.status is EvidenceStatus.FAIL for item in findings)
        else EvidenceStatus.INCONCLUSIVE
        if findings
        else EvidenceStatus.PASS
    )
    obligations = (
        tuple(
            f"{item.rule_id}: {item.message} ({item.path}:{item.line})"
            for item in findings
            if item.status is EvidenceStatus.INCONCLUSIVE
        )
        if status is EvidenceStatus.INCONCLUSIVE
        else ()
    )
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "source_manifest_sha256": source_manifest_sha256,
        "rules_sha256": rules_sha256,
        "case_key_manifest_sha256": case_key_manifest_sha256,
        "checks": tuple(check.check_sha256 for check in checks),
        "findings": tuple(item.finding_sha256 for item in findings),
        "status": status.value,
        "obligations": obligations,
    }
    return StaticAuditReport(
        _SCHEMA_VERSION,
        source_manifest_sha256,
        rules_sha256,
        case_key_manifest_sha256,
        checks,
        findings,
        status,
        obligations,
        _owned_hash("vfe4.h6.static-audit-report.v1", payload),
    )


__all__ = [
    "StaticAuditCheck",
    "StaticAuditFinding",
    "StaticAuditReport",
    "audit_h6_static_source",
]
