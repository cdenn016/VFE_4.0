from __future__ import annotations

import ast
import hashlib
import inspect
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import mpmath as mp
import pytest

import verification.mp_oracles.h7_covariance as h7_oracle
from verification.h7_budget import (
    H7BackwardOperandInput,
    H7BoundBudget,
    H7BudgetFormula,
    build_h7_backward_records,
    build_h7_budget,
    control_decisiveness_limit,
    require_control_decisive,
)
from verification.mp_oracles.h7_covariance import (
    H7_COMPLETE_LOCAL_TERM_IDS,
    H7OracleInconclusive,
    MPTrialResult,
    build_h7_scalar_probe_table_bytes,
    evaluate_h7_from_raw_bytes,
    evaluate_h7_task5_wiring,
    standard_normal_gauss_hermite,
)
from verification.mp_oracles.h7_budget_protocol import (
    H7BackwardResidualRecord as MPBackwardResidualRecord,
)
from vfe4.objective.h7_covariance import (
    H7_COMPLETE_LOCAL_TERM_IDS as TASK5_COMPLETE_LOCAL_TERM_IDS,
)
from vfe4.types.h7 import (
    H7AllowanceContribution,
    H7BudgetRecord,
    H7OperandRecord,
)


ROOT = Path(__file__).resolve().parents[2]
H1_FIXTURE = ROOT / "vfe4" / "validation" / "fixtures" / "h1_v1.json"
H7_FIXTURE = ROOT / "vfe4" / "validation" / "fixtures" / "h7_v1.json"
H7_PROBES = ROOT / "vfe4" / "validation" / "fixtures" / "h7_density_probes_v1.json"
ORACLE_SOURCE = ROOT / "verification" / "mp_oracles" / "h7_covariance.py"
ORACLE_BUDGET_PROTOCOL_SOURCE = (
    ROOT / "verification" / "mp_oracles" / "h7_budget_protocol.py"
)
TASK5_SOURCE = ROOT / "vfe4" / "objective" / "h7_covariance.py"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _mutated_json_bytes(
    raw: bytes,
    path: tuple[str | int, ...],
    replacement: object,
) -> bytes:
    value = json.loads(raw)
    cursor = value
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def test_h7_scalar_probe_builder_is_exact_and_self_validating() -> None:
    h1_bytes = H1_FIXTURE.read_bytes()

    table_bytes = build_h7_scalar_probe_table_bytes(h1_bytes)
    table = json.loads(table_bytes)

    assert table_bytes == (
        json.dumps(
            table,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )
    assert tuple(table) == (
        "anchor_provenance",
        "fixture_id",
        "ordered_source_path_ids",
        "probe_set_sha256",
        "probe_table_schema",
        "raw_fixture_sha256",
        "records",
        "scalar_trial_action_sha256",
    )
    assert table["probe_table_schema"] == "h7-scalar-density-probe-table-v1"
    assert table["fixture_id"] == "h1-v1"
    assert table["raw_fixture_sha256"] == hashlib.sha256(h1_bytes).hexdigest()
    assert table["ordered_source_path_ids"] == [
        "h1-path-0:a0-b0",
        "h1-path-1:a1-b0",
        "h1-path-2:a0-b1",
        "h1-path-3:a1-b1",
    ]
    assert len(table["scalar_trial_action_sha256"]) == 2
    assert len(table["records"]) == 8
    assert [record["probe_id"] for record in table["records"]] == [
        "scalar-base-transformed:h1.p.global.source_path:h1-path-0:a0-b0",
        "scalar-base-transformed:h1.p.global.source_path:h1-path-1:a1-b0",
        "scalar-base-transformed:h1.p.global.source_path:h1-path-2:a0-b1",
        "scalar-base-transformed:h1.p.global.source_path:h1-path-3:a1-b1",
        "scalar-internal-transformed:h1.p.global.source_path:h1-path-0:a0-b0",
        "scalar-internal-transformed:h1.p.global.source_path:h1-path-1:a1-b0",
        "scalar-internal-transformed:h1.p.global.source_path:h1-path-2:a0-b1",
        "scalar-internal-transformed:h1.p.global.source_path:h1-path-3:a1-b1",
    ]
    wrong_identity = dict(table)
    wrong_identity["fixture_id"] = "h7-v1"
    with pytest.raises(ValueError, match="identity/path order"):
        h7_oracle._validate_scalar_probe_table(
            wrong_identity,
            h7_oracle._parse_raw_json(h1_bytes),
            h7_oracle._h1_source_paths(h7_oracle._parse_raw_json(h1_bytes)),
        )
    with pytest.raises(ValueError, match="raw H1 fixture identity"):
        build_h7_scalar_probe_table_bytes(h1_bytes + b"\n")


def _local_import_graph(entry: Path) -> tuple[set[Path], set[str]]:
    pending = [entry.resolve()]
    visited: set[Path] = set()
    imported_roots: set[str] = set()

    def enqueue_module(module: str, *, relative_to: Path | None = None) -> None:
        parts = module.split(".") if module else []
        base = ROOT if relative_to is None else relative_to
        candidate = base.joinpath(*parts)
        module_file = candidate.with_suffix(".py")
        package_file = candidate / "__init__.py"
        target = (
            module_file
            if module_file.is_file()
            else package_file
            if package_file.is_file()
            else None
        )
        if target is not None and target.resolve() not in visited:
            pending.append(target.resolve())
        cursor = candidate if package_file.is_file() else candidate.parent
        while cursor != ROOT and ROOT in cursor.parents:
            init_file = cursor / "__init__.py"
            if init_file.is_file() and init_file.resolve() not in visited:
                pending.append(init_file.resolve())
            cursor = cursor.parent

    while pending:
        source_path = pending.pop()
        if source_path in visited:
            continue
        visited.add(source_path)
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_roots.add(alias.name.split(".", maxsplit=1)[0])
                    enqueue_module(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    relative_base = source_path.parent
                    for _ in range(node.level - 1):
                        relative_base = relative_base.parent
                    enqueue_module(module, relative_to=relative_base)
                elif module:
                    imported_roots.add(module.split(".", maxsplit=1)[0])
                    enqueue_module(module)
    return visited, imported_roots


def _operand(
    operand_id: str,
    *,
    category: str,
    role: str,
    shape: tuple[int, ...],
    scale: float,
    condition: float = 1.0,
    normalization: float | None = None,
    oracle_value: str | None = None,
) -> H7OperandRecord:
    return H7OperandRecord.create(
        operand_id=operand_id,
        category=category,
        role=role,
        dtype="decimal100" if role == "oracle" else "float64",
        shape=shape,
        value_sha256=_sha(operand_id),
        scale=scale,
        condition_number=condition,
        normalization=scale if normalization is None else normalization,
        oracle_value=oracle_value,
    )


def _formula(
    *,
    category: str,
    operation_kind: str,
    dimension_operand_id: str | None = None,
    compared_operand_ids: tuple[str, ...] = (),
    source_operand_ids: tuple[str, ...] = (),
    direct_action_operand_ids: tuple[str, ...] = (),
    spd_operand_ids: tuple[str, ...] = (),
    frame_operand_ids: tuple[str, ...] = (),
    link_operand_ids: tuple[str, ...] = (),
    signed_summand_operand_ids: tuple[str, ...] = (),
    child_budgets: tuple[H7BoundBudget, ...] = (),
    forward_budget: H7BoundBudget | None = None,
    inverse_action_budget: H7BoundBudget | None = None,
    quadrature_operand_ids: tuple[str, str] | None = None,
    reference_operand_id: str | None = None,
) -> H7BudgetFormula:
    return H7BudgetFormula(
        category=category,
        operation_kind=operation_kind,
        dimension_operand_id=dimension_operand_id,
        compared_operand_ids=compared_operand_ids,
        source_operand_ids=source_operand_ids,
        direct_action_operand_ids=direct_action_operand_ids,
        spd_operand_ids=spd_operand_ids,
        frame_operand_ids=frame_operand_ids,
        link_operand_ids=link_operand_ids,
        signed_summand_operand_ids=signed_summand_operand_ids,
        child_budgets=child_budgets,
        forward_budget=forward_budget,
        inverse_action_budget=inverse_action_budget,
        quadrature_operand_ids=quadrature_operand_ids,
        reference_operand_id=reference_operand_id,
    )


def _leaf_formula(
    prefix: str,
    category: str,
    *,
    operation_kind: str = "matrix_product",
) -> H7BudgetFormula:
    source_id = f"{prefix}.source"
    return _formula(
        category=category,
        operation_kind=operation_kind,
        dimension_operand_id=source_id,
        compared_operand_ids=(
            f"{prefix}.original",
            f"{prefix}.transformed",
        ),
        source_operand_ids=(source_id,),
        direct_action_operand_ids=(f"{prefix}.action",),
        spd_operand_ids=(
            (source_id,)
            if category in {"covariance", "precision", "second_moment"}
            else ()
        ),
    )


def _leaf_operands(
    prefix: str,
    category: str,
    shape: tuple[int, ...],
    *,
    source_condition: float = 1.0,
    action_condition: float = 1.0,
) -> tuple[H7OperandRecord, ...]:
    dimension = shape[-1]
    return (
        _operand(
            f"{prefix}.original",
            category=category,
            role="original",
            shape=shape,
            scale=2.0,
        ),
        _operand(
            f"{prefix}.transformed",
            category=category,
            role="transformed",
            shape=shape,
            scale=5.0,
        ),
        _operand(
            f"{prefix}.source",
            category=category,
            role="reference",
            shape=shape,
            scale=7.0,
            condition=source_condition,
        ),
        _operand(
            f"{prefix}.action",
            category=category,
            role="reference",
            shape=(dimension, dimension),
            scale=1.0,
            condition=action_condition,
        ),
    )


def _real_leaf_budget(
    prefix: str,
    category: str,
    shape: tuple[int, ...],
    *,
    operation_kind: str = "matrix_product",
) -> H7BoundBudget:
    return build_h7_budget(
        invariant_id=prefix,
        category=category,
        operands=_leaf_operands(prefix, category, shape),
        formula=_leaf_formula(
            prefix,
            category,
            operation_kind=operation_kind,
        ),
    )


def _gaussian_ids(prefix: str) -> tuple[str, ...]:
    return (
        f"{prefix}.mean",
        f"{prefix}.covariance",
        f"{prefix}.precision",
        f"{prefix}.h_information",
        f"{prefix}.J_precision",
        f"{prefix}.M_second_moment",
    )


def _complete_synthetic_backward_ids(prefix: str) -> tuple[str, ...]:
    ids: list[str] = [
        *_gaussian_ids(f"{prefix}.p.initial_joint"),
        *_gaussian_ids(f"{prefix}.q.initial_joint"),
        *(f"{prefix}.U[{index}]" for index in range(3)),
        *(
            f"{prefix}.Omega[{receiver}<-{source}]"
            for receiver in range(3)
            for source in range(3)
            if receiver != source
        ),
    ]
    for group in ("p", "q_model", "q_state"):
        for bank in ("model", "state"):
            for receiver_t, source_j in ((1, 0), (2, 1)):
                base = (
                    f"{prefix}.{group}.{bank}.receiver_{receiver_t}.source_{source_j}"
                )
                ids.append(f"{base}.parent_map")
                if bank == "state":
                    ids.append(f"{base}.B_model_map")
                ids.extend(_gaussian_ids(f"{base}.receiver_offset"))
    for receiver_t in (1, 2):
        ids.extend(
            (
                f"{prefix}.decoder[{receiver_t}].state_weight",
                f"{prefix}.decoder[{receiver_t}].model_weight",
            )
        )
    ids.extend(
        f"{prefix}.source_scorer.{channel}_history[{index}]"
        for channel in ("z", "m")
        for index in range(2)
    )
    ids.extend(
        f"{prefix}.source_scorer.{bank}[{receiver_t}<-{source_j}].{channel}_covector"
        for bank in ("model", "state")
        for receiver_t, source_j in ((1, 0), (2, 1))
        for channel in ("z", "m")
    )
    ids.extend(
        operand_id
        for role in ("q", "p")
        for operand_id in _gaussian_ids(
            f"{prefix}.{role}.global[matrix-singleton-path]"
        )
    )
    return tuple(ids)


def _injected_actions() -> tuple[mp.matrix, mp.matrix, mp.matrix]:
    return (
        mp.matrix([["1.1", "0.1"], ["0.0", "0.9"]]),
        mp.matrix([["0.95", "-0.05"], ["0.08", "1.05"]]),
        mp.matrix([["1.02", "0.04"], ["-0.03", "0.98"]]),
    )


def _injected_inventory(count: int):
    patterns = (
        (
            "vector",
            "left",
            (0,),
            mp.matrix(["0.25", "-0.5"]),
            True,
        ),
        (
            "map",
            "left",
            (1,),
            mp.matrix([["1.0", "0.1"], ["0.0", "0.9"]]),
            False,
        ),
        (
            "covariance",
            "covariance",
            (2,),
            mp.matrix([["1.5", "0.1"], ["0.1", "1.2"]]),
            False,
        ),
        (
            "precision",
            "precision",
            (0,),
            mp.matrix([["0.8", "0.05"], ["0.05", "1.1"]]),
            False,
        ),
        (
            "information",
            "information",
            (1,),
            mp.matrix(["0.4", "-0.2"]),
            True,
        ),
        (
            "map",
            "receiver_source",
            (2, 0),
            mp.matrix([["0.7", "0.2"], ["-0.1", "1.0"]]),
            False,
        ),
        (
            "decoder",
            "decoder",
            (2,),
            mp.matrix(
                [
                    ["0.5", "-0.2"],
                    ["0.1", "0.4"],
                    ["-0.3", "0.6"],
                ]
            ),
            False,
        ),
        (
            "offset",
            "left",
            (0,),
            mp.matrix(["0.3", "0.15"]),
            True,
        ),
        (
            "second_moment",
            "covariance",
            (1,),
            mp.matrix([["1.7", "0.2"], ["0.2", "1.4"]]),
            False,
        ),
    )
    inventory = []
    for index in range(count):
        category, transform_kind, action_indices, value, vector = patterns[
            index % len(patterns)
        ]
        condition_number = (
            h7_oracle._condition_spd(value)
            if category in {"covariance", "precision", "second_moment"}
            else None
        )
        record = h7_oracle._value_record(
            f"injected.operand[{index}]",
            value,
            vector=vector,
            condition_number=condition_number,
        )
        inventory.append(
            h7_oracle._MPInventoryOperand(
                record,
                category,
                transform_kind,
                action_indices,
            )
        )
    return tuple(inventory)


@dataclass(frozen=True)
class _InjectedOwnedTensor:
    snapshot_sha256: str

    def assert_intact(self) -> None:
        if len(self.snapshot_sha256) != 64:
            raise ValueError("injected tensor identity changed")


@dataclass(frozen=True)
class _InjectedAction:
    elements: tuple[
        _InjectedOwnedTensor,
        _InjectedOwnedTensor,
        _InjectedOwnedTensor,
    ]
    action_sha256: str

    def __post_init__(self) -> None:
        if len(self.elements) != 3 or len(self.action_sha256) != 64:
            raise ValueError("injected action identity changed")
        for item in self.elements:
            item.assert_intact()


@dataclass(frozen=True)
class _InjectedTrialSpec:
    trial_id: str
    fixture_id: str
    frame_profile: str
    decoder_policy: str
    action: _InjectedAction
    action_sha256: str

    def __post_init__(self) -> None:
        if self.action_sha256 != self.action.action_sha256:
            raise ValueError("injected trial/action identity changed")


@dataclass(frozen=True)
class _InjectedResidual:
    invariant_id: str
    value: float


@dataclass(frozen=True)
class _InjectedInitial:
    term_id: str
    original_value: float
    transformed_value: float
    residual: _InjectedResidual


@dataclass(frozen=True)
class _InjectedLocal:
    term_id: str
    original_value: float
    transformed_value: float
    residual: _InjectedResidual


@dataclass(frozen=True)
class _InjectedScalarEvidence:
    fixture_id: str
    raw_fixture_sha256: str
    action_sha256: str
    original_log_evidence: float
    transformed_log_evidence: float
    original_posterior_kl: float
    transformed_posterior_kl: float

    def __post_init__(self) -> None:
        if (
            self.fixture_id != "h1-v1"
            or len(self.raw_fixture_sha256) != 64
            or len(self.action_sha256) != 64
        ):
            raise ValueError("injected scalar evidence identity changed")


@dataclass(frozen=True)
class _InjectedTask5Evaluation:
    original_complete_local_value: float
    transformed_complete_local_value: float
    initial_joint_kl: _InjectedInitial
    local_terms: tuple[_InjectedLocal, ...]
    complete_local: _InjectedResidual
    complete_monolithic: _InjectedResidual
    p_density_shift: _InjectedResidual
    q_density_shift: _InjectedResidual
    log_ratio: _InjectedResidual
    entropy_shift: _InjectedResidual
    scalar_evidence: _InjectedScalarEvidence | None
    evidence: _InjectedResidual | None
    posterior_kl: _InjectedResidual | None
    not_applicable_reason: str | None

    def __post_init__(self) -> None:
        if (
            self.initial_joint_kl.term_id != "K0_joint_z0_m0"
            or tuple(item.term_id for item in self.local_terms)
            != H7_COMPLETE_LOCAL_TERM_IDS
        ):
            raise ValueError("injected Task-5 record inventory changed")


def _zero_scalar_items() -> tuple[tuple[str, str], ...]:
    local_ids = tuple(
        value_id
        for term_id in H7_COMPLETE_LOCAL_TERM_IDS
        for value_id in (
            term_id,
            f"transformed.{term_id}",
            f"residual.{term_id}",
        )
    )
    value_ids = (
        "complete_local_elbo",
        "transformed.complete_local_elbo",
        "residual.complete_local_elbo",
        "K0_joint_z0_m0",
        "transformed.K0_joint_z0_m0",
        "residual.K0_joint_z0_m0",
        *local_ids,
        "residual.complete_monolithic_elbo",
        "complete_local_monolithic_delta",
        "transformed.complete_local_monolithic_delta",
        "complete_pointwise_p_density_shift",
        "complete_pointwise_q_density_shift",
        "complete_pointwise_log_ratio",
        "residual.joint_recognition_entropy",
        "scalar_log_evidence",
        "transformed.scalar_log_evidence",
        "scalar_posterior_kl",
        "transformed.scalar_posterior_kl",
        "scalar_log_evidence_residual",
        "scalar_evidence_elbo_posterior_kl_residual",
        "transformed.scalar_evidence_elbo_posterior_kl_residual",
        "scalar_posterior_kl_residual",
    )
    return tuple((value_id, "0.0") for value_id in value_ids)


def test_h7_oracle_is_independent_and_uses_only_actual_rhs_solves() -> None:
    source = ORACLE_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = {
        alias.name.split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        node.module.split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    assert imported_roots.isdisjoint({"numpy", "torch", "vfe4"})
    assert "mp.inverse" not in source
    assert "_cofactor_inverse" not in source
    assert "adjugate" not in source
    assert "_minor(" not in source
    assert "lu_solve(coefficient, actual_rhs)" in source
    assert "mp.eye" not in source
    assert "identity_rhs" not in source

    previous = mp.mp.dps
    with mp.workdps(100):
        rule = standard_normal_gauss_hermite(51)
        assert rule.order == 51
        assert tuple(rule.jacobi[row, row] for row in range(51)) == (mp.mpf("0"),) * 51
        for k in range(1, 51):
            expected = mp.sqrt(mp.mpf(k) / 2)
            assert rule.jacobi[k - 1, k] == expected
            assert rule.jacobi[k, k - 1] == expected
        for degree in range(13):
            observed = mp.fsum(
                weight * node**degree
                for node, weight in zip(
                    rule.standard_normal_nodes,
                    rule.weights,
                    strict=True,
                )
            )
            expected_moment = (
                mp.mpf("0")
                if degree % 2
                else mp.factorial(degree)
                / (2 ** (degree // 2) * mp.factorial(degree // 2))
            )
            assert mp.almosteq(
                observed,
                expected_moment,
                rel_eps=mp.mpf("1e-90"),
                abs_eps=mp.mpf("1e-90"),
            )
    assert mp.mp.dps == previous


def test_h7_raw_contract_is_closed_and_unmeasured_calibration_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h1_bytes = H1_FIXTURE.read_bytes()
    h7_bytes = H7_FIXTURE.read_bytes()
    matrix_probe_bytes = H7_PROBES.read_bytes()
    result = evaluate_h7_from_raw_bytes(h1_bytes, h7_bytes, matrix_probe_bytes)
    assert result.status == "INCONCLUSIVE"
    assert result.trials == ()
    assert result.inventory_sha256 is None
    assert result.raw_fixture_sha256 == (
        hashlib.sha256(h1_bytes).hexdigest(),
        hashlib.sha256(h7_bytes).hexdigest(),
        hashlib.sha256(matrix_probe_bytes).hexdigest(),
    )
    assert tuple(path.path_id for path in result.h1_source_paths) == (
        "h1-path-0:a0-b0",
        "h1-path-1:a1-b0",
        "h1-path-2:a0-b1",
        "h1-path-3:a1-b1",
    )
    assert any(
        "scalar density-probe table is missing" in item
        for item in result.open_obligations
    )
    assert any("UNMEASURED" in item for item in result.open_obligations)

    with pytest.raises(H7OracleInconclusive, match="strict UTF-8 JSON") as malformed:
        evaluate_h7_from_raw_bytes(b"{", h7_bytes, matrix_probe_bytes)
    assert malformed.value.status == "INCONCLUSIVE"
    with pytest.raises(ValueError, match="exact closed field set"):
        evaluate_h7_from_raw_bytes(
            h1_bytes.replace(b"{", b'{"unexpected":0,', 1),
            h7_bytes,
            matrix_probe_bytes,
        )
    with pytest.raises(ValueError, match="H1 fixture identity"):
        evaluate_h7_from_raw_bytes(
            h1_bytes.replace(
                b'"observation_label_base": 1',
                b'"observation_label_base": 0',
            ),
            h7_bytes,
            matrix_probe_bytes,
        )
    with pytest.raises(ValueError, match="H7 fixture identity"):
        evaluate_h7_from_raw_bytes(
            h1_bytes,
            h7_bytes.replace(b'"horizon": 2', b'"horizon": 3'),
            matrix_probe_bytes,
        )
    with pytest.raises(ValueError, match="identity/order/hash"):
        evaluate_h7_from_raw_bytes(
            h1_bytes,
            h7_bytes,
            matrix_probe_bytes.replace(b'"row_index":0', b'"row_index":7', 1),
        )
    with pytest.raises(ValueError, match="table identity"):
        evaluate_h7_from_raw_bytes(
            h1_bytes,
            h7_bytes,
            matrix_probe_bytes.replace(
                b'"probe_set_sha256":"f002',
                b'"probe_set_sha256":"0002',
                1,
            ),
        )
    with pytest.raises(ValueError, match="identity/order/hash"):
        evaluate_h7_from_raw_bytes(
            h1_bytes,
            h7_bytes,
            matrix_probe_bytes.replace(
                b'"anchor_sha256":"369a',
                b'"anchor_sha256":"069a',
                1,
            ),
        )
    with pytest.raises(ValueError, match="probe_sha256 does not bind"):
        evaluate_h7_from_raw_bytes(
            h1_bytes,
            h7_bytes,
            matrix_probe_bytes.replace(
                b'"probe_sha256":"216ed2',
                b'"probe_sha256":"316ed2',
                1,
            ),
        )
    with pytest.raises(ValueError, match="action relation"):
        evaluate_h7_from_raw_bytes(
            h1_bytes,
            h7_bytes,
            matrix_probe_bytes.replace(
                b'"x_prime":[0.21999999999999997',
                b'"x_prime":[0.22999999999999997',
                1,
            ),
        )
    with pytest.raises(ValueError, match="scalar_probe_table"):
        evaluate_h7_from_raw_bytes(
            h1_bytes,
            h7_bytes,
            matrix_probe_bytes,
            b"{}",
        )
    malformed_h1_paths = (
        ("continuous_order",),
        ("vocabulary_labels",),
        ("observation_labels",),
        ("recognition", "model_kernels"),
        ("model_source_priors", 1),
        ("recognition", "state_source_probabilities_given_model_source", 1, 0),
        ("initial_joint", "covariance"),
    )
    malformed_h1_values = (
        None,
        {},
        None,
        None,
        ["0.2", "0.2"],
        ["0.0", "1.0"],
        [["1.0", "2.0"], ["2.0", "1.0"]],
    )
    for path, value in zip(
        malformed_h1_paths,
        malformed_h1_values,
        strict=True,
    ):
        with pytest.raises(H7OracleInconclusive) as malformed_external:
            evaluate_h7_from_raw_bytes(
                _mutated_json_bytes(h1_bytes, path, value),
                h7_bytes,
                matrix_probe_bytes,
            )
        assert type(malformed_external.value) is H7OracleInconclusive
    malformed_h7_cases = (
        (("continuous_order",), None),
        (("observation_labels",), {}),
        (("state_parent_sets",), None),
        (("generative", "decoder"), {}),
        (("recognition", "model_parent_maps"), None),
        (("density_probes", "components"), None),
        (("oracle", "gauss_hermite_orders"), {}),
        (("generative", "model_source_probabilities", 1), ["0.0"]),
        (
            ("generative", "initial_covariance"),
            [
                ["1.0", "0.0", "0.0", "0.0"],
                ["0.0", "1.0", "0.0", "0.0"],
                ["0.0", "0.0", "1.0", "0.0"],
                ["0.0", "0.0", "0.0", "-1.0"],
            ],
        ),
        (
            ("recognition", "model_receiver_covariances", 0),
            [["1.0", "2.0"], ["2.0", "1.0"]],
        ),
    )
    for path, value in malformed_h7_cases:
        with pytest.raises(H7OracleInconclusive) as malformed_external:
            evaluate_h7_from_raw_bytes(
                h1_bytes,
                _mutated_json_bytes(h7_bytes, path, value),
                matrix_probe_bytes,
            )
        assert type(malformed_external.value) is H7OracleInconclusive
    with pytest.raises(H7OracleInconclusive) as malformed_external:
        evaluate_h7_from_raw_bytes(
            h1_bytes,
            h7_bytes,
            _mutated_json_bytes(matrix_probe_bytes, ("records",), None),
        )
    assert type(malformed_external.value) is H7OracleInconclusive

    def programming_error(_value: object) -> None:
        raise RuntimeError("programming defect")

    monkeypatch.setattr(h7_oracle, "_validate_h1_fixture", programming_error)
    with pytest.raises(RuntimeError, match="programming defect"):
        evaluate_h7_from_raw_bytes(h1_bytes, h7_bytes, matrix_probe_bytes)


def test_h7_oracle_transitive_import_graph_is_independent() -> None:
    closure, imported_roots = _local_import_graph(ORACLE_SOURCE)
    relative_closure = {path.relative_to(ROOT).as_posix() for path in closure}
    assert "verification/mp_oracles/h7_covariance.py" in relative_closure
    assert (
        ORACLE_BUDGET_PROTOCOL_SOURCE.resolve().relative_to(ROOT).as_posix()
        in relative_closure
    )
    assert imported_roots.isdisjoint(
        {"vfe4", "torch", "numpy", "verification.h7_budget"}
    )
    assert all("verification/h7_budget.py" not in path for path in relative_closure)
    protocol_tree = ast.parse(ORACLE_BUDGET_PROTOCOL_SOURCE.read_text(encoding="utf-8"))
    protocol_import_roots = {
        alias.name.split(".", maxsplit=1)[0]
        for node in ast.walk(protocol_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    protocol_import_roots.update(
        node.module.split(".", maxsplit=1)[0]
        for node in ast.walk(protocol_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    assert protocol_import_roots == {
        "__future__",
        "dataclasses",
        "decimal",
        "hashlib",
        "json",
        "math",
        "typing",
    }

    source = ORACLE_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    gaussian_source = ast.unparse(functions["_gaussian_inventory_records"])
    gaussian_assignments = {
        target.id: node.value
        for node in ast.walk(functions["_gaussian_inventory_records"])
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    precision_expression = gaussian_assignments["precision"]
    assert isinstance(precision_expression, ast.Call)
    assert isinstance(precision_expression.func, ast.Attribute)
    assert ast.unparse(precision_expression.func) == "precision_source.consume"
    information_expression = gaussian_assignments["information"]
    assert isinstance(information_expression, ast.BinOp)
    assert isinstance(information_expression.op, ast.Mult)
    assert ast.unparse(information_expression) == "precision * mean"
    assert "precision_source.consume" in gaussian_source


def test_h7_real_218_transform_recovery_and_budget_path() -> None:
    with mp.workdps(100):
        actions = _injected_actions()
        original = _injected_inventory(218)
        transformed = h7_oracle._transform_inventory(
            original,
            actions,
            decoder_policy="transform",
        )
        recovered = h7_oracle._recover_inventory(
            transformed,
            actions,
            decoder_policy="transform",
        )
        aggregate = h7_oracle._build_bound_backward_records(
            original,
            transformed,
            recovered,
            actions,
            decoder_policy="transform",
        )

    expected_ids = tuple(item.record.operand_id for item in original)
    assert len(original) == len(transformed) == len(recovered) == 218
    assert tuple(item.record.operand_id for item in transformed) == expected_ids
    assert tuple(item.record.operand_id for item in recovered) == expected_ids
    assert tuple(item.operand_id for item in aggregate.records) == expected_ids
    assert len(aggregate.records) == len(aggregate.bound_budgets) == 218
    assert all(type(item) is MPBackwardResidualRecord for item in aggregate.records)
    assert aggregate.maximum == max(item.value for item in aggregate.records)
    assert len({item.backward_sha256 for item in aggregate.records}) == 218
    assert len({item.bound_sha256 for item in aggregate.bound_budgets}) == 218
    assert {item.category for item in original} == {
        "vector",
        "information",
        "offset",
        "decoder",
        "covariance",
        "precision",
        "second_moment",
        "map",
    }
    assert {item.transform_kind for item in original} == {
        "left",
        "covariance",
        "precision",
        "information",
        "receiver_source",
        "decoder",
    }
    for record, bound in zip(
        aggregate.records,
        aggregate.bound_budgets,
        strict=True,
    ):
        assert record.budget.budget_sha256 == bound.budget.budget_sha256
        assert bound.formula.forward_budget is not None
        assert bound.formula.inverse_action_budget is not None
        assert (
            bound.budget.contributions[0].operation_count
            == bound.formula.forward_budget.budget.contributions[0].operation_count
            + bound.formula.inverse_action_budget.budget.contributions[
                0
            ].operation_count
        )


def test_h7_task5_wiring_executes_closed_named_inventory_and_error_split() -> None:
    action_sha256 = _sha("injected-scalar-action")
    action = _InjectedAction(
        elements=tuple(
            _InjectedOwnedTensor(_sha(f"injected-action-element-{index}"))
            for index in range(3)
        ),
        action_sha256=action_sha256,
    )
    trial_spec = _InjectedTrialSpec(
        trial_id="scalar-base-transformed",
        fixture_id="h1-v1",
        frame_profile="h1_v1",
        decoder_policy="transform",
        action=action,
        action_sha256=action_sha256,
    )
    with mp.workdps(100):
        actions = _injected_actions()
        original_inventory = _injected_inventory(1)
        transformed_inventory = h7_oracle._transform_inventory(
            original_inventory,
            actions,
            decoder_policy="transform",
        )
        recovered_inventory = h7_oracle._recover_inventory(
            transformed_inventory,
            actions,
            decoder_policy="transform",
        )
        backward = h7_oracle._build_bound_backward_records(
            original_inventory,
            transformed_inventory,
            recovered_inventory,
            actions,
            decoder_policy="transform",
        )
    oracle_trial = MPTrialResult(
        trial_id=trial_spec.trial_id,
        fixture_id="h1-v1",
        frame_profile="h1_v1",
        decoder_policy="transform",
        action_sha256=action_sha256,
        recognition_families=("scalar_h1_v1",),
        source_paths=(),
        original=h7_oracle._inventory_values(original_inventory),
        transformed=h7_oracle._inventory_values(transformed_inventory),
        recovered=h7_oracle._inventory_values(recovered_inventory),
        backward_records=backward.records,
        backward_bound_budgets=backward.bound_budgets,
        backward_inventory_size=1,
        r_back_max=h7_oracle._decimal(mp.mpf(str(backward.maximum))),
        scalar_items=_zero_scalar_items(),
        status_items=(),
        scorer_rows=(),
        probe_evaluations=(),
    )
    scalar_evidence = _InjectedScalarEvidence(
        fixture_id="h1-v1",
        raw_fixture_sha256=h7_oracle._H1_RAW_SHA256,
        action_sha256=action_sha256,
        original_log_evidence=0.0,
        transformed_log_evidence=0.0,
        original_posterior_kl=0.0,
        transformed_posterior_kl=0.0,
    )
    local_terms = tuple(
        _InjectedLocal(
            term_id=term_id,
            original_value=0.0,
            transformed_value=0.0,
            residual=_InjectedResidual(term_id, 0.0),
        )
        for term_id in H7_COMPLETE_LOCAL_TERM_IDS
    )
    production = _InjectedTask5Evaluation(
        original_complete_local_value=0.0,
        transformed_complete_local_value=0.0,
        initial_joint_kl=_InjectedInitial(
            term_id="K0_joint_z0_m0",
            original_value=0.0,
            transformed_value=0.0,
            residual=_InjectedResidual("K0_joint_z0_m0", 0.0),
        ),
        local_terms=local_terms,
        complete_local=_InjectedResidual("complete_local_elbo", 0.0),
        complete_monolithic=_InjectedResidual(
            "complete_monolithic_elbo",
            0.0,
        ),
        p_density_shift=_InjectedResidual(
            "complete_pointwise_p_density_shift",
            0.0,
        ),
        q_density_shift=_InjectedResidual(
            "complete_pointwise_q_density_shift",
            0.0,
        ),
        log_ratio=_InjectedResidual("complete_pointwise_log_ratio", 0.0),
        entropy_shift=_InjectedResidual(
            "joint_recognition_entropy_shift",
            0.0,
        ),
        scalar_evidence=scalar_evidence,
        evidence=_InjectedResidual(
            "scalar_log_evidence_and_elbo_kl_identity",
            0.0,
        ),
        posterior_kl=_InjectedResidual(
            "scalar_posterior_kl_invariance",
            0.0,
        ),
        not_applicable_reason=None,
    )
    fixture = SimpleNamespace(
        fixture_id="h1-v1",
        raw_fixture_sha256=h7_oracle._H1_RAW_SHA256,
    )
    evaluator_calls: list[tuple[object, object, object]] = []

    def injected_evaluator(
        original: object,
        transformed: object,
        supplied_action: object,
        **_kwargs: object,
    ) -> _InjectedTask5Evaluation:
        evaluator_calls.append((original, transformed, supplied_action))
        return production

    def wire(
        evaluator: Callable[..., object],
        supplied_action: _InjectedAction = action,
    ):
        return evaluate_h7_task5_wiring(
            fixture,
            fixture,
            supplied_action,
            trial_spec=trial_spec,
            original_factor_trace=object(),
            transformed_factor_trace=object(),
            density_probe_pairs=(),
            quadrature_orders=(41, 51),
            budgets_by_invariant={},
            oracle_trial=oracle_trial,
            task5_evaluator=evaluator,
            scalar_evidence=scalar_evidence,
        )

    result = wire(injected_evaluator)
    comparison_ids = tuple(item.value_id for item in result.comparisons)
    assert evaluator_calls == [(fixture, fixture, action)]
    assert len(comparison_ids) == len(set(comparison_ids)) == 50
    assert {
        "complete_monolithic_elbo",
        "complete_pointwise_p_density_shift",
        "complete_pointwise_q_density_shift",
        "complete_pointwise_log_ratio",
        "residual.joint_recognition_entropy",
        "scalar_log_evidence",
        "transformed.scalar_log_evidence",
        "scalar_posterior_kl",
        "transformed.scalar_posterior_kl",
        "scalar_log_evidence_and_elbo_kl_identity",
        "scalar_posterior_kl_residual",
    }.issubset(comparison_ids)
    assert all(mp.mpf(item.absolute_delta) == 0 for item in result.comparisons)
    assert result.status_comparisons == ()

    wrong_action = replace(
        action,
        action_sha256=_sha("same-fixture-cross-trial-action"),
    )
    with pytest.raises(H7OracleInconclusive) as cross_trial:
        wire(injected_evaluator, wrong_action)
    assert type(cross_trial.value) is H7OracleInconclusive

    malformed = SimpleNamespace(
        **{
            **production.__dict__,
            "local_terms": production.local_terms[:-1],
            "__post_init__": lambda: None,
        }
    )

    def malformed_evaluator(
        *_args: object,
        **_kwargs: object,
    ) -> object:
        return malformed

    with pytest.raises(H7OracleInconclusive) as malformed_external:
        wire(malformed_evaluator)
    assert type(malformed_external.value) is H7OracleInconclusive

    def programming_error(
        *_args: object,
        **_kwargs: object,
    ) -> _InjectedTask5Evaluation:
        raise ValueError("plain Task-5 programming defect")

    with pytest.raises(ValueError, match="plain Task-5 programming defect") as programming:
        wire(programming_error)
    assert type(programming.value) is ValueError


def test_h7_source_freezes_probe_anchor_kl_and_task5_wiring() -> None:
    source = ORACLE_SOURCE.read_text(encoding="utf-8")
    task5_source = TASK5_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    posterior = functions["_scalar_posterior_kl"]
    posterior_calls = {
        node.func.id
        for node in ast.walk(posterior)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    scalar_validator = functions["_validate_scalar_probe_table"]
    scalar_validator_calls = {
        node.func.id
        for node in ast.walk(scalar_validator)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert H7_COMPLETE_LOCAL_TERM_IDS == TASK5_COMPLETE_LOCAL_TERM_IDS
    assert 'root["records"], 486' in source
    assert 'root["records"], 8' in source
    assert {"_gaussian_kl", "_expected_log_emission"}.issubset(posterior_calls)
    assert "_joint_moments" in scalar_validator_calls
    assert "_make_h1_law" in scalar_validator_calls
    assert "direct H1 generative conditional" in source
    assert "complete_local_elbo" not in ast.unparse(posterior)
    assert "log_evidence -" not in ast.unparse(posterior)
    assert "scalar_evidence_elbo_posterior_kl_residual" in source
    assert ".precision" in source
    assert ".J_precision" in source
    assert not any(
        (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.startswith("vfe4")
        )
        or (
            isinstance(node, ast.Import)
            and any(alias.name.startswith("vfe4") for alias in node.names)
        )
        for node in tree.body
    )
    assert "task5_evaluator(" in ast.unparse(functions["evaluate_h7_task5_wiring"])
    assert "_scalar_posterior_kl" not in task5_source


def test_h7_leaf_budget_formulas_use_exact_closed_shapes_and_scopes() -> None:
    cases = (
        ("vector", (2,), 128),
        ("information", (2,), 128),
        ("offset", (2,), 128),
        ("decoder", (3, 2), 128),
        ("covariance", (2, 2), 1408),
        ("precision", (2, 2), 1408),
        ("second_moment", (2, 2), 1408),
        ("map", (2, 2), 1408),
    )
    for category, shape, expected_count in cases:
        prefix = f"leaf.{category}"
        operands = _leaf_operands(
            prefix,
            category,
            shape,
            source_condition=11.0,
            action_condition=3.0,
        )
        formula = _leaf_formula(prefix, category)
        bound = build_h7_budget(
            invariant_id=prefix,
            category=category,
            operands=operands,
            formula=formula,
        )
        operation = bound.budget.contributions[0]
        expected_condition = (
            33.0 if category in {"covariance", "precision", "second_moment"} else 3.0
        )
        assert operation.operation_count == expected_count
        assert operation.value == pytest.approx(
            operation.unit_allowance * expected_condition * 7.0
        )
        assert bound.budget.comparison_normalization == 7.0
        assert formula.condition_operand_ids == (
            (
                f"{prefix}.action",
                f"{prefix}.source",
            )
            if expected_condition == 33.0
            else (f"{prefix}.action",)
        )
        assert formula.scale_operand_ids == (
            f"{prefix}.original",
            f"{prefix}.transformed",
            f"{prefix}.source",
        )
        assert len(bound.formula_sha256) == 64
        assert len(bound.bound_sha256) == 64

        irrelevant = _operand(
            f"{prefix}.irrelevant",
            category=category,
            role="reference",
            shape=(2, 2),
            scale=1e30,
            condition=1e20,
        )
        with pytest.raises(ValueError, match="outside the closed scope"):
            build_h7_budget(
                invariant_id=f"{prefix}.extra",
                category=category,
                operands=(*operands, irrelevant),
                formula=formula,
            )

    bad_shape_operands = (
        _operand(
            "bad.original",
            category="vector",
            role="original",
            shape=(2, 2),
            scale=1.0,
        ),
        _operand(
            "bad.transformed",
            category="vector",
            role="transformed",
            shape=(2, 2),
            scale=1.0,
        ),
        _operand(
            "bad.source",
            category="vector",
            role="reference",
            shape=(2, 2),
            scale=1.0,
        ),
        _operand(
            "bad.action",
            category="vector",
            role="reference",
            shape=(2, 2),
            scale=1.0,
        ),
    )
    with pytest.raises(ValueError, match="vector-like dimension"):
        build_h7_budget(
            invariant_id="bad.shape",
            category="vector",
            operands=bad_shape_operands,
            formula=_leaf_formula("bad", "vector"),
        )
    bad_action_operands = _leaf_operands("bad_action", "vector", (2,))
    with pytest.raises(ValueError, match="direct action"):
        build_h7_budget(
            invariant_id="bad.action",
            category="vector",
            operands=bad_action_operands,
            formula=replace(
                _leaf_formula("bad_action", "vector"),
                direct_action_operand_ids=("bad_action.source",),
                source_operand_ids=("bad_action.action",),
                dimension_operand_id="bad_action.action",
            ),
        )
    with pytest.raises(ValueError, match="semantic operand groups"):
        replace(
            _leaf_formula("duplicate", "covariance"),
            direct_action_operand_ids=("duplicate.source",),
        )


def test_h7_cocycle_density_and_real_composite_budgets_match_frozen_rows() -> None:
    cocycle_operands = tuple(
        _operand(
            f"cocycle.{name}",
            category="cocycle",
            role=role,
            shape=(2, 2),
            scale=scale,
            condition=condition,
        )
        for name, role, scale, condition in (
            ("link_21", "original", 2.0, 1.0),
            ("link_10", "original", 3.0, 1.0),
            ("composite", "transformed", 5.0, 1.0),
            ("expected_endpoint", "reference", 7.0, 1.0),
            ("action_2", "reference", 1.0, 2.0),
            ("action_1", "reference", 1.0, 3.0),
            ("action_0", "reference", 1.0, 5.0),
            ("frame_2", "reference", 1.0, 7.0),
            ("frame_1", "reference", 1.0, 11.0),
            ("frame_0", "reference", 1.0, 13.0),
        )
    )
    cocycle = build_h7_budget(
        invariant_id="cocycle.2<-1<-0",
        category="cocycle",
        operands=cocycle_operands,
        formula=_formula(
            category="cocycle",
            operation_kind="matrix_product",
            dimension_operand_id="cocycle.composite",
            compared_operand_ids=(
                "cocycle.composite",
                "cocycle.expected_endpoint",
            ),
            direct_action_operand_ids=(
                "cocycle.action_2",
                "cocycle.action_1",
                "cocycle.action_0",
            ),
            frame_operand_ids=(
                "cocycle.frame_2",
                "cocycle.frame_1",
                "cocycle.frame_0",
            ),
            link_operand_ids=("cocycle.link_21", "cocycle.link_10"),
        ),
    )
    assert cocycle.budget.contributions[0].operation_count == 1536
    assert cocycle.budget.comparison_normalization == 7.0

    density_operands = tuple(
        _operand(
            f"density.{name}",
            category="density",
            role=role,
            shape=shape,
            scale=scale,
            condition=condition,
        )
        for name, role, shape, scale, condition in (
            ("log_original", "original", (), 2.0, 1.0),
            ("log_shifted", "transformed", (), 3.0, 1.0),
            ("quadratic", "reference", (), 5.0, 1.0),
            ("logdet", "reference", (), 7.0, 1.0),
            ("covariance", "reference", (2, 2), 1.0, 11.0),
            ("action", "reference", (2, 2), 1.0, 13.0),
        )
    )
    density = build_h7_budget(
        invariant_id="density.probe",
        category="density",
        operands=density_operands,
        formula=_formula(
            category="density",
            operation_kind="analytic_density",
            dimension_operand_id="density.covariance",
            compared_operand_ids=(
                "density.log_original",
                "density.log_shifted",
            ),
            direct_action_operand_ids=("density.action",),
            spd_operand_ids=("density.covariance",),
            signed_summand_operand_ids=(
                "density.quadratic",
                "density.logdet",
            ),
        ),
    )
    assert density.budget.contributions[0].operation_count == 2560
    assert density.budget.comparison_normalization == 7.0

    child_vector = _real_leaf_budget("child.vector", "vector", (2,))
    child_matrix = _real_leaf_budget("child.matrix", "covariance", (2, 2))
    local_operands = tuple(
        _operand(
            f"local.{name}",
            category="local_term",
            role=role,
            shape=(2, 2) if name == "spd" else (),
            scale=scale,
            condition=condition,
        )
        for name, role, scale, condition in (
            ("original", "original", 2.0, 1.0),
            ("transformed", "transformed", 3.0, 1.0),
            ("summand_0", "reference", 5.0, 1.0),
            ("summand_1", "reference", 7.0, 1.0),
            ("spd", "reference", 1.0, 17.0),
        )
    )
    local_formula = _formula(
        category="local_term",
        operation_kind="pair_comparison",
        compared_operand_ids=("local.original", "local.transformed"),
        spd_operand_ids=("local.spd",),
        signed_summand_operand_ids=(
            "local.summand_0",
            "local.summand_1",
        ),
        child_budgets=(child_vector, child_matrix),
    )
    local = build_h7_budget(
        invariant_id="local.expected_log_emission",
        category="local_term",
        operands=local_operands,
        formula=local_formula,
    )
    assert local.budget.contributions[0].operation_count == 1664

    complete_operands = tuple(
        _operand(
            f"complete.{name}",
            category="complete_objective",
            role=role,
            shape=(2, 2) if name == "spd" else (),
            scale=scale,
            condition=condition,
        )
        for name, role, scale, condition in (
            ("original", "original", 2.0, 1.0),
            ("transformed", "transformed", 3.0, 1.0),
            ("term_0", "reference", 5.0, 1.0),
            ("term_1", "reference", 7.0, 1.0),
            ("spd", "reference", 1.0, 19.0),
        )
    )
    complete = build_h7_budget(
        invariant_id="complete.local",
        category="complete_objective",
        operands=complete_operands,
        formula=_formula(
            category="complete_objective",
            operation_kind="pair_comparison",
            compared_operand_ids=(
                "complete.original",
                "complete.transformed",
            ),
            spd_operand_ids=("complete.spd",),
            signed_summand_operand_ids=(
                "complete.term_0",
                "complete.term_1",
            ),
            child_budgets=(local, density),
        ),
    )
    assert complete.budget.contributions[0].operation_count == 4352

    raw_contribution = child_vector.budget.contributions[0]
    with pytest.raises(ValueError, match="formula-bound"):
        _formula(
            category="local_term",
            operation_kind="pair_comparison",
            compared_operand_ids=("local.original", "local.transformed"),
            signed_summand_operand_ids=("local.summand_0",),
            child_budgets=(raw_contribution,),
        )
    operation = child_vector.budget.contributions[0]
    tampered_operation = H7AllowanceContribution.create(
        kind=operation.kind,
        operation_id=operation.operation_id,
        operation_kind=operation.operation_kind,
        operation_count=operation.operation_count + 1,
        quadrature_order=operation.quadrature_order,
        unit_allowance=operation.unit_allowance,
        value=operation.value,
    )
    tampered_budget = H7BudgetRecord.create(
        invariant_id=child_vector.budget.invariant_id,
        category=child_vector.budget.category,
        operands=child_vector.budget.operands,
        contributions=(tampered_operation,),
        comparison_normalization=child_vector.budget.comparison_normalization,
        total_allowance=tampered_operation.value,
    )
    with pytest.raises(ValueError, match="not reproduced"):
        H7BoundBudget.create(tampered_budget, child_vector.formula)


def test_h7_quadrature_contribution_is_exact_2d_and_boundary_is_inclusive() -> None:
    child_vector = _real_leaf_budget("gh.child.vector", "vector", (2,))
    child_matrix = _real_leaf_budget(
        "gh.child.matrix",
        "covariance",
        (2, 2),
    )

    def build(value51: str) -> H7BoundBudget:
        operands = (
            _operand(
                "gh.original",
                category="local_term",
                role="original",
                shape=(),
                scale=2.0,
            ),
            _operand(
                "gh.transformed",
                category="local_term",
                role="transformed",
                shape=(),
                scale=3.0,
            ),
            _operand(
                "gh.summand_0",
                category="local_term",
                role="reference",
                shape=(),
                scale=5.0,
            ),
            _operand(
                "gh.summand_1",
                category="local_term",
                role="reference",
                shape=(),
                scale=7.0,
            ),
            _operand(
                "gh.spd",
                category="local_term",
                role="reference",
                shape=(2, 2),
                scale=1.0,
                condition=11.0,
            ),
            _operand(
                "gh.oracle.41",
                category="local_term",
                role="oracle",
                shape=(),
                scale=1.0,
                oracle_value="0.0",
            ),
            _operand(
                "gh.oracle.51",
                category="local_term",
                role="oracle",
                shape=(),
                scale=1.0,
                oracle_value=value51,
            ),
        )
        return build_h7_budget(
            invariant_id="gh.expected_log_emission",
            category="local_term",
            operands=operands,
            formula=_formula(
                category="local_term",
                operation_kind="gauss_hermite",
                compared_operand_ids=("gh.original", "gh.transformed"),
                spd_operand_ids=("gh.spd",),
                signed_summand_operand_ids=(
                    "gh.summand_0",
                    "gh.summand_1",
                ),
                child_budgets=(child_vector, child_matrix),
                quadrature_operand_ids=("gh.oracle.41", "gh.oracle.51"),
                reference_operand_id="gh.oracle.51",
            ),
        )

    converged = build("0.0000000000000000001")
    assert tuple(item.kind for item in converged.budget.contributions) == (
        "operation_rounding",
        "quadrature_convergence",
        "reference_rounding",
    )
    quadrature = converged.budget.contributions[1]
    assert quadrature.quadrature_order == 51
    assert quadrature.operation_count == 2601
    assert quadrature.unit_allowance == 2.0
    assert quadrature.value == pytest.approx(2e-19)
    assert build("0.000000000000000001").budget.contributions[1].value == (
        pytest.approx(2e-18)
    )
    with pytest.raises(ValueError, match="GH41/GH51 boundary"):
        build("0.0000000000000000010000000000000001")


def test_h7_backward_records_cover_full_inventory_and_bound_provenance() -> None:
    required_ids = _complete_synthetic_backward_ids("structured")
    assert len(required_ids) > 100
    assert len(set(required_ids)) == len(required_ids)
    assert sum(".Omega[" in item for item in required_ids) == 6
    assert any(item.endswith(".precision") for item in required_ids)
    assert any(item.endswith(".J_precision") for item in required_ids)
    assert all(
        f"structured.source_scorer.{bank}[{receiver_t}<-{source_j}]"
        f".{channel}_covector" in required_ids
        for bank in ("model", "state")
        for receiver_t, source_j in ((1, 0), (2, 1))
        for channel in ("z", "m")
    )
    forward = _real_leaf_budget(
        "backward.forward",
        "covariance",
        (2, 2),
    )
    inverse_action = _real_leaf_budget(
        "backward.inverse_action",
        "vector",
        (2,),
        operation_kind="direct_solve",
    )
    with pytest.raises(ValueError, match="closed operand groups"):
        _formula(
            category="backward",
            operation_kind="direct_solve",
            compared_operand_ids=(
                "same.original",
                "same.transformed",
                "same.recovered",
            ),
            direct_action_operand_ids=("same.action",),
            forward_budget=forward,
            inverse_action_budget=forward,
        )

    def backward_input(index: int, operand_id: str) -> H7BackwardOperandInput:
        prefix = f"backward.{index}"
        operands = (
            _operand(
                f"{prefix}.original",
                category="backward",
                role="original",
                shape=(2, 2),
                scale=2.0,
            ),
            _operand(
                f"{prefix}.transformed",
                category="backward",
                role="transformed",
                shape=(2, 2),
                scale=3.0,
            ),
            _operand(
                f"{prefix}.recovered",
                category="backward",
                role="recovered",
                shape=(2, 2),
                scale=5.0,
            ),
            _operand(
                f"{prefix}.action",
                category="backward",
                role="reference",
                shape=(2, 2),
                scale=1.0,
                condition=7.0,
            ),
        )
        return H7BackwardOperandInput(
            operand_id=operand_id,
            original_sha256=operands[0].value_sha256,
            transformed_sha256=operands[1].value_sha256,
            recovered_sha256=operands[2].value_sha256,
            numerator=float(index + 1) * 1e-15,
            normalization=2.0,
            operands=operands,
            formula=_formula(
                category="backward",
                operation_kind="direct_solve",
                compared_operand_ids=(
                    f"{prefix}.original",
                    f"{prefix}.transformed",
                    f"{prefix}.recovered",
                ),
                direct_action_operand_ids=(f"{prefix}.action",),
                forward_budget=forward,
                inverse_action_budget=inverse_action,
            ),
        )

    inputs = tuple(
        backward_input(index, operand_id)
        for index, operand_id in enumerate(required_ids)
    )
    aggregate = build_h7_backward_records(
        inputs,
        required_operand_ids=required_ids,
    )
    assert tuple(item.operand_id for item in aggregate.records) == required_ids
    assert aggregate.maximum == max(item.value for item in aggregate.records)
    assert all(
        item.budget.contributions[0].operation_count == 1536
        for item in aggregate.records
    )
    assert all(
        record.original_sha256 == input_record.original_sha256
        and record.transformed_sha256 == input_record.transformed_sha256
        and record.recovered_sha256 == input_record.recovered_sha256
        and record.numerator == input_record.numerator
        and record.normalization == input_record.normalization
        for record, input_record in zip(aggregate.records, inputs, strict=True)
    )
    assert all(
        record.passed == (record.value <= record.budget.total_allowance)
        for record in aggregate.records
    )
    assert len({item.bound_sha256 for item in aggregate.bound_budgets}) == len(
        required_ids
    )
    with pytest.raises(ValueError, match="missing, extra, or reordered"):
        build_h7_backward_records(
            inputs,
            required_operand_ids=tuple(reversed(required_ids)),
        )

    bound = aggregate.bound_budgets[0]
    assert tuple(inspect.signature(control_decisiveness_limit).parameters) == (
        "correct_budget",
    )
    limit = control_decisiveness_limit(bound)
    with pytest.raises(ValueError, match="boundary"):
        require_control_decisive(limit, bound)
    assert (
        require_control_decisive(
            limit + max(1e-30, limit * 1e-12),
            bound,
        )
        == limit
    )
    switched_formula = replace(
        bound.formula,
        compared_operand_ids=(
            "backward.0.original",
            "backward.0.transformed",
            "backward.0.action",
        ),
    )
    with pytest.raises(ValueError, match="identity is inconsistent"):
        H7BoundBudget(
            budget=bound.budget,
            formula=switched_formula,
            formula_sha256=bound.formula_sha256,
            bound_sha256=bound.bound_sha256,
        )
