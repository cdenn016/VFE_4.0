"""Fail-closed H2 information--moment promotion gate."""

from __future__ import annotations

import hashlib
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping
from unittest.mock import patch

import numpy as np
import torch

from verification.h2_budget import (
    backward_residual_allowance,
    complete_elbo_allowance,
    infinity_norm,
    pair_allowance,
    path_allowance,
)
from verification.numpy_oracles.h2_moment import (
    H2MomentOracleEvaluation,
    evaluate_h2_moment_oracle,
)
from vfe4.config import ResolvedConfig
from vfe4.generative import H1GenerativeModel, assemble_generative_information
from vfe4.numerics.information import InformationGaussian
from vfe4.numerics.precision import DenseCholeskyPrecision
from vfe4.objective import (
    H2InformationEvaluation,
    evaluate_information_elbo,
    evaluate_local_elbo,
    evaluate_monolithic_elbo,
)
from vfe4.recognition import H1RecognitionLaw, assemble_recognition_information
from vfe4.types import GateResult, GateStatus, InvariantResult, MatrixBlock, SourcePath
from vfe4.types.h1 import GaussianLaw
from vfe4.validation import load_h1_fixture


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "vfe4" / "validation" / "fixtures" / "h1_v1.json"
EXPECTED_H1_FIXTURE_SHA256 = (
    "388e38cc8c16d8b5e2c61919c1e712a134d88fb0bbd8ec1f2939b9859c9a583b"
)
CONTROL_DECISIVENESS = 1.0e-3
_DIMENSION = 6
_ALL_INDICES = tuple(range(_DIMENSION))
_PATHS = tuple(
    SourcePath((0, state_source), (0, model_source))
    for model_source in range(2)
    for state_source in range(2)
)
H2_NEGATIVE_CONTROL_NAMES = (
    "misread_h_as_mu",
    "reversed_log_determinant_ratio",
    "diagonal_inverse_emission_marginal",
    "forbidden_inverse_path",
)


@dataclass(frozen=True)
class H2Comparison:
    left: object
    right: object
    residual: float
    allowance: float
    passed: bool


@dataclass(frozen=True)
class H2ControlResidual:
    label: str
    correct_value: object
    wrong_value: object
    residual: float
    scale: float
    decisiveness_limit: float
    margin: float
    passed: bool


@dataclass(frozen=True)
class H2NegativeControl:
    passed: bool
    detected: bool
    residual: float
    decisiveness_limit: float
    correct_value: object
    wrong_value: object
    forbidden_attempts: int = 0
    injected_attempts: int = 0
    solve_rhs_widths: tuple[int, ...] = ()
    selected_column_sets: tuple[tuple[int, ...], ...] = ()
    weakest_margin: float = 0.0
    residual_records: tuple[H2ControlResidual, ...] = ()


@dataclass(frozen=True)
class H2MomentEvaluation:
    q_components: tuple[GaussianLaw, GaussianLaw, GaussianLaw, GaussianLaw]
    p_components: tuple[GaussianLaw, GaussianLaw, GaussianLaw, GaussianLaw]
    monolithic: object
    local_terms: object


@dataclass(frozen=True)
class H2GateEvaluation:
    result: GateResult
    fixture_observed_sha256: str | None
    information: H2InformationEvaluation | None
    moment: H2MomentEvaluation | None
    oracle: H2MomentOracleEvaluation | None
    comparisons: Mapping[str, H2Comparison]
    negative_controls: Mapping[str, H2NegativeControl]


def _block_inventory(path: SourcePath) -> tuple[tuple[int, ...], ...]:
    result: list[tuple[int, ...]] = [(0, 1)]
    for time in (1, 2):
        a, b = path.a[time - 1], path.b[time - 1]
        result.extend(((2 * b + 1, 2 * time + 1), (2 * a, 2 * time + 1, 2 * time)))
    result.extend(((2, 3), (4, 5)))
    return tuple(result)


def _comparison_names() -> tuple[str, ...]:
    names: list[str] = []
    for component_index, path in enumerate(_PATHS):
        for law in ("q", "p"):
            prefix = f"component.{component_index}.{law}"
            names.extend(
                (
                    f"{prefix}.mean.info_vs_h1",
                    f"{prefix}.mean.info_vs_numpy",
                    f"{prefix}.backward.mean",
                )
            )
            for indices in _block_inventory(path):
                label = "-".join(str(index) for index in indices)
                names.extend(
                    (
                        f"{prefix}.covariance[{label}].info_vs_h1",
                        f"{prefix}.covariance[{label}].info_vs_numpy",
                        f"{prefix}.backward.covariance[{label}]",
                    )
                )
        names.extend(
            (
                f"component.{component_index}.q_log_normalizer.info_vs_numpy",
                f"component.{component_index}.p_log_normalizer.info_vs_numpy",
                f"component.{component_index}.q_entropy.info_vs_numpy",
                f"component.{component_index}.gaussian_kl.info_vs_h1",
                f"component.{component_index}.gaussian_kl.info_vs_numpy",
                f"component.{component_index}.gaussian_log_ratio.info_vs_h1",
                f"component.{component_index}.gaussian_log_ratio.info_vs_numpy",
                f"component.{component_index}.source_log_ratio.info_vs_h1",
                f"component.{component_index}.source_log_ratio.info_vs_numpy",
                f"component.{component_index}.emission[0].info_vs_h1",
                f"component.{component_index}.emission[0].info_vs_numpy",
                f"component.{component_index}.emission[1].info_vs_h1",
                f"component.{component_index}.emission[1].info_vs_numpy",
                f"component.{component_index}.complete_value.info_vs_h1",
                f"component.{component_index}.complete_value.info_vs_numpy",
            )
        )
    for name in _LOCAL_NAMES:
        names.extend(
            (
                f"aggregate.local.{name}.info_vs_h1",
                f"aggregate.local.{name}.info_vs_numpy",
            )
        )
    names.extend(
        (
            "aggregate.complete_elbo.info_vs_h1_monolithic",
            "aggregate.complete_elbo.info_vs_h1_local",
            "aggregate.complete_elbo.info_vs_numpy_component",
            "aggregate.complete_elbo.info_local_vs_numpy_local",
        )
    )
    return tuple(names)


_LOCAL_NAMES = (
    "expected_log_emission[0]",
    "expected_log_emission[1]",
    "initial_model_kl",
    "initial_state_kl",
    "model_source_kl[0]",
    "model_transition_kl[0]",
    "model_source_kl[1]",
    "model_transition_kl[1]",
    "state_source_kl[0]",
    "state_transition_kl[0]",
    "state_source_kl[1]",
    "state_transition_kl[1]",
    "joint_recognition_entropy",
)
_COMPARISON_NAMES = _comparison_names()
_CONDITION_NAMES = tuple(
    f"condition.component.{index}.{law}" for index in range(4) for law in ("q", "p")
)
H2_INVARIANT_NAMES = (
    "fixture.sha256",
    *_COMPARISON_NAMES,
    *_CONDITION_NAMES,
    *(f"negative.{name}" for name in H2_NEGATIVE_CONTROL_NAMES),
)


@dataclass
class _InverseAudit:
    solve_rhs_widths: list[int]
    selected_column_sets: list[tuple[int, ...]]
    factors: list[DenseCholeskyPrecision]
    forbidden_attempts: int = 0
    injected_attempts: int = 0
    injecting: bool = False


def evaluate_h2(
    config: ResolvedConfig, *, fixture_bytes: bytes | None = None
) -> H2GateEvaluation:
    """Evaluate H2 without publishing any run artifact."""

    observed: str | None = None
    temporary: tempfile.TemporaryDirectory[str] | None = None
    try:
        _validate_config(config)
        if fixture_bytes is None:
            captured_bytes = FIXTURE_PATH.read_bytes()
            fixture_path = FIXTURE_PATH
        elif type(fixture_bytes) is bytes:
            captured_bytes = fixture_bytes
            temporary = tempfile.TemporaryDirectory(prefix="vfe4-h2-fixture-")
            fixture_path = Path(temporary.name) / "h1_v1.json"
            fixture_path.write_bytes(captured_bytes)
        else:
            raise _Inconclusive("fixture bytes must be immutable bytes")
        observed = hashlib.sha256(captured_bytes).hexdigest()
        if observed != EXPECTED_H1_FIXTURE_SHA256:
            raise _Inconclusive("fixture/hash mismatch")
        fixture = load_h1_fixture(fixture_path)
        model = H1GenerativeModel.from_fixture(fixture)
        recognition = H1RecognitionLaw.from_fixture(fixture)
        audit = _InverseAudit([], [], [])
        with _instrument_inverse_paths(audit):
            with torch.no_grad():
                information = evaluate_information_elbo(
                    model,
                    recognition,
                    quadrature_order=config.validation.quadrature_order,
                )
                q_infos = tuple(
                    assemble_recognition_information(recognition.factors, path)
                    for path in _PATHS
                )
                p_infos = tuple(
                    assemble_generative_information(model.factors, path) for path in _PATHS
                )
        inverse_control = _inject_forbidden_identity_solve(audit)
        if audit.forbidden_attempts != 0 or not inverse_control.detected:
            raise _Inconclusive("inverse-path instrumentation was unavailable or observed a real forbidden attempt")

        moment = H2MomentEvaluation(
            q_components=tuple(recognition.joint_component(path) for path in _PATHS),  # type: ignore[arg-type]
            p_components=tuple(model.joint_component(path) for path in _PATHS),  # type: ignore[arg-type]
            monolithic=evaluate_monolithic_elbo(
                model,
                recognition,
                quadrature_order=config.validation.quadrature_order,
                convergence_check_order=config.validation.convergence_check_order,
            ),
            local_terms=evaluate_local_elbo(
                model,
                recognition,
                quadrature_order=config.validation.quadrature_order,
                convergence_check_order=config.validation.convergence_check_order,
            ),
        )
        oracle = evaluate_h2_moment_oracle(
            fixture_path, quadrature_order=config.validation.quadrature_order
        )
        comparisons = _comparisons(information, moment, oracle, q_infos, p_infos)
        if tuple(comparisons) != _COMPARISON_NAMES:
            raise ValueError("H2 comparison inventory mismatch")
        controls = _negative_controls(q_infos, p_infos, oracle, inverse_control)
        if tuple(controls) != H2_NEGATIVE_CONTROL_NAMES:
            raise ValueError("H2 negative-control inventory mismatch")
        conditions = _conditions(information)

        invariants: list[InvariantResult] = [
            InvariantResult(
                "fixture.sha256", True, 0.0, 0.0, "observed SHA-256 equals preregistered SHA-256"
            )
        ]
        invariants.extend(
            InvariantResult(
                name,
                comparison.passed,
                comparison.residual,
                comparison.allowance,
                "absolute residual <= invariant-specific allowance",
            )
            for name, comparison in comparisons.items()
        )
        invariants.extend(conditions.values())
        invariants.extend(
            InvariantResult(
                f"negative.{name}",
                control.passed,
                control.residual,
                control.decisiveness_limit,
                "wrong path must be decisive or injected forbidden path must be detected",
            )
            for name, control in controls.items()
        )
        if tuple(item.name for item in invariants) != H2_INVARIANT_NAMES:
            raise ValueError("H2 invariant inventory mismatch")

        status, obligations = _status_and_obligations(tuple(invariants), controls)
        largest = max(comparisons.values(), key=lambda item: item.residual)
        result = GateResult(
            gate="H2",
            status=status,
            fixture_id="h1-v1",
            residual=None if status is GateStatus.INCONCLUSIVE else largest.residual,
            calibrated_allowance=(
                None if status is GateStatus.INCONCLUSIVE else largest.allowance
            ),
            measurements=(
                {"information_elbo": None, "h1_moment_elbo": None, "numpy_elbo": None}
                if status is GateStatus.INCONCLUSIVE
                else {
                    "information_elbo": information.complete_elbo,
                    "h1_moment_elbo": float(moment.monolithic.value),
                    "numpy_elbo": oracle.complete_elbo,
                }
            ),
            invariants=tuple(invariants),
            obligations=obligations,
        )
        return H2GateEvaluation(
            result,
            observed,
            information,
            moment,
            oracle,
            MappingProxyType(comparisons),
            MappingProxyType(controls),
        )
    except _Inconclusive as error:
        return _inconclusive(observed, str(error))
    except Exception as error:
        return _inconclusive(observed, f"H2 computation requires resolution: {error}")
    finally:
        if temporary is not None:
            temporary.cleanup()


class _Inconclusive(RuntimeError):
    pass


def _inconclusive(observed: str | None, reason: str) -> H2GateEvaluation:
    result = GateResult(
        gate="H2",
        status=GateStatus.INCONCLUSIVE,
        fixture_id="h1-v1",
        residual=None,
        calibrated_allowance=None,
        measurements={"information_elbo": None, "h1_moment_elbo": None, "numpy_elbo": None},
        invariants=(),
        obligations=(reason,),
    )
    return H2GateEvaluation(
        result,
        observed,
        None,
        None,
        None,
        MappingProxyType({}),
        MappingProxyType({}),
    )


def _instrument_inverse_paths(audit: _InverseAudit):
    original_solve = DenseCholeskyPrecision.solve
    original_selected = DenseCholeskyPrecision.selected_inverse

    def solve(self: DenseCholeskyPrecision, rhs: torch.Tensor):
        width = 1 if rhs.ndim == 1 else int(rhs.shape[1])
        audit.solve_rhs_widths.append(width)
        if self not in audit.factors:
            audit.factors.append(self)
        if width == _DIMENSION:
            if audit.injecting:
                audit.injected_attempts += 1
            else:
                audit.forbidden_attempts += 1
            raise ValueError("forbidden full-identity precision solve")
        return original_solve(self, rhs)

    def selected(self: DenseCholeskyPrecision, blocks: object):
        columns: list[int] = []
        for block in blocks:  # type: ignore[union-attr]
            for column in block.columns:
                if column not in columns:
                    columns.append(column)
        audit.selected_column_sets.append(tuple(columns))
        if len(columns) == _DIMENSION:
            audit.forbidden_attempts += 1
            raise ValueError("forbidden all-column selected inverse")
        return original_selected(self, blocks)  # type: ignore[arg-type]

    def forbidden(*args: object, **kwargs: object):
        audit.forbidden_attempts += 1
        raise ValueError("forbidden explicit inverse path")

    return _PatchGroup(
        patch.object(DenseCholeskyPrecision, "solve", solve),
        patch.object(DenseCholeskyPrecision, "selected_inverse", selected),
        patch.object(torch.linalg, "inv", forbidden),
        patch.object(torch.linalg, "pinv", forbidden),
        patch.object(torch, "cholesky_inverse", forbidden),
    )


class _PatchGroup:
    def __init__(self, *patchers: object) -> None:
        self.patchers = patchers

    def __enter__(self) -> None:
        for patcher in self.patchers:
            patcher.start()  # type: ignore[attr-defined]

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        for patcher in reversed(self.patchers):
            patcher.stop()  # type: ignore[attr-defined]


def _inject_forbidden_identity_solve(audit: _InverseAudit) -> H2NegativeControl:
    if not audit.factors:
        raise _Inconclusive("inverse-path instrumentation captured no production factor")
    original = DenseCholeskyPrecision.solve
    audit.injecting = True
    detected = False
    try:
        def injected(self: DenseCholeskyPrecision, rhs: torch.Tensor):
            width = 1 if rhs.ndim == 1 else int(rhs.shape[1])
            if width == _DIMENSION:
                audit.injected_attempts += 1
                raise ValueError("forbidden full-identity precision solve")
            return original(self, rhs)

        with patch.object(DenseCholeskyPrecision, "solve", injected):
            try:
                audit.factors[0].solve(torch.eye(_DIMENSION, dtype=torch.float64))
            except ValueError:
                detected = True
    finally:
        audit.injecting = False
    return H2NegativeControl(
        passed=detected and audit.forbidden_attempts == 0 and audit.injected_attempts == 1,
        detected=detected,
        residual=float(audit.injected_attempts),
        decisiveness_limit=1.0,
        correct_value=0.0,
        wrong_value=float(audit.injected_attempts),
        forbidden_attempts=audit.forbidden_attempts,
        injected_attempts=audit.injected_attempts,
        solve_rhs_widths=tuple(audit.solve_rhs_widths),
        selected_column_sets=tuple(audit.selected_column_sets),
    )


def _comparisons(
    information: H2InformationEvaluation,
    moment: H2MomentEvaluation,
    oracle: H2MomentOracleEvaluation,
    q_infos: tuple[InformationGaussian, ...],
    p_infos: tuple[InformationGaussian, ...],
) -> dict[str, H2Comparison]:
    result: dict[str, H2Comparison] = {}

    def add(name: str, left: object, right: object, left_allowance: float, right_allowance: float) -> None:
        residual = infinity_norm(np.asarray(left) - np.asarray(right))
        allowance = pair_allowance(
            _DIMENSION, left_allowance, right_allowance, left, right
        )
        result[name] = H2Comparison(left, right, residual, allowance, residual <= allowance)

    for index, path in enumerate(_PATHS):
        for law_name, info_law, h1_law, np_law in (
            ("q", q_infos[index], moment.q_components[index], oracle.components[index].q),
            ("p", p_infos[index], moment.p_components[index], oracle.components[index].p),
        ):
            prefix = f"component.{index}.{law_name}"
            mean = info_law.mean().numpy()
            h1_mean = h1_law.mean.numpy()
            kappa = info_law.factor.diagnostics.kappa_2
            info_allow = _simple_allowance(mean, kappa)
            add(f"{prefix}.mean.info_vs_h1", mean, h1_mean, info_allow, _simple_allowance(h1_mean, kappa))
            add(f"{prefix}.mean.info_vs_numpy", mean, np_law.mean, info_allow, _oracle_allowance(np_law, "mean"))
            mean_residual = infinity_norm(info_law.J.numpy() @ mean - info_law.h.numpy())
            mean_limit = backward_residual_allowance(
                _DIMENSION, info_law.J.numpy(), mean, info_law.h.numpy()
            )
            result[f"{prefix}.backward.mean"] = H2Comparison(
                info_law.J.numpy() @ mean,
                info_law.h.numpy(),
                mean_residual,
                mean_limit,
                mean_residual <= mean_limit,
            )
            for indices in _block_inventory(path):
                label = "-".join(str(item) for item in indices)
                principal = MatrixBlock(indices, indices)
                second = info_law.selected_moment_blocks((principal,))[principal].numpy()
                selected_mean = mean[list(indices)]
                covariance = second - np.outer(selected_mean, selected_mean)
                h1_covariance = h1_law.covariance.numpy()[np.ix_(indices, indices)]
                np_covariance = np_law.covariance[np.ix_(indices, indices)]
                local_allow = _simple_allowance(covariance, kappa)
                add(
                    f"{prefix}.covariance[{label}].info_vs_h1",
                    covariance,
                    h1_covariance,
                    local_allow,
                    _simple_allowance(h1_covariance, kappa),
                )
                add(
                    f"{prefix}.covariance[{label}].info_vs_numpy",
                    covariance,
                    np_covariance,
                    local_allow,
                    _oracle_allowance(np_law, "covariance"),
                )
                columns = MatrixBlock(_ALL_INDICES, indices)
                selected_covariance = info_law.factor.selected_inverse((columns,))[columns].numpy()
                selector = np.zeros((_DIMENSION, len(indices)), dtype=np.float64)
                selector[list(indices), np.arange(len(indices))] = 1.0
                product = info_law.J.numpy() @ selected_covariance
                residual = infinity_norm(product - selector)
                allowance = backward_residual_allowance(
                    _DIMENSION, info_law.J.numpy(), selected_covariance, selector
                )
                result[f"{prefix}.backward.covariance[{label}]"] = H2Comparison(
                    product, selector, residual, allowance, residual <= allowance
                )

        info_component = information.components[index]
        h1 = moment.monolithic
        np_component = oracle.components[index]
        for suffix, left, right, key, oracle_law, oracle_key in (
            ("q_log_normalizer.info_vs_numpy", info_component.q_log_normalizer, np_component.q.log_normalizer, "q_log_normalizer", np_component.q, "log_normalizer"),
            ("p_log_normalizer.info_vs_numpy", info_component.p_log_normalizer, np_component.p.log_normalizer, "p_log_normalizer", np_component.p, "log_normalizer"),
            ("q_entropy.info_vs_numpy", info_component.q_entropy, np_component.q.entropy, "q_entropy", np_component.q, "entropy"),
        ):
            add(
                f"component.{index}.{suffix}", left, right,
                _rounding_allowance(info_component.rounding_inputs[key]),
                _oracle_allowance(oracle_law, oracle_key),
            )
        pairs = (
            ("gaussian_kl.info_vs_h1", info_component.gaussian_kl, -h1.component_gaussian_log_ratios[index], "gaussian_kl"),
            ("gaussian_kl.info_vs_numpy", info_component.gaussian_kl, np_component.gaussian_kl, "gaussian_kl"),
            ("gaussian_log_ratio.info_vs_h1", info_component.gaussian_log_ratio, h1.component_gaussian_log_ratios[index], "gaussian_log_ratio"),
            ("gaussian_log_ratio.info_vs_numpy", info_component.gaussian_log_ratio, np_component.gaussian_log_ratio, "gaussian_log_ratio"),
            ("source_log_ratio.info_vs_h1", info_component.source_log_ratio, h1.component_source_log_ratios[index], "source_log_ratio"),
            ("source_log_ratio.info_vs_numpy", info_component.source_log_ratio, np_component.source_log_ratio, "source_log_ratio"),
            ("emission[0].info_vs_h1", info_component.expected_log_emission[0], h1.component_emission_values[index][0], "expected_log_emission[0]"),
            ("emission[0].info_vs_numpy", info_component.expected_log_emission[0], np_component.expected_log_emission[0], "expected_log_emission[0]"),
            ("emission[1].info_vs_h1", info_component.expected_log_emission[1], h1.component_emission_values[index][1], "expected_log_emission[1]"),
            ("emission[1].info_vs_numpy", info_component.expected_log_emission[1], np_component.expected_log_emission[1], "expected_log_emission[1]"),
            ("complete_value.info_vs_h1", info_component.complete_value, h1.component_values[index], "complete_value"),
            ("complete_value.info_vs_numpy", info_component.complete_value, np_component.complete_value, "complete_value"),
        )
        for suffix, left, right, key in pairs:
            right_allowance = (
                _oracle_component_allowance(np_component, key, right)
                if suffix.endswith("info_vs_numpy")
                else _simple_allowance(
                    right, max(np_component.q.kappa_2, np_component.p.kappa_2)
                )
            )
            add(
                f"component.{index}.{suffix}", left, right,
                _rounding_allowance(info_component.rounding_inputs[key]),
                right_allowance,
            )

    info_local = _local_values(information.local_terms)
    h1_local = _local_values(moment.local_terms)
    np_local = _local_values(oracle.local_terms)
    info_allowances = _local_allowances_info(information)
    h1_allowances = _local_allowances_simple(h1_local)
    np_allowances = _local_allowances_oracle(oracle)
    for name in _LOCAL_NAMES:
        add(
            f"aggregate.local.{name}.info_vs_h1",
            info_local[name], h1_local[name], info_allowances[name], h1_allowances[name]
        )
        add(
            f"aggregate.local.{name}.info_vs_numpy",
            info_local[name], np_local[name], info_allowances[name], np_allowances[name]
        )
    add(
        "aggregate.complete_elbo.info_vs_h1_monolithic",
        information.complete_elbo, moment.monolithic.value,
        info_allowances["complete_elbo"], _simple_allowance(moment.monolithic.value, 1.0)
    )
    add(
        "aggregate.complete_elbo.info_vs_h1_local",
        information.complete_elbo, moment.local_terms.complete_elbo,
        info_allowances["complete_elbo"], h1_allowances["complete_elbo"]
    )
    add(
        "aggregate.complete_elbo.info_vs_numpy_component",
        information.complete_elbo, oracle.complete_elbo,
        info_allowances["complete_elbo"], np_allowances["complete_elbo"]
    )
    add(
        "aggregate.complete_elbo.info_local_vs_numpy_local",
        information.local_terms.complete_elbo, oracle.local_terms.complete_elbo,
        info_allowances["complete_elbo"], np_allowances["complete_elbo"]
    )
    return result


def _simple_allowance(value: object, kappa: float) -> float:
    norm = infinity_norm(value)
    return path_allowance(_DIMENSION, (kappa,), norm, norm)


def _oracle_allowance(law: object, key: str) -> float:
    value = getattr(law, key if key in ("mean", "covariance", "precision", "h") else key)
    return path_allowance(
        _DIMENSION,
        (law.kappa_2,),
        value,
        law.absolute_summand_accumulation[key],
    )


def _oracle_component_allowance(component: object, key: str, value: object) -> float:
    return path_allowance(
        _DIMENSION,
        (component.q.kappa_2, component.p.kappa_2),
        value,
        component.absolute_summand_accumulation[key],
    )


def _rounding_allowance(rounding: object) -> float:
    return path_allowance(
        _DIMENSION,
        (rounding.spd_kappa2,),
        rounding.output_inf_norm,
        rounding.absolute_summand_accumulation_inf,
    )


def _local_values(record: object) -> dict[str, float]:
    return {
        "expected_log_emission[0]": float(record.expected_log_emission[0]),
        "expected_log_emission[1]": float(record.expected_log_emission[1]),
        "initial_model_kl": float(record.initial_model_kl),
        "initial_state_kl": float(record.initial_state_kl),
        "model_source_kl[0]": float(record.model_source_kl[0]),
        "model_transition_kl[0]": float(record.model_transition_kl[0]),
        "model_source_kl[1]": float(record.model_source_kl[1]),
        "model_transition_kl[1]": float(record.model_transition_kl[1]),
        "state_source_kl[0]": float(record.state_source_kl[0]),
        "state_transition_kl[0]": float(record.state_transition_kl[0]),
        "state_source_kl[1]": float(record.state_source_kl[1]),
        "state_transition_kl[1]": float(record.state_transition_kl[1]),
        "joint_recognition_entropy": float(record.joint_recognition_entropy),
        "complete_elbo": float(record.complete_elbo),
    }


def _local_allowances_info(information: H2InformationEvaluation) -> dict[str, float]:
    result = {
        name: _rounding_allowance(information.rounding_inputs[f"local.{name}"])
        for name in _LOCAL_NAMES
    }
    signed_names = _LOCAL_NAMES[:12]
    signed_values = tuple(_local_values(information.local_terms)[name] * (1.0 if name.startswith("expected") else -1.0) for name in signed_names)
    result["complete_elbo"] = complete_elbo_allowance(
        signed_values, tuple(result[name] for name in signed_names)
    )
    return result


def _local_allowances_simple(values: Mapping[str, float]) -> dict[str, float]:
    result = {name: _simple_allowance(values[name], 1.0) for name in _LOCAL_NAMES}
    signed_names = _LOCAL_NAMES[:12]
    result["complete_elbo"] = complete_elbo_allowance(
        tuple(values[name] * (1.0 if name.startswith("expected") else -1.0) for name in signed_names),
        tuple(result[name] for name in signed_names),
    )
    return result


def _local_allowances_oracle(oracle: H2MomentOracleEvaluation) -> dict[str, float]:
    values = _local_values(oracle.local_terms)
    result = {
        name: path_allowance(
            _DIMENSION,
            oracle.local_terms.spd_operand_kappas[name],
            values[name],
            oracle.local_terms.absolute_summand_accumulation[name],
        )
        for name in _LOCAL_NAMES
    }
    result["complete_elbo"] = complete_elbo_allowance(
        oracle.signed_local_terms,
        tuple(result[name] for name in _LOCAL_NAMES[:12]),
    )
    return result


def _conditions(information: H2InformationEvaluation) -> dict[str, InvariantResult]:
    records: dict[str, InvariantResult] = {}
    for index, diagnostics in enumerate(information.component_diagnostics):
        for law, precision, mean_inf in (
            ("q", diagnostics.q_precision, diagnostics.q_mean_inf_norm),
            ("p", diagnostics.p_precision, diagnostics.p_mean_inf_norm),
        ):
            passed = (
                precision.dimension <= 6
                and precision.min_cholesky_pivot > 0.0
                and precision.lambda_min >= 1.0e-4
                and precision.lambda_max <= 1.0e4
                and precision.kappa_2 <= 1.0e6
                and mean_inf <= 4.0
            )
            value = max(
                precision.lambda_max / 1.0e4,
                precision.kappa_2 / 1.0e6,
                mean_inf / 4.0,
                1.0e-4 / precision.lambda_min,
            )
            records[f"condition.component.{index}.{law}"] = InvariantResult(
                f"condition.component.{index}.{law}",
                passed,
                value,
                1.0,
                (
                    f"pivot={precision.min_cholesky_pivot}; lambda_min={precision.lambda_min}; "
                    f"lambda_max={precision.lambda_max}; kappa_2={precision.kappa_2}; mean_inf={mean_inf}"
                ),
            )
    return records


def _negative_controls(
    q_infos: tuple[InformationGaussian, ...],
    p_infos: tuple[InformationGaussian, ...],
    oracle: H2MomentOracleEvaluation,
    inverse: H2NegativeControl,
) -> dict[str, H2NegativeControl]:
    misread_values: list[tuple[str, object, object]] = []
    determinant_values: list[tuple[str, object, object]] = []
    marginal_values: list[tuple[str, object, object]] = []
    for index, (q_info, p_info, component) in enumerate(zip(q_infos, p_infos, oracle.components)):
        for law_name, info_law in (("q", q_info), ("p", p_info)):
            correct_mean = info_law.mean().numpy()
            wrong_mean = info_law.factor.solve(info_law.mean()).numpy()
            misread_values.append(
                (f"component.{index}.{law_name}", correct_mean, wrong_mean)
            )

        q, p = component.q, component.p
        delta = p.mean - q.mean
        trace = float(np.trace(p.precision @ q.covariance))
        quadratic = float(delta @ p.precision @ delta)
        q_logdet = float(np.linalg.slogdet(q.precision)[1])
        p_logdet = float(np.linalg.slogdet(p.precision)[1])
        wrong = 0.5 * math.fsum((trace, quadratic, -_DIMENSION, p_logdet, -q_logdet))
        correct = component.gaussian_kl
        determinant_values.append((f"component.{index}", correct, wrong))

        for marginal in component.emission_marginals:
            indices = list(marginal.indices)
            wrong_covariance = np.diag(1.0 / np.diag(q_info.J.numpy())[indices])
            marginal_values.append(
                (
                    f"component.{index}.emission[{len(marginal_values) % 2}]",
                    marginal.covariance,
                    wrong_covariance,
                )
            )
    controls = {
        "misread_h_as_mu": _decisive_control(misread_values),
        "reversed_log_determinant_ratio": _decisive_control(determinant_values),
        "diagonal_inverse_emission_marginal": _decisive_control(marginal_values),
        "forbidden_inverse_path": inverse,
    }
    return controls


def _decisive_control(
    values: tuple[tuple[str, object, object], ...]
    | list[tuple[str, object, object]],
) -> H2NegativeControl:
    if not math.isfinite(CONTROL_DECISIVENESS) or CONTROL_DECISIVENESS < 0.0:
        raise _Inconclusive("negative control decisiveness threshold is indecisive")
    if not values:
        raise _Inconclusive("negative control has no affected residuals")
    records: list[H2ControlResidual] = []
    for label, correct, wrong in values:
        if type(label) is not str or not label:
            raise _Inconclusive("negative control residual label is unavailable")
        correct_array = np.asarray(correct, dtype=np.float64)
        wrong_array = np.asarray(wrong, dtype=np.float64)
        if correct_array.shape != wrong_array.shape:
            raise _Inconclusive("negative control operands have different shapes")
        residual = infinity_norm(wrong_array - correct_array)
        scale = max(1.0, infinity_norm(correct_array), infinity_norm(wrong_array))
        limit = CONTROL_DECISIVENESS * scale
        margin = residual - limit
        passed = (
            math.isfinite(residual)
            and math.isfinite(limit)
            and math.isfinite(margin)
            and residual >= limit
        )
        records.append(
            H2ControlResidual(
                label=label,
                correct_value=_control_value(correct_array),
                wrong_value=_control_value(wrong_array),
                residual=residual,
                scale=scale,
                decisiveness_limit=limit,
                margin=margin,
                passed=passed,
            )
        )
    weakest = min(records, key=lambda record: record.margin)
    detected = all(record.passed for record in records)
    return H2NegativeControl(
        passed=detected,
        detected=detected,
        residual=weakest.residual,
        decisiveness_limit=weakest.decisiveness_limit,
        correct_value=weakest.correct_value,
        wrong_value=weakest.wrong_value,
        weakest_margin=weakest.margin,
        residual_records=tuple(records),
    )


def _control_value(value: np.ndarray) -> object:
    if value.ndim == 0:
        return float(value)
    return tuple(_control_value(item) for item in value)


def _status_and_obligations(
    invariants: tuple[InvariantResult, ...],
    controls: Mapping[str, H2NegativeControl],
) -> tuple[GateStatus, tuple[str, ...]]:
    finite_failures = tuple(
        invariant.name
        for invariant in invariants
        if (
            not invariant.name.startswith("negative.")
            and not invariant.passed
            and invariant.value is not None
            and invariant.limit is not None
        )
    )
    if finite_failures:
        return GateStatus.FAIL, ()
    unavailable = tuple(
        invariant.name
        for invariant in invariants
        if not invariant.passed and not invariant.name.startswith("negative.")
    )
    indecisive = tuple(name for name, control in controls.items() if not control.passed)
    obligations = tuple(
        [f"invariant {name} lacks closure evidence" for name in unavailable]
        + [f"negative control {name} is indecisive" for name in indecisive]
    )
    if obligations:
        return GateStatus.INCONCLUSIVE, obligations
    return GateStatus.PASS, ()


def _validate_config(config: object) -> None:
    if not isinstance(config, ResolvedConfig):
        raise _Inconclusive("config must be a ResolvedConfig")
    if (
        config.validation.fixture_id != "h1-v1"
        or config.validation.quadrature_order != 21
        or config.run.dtype != "float64"
        or config.run.device != "cpu"
    ):
        raise _Inconclusive("config does not match the frozen H2 profile")


def h2_validation_payload(evaluation: H2GateEvaluation) -> dict[str, object]:
    """Return the complete evaluate-only H2 payload; publication belongs to Task 5."""

    if not isinstance(evaluation, H2GateEvaluation):
        raise ValueError("evaluation must be an H2GateEvaluation")
    payload = {
        "gate_result": evaluation.result,
        "fixture_observed_sha256": evaluation.fixture_observed_sha256,
        "information": evaluation.information,
        "moment": evaluation.moment,
        "oracle": evaluation.oracle,
        "comparisons": evaluation.comparisons,
        "negative_controls": evaluation.negative_controls,
    }
    return payload


__all__ = [
    "CONTROL_DECISIVENESS",
    "EXPECTED_H1_FIXTURE_SHA256",
    "H2Comparison",
    "H2ControlResidual",
    "H2GateEvaluation",
    "H2NegativeControl",
    "H2MomentEvaluation",
    "H2_INVARIANT_NAMES",
    "H2_NEGATIVE_CONTROL_NAMES",
    "evaluate_h2",
    "h2_validation_payload",
]
