from __future__ import annotations

import ast
import json
import math
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path

import pytest
import torch

from vfe4.generative import H1GenerativeModel
import vfe4.objective as public_objective
import vfe4.objective.h5_complete as complete_module
from vfe4.objective.h1_local import _evaluate_order as evaluate_h1_order
from vfe4.objective.h5_complete import (
    CacheDisposition,
    CompleteElboEvaluation,
    FactorCacheEntry,
    FactorCacheKey,
    FactorEvaluationRecord,
    FactorInputHashRecord,
    StaleFactorCacheError,
    evaluate_h5_complete_elbo,
)
from vfe4.recognition import H1RecognitionLaw
from vfe4.types.h5_schema import (
    H5_ANALYTIC_FACTOR_OPERATION_COUNTS,
    H5_C,
    H5_FACTOR_INPUT_DOMAIN,
    H5_FACTOR_INPUT_FIELDS,
    H5_FACTOR_INPUT_SCHEMA_SHA256,
    H5_FACTOR_UNIVERSE,
    H5_OBJECTIVE_SCHEMA_SHA256,
    H5_PARAMETER_DEPENDENCY_ROWS,
    H5_QUADRATURE_ORDERS,
    H5_SIGNED_TERM_IDS,
    H5_VARIABLE_DEPENDENCY_ROWS,
    emission_operation_count,
    gamma_n,
)
from vfe4.types.updates import (
    FrozenTensorValue,
    H5LiveState,
    H5ModelSnapshot,
    ModelParameterBlock,
    RecognitionSnapshot,
    h5_semantic_state_sha256,
    initial_live,
)
from vfe4.validation import load_h1_fixture
from vfe4.validation.h5_update_spec import build_h5_reference_state


ROOT = Path(__file__).parents[2]
H1_BYTES = (ROOT / "vfe4/validation/fixtures/h1_v1.json").read_bytes()
H5_BYTES = (
    ROOT / "vfe4/validation/fixtures/h5_conditional_update_v1.json"
).read_bytes()
COMPLEMENT = "ab" * 32


def _reference():
    return build_h5_reference_state(H1_BYTES, H5_BYTES)


def _signed_term_values(order_evaluation) -> tuple[float, ...]:
    return (
        order_evaluation.expected_log_emission[0].value,
        order_evaluation.expected_log_emission[1].value,
        order_evaluation.initial_model_kl.value,
        order_evaluation.initial_state_kl.value,
        order_evaluation.model_source_kl[0].value,
        order_evaluation.model_source_kl[1].value,
        order_evaluation.model_transition_kl[0].value,
        order_evaluation.model_transition_kl[1].value,
        order_evaluation.state_source_kl[0].value,
        order_evaluation.state_source_kl[1].value,
        order_evaluation.state_transition_kl[0].value,
        order_evaluation.state_transition_kl[1].value,
    )


def _cache(evaluation):
    return {
        FactorCacheKey(
            record.factor_id,
            record.input_hash,
            H5_QUADRATURE_ORDERS,
            record.frozen_complement_sha256,
        ): FactorCacheEntry(
            FactorCacheKey(
                record.factor_id,
                record.input_hash,
                H5_QUADRATURE_ORDERS,
                record.frozen_complement_sha256,
            ),
            record,
        )
        for record in evaluation.factor_records
    }


def _replace_gaussian_value(
    recognition: RecognitionSnapshot,
    coordinate_id: str,
    field_name: str,
    value: float,
) -> RecognitionSnapshot:
    rows = tuple(
        replace(
            coordinate,
            **{field_name: FrozenTensorValue("float64", (), (value,))},
        )
        if coordinate.coordinate_id == coordinate_id
        else coordinate
        for coordinate in recognition.gaussians
    )
    return replace(recognition, gaussians=rows)


def _replace_categorical(
    recognition: RecognitionSnapshot,
    coordinate_id: str,
    probabilities: tuple[float, ...],
) -> RecognitionSnapshot:
    rows = tuple(
        replace(
            coordinate,
            probabilities=FrozenTensorValue(
                "float64", (len(probabilities),), probabilities
            ),
        )
        if coordinate.coordinate_id == coordinate_id
        else coordinate
        for coordinate in recognition.categoricals
    )
    return replace(recognition, categoricals=rows)


def _replace_model_value(
    model: H5ModelSnapshot,
    block_id: str,
    field_name: str,
    values: tuple[float, ...],
) -> H5ModelSnapshot:
    blocks: list[ModelParameterBlock] = []
    for block in model.parameter_blocks:
        if block.block_id != block_id:
            blocks.append(block)
            continue
        changed = tuple(
            (
                name,
                FrozenTensorValue(value.dtype, value.shape, values)
                if name == field_name
                else value,
            )
            for name, value in block.values
        )
        blocks.append(replace(block, values=changed))
    return replace(model, parameter_blocks=tuple(blocks))


def _hash_differences(before, after) -> tuple[str, ...]:
    return tuple(
        factor_id
        for factor_id, left, right in zip(
            H5_FACTOR_UNIVERSE,
            before.factor_records,
            after.factor_records,
            strict=True,
        )
        if left.input_hash.input_sha256 != right.input_hash.input_sha256
    )


def _entropy(probabilities: tuple[float, ...]) -> float:
    return math.fsum(-value * math.log(value) for value in probabilities)


def _independent_raw_factor_values(reference, order: int) -> tuple[float, ...]:
    h1_model = H1GenerativeModel.from_fixture(load_h1_fixture())
    h1_recognition = H1RecognitionLaw(
        reference.specification.as_h1_recognition_record()
    )
    evaluated = evaluate_h1_order(h1_model, h1_recognition, order)
    gaussians = {
        coordinate.coordinate_id: (
            coordinate.mean.values[0],
            coordinate.variance.values[0],
        )
        for coordinate in reference.initial_recognition.gaussians
    }
    categoricals = {
        coordinate.coordinate_id: tuple(coordinate.probabilities.values)
        for coordinate in reference.initial_recognition.categoricals
    }
    gaussian_entropy = {
        coordinate_id: 0.5 * math.log(2.0 * math.pi * math.e * variance)
        for coordinate_id, (_, variance) in gaussians.items()
    }
    gamma_entropy = {
        time: _entropy(categoricals[f"q[model_source_b{time}]"])
        for time in (1, 2)
    }
    conditional_entropy = {
        1: _entropy(categoricals["q[state_source_a1_b0]"]),
        2: math.fsum(
            categoricals["q[model_source_b2]"][b] * _entropy(row)
            for b, row in enumerate(
                (
                    categoricals["q[source_row_a2]"],
                    categoricals["q[state_source_a2_b1]"],
                )
            )
        ),
    }
    recognition_entropy = math.fsum(
        (
            *gaussian_entropy.values(),
            gamma_entropy[1],
            conditional_entropy[1],
            gamma_entropy[2],
            conditional_entropy[2],
        )
    )
    return (
        -gaussian_entropy["q[z0]"]
        - gaussian_entropy["q[m0]"]
        - evaluated.initial_model_kl.value
        - evaluated.initial_state_kl.value,
        -gamma_entropy[1] - evaluated.model_source_kl[0].value,
        -gaussian_entropy["q[m1]"] - evaluated.model_transition_kl[0].value,
        -conditional_entropy[1] - evaluated.state_source_kl[0].value,
        -gaussian_entropy["q[z1]"] - evaluated.state_transition_kl[0].value,
        evaluated.expected_log_emission[0].value,
        -gamma_entropy[2] - evaluated.model_source_kl[1].value,
        -gaussian_entropy["q[m2]"] - evaluated.model_transition_kl[1].value,
        -conditional_entropy[2] - evaluated.state_source_kl[1].value,
        -gaussian_entropy["q[z2]"] - evaluated.state_transition_kl[1].value,
        evaluated.expected_log_emission[1].value,
        recognition_entropy,
    )


def _factor_comparison_allowance(record, order: int) -> float:
    if order == 21:
        value = record.value_order_21
        summands = record.absolute_summands_order_21
        conditions = record.condition_numbers_order_21
        operation_count = record.operation_count_order_21
    else:
        value = record.value_order_17
        summands = record.absolute_summands_order_17
        conditions = record.condition_numbers_order_17
        operation_count = record.operation_count_order_17
    one_path = (
        H5_C
        * gamma_n(operation_count)
        * max(1.0, *conditions)
        * max(1.0, abs(value), math.fsum(summands))
    )
    return 2.0 * one_path


def test_h5_complete_objective_has_one_scalar_and_both_order_factor_trace() -> None:
    reference = _reference()
    live = initial_live(reference)
    evaluation = evaluate_h5_complete_elbo(
        reference,
        live,
        frozen_complement_sha256=COMPLEMENT,
    )
    assert tuple(record.factor_id for record in evaluation.factor_records) == (
        H5_FACTOR_UNIVERSE
    )
    assert tuple(item.term_id for item in evaluation.term_allowances) == (
        H5_SIGNED_TERM_IDS
    )
    assert tuple(item.term_id for item in evaluation.diagnostic_allowances) == (
        "joint_recognition_entropy",
    )
    assert evaluation.objective_schema_sha256 == H5_OBJECTIVE_SCHEMA_SHA256
    assert evaluation.evaluated_state_sha256 == h5_semantic_state_sha256(
        live.recognition, live.model
    )
    assert evaluation.complete_allowance.stochastic_contribution == 0.0
    assert all(
        record.cache_disposition is CacheDisposition.REEVALUATED
        for record in evaluation.factor_records
    )
    assert math.fsum(
        record.value_order_21 for record in evaluation.factor_records
    ) == pytest.approx(
        evaluation.terms.complete_elbo,
        abs=evaluation.complete_allowance.total,
    )

    h1_model = H1GenerativeModel.from_fixture(load_h1_fixture())
    h1_recognition = H1RecognitionLaw(
        reference.specification.as_h1_recognition_record()
    )
    expected_21 = evaluate_h1_order(h1_model, h1_recognition, 21)
    expected_17 = evaluate_h1_order(h1_model, h1_recognition, 17)
    assert tuple(item.value_order_21 for item in evaluation.term_allowances) == (
        pytest.approx(_signed_term_values(expected_21), rel=0.0, abs=2e-14)
    )
    assert tuple(item.value_order_17 for item in evaluation.term_allowances) == (
        pytest.approx(_signed_term_values(expected_17), rel=0.0, abs=2e-14)
    )
    assert evaluation.terms.joint_recognition_entropy == pytest.approx(
        expected_21.joint_recognition_entropy.value, rel=0.0, abs=2e-14
    )
    assert evaluation.terms.complete_elbo == pytest.approx(
        expected_21.complete_elbo.value, rel=0.0, abs=2e-14
    )


@pytest.mark.parametrize("order", (21, 17))
def test_all_twelve_raw_factor_values_match_an_independent_reconstruction(
    order: int,
) -> None:
    reference = _reference()
    evaluation = evaluate_h5_complete_elbo(
        reference,
        initial_live(reference),
        frozen_complement_sha256=COMPLEMENT,
    )
    expected = _independent_raw_factor_values(reference, order)
    assert len(expected) == len(H5_FACTOR_UNIVERSE) == 12
    for record, independent in zip(evaluation.factor_records, expected, strict=True):
        actual = record.value_order_21 if order == 21 else record.value_order_17
        assert actual == pytest.approx(
            independent,
            rel=0.0,
            abs=_factor_comparison_allowance(record, order),
        )


def test_initial_analytic_terms_use_unit_condition_while_raw_joint_uses_p_condition() -> None:
    reference = _reference()
    evaluation = evaluate_h5_complete_elbo(
        reference,
        initial_live(reference),
        frozen_complement_sha256=COMPLEMENT,
    )
    allowances = {item.term_id: item for item in evaluation.term_allowances}
    for term_id in ("initial_model_kl", "initial_state_kl"):
        assert allowances[term_id].condition_numbers_order_21 == (1.0,)
        assert allowances[term_id].condition_numbers_order_17 == (1.0,)
    raw_initial = evaluation.factor_records[0]
    initial = json.loads(H1_BYTES)["initial_joint"]
    expected_condition = float(
        torch.linalg.cond(
            torch.tensor(initial["covariance"], dtype=torch.float64)
        ).item()
    )
    assert raw_initial.condition_numbers_order_21 == (expected_condition,)
    assert raw_initial.condition_numbers_order_17 == (expected_condition,)


@pytest.mark.parametrize("order", (21, 17))
def test_raw_trace_rejects_perturbation_above_its_reduction_rounding_only(
    order: int,
) -> None:
    reference = _reference()
    live = initial_live(reference)
    evaluation = evaluate_h5_complete_elbo(
        reference, live, frozen_complement_sha256=COMPLEMENT
    )
    signed_17 = tuple(
        item.objective_sign * item.value_order_17
        for item in evaluation.term_allowances
    )
    reduction = (
        evaluation.complete_allowance.reduction_rounding
        if order == 21
        else H5_C
        * gamma_n(13)
        * max(1.0, math.fsum(abs(value) for value in signed_17))
    )
    assert reduction < evaluation.complete_allowance.total
    perturbation = 0.5 * (reduction + evaluation.complete_allowance.total)
    records = list(evaluation.factor_records)
    emission = records[5]
    records[5] = replace(
        emission,
        **{
            f"value_order_{order}": (
                getattr(emission, f"value_order_{order}") + perturbation
            )
        },
    )
    with pytest.raises(ValueError, match=f"raw order-{order} factor trace"):
        CompleteElboEvaluation.build(
            state=live,
            terms=evaluation.terms,
            factor_records=tuple(records),
            term_allowances=evaluation.term_allowances,
            diagnostic_allowances=evaluation.diagnostic_allowances,
            complete_allowance=evaluation.complete_allowance,
            frozen_complement_sha256=COMPLEMENT,
        )


def test_factor_inputs_are_closed_ordered_and_operand_records_are_complete() -> None:
    evaluation = evaluate_h5_complete_elbo(
        _reference(),
        initial_live(_reference()),
        frozen_complement_sha256=COMPLEMENT,
    )
    for record in evaluation.factor_records:
        input_hash = record.input_hash
        assert input_hash.input_schema_sha256 == H5_FACTOR_INPUT_SCHEMA_SHA256
        assert input_hash.canonical_input_bytes.startswith(H5_FACTOR_INPUT_DOMAIN)
        decoded = json.loads(
            input_hash.canonical_input_bytes[len(H5_FACTOR_INPUT_DOMAIN) :]
        )
        assert set(decoded) == set(H5_FACTOR_INPUT_FIELDS)
        assert decoded["schema_version"] == "h5-factor-input-v1"
        assert decoded["factor_id"] == record.factor_id
        assert record.frozen_complement_sha256 == COMPLEMENT
        assert record.absolute_summands_order_21
        assert record.absolute_summands_order_17
        assert math.fsum(record.absolute_summands_order_21) >= abs(
            record.value_order_21
        ) - 2e-14
        assert math.fsum(record.absolute_summands_order_17) >= abs(
            record.value_order_17
        ) - 2e-14
        assert all(value >= 1.0 for value in record.condition_numbers_order_21)
        assert all(value >= 1.0 for value in record.condition_numbers_order_17)
        if record.factor_id.startswith("emission"):
            assert record.operation_count_order_21 == emission_operation_count(21)
            assert record.operation_count_order_17 == emission_operation_count(17)
        else:
            assert (
                record.operation_count_order_21
                == H5_ANALYTIC_FACTOR_OPERATION_COUNTS[record.factor_id]
            )
            assert record.operation_count_order_17 == record.operation_count_order_21


@pytest.mark.parametrize(
    ("coordinate_id", "probabilities"),
    (
        ("q[model_source_b2]", (0.45, 0.55)),
        ("q[source_row_a2]", (0.7, 0.3)),
        ("q[state_source_a2_b1]", (0.25, 0.75)),
    ),
)
def test_categorical_input_hash_changes_equal_the_dependency_row(
    coordinate_id: str, probabilities: tuple[float, ...]
) -> None:
    reference = _reference()
    live = initial_live(reference)
    before = evaluate_h5_complete_elbo(
        reference, live, frozen_complement_sha256=COMPLEMENT
    )
    changed = replace(
        live,
        recognition=_replace_categorical(
            live.recognition, coordinate_id, probabilities
        ),
    )
    after = evaluate_h5_complete_elbo(
        reference, changed, frozen_complement_sha256=COMPLEMENT
    )
    assert _hash_differences(before, after) == dict(H5_VARIABLE_DEPENDENCY_ROWS)[
        coordinate_id
    ]


@pytest.mark.parametrize(
    ("coordinate_id", "field_name"),
    tuple(
        (f"q[{name}{time}]", field_name)
        for time in range(3)
        for name in ("z", "m")
        for field_name in ("mean", "variance")
    ),
)
def test_every_gaussian_field_hash_change_equals_its_dependency_row_and_reuses_unaffected(
    coordinate_id: str, field_name: str,
) -> None:
    reference = _reference()
    live = initial_live(reference)
    before = evaluate_h5_complete_elbo(
        reference, live, frozen_complement_sha256=COMPLEMENT
    )
    current = next(
        getattr(coordinate, field_name).values[0]
        for coordinate in live.recognition.gaussians
        if coordinate.coordinate_id == coordinate_id
    )
    changed = replace(
        live,
        recognition=_replace_gaussian_value(
            live.recognition, coordinate_id, field_name, current + 0.03125
        ),
    )
    after = evaluate_h5_complete_elbo(
        reference,
        changed,
        frozen_complement_sha256=COMPLEMENT,
        cache=_cache(before),
    )
    expected = dict(H5_VARIABLE_DEPENDENCY_ROWS)[
        coordinate_id
    ]
    assert _hash_differences(before, after) == expected
    assert tuple(
        record.cache_disposition for record in after.factor_records
    ) == tuple(
        CacheDisposition.REEVALUATED
        if factor_id in expected
        else CacheDisposition.REUSED
        for factor_id in H5_FACTOR_UNIVERSE
    )


@pytest.mark.parametrize(
    ("block_id", "field_name"),
    (
        ("theta[state_transition_2]", "alpha_0"),
        ("theta[state_transition_2]", "alpha_1"),
        ("theta[state_transition_2]", "B_base"),
        ("theta[state_transition_2]", "c"),
        ("theta[state_transition_2]", "R"),
        ("theta[emission_1]", "w_z"),
        ("theta[emission_1]", "w_m"),
        ("theta[emission_1]", "bias"),
        ("theta[shared_decoder_transition]", "s"),
    ),
)
def test_every_mutable_model_field_hash_change_equals_its_dependency_row_and_reuses_unaffected(
    block_id: str, field_name: str
) -> None:
    reference = _reference()
    live = initial_live(reference)
    before = evaluate_h5_complete_elbo(
        reference, live, frozen_complement_sha256=COMPLEMENT
    )
    block = next(item for item in live.model.parameter_blocks if item.block_id == block_id)
    frozen = dict(block.values)[field_name]
    changed_values = (frozen.values[0] + 0.03125,) + frozen.values[1:]
    changed = replace(
        live,
        model=_replace_model_value(
            live.model, block_id, field_name, changed_values
        ),
    )
    after = evaluate_h5_complete_elbo(
        reference,
        changed,
        frozen_complement_sha256=COMPLEMENT,
        cache=_cache(before),
    )
    graph_row = dict(H5_PARAMETER_DEPENDENCY_ROWS)[block_id]
    expected = tuple(
        factor_id for factor_id in H5_FACTOR_UNIVERSE if factor_id in graph_row
    )
    assert _hash_differences(before, after) == expected
    assert tuple(
        record.cache_disposition for record in after.factor_records
    ) == tuple(
        CacheDisposition.REEVALUATED
        if factor_id in expected
        else CacheDisposition.REUSED
        for factor_id in H5_FACTOR_UNIVERSE
    )


def test_shared_parameter_has_one_identity_and_hashes_all_three_consumers() -> None:
    reference = _reference()
    live = initial_live(reference)
    before = evaluate_h5_complete_elbo(
        reference, live, frozen_complement_sha256=COMPLEMENT
    )
    changed_model = _replace_model_value(
        live.model,
        "theta[shared_decoder_transition]",
        "s",
        (0.125,),
    )
    changed = replace(live, model=changed_model)
    after = evaluate_h5_complete_elbo(
        reference, changed, frozen_complement_sha256=COMPLEMENT
    )
    expected = dict(H5_PARAMETER_DEPENDENCY_ROWS)[
        "theta[shared_decoder_transition]"
    ]
    assert _hash_differences(before, after) == tuple(
        factor_id for factor_id in H5_FACTOR_UNIVERSE if factor_id in expected
    )
    decoded = {
        record.factor_id: json.loads(
            record.input_hash.canonical_input_bytes[len(H5_FACTOR_INPUT_DOMAIN) :]
        )["normalized_factor"]
        for record in after.factor_records
        if record.factor_id in expected
    }
    identities = tuple(
        decoded[factor_id]["shared_parameter"] for factor_id in expected
    )
    assert identities == (
        {
            "source": "theta[shared_decoder_transition].s",
            "value": float(0.125).hex(),
        },
    ) * 3
    assert float.fromhex(decoded["state_transition[2]"]["B"]) == pytest.approx(
        -0.35 + 0.125
    )
    assert float.fromhex(decoded["emission[1]"]["w_z"][0]) == pytest.approx(
        0.2 + 0.125
    )
    assert float.fromhex(decoded["emission[2]"]["w_z"][0]) == pytest.approx(
        -0.1 + 0.125
    )


def test_semantic_hash_is_provenance_not_an_objective_acceptance_substitute() -> None:
    reference = _reference()
    plus = initial_live(reference)
    minus = replace(
        plus,
        model=_replace_model_value(
            plus.model,
            "theta[shared_decoder_transition]",
            "s",
            (-0.0,),
        ),
    )
    plus_eval = evaluate_h5_complete_elbo(
        reference, plus, frozen_complement_sha256=COMPLEMENT
    )
    minus_eval = evaluate_h5_complete_elbo(
        reference, minus, frozen_complement_sha256=COMPLEMENT
    )
    assert plus_eval.evaluated_state_sha256 != minus_eval.evaluated_state_sha256
    assert _hash_differences(plus_eval, minus_eval) == (
        "emission[1]",
        "state_transition[2]",
        "emission[2]",
    )
    assert plus_eval.terms.complete_elbo.hex() == minus_eval.terms.complete_elbo.hex()
    assert tuple(
        (record.value_order_21.hex(), record.value_order_17.hex())
        for record in plus_eval.factor_records
    ) == tuple(
        (record.value_order_21.hex(), record.value_order_17.hex())
        for record in minus_eval.factor_records
    )


def test_exact_cache_hit_partial_key_miss_and_stale_entry_are_distinct() -> None:
    reference = _reference()
    live = initial_live(reference)
    first = evaluate_h5_complete_elbo(
        reference, live, frozen_complement_sha256=COMPLEMENT
    )
    cache = _cache(first)
    reused = evaluate_h5_complete_elbo(
        reference,
        live,
        frozen_complement_sha256=COMPLEMENT,
        cache=cache,
    )
    assert all(
        record.cache_disposition is CacheDisposition.REUSED
        for record in reused.factor_records
    )
    assert tuple(
        (record.value_order_21.hex(), record.value_order_17.hex())
        for record in reused.factor_records
    ) == tuple(
        (record.value_order_21.hex(), record.value_order_17.hex())
        for record in first.factor_records
    )

    missed = evaluate_h5_complete_elbo(
        reference,
        live,
        frozen_complement_sha256="cd" * 32,
        cache=cache,
    )
    assert all(
        record.cache_disposition is CacheDisposition.REEVALUATED
        for record in missed.factor_records
    )

    key = next(iter(cache))
    stale_record = replace(
        cache[key].record,
        frozen_complement_sha256="ef" * 32,
    )
    stale = dict(cache)
    stale[key] = FactorCacheEntry(key, stale_record)
    with pytest.raises(StaleFactorCacheError):
        evaluate_h5_complete_elbo(
            reference,
            live,
            frozen_complement_sha256=COMPLEMENT,
            cache=stale,
        )

    present_none = dict(cache)
    present_none[key] = None  # type: ignore[assignment]
    with pytest.raises(StaleFactorCacheError):
        evaluate_h5_complete_elbo(
            reference,
            live,
            frozen_complement_sha256=COMPLEMENT,
            cache=present_none,  # type: ignore[arg-type]
        )


def test_complete_objective_has_no_verification_dependency() -> None:
    source = (ROOT / "vfe4/objective/h5_complete.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = tuple(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ) + tuple(
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    assert not any("verification" in module for module in imported)


def test_complete_objective_public_fields_are_exact_frozen_owned_and_exported() -> None:
    assert tuple(field.name for field in fields(FactorInputHashRecord)) == (
        "factor_id",
        "input_schema_version",
        "input_schema_sha256",
        "canonical_input_bytes",
        "input_sha256",
    )
    assert tuple(field.name for field in fields(FactorEvaluationRecord)) == (
        "factor_id",
        "input_hash",
        "frozen_complement_sha256",
        "value_order_21",
        "value_order_17",
        "absolute_summands_order_21",
        "absolute_summands_order_17",
        "condition_numbers_order_21",
        "condition_numbers_order_17",
        "operation_count_order_21",
        "operation_count_order_17",
        "cache_disposition",
    )
    assert tuple(field.name for field in fields(FactorCacheKey)) == (
        "factor_id",
        "input_hash",
        "quadrature_orders",
        "frozen_complement_sha256",
    )
    assert tuple(field.name for field in fields(FactorCacheEntry)) == (
        "key",
        "record",
    )
    assert tuple(field.name for field in fields(CompleteElboEvaluation)) == (
        "terms",
        "factor_records",
        "term_allowances",
        "diagnostic_allowances",
        "complete_allowance",
        "objective_schema_sha256",
        "evaluated_state_sha256",
        "frozen_complement_sha256",
    )
    for record_type in (
        FactorInputHashRecord,
        FactorEvaluationRecord,
        FactorCacheKey,
        FactorCacheEntry,
        CompleteElboEvaluation,
    ):
        assert record_type.__dataclass_params__.frozen is True

    reference = _reference()
    evaluation = evaluate_h5_complete_elbo(
        reference,
        initial_live(reference),
        frozen_complement_sha256=COMPLEMENT,
    )
    with pytest.raises(FrozenInstanceError):
        evaluation.factor_records[0].factor_id = "changed"  # type: ignore[misc]
    assert evaluation.factor_records is not list(evaluation.factor_records)
    assert evaluation.factor_records[0].input_hash.canonical_input_bytes == bytes(
        evaluation.factor_records[0].input_hash.canonical_input_bytes
    )

    expected_exports = (
        "CacheDisposition",
        "CompleteElboEvaluation",
        "CompleteElboEvaluator",
        "FactorCacheEntry",
        "FactorCacheKey",
        "FactorEvaluationRecord",
        "FactorInputHashRecord",
        "StaleFactorCacheError",
        "evaluate_h5_complete_elbo",
    )
    assert complete_module.__all__ == list(expected_exports)
    for name in expected_exports:
        assert name in public_objective.__all__
        assert getattr(public_objective, name) is getattr(complete_module, name)
