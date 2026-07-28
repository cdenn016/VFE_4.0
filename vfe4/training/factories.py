"""Exact WikiText-103 arm constructors and corpus-free A0 matching."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Literal

import torch

from vfe4.types.results import GateStatus
from vfe4.types.training import (
    A0ArchitectureProfile,
    A0FormulaRecord,
    EndpointInventory,
    WT103_A0_HIDDEN_WIDTH_CANDIDATES,
    WT103ArmSpec,
    WT103ExperimentProfile,
    default_wt103_arm_specs,
    owned_sha256,
)

from .formulas import (
    A0FlopLedger,
    A0FlopWorkload,
    reconstruct_a0_flops,
    reconstruct_a0_parameters,
)
from .wt103_models import (
    BuiltWT103Arm,
    ExecutionScope,
    OptimizerParameterBinding,
    WT103A0Model,
    WT103ArmBuildRecord,
    WT103ArmRuntimeComponents,
)


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive exact int")
    return value


def _sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _validate_profile(profile: object) -> WT103ExperimentProfile:
    if type(profile) is not WT103ExperimentProfile:
        raise ValueError("profile must be an exact WT103ExperimentProfile")
    profile.__post_init__()
    return profile


def _validate_endpoint_inventory(value: object) -> EndpointInventory:
    if type(value) is not EndpointInventory:
        raise ValueError("endpoint_inventory must be exact")
    value.__post_init__()
    return value


def _exact_spec(
    value: object,
    *,
    permitted_indices: tuple[int, ...],
    constructor_id: str,
) -> WT103ArmSpec:
    if type(value) is not WT103ArmSpec:
        raise ValueError("spec must be an exact WT103ArmSpec")
    value.__post_init__()
    exact = default_wt103_arm_specs()
    permitted = tuple(exact[index] for index in permitted_indices)
    if value.arm_spec_sha256 not in tuple(
        item.arm_spec_sha256 for item in permitted
    ):
        raise ValueError(
            f"{constructor_id} cannot construct arm {value.arm_id}"
        )
    return value


@dataclass(frozen=True, slots=True)
class A0MatchRow:
    """One deterministic, corpus-free A0 width comparison."""

    hidden_width: int
    parameter_count: int
    semantic_train_flops: int
    parameter_ratio: float
    flop_ratio: float
    parameter_relative_error: float
    flop_relative_error: float
    optimizer_access_exact: bool
    filler_parameter_count: Literal[0]
    eligible: bool
    selection_key: tuple[float, float, int]
    formula_sha256: str

    def __post_init__(self) -> None:
        if self.hidden_width not in WT103_A0_HIDDEN_WIDTH_CANDIDATES:
            raise ValueError("matching row width is not a frozen candidate")
        _positive_int(self.parameter_count, "parameter_count")
        _positive_int(self.semantic_train_flops, "semantic_train_flops")
        for name in (
            "parameter_ratio",
            "flop_ratio",
            "parameter_relative_error",
            "flop_relative_error",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite exact float")
        if (
            type(self.optimizer_access_exact) is not bool
            or type(self.eligible) is not bool
            or type(self.filler_parameter_count) is not int
            or self.filler_parameter_count != 0
        ):
            raise ValueError("matching eligibility fields are invalid")
        expected_eligible = (
            self.parameter_relative_error <= 0.01
            and self.flop_relative_error <= 0.05
            and self.optimizer_access_exact
        )
        if self.eligible is not expected_eligible:
            raise ValueError("matching row eligibility is not deterministic")
        expected_key = (
            abs(math.log(self.parameter_ratio)),
            abs(math.log(self.flop_ratio)),
            self.hidden_width,
        )
        if self.selection_key != expected_key:
            raise ValueError("matching row selection key changed")
        _sha256(self.formula_sha256, "formula_sha256")


@dataclass(frozen=True, slots=True)
class ArmMatchingReport:
    """Closed finite A0 search against the unique PRIMARY endpoint."""

    schema_version: Literal["wt103-a0-arm-matching-v1"]
    primary_arm_spec_sha256: str
    endpoint_inventory_sha256: str
    primary_parameter_count: int
    primary_semantic_train_flops: int
    parameter_relative_tolerance: Literal[0.01]
    flop_relative_tolerance: Literal[0.05]
    candidate_hidden_widths: tuple[int, ...]
    rows: tuple[A0MatchRow, ...]
    selected_hidden_width: int | None
    status: GateStatus
    obligations: tuple[str, ...]
    matching_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in tuple(self.__dataclass_fields__)[:-1]
        }

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-a0-arm-matching-v1":
            raise ValueError("unsupported A0 matching schema")
        _sha256(self.primary_arm_spec_sha256, "primary_arm_spec_sha256")
        _sha256(
            self.endpoint_inventory_sha256,
            "endpoint_inventory_sha256",
        )
        _positive_int(
            self.primary_parameter_count,
            "primary_parameter_count",
        )
        _positive_int(
            self.primary_semantic_train_flops,
            "primary_semantic_train_flops",
        )
        if (
            type(self.parameter_relative_tolerance) is not float
            or self.parameter_relative_tolerance != 0.01
            or type(self.flop_relative_tolerance) is not float
            or self.flop_relative_tolerance != 0.05
            or self.candidate_hidden_widths
            != WT103_A0_HIDDEN_WIDTH_CANDIDATES
        ):
            raise ValueError("A0 matching policy changed")
        if (
            type(self.rows) is not tuple
            or tuple(item.hidden_width for item in self.rows)
            != self.candidate_hidden_widths
            or any(type(item) is not A0MatchRow for item in self.rows)
        ):
            raise ValueError("A0 matching rows are not the exact finite search")
        eligible = tuple(item for item in self.rows if item.eligible)
        selected = (
            min(eligible, key=lambda item: item.selection_key)
            if eligible
            else None
        )
        if (
            self.selected_hidden_width
            != (None if selected is None else selected.hidden_width)
            or type(self.status) is not GateStatus
        ):
            raise ValueError("A0 matching selection is inconsistent")
        if (
            (selected is not None and self.status is not GateStatus.PASS)
            or (selected is None and self.status is not GateStatus.INCONCLUSIVE)
        ):
            raise ValueError("A0 matching status is inconsistent")
        if (
            type(self.obligations) is not tuple
            or any(type(item) is not str or not item for item in self.obligations)
            or (self.status is GateStatus.PASS and self.obligations)
            or (self.status is not GateStatus.PASS and not self.obligations)
        ):
            raise ValueError("A0 matching obligations are inconsistent")
        expected = owned_sha256(
            "vfe4.wt103.a0-arm-matching.v1",
            self.semantic_payload(),
        )
        _sha256(self.matching_sha256, "matching_sha256")
        if self.matching_sha256 != expected:
            raise ValueError("A0 matching hash does not match")

    @classmethod
    def create(
        cls,
        *,
        primary_arm_spec_sha256: str,
        endpoint_inventory_sha256: str,
        primary_parameter_count: int,
        primary_semantic_train_flops: int,
        rows: tuple[A0MatchRow, ...],
    ) -> "ArmMatchingReport":
        eligible = tuple(item for item in rows if item.eligible)
        selected = (
            min(eligible, key=lambda item: item.selection_key)
            if eligible
            else None
        )
        payload = {
            "schema_version": "wt103-a0-arm-matching-v1",
            "primary_arm_spec_sha256": primary_arm_spec_sha256,
            "endpoint_inventory_sha256": endpoint_inventory_sha256,
            "primary_parameter_count": primary_parameter_count,
            "primary_semantic_train_flops": primary_semantic_train_flops,
            "parameter_relative_tolerance": 0.01,
            "flop_relative_tolerance": 0.05,
            "candidate_hidden_widths": WT103_A0_HIDDEN_WIDTH_CANDIDATES,
            "rows": rows,
            "selected_hidden_width": (
                None if selected is None else selected.hidden_width
            ),
            "status": (
                GateStatus.INCONCLUSIVE
                if selected is None
                else GateStatus.PASS
            ),
            "obligations": (
                ("no_a0_candidate_satisfies_parameter_and_flop_margins",)
                if selected is None
                else ()
            ),
        }
        return cls(
            **payload,
            matching_sha256=owned_sha256(
                "vfe4.wt103.a0-arm-matching.v1",
                payload,
            ),
        )  # type: ignore[arg-type]


def _a0_parameter_count(
    *,
    vocabulary_size: int,
    positional_capacity: int,
    hidden_width: int,
) -> int:
    return (
        2 * vocabulary_size * hidden_width
        + positional_capacity * hidden_width
        + 12 * hidden_width**2
        + 15 * hidden_width
        + vocabulary_size
    )


def audit_arm_matching(
    *,
    profile: WT103ExperimentProfile,
    endpoint_inventory: EndpointInventory,
    primary_parameter_count: int,
    primary_semantic_train_flops: int,
    workload_template: A0FlopWorkload,
    optimizer_access_exact: bool,
) -> ArmMatchingReport:
    """Evaluate all frozen widths without using corpus tensors or outcomes."""

    profile = _validate_profile(profile)
    endpoint_inventory = _validate_endpoint_inventory(endpoint_inventory)
    _positive_int(primary_parameter_count, "primary_parameter_count")
    _positive_int(
        primary_semantic_train_flops,
        "primary_semantic_train_flops",
    )
    if type(workload_template) is not A0FlopWorkload:
        raise ValueError("workload_template must be an exact A0FlopWorkload")
    workload_template.__post_init__()
    if (
        workload_template.vocabulary_size != profile.vocabulary_size
        or type(optimizer_access_exact) is not bool
    ):
        raise ValueError("A0 matching inputs do not match the literal profile")
    primary = tuple(
        item
        for item in endpoint_inventory.arms
        if item.result_role == "PRIMARY_ENDPOINT"
    )
    if (
        len(primary) != 1
        or primary[0].arm_id
        != "WT103-A5-PARENT-SPECIFIC-PREFIX-COMPLETE-v1"
    ):
        raise ValueError("matching requires the unique exact PRIMARY endpoint")

    rows: list[A0MatchRow] = []
    for hidden_width in WT103_A0_HIDDEN_WIDTH_CANDIDATES:
        parameter_count = _a0_parameter_count(
            vocabulary_size=profile.vocabulary_size,
            positional_capacity=128,
            hidden_width=hidden_width,
        )
        workload = dataclasses.replace(
            workload_template,
            hidden_width=hidden_width,
            parameter_count=parameter_count,
        )
        ledger = reconstruct_a0_flops(workload)
        parameter_ratio = parameter_count / primary_parameter_count
        flop_ratio = (
            ledger.semantic_train_flops / primary_semantic_train_flops
        )
        parameter_error = abs(parameter_ratio - 1.0)
        flop_error = abs(flop_ratio - 1.0)
        rows.append(
            A0MatchRow(
                hidden_width=hidden_width,
                parameter_count=parameter_count,
                semantic_train_flops=ledger.semantic_train_flops,
                parameter_ratio=parameter_ratio,
                flop_ratio=flop_ratio,
                parameter_relative_error=parameter_error,
                flop_relative_error=flop_error,
                optimizer_access_exact=optimizer_access_exact,
                filler_parameter_count=0,
                eligible=(
                    parameter_error <= 0.01
                    and flop_error <= 0.05
                    and optimizer_access_exact
                ),
                selection_key=(
                    abs(math.log(parameter_ratio)),
                    abs(math.log(flop_ratio)),
                    hidden_width,
                ),
                formula_sha256=owned_sha256(
                    "vfe4.wt103.a0-candidate-formula.v1",
                    {
                        "hidden_width": hidden_width,
                        "parameter_count": parameter_count,
                        "flop_ledger_sha256": ledger.ledger_sha256,
                    },
                ),
            )
        )
    return ArmMatchingReport.create(
        primary_arm_spec_sha256=primary[0].arm_spec_sha256,
        endpoint_inventory_sha256=(
            endpoint_inventory.endpoint_inventory_sha256
        ),
        primary_parameter_count=primary_parameter_count,
        primary_semantic_train_flops=primary_semantic_train_flops,
        rows=tuple(rows),
    )


@dataclass(frozen=True, slots=True)
class A0FactoryInputs:
    """All scientific identities needed by the direct A0 constructor."""

    architecture: A0ArchitectureProfile
    formula: A0FormulaRecord
    flop_ledger: A0FlopLedger
    matching: ArmMatchingReport
    device: torch.device
    dtype: torch.dtype

    def __post_init__(self) -> None:
        if type(self.architecture) is not A0ArchitectureProfile:
            raise ValueError("architecture must be exact")
        if type(self.formula) is not A0FormulaRecord:
            raise ValueError("formula must be exact")
        if type(self.flop_ledger) is not A0FlopLedger:
            raise ValueError("flop_ledger must be exact")
        if type(self.matching) is not ArmMatchingReport:
            raise ValueError("matching must be exact")
        self.architecture.__post_init__()
        self.formula.__post_init__()
        self.flop_ledger.__post_init__()
        self.matching.__post_init__()
        width = self.matching.selected_hidden_width
        if (
            self.matching.status is not GateStatus.PASS
            or width is None
            or self.architecture.hidden_width != width
            or self.formula.hidden_width != width
            or self.flop_ledger.hidden_width != width
            or self.architecture.formula_sha256
            != self.formula.formula_sha256
            or self.formula.parameter_count
            != self.flop_ledger.parameter_count
            or self.formula.semantic_train_flops
            != self.flop_ledger.semantic_train_flops
        ):
            raise ValueError("A0 factory identities are not match-bound")
        if type(self.device) is not torch.device:
            raise ValueError("device must be an explicit torch.device")
        if type(self.dtype) is not torch.dtype:
            raise ValueError("dtype must be an explicit torch.dtype")


def _build_record(
    *,
    spec: WT103ArmSpec,
    runtime: WT103ArmRuntimeComponents,
    constructor_id: str,
    execution_scope: ExecutionScope,
    model_family_id: str,
    architecture_sha256: str | None,
    formula_sha256: str | None,
    flop_ledger_sha256: str | None,
) -> BuiltWT103Arm:
    runtime.__post_init__()
    record = WT103ArmBuildRecord.create(
        spec=spec,
        constructor_id=constructor_id,
        execution_scope=execution_scope,
        model_family_id=model_family_id,
        training_objective=spec.training_objective,
        scorer_kind=spec.scorer_kind,
        update_phases=spec.update_phases,
        model_parameter_names=runtime.model_parameter_names,
        latent_parameter_names=runtime.latent_parameter_names,
        source_parameter_names=runtime.source_parameter_names,
        frame_parameter_names=runtime.frame_parameter_names,
        recognition_parameter_names=runtime.recognition_parameter_names,
        optimizer_bindings=runtime.optimizer_bindings,
        filler_parameter_names=runtime.filler_parameter_names,
        dormant_parameter_names=runtime.dormant_parameter_names,
        architecture_sha256=architecture_sha256,
        formula_sha256=formula_sha256,
        flop_ledger_sha256=flop_ledger_sha256,
    )
    return BuiltWT103Arm(record=record, runtime=runtime)


def build_wt103_a0(
    *,
    spec: WT103ArmSpec,
    profile: WT103ExperimentProfile,
    inputs: A0FactoryInputs,
    execution_scope: ExecutionScope,
) -> BuiltWT103Arm:
    """Construct the exact nonlatent, two-head, Flash-only A0."""

    spec = _exact_spec(
        spec,
        permitted_indices=(0,),
        constructor_id="build_wt103_a0",
    )
    profile = _validate_profile(profile)
    if type(inputs) is not A0FactoryInputs:
        raise ValueError("inputs must be exact A0FactoryInputs")
    inputs.__post_init__()
    if execution_scope not in (
        "nonproduction_synthetic_smoke",
        "production_source_lock_verified",
    ):
        raise ValueError("unknown WT103 execution scope")
    if (
        execution_scope == "production_source_lock_verified"
        and inputs.architecture.source_lock_scope
        != "production_source_lock_verified"
    ):
        raise ValueError("production A0 requires a verified source lock")
    if (
        inputs.formula.vocabulary_size != profile.vocabulary_size
        or inputs.architecture.positional_capacity != profile.sequence_length
    ):
        raise ValueError("A0 inputs disagree with the experiment profile")
    model = WT103A0Model(
        vocabulary_size=profile.vocabulary_size,
        positional_capacity=inputs.architecture.positional_capacity,
        hidden_width=inputs.architecture.hidden_width,
        attention_heads=inputs.architecture.attention_heads,
        layer_norm_epsilon=1.0e-5,
        device=inputs.device,
        dtype=inputs.dtype,
    )
    inventory = reconstruct_a0_parameters(
        model,
        vocabulary_size=profile.vocabulary_size,
        positional_capacity=inputs.architecture.positional_capacity,
        hidden_width=inputs.architecture.hidden_width,
    )
    if inventory.parameter_count != inputs.formula.parameter_count:
        raise ValueError("live A0 parameters disagree with the formula")
    names = tuple(item.name for item in inventory.parameters)
    runtime = WT103ArmRuntimeComponents.create(
        model=model,
        model_parameter_names=names,
        latent_parameter_names=(),
        source_parameter_names=(),
        frame_parameter_names=(),
        recognition_parameter_names=(),
        optimizer_bindings=(
            OptimizerParameterBinding(
                optimizer_id="model_adamw",
                parameter_names=names,
            ),
        ),
        filler_parameter_names=(),
        dormant_parameter_names=(),
    )
    return _build_record(
        spec=spec,
        runtime=runtime,
        constructor_id="build_wt103_a0",
        execution_scope=execution_scope,
        model_family_id="wt103-a0-one-block-two-head-flash-v1",
        architecture_sha256=inputs.architecture.architecture_sha256,
        formula_sha256=inputs.formula.formula_sha256,
        flop_ledger_sha256=inputs.flop_ledger.ledger_sha256,
    )


def _validate_latent_runtime(
    runtime: object,
) -> WT103ArmRuntimeComponents:
    if type(runtime) is not WT103ArmRuntimeComponents:
        raise ValueError("runtime must be exact WT103ArmRuntimeComponents")
    runtime.__post_init__()
    optimizer_ids = tuple(
        item.optimizer_id for item in runtime.optimizer_bindings
    )
    if (
        not runtime.latent_parameter_names
        or not runtime.source_parameter_names
        or not runtime.recognition_parameter_names
        or set(optimizer_ids) != {"model_adamw", "recognition_adamw"}
        or len(optimizer_ids) != 2
    ):
        raise ValueError(
            "latent A5 requires active latent/source/recognition state "
            "and exactly two optimizers"
        )
    recognition_owned = next(
        item.parameter_names
        for item in runtime.optimizer_bindings
        if item.optimizer_id == "recognition_adamw"
    )
    if set(recognition_owned) != set(runtime.recognition_parameter_names):
        raise ValueError(
            "recognition optimizer must own exactly recognition parameters"
        )
    return runtime


def build_wt103_a5_parent_specific(
    *,
    spec: WT103ArmSpec,
    profile: WT103ExperimentProfile,
    runtime: WT103ArmRuntimeComponents,
    execution_scope: ExecutionScope,
) -> BuiltWT103Arm:
    """Construct either exact parent-specific arm from its immutable spec."""

    spec = _exact_spec(
        spec,
        permitted_indices=(1, 3),
        constructor_id="build_wt103_a5_parent_specific",
    )
    _validate_profile(profile)
    runtime = _validate_latent_runtime(runtime)
    return _build_record(
        spec=spec,
        runtime=runtime,
        constructor_id="build_wt103_a5_parent_specific",
        execution_scope=execution_scope,
        model_family_id="wt103-a5-parent-specific-prefix-v1",
        architecture_sha256=None,
        formula_sha256=None,
        flop_ledger_sha256=None,
    )


def build_wt103_a5_fixed(
    *,
    spec: WT103ArmSpec,
    profile: WT103ExperimentProfile,
    runtime: WT103ArmRuntimeComponents,
    execution_scope: ExecutionScope,
) -> BuiltWT103Arm:
    """Construct the exact fixed-prior complete-ELBO control."""

    spec = _exact_spec(
        spec,
        permitted_indices=(2,),
        constructor_id="build_wt103_a5_fixed",
    )
    _validate_profile(profile)
    runtime = _validate_latent_runtime(runtime)
    return _build_record(
        spec=spec,
        runtime=runtime,
        constructor_id="build_wt103_a5_fixed",
        execution_scope=execution_scope,
        model_family_id="wt103-a5-fixed-prior-v1",
        architecture_sha256=None,
        formula_sha256=None,
        flop_ledger_sha256=None,
    )


def build_wt103_a5_nolatent(
    *,
    spec: WT103ArmSpec,
    profile: WT103ExperimentProfile,
    runtime: WT103ArmRuntimeComponents,
    execution_scope: ExecutionScope,
) -> BuiltWT103Arm:
    """Construct the exact no-latent control with no dormant A5 state."""

    spec = _exact_spec(
        spec,
        permitted_indices=(4,),
        constructor_id="build_wt103_a5_nolatent",
    )
    _validate_profile(profile)
    if type(runtime) is not WT103ArmRuntimeComponents:
        raise ValueError("runtime must be exact WT103ArmRuntimeComponents")
    runtime.__post_init__()
    if (
        runtime.latent_parameter_names
        or runtime.source_parameter_names
        or runtime.frame_parameter_names
        or runtime.recognition_parameter_names
        or runtime.optimizer_bindings
        != (
            OptimizerParameterBinding(
                optimizer_id="model_adamw",
                parameter_names=runtime.model_parameter_names,
            ),
        )
    ):
        raise ValueError("no-latent arm contains latent or dormant A5 state")
    return _build_record(
        spec=spec,
        runtime=runtime,
        constructor_id="build_wt103_a5_nolatent",
        execution_scope=execution_scope,
        model_family_id="wt103-a5-nolatent-v1",
        architecture_sha256=None,
        formula_sha256=None,
        flop_ledger_sha256=None,
    )


def build_wt103_arm(
    *,
    spec: WT103ArmSpec,
    profile: WT103ExperimentProfile,
    a0_inputs: A0FactoryInputs | None,
    runtime: WT103ArmRuntimeComponents | None,
    execution_scope: ExecutionScope,
) -> BuiltWT103Arm:
    """Dispatch only the five exact factory IDs through direct calls."""

    if type(spec) is not WT103ArmSpec:
        raise ValueError("spec must be exact")
    if (
        spec.factory_id == "build_wt103_a0@wt103-arm-v1"
        and spec.arm_id == "WT103-A0-AR-v1"
    ):
        if a0_inputs is None or runtime is not None:
            raise ValueError("A0 dispatch requires only a0_inputs")
        return build_wt103_a0(
            spec=spec,
            profile=profile,
            inputs=a0_inputs,
            execution_scope=execution_scope,
        )
    if (
        spec.factory_id
        == "build_wt103_a5_parent_specific@wt103-arm-v1"
        and spec.arm_id
        in (
            "WT103-A5-PARENT-SPECIFIC-PREFIX-COMPLETE-v1",
            "WT103-A5-PARENT-SPECIFIC-PREFIX-EMISSION-v1",
        )
    ):
        if a0_inputs is not None or runtime is None:
            raise ValueError(
                "parent-specific dispatch requires only its runtime"
            )
        return build_wt103_a5_parent_specific(
            spec=spec,
            profile=profile,
            runtime=runtime,
            execution_scope=execution_scope,
        )
    if (
        spec.factory_id == "build_wt103_a5_fixed@wt103-arm-v1"
        and spec.arm_id == "WT103-A5-FIXED-COMPLETE-v1"
    ):
        if a0_inputs is not None or runtime is None:
            raise ValueError("fixed dispatch requires only its runtime")
        return build_wt103_a5_fixed(
            spec=spec,
            profile=profile,
            runtime=runtime,
            execution_scope=execution_scope,
        )
    if (
        spec.factory_id == "build_wt103_a5_nolatent@wt103-arm-v1"
        and spec.arm_id == "WT103-A5-NOLATENT-v1"
    ):
        if a0_inputs is not None or runtime is None:
            raise ValueError("no-latent dispatch requires only its runtime")
        return build_wt103_a5_nolatent(
            spec=spec,
            profile=profile,
            runtime=runtime,
            execution_scope=execution_scope,
        )
    raise ValueError("arm spec does not name an exact WT103 factory row")


def scorer_dispatch(build: BuiltWT103Arm) -> str:
    """Return the scorer frozen by the exact arm spec, never by family."""

    if type(build) is not BuiltWT103Arm:
        raise ValueError("build must be an exact BuiltWT103Arm")
    build.__post_init__()
    return build.record.spec.scorer_kind


@dataclass(frozen=True, slots=True)
class WT103FactorySetIdentity:
    """Ordered identity of all five builds through exactly four constructors."""

    schema_version: Literal["wt103-factory-set-v1"]
    arm_spec_sha256s: tuple[str, ...]
    constructor_ids: tuple[str, ...]
    build_sha256s: tuple[str, ...]
    factory_set_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in tuple(self.__dataclass_fields__)[:-1]
        }

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-factory-set-v1":
            raise ValueError("unsupported WT103 factory-set schema")
        exact_specs = default_wt103_arm_specs()
        if (
            self.arm_spec_sha256s
            != tuple(item.arm_spec_sha256 for item in exact_specs)
            or self.constructor_ids
            != (
                "build_wt103_a0",
                "build_wt103_a5_parent_specific",
                "build_wt103_a5_fixed",
                "build_wt103_a5_parent_specific",
                "build_wt103_a5_nolatent",
            )
            or len(set(self.constructor_ids)) != 4
            or type(self.build_sha256s) is not tuple
            or len(self.build_sha256s) != 5
        ):
            raise ValueError("factory set is not the exact ordered five arms")
        for index, value in enumerate(self.build_sha256s):
            _sha256(value, f"build_sha256s[{index}]")
        expected = owned_sha256(
            "vfe4.wt103.factory-set.v1",
            self.semantic_payload(),
        )
        _sha256(self.factory_set_sha256, "factory_set_sha256")
        if self.factory_set_sha256 != expected:
            raise ValueError("factory_set_sha256 does not match")

    @classmethod
    def bind(
        cls,
        *,
        arm_specs: tuple[WT103ArmSpec, ...],
        constructor_ids: tuple[str, ...],
        build_sha256s: tuple[str, ...],
    ) -> "WT103FactorySetIdentity":
        payload = {
            "schema_version": "wt103-factory-set-v1",
            "arm_spec_sha256s": tuple(
                item.arm_spec_sha256 for item in arm_specs
            ),
            "constructor_ids": constructor_ids,
            "build_sha256s": build_sha256s,
        }
        return cls(
            **payload,
            factory_set_sha256=owned_sha256(
                "vfe4.wt103.factory-set.v1",
                payload,
            ),
        )  # type: ignore[arg-type]

    @classmethod
    def create(
        cls,
        builds: tuple[BuiltWT103Arm, ...],
    ) -> "WT103FactorySetIdentity":
        if (
            type(builds) is not tuple
            or len(builds) != 5
            or any(type(item) is not BuiltWT103Arm for item in builds)
        ):
            raise ValueError("builds must be five exact BuiltWT103Arm records")
        for build in builds:
            build.__post_init__()
        return cls.bind(
            arm_specs=tuple(item.record.spec for item in builds),
            constructor_ids=tuple(
                item.record.constructor_id for item in builds
            ),
            build_sha256s=tuple(
                item.record.build_sha256 for item in builds
            ),
        )


__all__ = [
    "A0FactoryInputs",
    "A0MatchRow",
    "ArmMatchingReport",
    "WT103FactorySetIdentity",
    "audit_arm_matching",
    "build_wt103_a0",
    "build_wt103_a5_fixed",
    "build_wt103_a5_nolatent",
    "build_wt103_a5_parent_specific",
    "build_wt103_arm",
    "scorer_dispatch",
]
