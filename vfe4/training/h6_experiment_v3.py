"""Outcome-blind H6-Prediction v3 experiment planning.

The planner is deliberately pure.  It consumes only already-authenticated
configuration authorities and reconstructs the frozen endpoint, matching,
schedule, tuning, seed, and attempt inventory.  Corpus bytes and experimental
outcomes are not part of its interface.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import Literal

from vfe4.data.windows import frozen_batch_schedule
from vfe4.types.h6 import (
    ArmConfig,
    H6ArmPhaseSchedule,
    TrainingPhase,
    canonical_json_bytes,
)
from vfe4.types.h6_prediction_v3 import (
    H6AttemptSpecV3,
    H6PredictionRuntimeIdentity,
    H6PredictionV3ReadinessToken,
    H6TrainingScheduleV3,
    H6_COUNTER_MAPPING_SHA256,
    H6_NO_COUNTER_CONSUMPTION_SHA256,
)

from .h6_matching_v3 import (
    H6_MATCHING_POLICY_V3,
    H6_MATCHING_V3_ENDPOINT_CONFIG_IDS,
    H6MatchingSetV3,
    H6TrainingWorkloadV3,
)


H6_TUNING_CELLS_V3: tuple[tuple[float, float], ...] = (
    (1.0e-4, 0.0),
    (1.0e-4, 1.0e-2),
    (3.0e-4, 0.0),
    (3.0e-4, 1.0e-2),
    (1.0e-3, 0.0),
    (1.0e-3, 1.0e-2),
)
H6_TUNING_SEEDS_V3 = (2026072199, 2026072200)
H6_CONFIRMATORY_SEEDS_V3 = tuple(range(2026072101, 2026072109))
H6_TUNED_ENDPOINT_CONFIG_IDS_V3 = (
    *H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[:5],
    H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[7],
)

_LOWER_HEX = frozenset("0123456789abcdef")
_TRAINING_NORMAL_DOMAIN = b"vfe4.h6.training-rmc-normal.v1\x00"
_TRAINING_COUNTER_CONSUMPTION_DOMAIN = (
    b"vfe4.h6.training-counter-consumption.v1\x00"
)


def _hash(domain: str, payload: object) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + canonical_json_bytes(payload)
    ).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _terminal_counter_identity(
    *,
    attempt_spec_sha256: str,
    pass_index: int,
    batch_index: int,
    example_ordinal: int,
    draw_block: int,
    receiver_count: int,
    latent_dimension: int,
) -> tuple[str, str]:
    """Purely reconstruct the last latent model-phase counter consumption."""

    key_payload = {
        "attempt_spec_sha256": attempt_spec_sha256,
        "pass_index": pass_index,
        "batch_index": batch_index,
        "phase": TrainingPhase.MODEL_ADAMW.value,
        "example_ordinal": example_ordinal,
        "sample_ordinal": 0,
        "draw_block": draw_block,
    }
    key_bytes = canonical_json_bytes(key_payload)
    key_sha256 = hashlib.sha256(
        _TRAINING_NORMAL_DOMAIN + key_bytes
    ).hexdigest()
    values: list[float] = []
    count = receiver_count * latent_dimension
    for pair_index in range((count + 1) // 2):
        uniforms: list[float] = []
        for draw_index in (2 * pair_index, 2 * pair_index + 1):
            block_index, word_index = divmod(draw_index, 4)
            digest = hashlib.sha256(
                _TRAINING_NORMAL_DOMAIN
                + key_bytes
                + block_index.to_bytes(8, "little")
            ).digest()
            offset = 8 * word_index
            word = int.from_bytes(
                digest[offset : offset + 8],
                "little",
            )
            uniform = (float(word) + 0.5) / float(2**64)
            if uniform <= 0.0:
                uniform = math.nextafter(0.0, 1.0)
            elif uniform >= 1.0:
                uniform = math.nextafter(1.0, 0.0)
            uniforms.append(uniform)
        radius = math.sqrt(-2.0 * math.log(uniforms[0]))
        angle = 2.0 * math.pi * uniforms[1]
        values.append(radius * math.cos(angle))
        if len(values) < count:
            values.append(radius * math.sin(angle))
    raw_bytes = b"".join(struct.pack("<d", value) for value in values)
    consumption_sha256 = hashlib.sha256(
        _TRAINING_COUNTER_CONSUMPTION_DOMAIN
        + bytes.fromhex(H6_COUNTER_MAPPING_SHA256)
        + bytes.fromhex(key_sha256)
        + receiver_count.to_bytes(8, "little")
        + latent_dimension.to_bytes(8, "little")
        + raw_bytes
    ).hexdigest()
    return key_sha256, consumption_sha256


@dataclass(frozen=True, slots=True)
class H6TuningCellV3:
    """One of the six frozen AdamW tuning cells."""

    learning_rate: float
    weight_decay: float
    cell_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
        }

    def __post_init__(self) -> None:
        if (self.learning_rate, self.weight_decay) not in H6_TUNING_CELLS_V3:
            raise ValueError("tuning cell is outside the frozen six-cell grid")
        if self.cell_sha256 != _hash(
            "vfe4.h6.tuning-cell.v3", self.canonical_payload()
        ):
            raise ValueError("tuning-cell identity is stale")

    @classmethod
    def create(cls, *, learning_rate: float, weight_decay: float) -> "H6TuningCellV3":
        payload = {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        }
        return cls(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            cell_sha256=_hash("vfe4.h6.tuning-cell.v3", payload),
        )


@dataclass(frozen=True, slots=True)
class H6PlannedAttemptV3:
    """An exact attempt spec plus the planning facts absent from its v3 schema."""

    stage: Literal["tuning", "confirmatory"]
    endpoint_config_id: str
    endpoint_config_sha256: str
    matching_policy_sha256: str
    matching_set_sha256: str
    matching_ledger_sha256: str
    matching_report_sha256s: tuple[str, ...]
    receiver_count: int
    state_categorical_enabled: bool
    model_categorical_enabled: bool
    tuning_cell: H6TuningCellV3 | None
    tuning_cell_source: str
    training_seed: int
    terminal_pass_index: int
    terminal_batch_index: int
    terminal_example_ordinal: int
    terminal_draw_block: int
    terminal_counter_key_sha256: str | None
    terminal_counter_consumption_sha256: str
    consumed_permutation_sha256s: tuple[str, ...]
    terminal_permutation_sha256: str
    terminal_recognition_update_count: int
    terminal_model_update_count: int
    terminal_validation_boundary_count: int
    terminal_checkpoint_boundary_count: int
    attempt_spec: H6AttemptSpecV3
    planned_attempt_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "endpoint_config_id": self.endpoint_config_id,
            "endpoint_config_sha256": self.endpoint_config_sha256,
            "matching_policy_sha256": self.matching_policy_sha256,
            "matching_set_sha256": self.matching_set_sha256,
            "matching_ledger_sha256": self.matching_ledger_sha256,
            "matching_report_sha256s": self.matching_report_sha256s,
            "receiver_count": self.receiver_count,
            "state_categorical_enabled": self.state_categorical_enabled,
            "model_categorical_enabled": self.model_categorical_enabled,
            "tuning_cell_sha256": (
                None if self.tuning_cell is None else self.tuning_cell.cell_sha256
            ),
            "tuning_cell_source": self.tuning_cell_source,
            "training_seed": self.training_seed,
            "terminal_pass_index": self.terminal_pass_index,
            "terminal_batch_index": self.terminal_batch_index,
            "terminal_example_ordinal": self.terminal_example_ordinal,
            "terminal_draw_block": self.terminal_draw_block,
            "terminal_counter_key_sha256": (
                self.terminal_counter_key_sha256
            ),
            "terminal_counter_consumption_sha256": (
                self.terminal_counter_consumption_sha256
            ),
            "consumed_permutation_sha256s": (
                self.consumed_permutation_sha256s
            ),
            "terminal_permutation_sha256": (
                self.terminal_permutation_sha256
            ),
            "terminal_recognition_update_count": (
                self.terminal_recognition_update_count
            ),
            "terminal_model_update_count": self.terminal_model_update_count,
            "terminal_validation_boundary_count": (
                self.terminal_validation_boundary_count
            ),
            "terminal_checkpoint_boundary_count": (
                self.terminal_checkpoint_boundary_count
            ),
            "attempt_spec_sha256": self.attempt_spec.attempt_spec_sha256,
        }

    def __post_init__(self) -> None:
        if self.stage not in ("tuning", "confirmatory"):
            raise ValueError("planned attempt stage is invalid")
        if type(self.endpoint_config_id) is not str or not self.endpoint_config_id:
            raise ValueError("planned endpoint ID must be nonempty")
        for name in (
            "endpoint_config_sha256",
            "matching_policy_sha256",
            "matching_set_sha256",
            "matching_ledger_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if type(self.receiver_count) is not int or self.receiver_count < 2:
            raise ValueError("planned attempt receiver count must represent 0..T")
        if (
            type(self.state_categorical_enabled) is not bool
            or type(self.model_categorical_enabled) is not bool
        ):
            raise ValueError("planned attempt categorical topology is malformed")
        for name in (
            "terminal_pass_index",
            "terminal_batch_index",
            "terminal_example_ordinal",
            "terminal_draw_block",
            "terminal_recognition_update_count",
            "terminal_model_update_count",
            "terminal_validation_boundary_count",
            "terminal_checkpoint_boundary_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        _require_sha256(
            self.terminal_counter_consumption_sha256,
            "terminal_counter_consumption_sha256",
        )
        _require_sha256(
            self.terminal_permutation_sha256,
            "terminal_permutation_sha256",
        )
        if self.terminal_counter_key_sha256 is not None:
            _require_sha256(
                self.terminal_counter_key_sha256,
                "terminal_counter_key_sha256",
            )
        expected_permutation_count = 1 if self.stage == "tuning" else 2
        if (
            type(self.consumed_permutation_sha256s) is not tuple
            or len(self.consumed_permutation_sha256s)
            != expected_permutation_count
            or len(set(self.consumed_permutation_sha256s))
            != expected_permutation_count
            or self.terminal_permutation_sha256
            != self.consumed_permutation_sha256s[-1]
        ):
            raise ValueError(
                "planned terminal permutation inventory is inconsistent"
            )
        for digest in self.consumed_permutation_sha256s:
            _require_sha256(digest, "consumed permutation SHA-256")
        if (
            self.terminal_example_ordinal != 0
            or self.terminal_model_update_count <= 0
            or self.terminal_validation_boundary_count <= 0
            or self.terminal_checkpoint_boundary_count != 1
            or (
                self.attempt_spec.recognition_factory_sha256 is None
                and self.terminal_recognition_update_count != 0
            )
            or (
                self.attempt_spec.recognition_factory_sha256 is not None
                and self.terminal_recognition_update_count
                != self.terminal_model_update_count
            )
        ):
            raise ValueError("planned terminal cursor contract is inconsistent")
        if (
            type(self.matching_report_sha256s) is not tuple
            or not self.matching_report_sha256s
            or len(set(self.matching_report_sha256s))
            != len(self.matching_report_sha256s)
        ):
            raise ValueError(
                "planned attempt requires a unique matching-report inventory"
            )
        for digest in self.matching_report_sha256s:
            _require_sha256(digest, "matching report SHA-256")
        if type(self.attempt_spec) is not H6AttemptSpecV3:
            raise ValueError("planned attempt requires an exact v3 attempt spec")
        self.attempt_spec.__post_init__()
        latent = self.attempt_spec.recognition_factory_sha256 is not None
        if (
            latent
            and (
                self.terminal_draw_block
                != 2 * self.terminal_model_update_count
                or self.terminal_counter_key_sha256 is None
                or self.terminal_counter_consumption_sha256
                == H6_NO_COUNTER_CONSUMPTION_SHA256
            )
        ) or (
            not latent
            and (
                self.terminal_draw_block != 0
                or self.terminal_counter_key_sha256 is not None
                or self.terminal_counter_consumption_sha256
                != H6_NO_COUNTER_CONSUMPTION_SHA256
            )
        ):
            raise ValueError(
                "planned terminal counter-consumption contract is inconsistent"
            )
        if (
            self.attempt_spec.endpoint_id != self.endpoint_config_id
            or self.attempt_spec.endpoint_config_sha256 != self.endpoint_config_sha256
            or self.attempt_spec.training_seed != self.training_seed
        ):
            raise ValueError("planned attempt and attempt spec disagree")
        if self.stage == "tuning":
            if (
                type(self.tuning_cell) is not H6TuningCellV3
                or self.tuning_cell_source != "literal-six-cell-v1"
                or self.training_seed not in H6_TUNING_SEEDS_V3
            ):
                raise ValueError("tuning attempt does not bind its literal cell")
            self.tuning_cell.__post_init__()
        elif (
            self.tuning_cell is not None
            or not self.tuning_cell_source.startswith("selected:")
            or self.training_seed not in H6_CONFIRMATORY_SEEDS_V3
        ):
            raise ValueError("confirmatory attempt selection binding is invalid")
        if self.planned_attempt_sha256 != _hash(
            "vfe4.h6.planned-attempt.v3", self.canonical_payload()
        ):
            raise ValueError("planned-attempt identity is stale")


@dataclass(frozen=True, slots=True)
class H6ExperimentPlanV3:
    """Complete immutable pre-outcome H6 v3 execution plan."""

    plan_schema: Literal["h6-experiment-plan-v3"]
    git_head: str
    dirty_digest: str
    experiment_config_sha256: str
    readiness_sha256: str
    matching_policy_sha256: str
    matching_set_sha256: str
    endpoint_configs: tuple[ArmConfig, ...]
    matching_report_sha256s: tuple[str, ...]
    tuning_cells: tuple[H6TuningCellV3, ...]
    tuning_seeds: tuple[int, int]
    confirmatory_seeds: tuple[int, ...]
    training_schedule: H6TrainingScheduleV3
    tuning_attempts: tuple[H6PlannedAttemptV3, ...]
    confirmatory_attempts: tuple[H6PlannedAttemptV3, ...]
    plan_sha256: str

    @property
    def endpoint_config_ids(self) -> tuple[str, ...]:
        return tuple(config.config_id for config in self.endpoint_configs)

    @property
    def attempts(self) -> tuple[H6PlannedAttemptV3, ...]:
        return self.tuning_attempts + self.confirmatory_attempts

    def canonical_payload(self) -> dict[str, object]:
        return {
            "plan_schema": self.plan_schema,
            "git_head": self.git_head,
            "dirty_digest": self.dirty_digest,
            "experiment_config_sha256": self.experiment_config_sha256,
            "readiness_sha256": self.readiness_sha256,
            "matching_policy_sha256": self.matching_policy_sha256,
            "matching_set_sha256": self.matching_set_sha256,
            "endpoint_config_sha256s": tuple(
                config.config_sha256 for config in self.endpoint_configs
            ),
            "matching_report_sha256s": self.matching_report_sha256s,
            "tuning_cell_sha256s": tuple(
                cell.cell_sha256 for cell in self.tuning_cells
            ),
            "tuning_seeds": self.tuning_seeds,
            "confirmatory_seeds": self.confirmatory_seeds,
            "training_schedule_sha256": self.training_schedule.schedule_sha256,
            "tuning_attempt_sha256s": tuple(
                attempt.planned_attempt_sha256 for attempt in self.tuning_attempts
            ),
            "confirmatory_attempt_sha256s": tuple(
                attempt.planned_attempt_sha256 for attempt in self.confirmatory_attempts
            ),
        }

    def __post_init__(self) -> None:
        if self.plan_schema != "h6-experiment-plan-v3":
            raise ValueError("unsupported H6 experiment plan schema")
        if self.endpoint_config_ids != H6_MATCHING_V3_ENDPOINT_CONFIG_IDS:
            raise ValueError("experiment plan endpoint inventory is not exact")
        if self.tuning_seeds != H6_TUNING_SEEDS_V3:
            raise ValueError("experiment plan tuning seeds are not frozen")
        if self.confirmatory_seeds != H6_CONFIRMATORY_SEEDS_V3:
            raise ValueError("experiment plan confirmatory seeds are not frozen")
        if (
            tuple((cell.learning_rate, cell.weight_decay) for cell in self.tuning_cells)
            != H6_TUNING_CELLS_V3
        ):
            raise ValueError("experiment plan tuning cells are not frozen")
        if len(self.tuning_attempts) != 72 or len(self.confirmatory_attempts) != 96:
            raise ValueError("experiment plan attempt inventory is incomplete")
        for attempt in self.attempts:
            attempt.__post_init__()
        if self.plan_sha256 != _hash(
            "vfe4.h6.experiment-plan.v3", self.canonical_payload()
        ):
            raise ValueError("experiment-plan identity is stale")


def _factory_sha256(config: ArmConfig, *, recognition: bool) -> str | None:
    if recognition and not config.latent_enabled:
        return None
    kind = "recognition" if recognition else "model"
    factory = (
        "none"
        if recognition and not config.latent_enabled
        else f"build_{config.arm.value.lower()}@h6-arm-v1"
    )
    return _hash(
        f"vfe4.h6.{kind}-factory.v3",
        {
            "factory": factory,
            "endpoint_config_sha256": config.config_sha256,
        },
    )


def model_factory_sha256_v3(config: ArmConfig) -> str:
    """Return the canonical model-only factory identity for one endpoint."""

    if type(config) is not ArmConfig:
        raise ValueError("model factory identity requires an exact ArmConfig")
    config.__post_init__()
    digest = _factory_sha256(config, recognition=False)
    assert digest is not None
    return digest


def _attempt(
    *,
    stage: Literal["tuning", "confirmatory"],
    config: ArmConfig,
    matching_policy_sha256: str,
    matching_set_sha256: str,
    ledger_sha256: str,
    matching_report_sha256s: tuple[str, ...],
    phase_schedule: H6ArmPhaseSchedule,
    cell: H6TuningCellV3 | None,
    tuning_cell_source: str,
    seed: int,
    readiness: H6PredictionV3ReadinessToken,
    schedule: H6TrainingScheduleV3,
    runtime: H6PredictionRuntimeIdentity,
    workload: H6TrainingWorkloadV3,
) -> H6PlannedAttemptV3:
    workload.__post_init__()
    workload_sha256 = workload.workload_sha256
    initialization_sha256 = _hash(
        "vfe4.h6.canonical-initialization-plan.v3",
        {
            "endpoint_config_sha256": config.config_sha256,
            "training_seed": seed,
        },
    )
    window_schedule_sha256 = _hash(
        "vfe4.h6.window-schedule-plan.v3",
        {
            "workload_sha256": workload_sha256,
            "stage": stage,
            "coverage": "quarter-pass" if stage == "tuning" else "two-full-passes",
        },
    )
    batch_schedule_sha256 = _hash(
        "vfe4.h6.batch-schedule-plan.v3",
        {
            "window_schedule_sha256": window_schedule_sha256,
            "training_seed": seed,
            "batch_size": 8,
            "drop_last": False,
        },
    )
    spec = H6AttemptSpecV3.create(
        git_head=readiness.git_head,
        dirty_digest=readiness.dirty_digest,
        readiness_sha256=readiness.readiness_sha256,
        experiment_config_sha256=readiness.experiment_config_sha256,
        endpoint_id=config.config_id,
        arm_id=config.arm.value,
        endpoint_config_sha256=config.config_sha256,
        objective_kind=config.objective_kind,
        model_factory_sha256=model_factory_sha256_v3(config),
        recognition_factory_sha256=_factory_sha256(config, recognition=True),
        initialization_sha256=initialization_sha256,
        optimizer_policy_sha256=schedule.outer.optimizer_policy_sha256,
        training_seed=seed,
        data_identity_sha256=readiness.data_identity_sha256,
        window_schedule_sha256=window_schedule_sha256,
        batch_schedule_sha256=batch_schedule_sha256,
        phase_schedule_sha256=phase_schedule.phase_schedule_sha256,
        training_schedule_sha256=schedule.schedule_sha256,
        recognition_estimator_sha256=readiness.recognition_estimator_sha256,
        runtime_identity_sha256=runtime.runtime_identity_sha256,
    )
    if stage == "tuning":
        terminal_batches = (workload.batches_per_pass + 3) // 4
        terminal_pass_index = 0
        terminal_batch_index = terminal_batches
        consumed_pass_indices = (0,)
        final_consumed_batch_index = terminal_batches - 1
        terminal_model_updates = terminal_batches
        terminal_validation_boundaries = sum(
            boundary <= terminal_batches
            for boundary in workload.validation_boundaries_per_pass
        )
    else:
        terminal_pass_index = workload.full_passes
        terminal_batch_index = 0
        consumed_pass_indices = tuple(range(workload.full_passes))
        final_consumed_batch_index = workload.batches_per_pass - 1
        terminal_model_updates = workload.model_update_opportunities
        terminal_validation_boundaries = (
            workload.full_passes
            * len(workload.validation_boundaries_per_pass)
        )
    consumed_permutation_sha256s = tuple(
        frozen_batch_schedule(
            window_count=workload.window_count,
            zero_based_pass_index=pass_index,
        ).schedule_sha256
        for pass_index in consumed_pass_indices
    )
    terminal_example_ordinal = 0
    if config.latent_enabled:
        latent_dimension = config.capacity_allocation.latent_width
        if type(latent_dimension) is not int or latent_dimension <= 0:
            raise ValueError(
                "latent terminal counter plan requires a positive latent width"
            )
        terminal_draw_block = 2 * terminal_model_updates
        (
            terminal_counter_key_sha256,
            terminal_counter_consumption_sha256,
        ) = _terminal_counter_identity(
            attempt_spec_sha256=spec.attempt_spec_sha256,
            pass_index=consumed_pass_indices[-1],
            batch_index=final_consumed_batch_index,
            example_ordinal=terminal_example_ordinal,
            draw_block=terminal_draw_block - 1,
            receiver_count=config.horizon + 1,
            latent_dimension=latent_dimension,
        )
    else:
        terminal_draw_block = 0
        terminal_counter_key_sha256 = None
        terminal_counter_consumption_sha256 = (
            H6_NO_COUNTER_CONSUMPTION_SHA256
        )
    values = {
        "stage": stage,
        "endpoint_config_id": config.config_id,
        "endpoint_config_sha256": config.config_sha256,
        "matching_policy_sha256": matching_policy_sha256,
        "matching_set_sha256": matching_set_sha256,
        "matching_ledger_sha256": ledger_sha256,
        "matching_report_sha256s": matching_report_sha256s,
        "receiver_count": config.horizon + 1,
        "state_categorical_enabled": (
            config.source_mode == "categorical" and config.state_channel_enabled
        ),
        "model_categorical_enabled": (
            config.source_mode == "categorical" and config.model_channel_enabled
        ),
        "tuning_cell": cell,
        "tuning_cell_source": tuning_cell_source,
        "training_seed": seed,
        "terminal_pass_index": terminal_pass_index,
        "terminal_batch_index": terminal_batch_index,
        "terminal_example_ordinal": terminal_example_ordinal,
        "terminal_draw_block": terminal_draw_block,
        "terminal_counter_key_sha256": terminal_counter_key_sha256,
        "terminal_counter_consumption_sha256": (
            terminal_counter_consumption_sha256
        ),
        "consumed_permutation_sha256s": (
            consumed_permutation_sha256s
        ),
        "terminal_permutation_sha256": (
            consumed_permutation_sha256s[-1]
        ),
        "terminal_recognition_update_count": (
            terminal_model_updates if config.latent_enabled else 0
        ),
        "terminal_model_update_count": terminal_model_updates,
        "terminal_validation_boundary_count": (
            terminal_validation_boundaries
        ),
        "terminal_checkpoint_boundary_count": 1,
        "attempt_spec": spec,
    }
    provisional = object.__new__(H6PlannedAttemptV3)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    return H6PlannedAttemptV3(
        **values,  # type: ignore[arg-type]
        planned_attempt_sha256=_hash(
            "vfe4.h6.planned-attempt.v3",
            provisional.canonical_payload(),
        ),
    )


def plan_h6_experiment_v3(
    *,
    readiness: H6PredictionV3ReadinessToken,
    matching_set: H6MatchingSetV3,
    training_schedule: H6TrainingScheduleV3,
    runtime_identity: H6PredictionRuntimeIdentity,
) -> H6ExperimentPlanV3:
    """Reconstruct the complete immutable H6 v3 plan without data or outcomes."""

    if type(readiness) is not H6PredictionV3ReadinessToken:
        raise ValueError("plan requires an exact v3 readiness token")
    if type(matching_set) is not H6MatchingSetV3:
        raise ValueError("plan requires an exact v3 matching set")
    if type(training_schedule) is not H6TrainingScheduleV3:
        raise ValueError("plan requires an exact v3 training schedule")
    if type(runtime_identity) is not H6PredictionRuntimeIdentity:
        raise ValueError("plan requires an exact v3 runtime identity")

    # Compare identities before recomputing owned digests so drift gets a
    # boundary-specific refusal instead of a generic stale-record message.
    if (
        readiness.matching_set_sha256 != matching_set.matching_set_sha256
        or readiness.matching_policy_sha256 != matching_set.matching_policy_sha256
        or readiness.matching_policy_sha256 != H6_MATCHING_POLICY_V3.policy_sha256
    ):
        raise ValueError("readiness and matching-set identity drift")
    if readiness.training_schedule_sha256 != training_schedule.schedule_sha256:
        raise ValueError("readiness and training-schedule identity drift")
    if (
        readiness.runtime_identity_sha256 != runtime_identity.runtime_identity_sha256
        or training_schedule.runtime_identity_sha256
        != runtime_identity.runtime_identity_sha256
    ):
        raise ValueError("readiness/runtime identity drift")

    readiness.__post_init__()
    matching_set.__post_init__()
    training_schedule.__post_init__()
    runtime_identity.__post_init__()
    if (
        matching_set.status != "ELIGIBLE"
        or matching_set.git_head != readiness.git_head
        or matching_set.dirty_digest != readiness.dirty_digest
    ):
        raise ValueError("matching set is not current and eligible")
    phase_by_endpoint = {
        phase.endpoint_config_sha256: phase
        for phase in training_schedule.endpoint_phases
    }
    if tuple(phase_by_endpoint) != tuple(
        config.config_sha256 for config in matching_set.endpoint_configs
    ):
        raise ValueError("training schedule does not cover the exact twelve endpoints")

    cells = tuple(
        H6TuningCellV3.create(learning_rate=lr, weight_decay=wd)
        for lr, wd in H6_TUNING_CELLS_V3
    )
    config_by_id = {
        config.config_id: config for config in matching_set.endpoint_configs
    }
    ledger_by_id = {
        config.config_id: ledger.ledger_sha256
        for config, ledger in zip(
            matching_set.endpoint_configs,
            matching_set.endpoint_ledgers,
            strict=True,
        )
    }
    matching_report_sha256s = tuple(
        report.record_sha256 for report in matching_set.matrix_reports
    )
    tuning_attempts = tuple(
        _attempt(
            stage="tuning",
            config=config_by_id[config_id],
            matching_policy_sha256=matching_set.matching_policy_sha256,
            matching_set_sha256=matching_set.matching_set_sha256,
            ledger_sha256=ledger_by_id[config_id],
            matching_report_sha256s=matching_report_sha256s,
            phase_schedule=phase_by_endpoint[config_by_id[config_id].config_sha256],
            cell=cell,
            tuning_cell_source="literal-six-cell-v1",
            seed=seed,
            readiness=readiness,
            schedule=training_schedule,
            runtime=runtime_identity,
            workload=matching_set.workload,
        )
        for config_id in H6_TUNED_ENDPOINT_CONFIG_IDS_V3
        for cell in cells
        for seed in H6_TUNING_SEEDS_V3
    )
    confirmatory_attempts = tuple(
        _attempt(
            stage="confirmatory",
            config=config,
            matching_policy_sha256=matching_set.matching_policy_sha256,
            matching_set_sha256=matching_set.matching_set_sha256,
            ledger_sha256=ledger_by_id[config.config_id],
            matching_report_sha256s=matching_report_sha256s,
            phase_schedule=phase_by_endpoint[config.config_sha256],
            cell=None,
            tuning_cell_source=(
                f"selected:{config.config_id}"
                if config.config_id in H6_TUNED_ENDPOINT_CONFIG_IDS_V3
                else f"selected:{H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[7]}"
            ),
            seed=seed,
            readiness=readiness,
            schedule=training_schedule,
            runtime=runtime_identity,
            workload=matching_set.workload,
        )
        for config in matching_set.endpoint_configs
        for seed in H6_CONFIRMATORY_SEEDS_V3
    )
    values = {
        "plan_schema": "h6-experiment-plan-v3",
        "git_head": readiness.git_head,
        "dirty_digest": readiness.dirty_digest,
        "experiment_config_sha256": readiness.experiment_config_sha256,
        "readiness_sha256": readiness.readiness_sha256,
        "matching_policy_sha256": matching_set.matching_policy_sha256,
        "matching_set_sha256": matching_set.matching_set_sha256,
        "endpoint_configs": matching_set.endpoint_configs,
        "matching_report_sha256s": tuple(
            report.record_sha256 for report in matching_set.matrix_reports
        ),
        "tuning_cells": cells,
        "tuning_seeds": H6_TUNING_SEEDS_V3,
        "confirmatory_seeds": H6_CONFIRMATORY_SEEDS_V3,
        "training_schedule": training_schedule,
        "tuning_attempts": tuning_attempts,
        "confirmatory_attempts": confirmatory_attempts,
    }
    provisional = object.__new__(H6ExperimentPlanV3)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    return H6ExperimentPlanV3(
        **values,  # type: ignore[arg-type]
        plan_sha256=_hash(
            "vfe4.h6.experiment-plan.v3",
            provisional.canonical_payload(),
        ),
    )


__all__ = [
    "H6_CONFIRMATORY_SEEDS_V3",
    "H6_TUNED_ENDPOINT_CONFIG_IDS_V3",
    "H6_TUNING_CELLS_V3",
    "H6_TUNING_SEEDS_V3",
    "H6ExperimentPlanV3",
    "H6PlannedAttemptV3",
    "H6TuningCellV3",
    "model_factory_sha256_v3",
    "plan_h6_experiment_v3",
]
