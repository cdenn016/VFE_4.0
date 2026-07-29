"""Hermetic, readiness-ineligible WikiText-103 training smoke.

The smoke uses generated token and scalar CPU fixtures. It exercises the exact
five immutable arm specifications, four direct factory constructors, real
arm-specific forward/prior/scorer/evaluator paths, canonical metric recording,
checkpoint restoration against an uninterrupted oracle, and the Task 10 run
lifecycle. It never imports the live tokenizer package, maps corpus data,
initializes CUDA, or opens held-out test data.
"""

from __future__ import annotations

import hashlib
import math
import os
import pickle
import random
import subprocess
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn

from vfe4.artifacts.durability import (
    DurabilityBackend,
    PosixDurabilityBackend,
    WindowsDurabilityBackend,
    canonical_json_bytes_generic,
)
from vfe4.artifacts.manifest import ArtifactIntegrityRecord
from vfe4.artifacts.run_directory import (
    ExperimentPlan,
    ExperimentPlanIdentity,
    ResumeLineageEvent,
    RunManifestIdentity,
    consume_resume_execution_retry,
    finalize_run,
    publish_experiment_index,
    publish_experiment_plan,
    release_run_execution_lease,
    reserve_run,
)
from vfe4.checkpoint import (
    ResumeContract,
    WT103CheckpointIdentity,
    load_checkpoint,
    require_terminal_scoring,
    save_checkpoint,
)
from vfe4.checkpoint.serialization import scientific_state_sha256
from vfe4.config.schema import TrainingConfig
from vfe4.data.tokenizer import (
    SyntheticTokenizerFixtureContract,
    build_synthetic_fixture_tokenizer_spec,
    encode_fixture_split_record,
    issue_fixture_split_capability,
)
from vfe4.data.windows import (
    CausalPrefix,
    build_evaluation_schedule,
    materialize_causal_window_set,
)
from vfe4.evaluation.prior_nll import (
    WT103EstimatorStreamBinding,
    WT103EvaluationBatches,
    bind_wt103_prior_predictor,
    score_prior_nll,
    wt103_estimator_stream_seed,
    wt103_score_trace,
)
from vfe4.generative.language import LanguageGenerativeModel
from vfe4.generative.source_priors import (
    FixedSourcePrior,
    ParentSpecificPooledPrefixSourcePrior,
)
from vfe4.predictive import (
    AssimilationRecord,
    BootstrapSmcPredictor,
    EstimatorIdentity,
    EstimatorRecord,
    EstimatorStream,
    LanguageGenerativeProposalAdapter,
    PendingPrediction,
    PrefixCache,
    PrefixCacheKey,
    PriorPrediction,
    ProposalPopulation,
    ProposalStep,
    canonical_model_state_sha256,
    vocabulary_identity_sha256,
)
from vfe4.recording.metrics import (
    WT103_METRIC_SEMANTIC_BY_NAME,
    WT103_REQUIRED_METRIC_FAMILIES,
    WT103_SOURCE_KL_DIAGNOSTIC_REASON,
    WT103_UNAVAILABLE_ESTIMATOR_BOUND_REASON,
    append_metric,
    applicable_metric,
    create_metric_record,
    export_metrics_csv,
    metric_family_applicability,
    metric_family_units,
    not_applicable_metric,
    validate_metric_log,
    validate_required_metric_families,
)
from vfe4.training.engine import (
    ArmExecutionRuntime,
    ForwardTerms,
    RecognitionSnapshot,
    StepResult,
    train_attempt,
)
from vfe4.training.factories import (
    A0FactoryInputs,
    WT103FactorySetIdentity,
    audit_arm_matching,
    build_wt103_arm,
    scorer_dispatch,
)
from vfe4.training.formulas import (
    A0FlopWorkload,
    build_a0_architecture_profile,
    build_a0_formula_record,
    reconstruct_a0_flops,
    reconstruct_a0_parameters,
)
from vfe4.training.production_observability import (
    project_objective_metrics,
)
from vfe4.training.wt103_models import (
    BuiltWT103Arm,
    OptimizerParameterBinding,
    WT103A0Model,
    WT103ArmRuntimeComponents,
)
from vfe4.types.results import GateStatus
from vfe4.types.h6 import (
    CausalDag,
    CausalDagRow,
    EstimatorSpec,
    H6LanguageStructure,
    VocabularyIdentity,
    ZeroDimensionalBase,
)
from vfe4.types.training import (
    EstimatorProtocol,
    WT103NllTotals,
    WT103ArmSpec,
    owned_sha256,
)


_AUTHORITY = "nonproduction_synthetic_smoke"
_ZERO_SHA256 = "0" * 64
_RESUME_ARM_INDEX = 1
_SEED_ID = 2026072101
_RUN_ARTIFACT_PATHS = (
    "metrics.csv",
    "metrics.jsonl",
    "nonproduction-authority.json",
)
_THREAD_ENVIRONMENT_NAMES = (
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _digest(label: str) -> str:
    return hashlib.sha256(
        b"vfe4-task12-synthetic-smoke-v1\0" + label.encode("utf-8")
    ).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _semantic_payload(value: object, *, omit: tuple[str, ...]) -> dict[str, object]:
    return {
        field.name: getattr(value, field.name)
        for field in fields(value)
        if field.name not in omit
    }


@dataclass(frozen=True, slots=True)
class SyntheticArmExecutionTrace:
    """Canonical proof that one smoke arm ran its scientific path."""

    schema_version: Literal["wt103-synthetic-arm-execution-trace-v1"]
    forward_path: str
    prior_path: str
    scorer_path: Literal["exact_autoregressive", "weighted_smc"]
    evaluator_path: Literal["score_prior_nll"]
    counted_targets: int
    forward_evidence_sha256: str
    source_factor_sha256: str
    score_trace_sha256: str
    nll_totals_sha256: str
    trace_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != "wt103-synthetic-arm-execution-trace-v1"
            or type(self.forward_path) is not str
            or not self.forward_path
            or type(self.prior_path) is not str
            or not self.prior_path
            or self.scorer_path
            not in ("exact_autoregressive", "weighted_smc")
            or self.evaluator_path != "score_prior_nll"
            or type(self.counted_targets) is not int
            or self.counted_targets != 3
        ):
            raise ValueError("synthetic arm execution trace is inconsistent")
        for name in (
            "forward_evidence_sha256",
            "source_factor_sha256",
            "score_trace_sha256",
            "nll_totals_sha256",
            "trace_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        expected = owned_sha256(
            "vfe4.wt103.synthetic-arm-execution-trace.v1",
            _semantic_payload(self, omit=("trace_sha256",)),
        )
        if self.trace_sha256 != expected:
            raise ValueError("synthetic arm execution trace hash does not match")

    @classmethod
    def create(cls, **values: object) -> "SyntheticArmExecutionTrace":
        payload = {
            "schema_version": (
                "wt103-synthetic-arm-execution-trace-v1"
            ),
            **values,
        }
        return cls(
            **payload,
            trace_sha256=owned_sha256(
                "vfe4.wt103.synthetic-arm-execution-trace.v1",
                payload,
            ),
        )  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class SyntheticRuntimeObservation:
    """Runtime proof collected inside the isolated smoke worker."""

    schema_version: Literal["wt103-synthetic-runtime-observation-v1"]
    execution_mode: Literal["isolated_subprocess"]
    parent_process_id: int
    worker_process_id: int
    intraop_threads: Literal[1]
    interop_threads: Literal[1]
    thread_environment: tuple[tuple[str, str], ...]
    cuda_visible_devices: Literal["-1"]
    cuda_available: Literal[False]
    cuda_initialized_on_entry: Literal[False]
    cuda_initialized_on_exit: Literal[False]
    observation_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != "wt103-synthetic-runtime-observation-v1"
            or self.execution_mode != "isolated_subprocess"
            or type(self.parent_process_id) is not int
            or self.parent_process_id <= 0
            or type(self.worker_process_id) is not int
            or self.worker_process_id <= 0
            or self.parent_process_id == self.worker_process_id
            or self.intraop_threads != 1
            or self.interop_threads != 1
            or self.thread_environment
            != tuple((name, "1") for name in _THREAD_ENVIRONMENT_NAMES)
            or self.cuda_visible_devices != "-1"
            or self.cuda_available is not False
            or self.cuda_initialized_on_entry is not False
            or self.cuda_initialized_on_exit is not False
        ):
            raise ValueError("synthetic runtime observation is inconsistent")
        expected = owned_sha256(
            "vfe4.wt103.synthetic-runtime-observation.v1",
            _semantic_payload(self, omit=("observation_sha256",)),
        )
        _require_sha256(self.observation_sha256, "observation_sha256")
        if self.observation_sha256 != expected:
            raise ValueError("synthetic runtime observation hash does not match")

    @classmethod
    def create(cls, **values: object) -> "SyntheticRuntimeObservation":
        payload = {
            "schema_version": (
                "wt103-synthetic-runtime-observation-v1"
            ),
            "execution_mode": "isolated_subprocess",
            **values,
        }
        return cls(
            **payload,
            observation_sha256=owned_sha256(
                "vfe4.wt103.synthetic-runtime-observation.v1",
                payload,
            ),
        )  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class SyntheticSmokeArmResult:
    """Typed identity of one arm's bounded nonproduction run."""

    schema_version: Literal["wt103-synthetic-smoke-arm-v1"]
    authority: Literal["nonproduction_synthetic_smoke"]
    arm_id: str
    arm_spec_sha256: str
    constructor_id: str
    build_sha256: str
    scorer_kind: str
    execution_trace: SyntheticArmExecutionTrace
    update_phase_order: tuple[str, ...]
    accepted_update: bool
    validation_completed: bool
    terminal_checkpoint_role: Literal["terminal_scoring"]
    terminal_checkpoint_identity_sha256: str
    terminal_scientific_state_sha256: str
    metrics_jsonl_path: str
    metrics_jsonl_sha256: str
    metrics_csv_path: str
    metrics_csv_sha256: str
    run_manifest_path: str
    run_manifest_sha256: str
    resume_exercised: bool
    result_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != "wt103-synthetic-smoke-arm-v1"
            or self.authority != _AUTHORITY
            or self.constructor_id
            not in {
                "build_wt103_a0",
                "build_wt103_a5_parent_specific",
                "build_wt103_a5_fixed",
                "build_wt103_a5_nolatent",
            }
            or type(self.update_phase_order) is not tuple
            or not self.update_phase_order
            or type(self.execution_trace) is not SyntheticArmExecutionTrace
            or self.accepted_update is not True
            or self.validation_completed is not True
            or self.terminal_checkpoint_role != "terminal_scoring"
            or type(self.resume_exercised) is not bool
        ):
            raise ValueError("synthetic smoke arm result is inconsistent")
        self.execution_trace.__post_init__()
        if self.execution_trace.scorer_path != self.scorer_kind:
            raise ValueError("execution trace differs from factory scorer")
        for name in (
            "arm_spec_sha256",
            "build_sha256",
            "terminal_checkpoint_identity_sha256",
            "terminal_scientific_state_sha256",
            "metrics_jsonl_sha256",
            "metrics_csv_sha256",
            "run_manifest_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        for name in (
            "arm_id",
            "scorer_kind",
            "metrics_jsonl_path",
            "metrics_csv_path",
            "run_manifest_path",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be nonempty text")
        expected = owned_sha256(
            "vfe4.wt103.synthetic-smoke-arm-result.v1",
            _semantic_payload(self, omit=("result_sha256",)),
        )
        _require_sha256(self.result_sha256, "result_sha256")
        if self.result_sha256 != expected:
            raise ValueError("synthetic smoke arm result hash does not match")

    @classmethod
    def create(cls, **values: object) -> "SyntheticSmokeArmResult":
        payload = {
            "schema_version": "wt103-synthetic-smoke-arm-v1",
            "authority": _AUTHORITY,
            **values,
        }
        return cls(
            **payload,
            result_sha256=owned_sha256(
                "vfe4.wt103.synthetic-smoke-arm-result.v1",
                payload,
            ),
        )  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class WT103SyntheticSmokeResult:
    """Closed result of the hermetic Task 12 integration smoke."""

    schema_version: Literal["wt103-synthetic-smoke-result-v1"]
    authority: Literal["nonproduction_synthetic_smoke"]
    smoke_run_id: str
    config_sha256: str
    cache_fixture_path: str
    cache_fixture_sha256: str
    factory_set_sha256: str
    arm_results: tuple[SyntheticSmokeArmResult, ...]
    resume_checkpoint_role: Literal["resume_only"]
    resume_checkpoint_identity_sha256: str
    resume_identity_before_sha256: str
    resume_identity_after_sha256: str
    resume_scientific_state_before_sha256: str
    resume_scientific_state_after_sha256: str
    resume_next_predictions_equal: Literal[True]
    resume_oracle_passed: Literal[True]
    resume_uninterrupted_terminal_scientific_state_sha256: str
    resume_resumed_terminal_scientific_state_sha256: str
    resume_uninterrupted_metrics_jsonl_sha256: str
    resume_resumed_metrics_jsonl_sha256: str
    resume_uninterrupted_next_predictions_equal: Literal[True]
    runtime_observation: SyntheticRuntimeObservation
    experiment_plan_path: str
    experiment_plan_sha256: str
    experiment_index_path: str
    experiment_index_sha256: str
    experiment_index_stage: Literal["pretest"]
    production_readiness_eligible: Literal[False]
    heldout_test_opened: Literal[False]
    result_sha256: str

    def __post_init__(self) -> None:
        expected_constructors = (
            "build_wt103_a0",
            "build_wt103_a5_parent_specific",
            "build_wt103_a5_fixed",
            "build_wt103_a5_parent_specific",
            "build_wt103_a5_nolatent",
        )
        if (
            self.schema_version != "wt103-synthetic-smoke-result-v1"
            or self.authority != _AUTHORITY
            or type(self.smoke_run_id) is not str
            or not self.smoke_run_id
            or type(self.arm_results) is not tuple
            or len(self.arm_results) != 5
            or any(
                type(item) is not SyntheticSmokeArmResult
                for item in self.arm_results
            )
            or tuple(item.constructor_id for item in self.arm_results)
            != expected_constructors
            or len(set(expected_constructors)) != 4
            or sum(item.resume_exercised for item in self.arm_results) != 1
            or self.resume_checkpoint_role != "resume_only"
            or self.resume_identity_before_sha256
            != self.resume_identity_after_sha256
            or self.resume_scientific_state_before_sha256
            != self.resume_scientific_state_after_sha256
            or self.resume_next_predictions_equal is not True
            or self.resume_oracle_passed is not True
            or self.resume_uninterrupted_terminal_scientific_state_sha256
            != self.resume_resumed_terminal_scientific_state_sha256
            or self.resume_uninterrupted_metrics_jsonl_sha256
            != self.resume_resumed_metrics_jsonl_sha256
            or self.resume_uninterrupted_next_predictions_equal is not True
            or type(self.runtime_observation)
            is not SyntheticRuntimeObservation
            or self.experiment_index_stage != "pretest"
            or self.production_readiness_eligible is not False
            or self.heldout_test_opened is not False
        ):
            raise ValueError("synthetic smoke result is inconsistent")
        self.runtime_observation.__post_init__()
        for arm_result in self.arm_results:
            arm_result.__post_init__()
            if arm_result.authority != self.authority:
                raise ValueError("smoke arm authority differs from result")
        for name in (
            "config_sha256",
            "cache_fixture_sha256",
            "factory_set_sha256",
            "resume_checkpoint_identity_sha256",
            "resume_identity_before_sha256",
            "resume_identity_after_sha256",
            "resume_scientific_state_before_sha256",
            "resume_scientific_state_after_sha256",
            "resume_uninterrupted_terminal_scientific_state_sha256",
            "resume_resumed_terminal_scientific_state_sha256",
            "resume_uninterrupted_metrics_jsonl_sha256",
            "resume_resumed_metrics_jsonl_sha256",
            "experiment_plan_sha256",
            "experiment_index_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        for name in (
            "cache_fixture_path",
            "experiment_plan_path",
            "experiment_index_path",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be nonempty text")
        expected = owned_sha256(
            "vfe4.wt103.synthetic-smoke-result.v1",
            _semantic_payload(self, omit=("result_sha256",)),
        )
        _require_sha256(self.result_sha256, "result_sha256")
        if self.result_sha256 != expected:
            raise ValueError("synthetic smoke result hash does not match")

    @classmethod
    def create(cls, **values: object) -> "WT103SyntheticSmokeResult":
        payload = {
            "schema_version": "wt103-synthetic-smoke-result-v1",
            "authority": _AUTHORITY,
            "production_readiness_eligible": False,
            "heldout_test_opened": False,
            **values,
        }
        return cls(
            **payload,
            result_sha256=owned_sha256(
                "vfe4.wt103.synthetic-smoke-result.v1",
                payload,
            ),
        )  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class _ArmScienceEvidence:
    trace: SyntheticArmExecutionTrace
    totals: WT103NllTotals
    source_entropy: float
    source_support_size: int


@dataclass(frozen=True, slots=True)
class _ResumeOracleEvidence:
    uninterrupted_terminal_scientific_state_sha256: str
    resumed_terminal_scientific_state_sha256: str
    uninterrupted_metrics_jsonl_sha256: str
    resumed_metrics_jsonl_sha256: str
    uninterrupted_next_predictions_equal: Literal[True]

    def __post_init__(self) -> None:
        for name in (
            "uninterrupted_terminal_scientific_state_sha256",
            "resumed_terminal_scientific_state_sha256",
            "uninterrupted_metrics_jsonl_sha256",
            "resumed_metrics_jsonl_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if (
            self.uninterrupted_terminal_scientific_state_sha256
            != self.resumed_terminal_scientific_state_sha256
            or self.uninterrupted_metrics_jsonl_sha256
            != self.resumed_metrics_jsonl_sha256
            or self.uninterrupted_next_predictions_equal is not True
        ):
            raise ValueError("resume oracle did not reproduce uninterrupted state")


class _TinyTokenizerAdapter:
    distribution_name = "tiktoken"
    distribution_version = "0.12.0"
    encoding_name = "gpt2"
    vocabulary_size = 3
    special_tokens = (("<|endoftext|>", 2),)
    regex_pattern_sha256 = hashlib.sha256(
        b"task12-generated-tokenizer-regex"
    ).hexdigest()
    mergeable_ranks_sha256 = hashlib.sha256(
        b"task12-generated-tokenizer-ranks"
    ).hexdigest()
    ordinary_encoding_policy = "encode_ordinary_no_special_tokens"
    fitted_state_sha256 = None
    implementation_sha256 = hashlib.sha256(
        b"vfe4.task12.generated-tokenizer-adapter.v1"
    ).hexdigest()
    _ENCODE = {"a": 0, "b": 1, "c": 2}
    _DECODE = ("a", "b", "c")

    def encode_ordinary(self, text: str) -> tuple[int, ...]:
        return tuple(self._ENCODE[character] for character in text)

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return "".join(self._DECODE[token_id] for token_id in token_ids)


class _TinyCausalDecoder(nn.Module):
    """Small target-blind causal decoder used only by the generated smoke."""

    def __init__(self, vocabulary: VocabularyIdentity) -> None:
        super().__init__()
        self.vocabulary = vocabulary
        width = 4
        self.token_embedding = nn.Embedding(
            vocabulary.size, width, dtype=torch.float64
        )
        self.position_embedding = nn.Embedding(4, width, dtype=torch.float64)
        self.qkv = nn.Linear(width, 3 * width, dtype=torch.float64)
        self.output = nn.Linear(width, width, dtype=torch.float64)
        self.decoder = nn.Linear(width, vocabulary.size, dtype=torch.float64)

    def forward(self, prefix: CausalPrefix) -> torch.Tensor:
        prefix.__post_init__()
        if prefix.vocabulary != self.vocabulary:
            raise ValueError("causal decoder received a foreign vocabulary")
        token_ids = prefix.token_ids
        positions = torch.arange(token_ids.numel(), dtype=torch.int64)
        hidden = self.token_embedding(token_ids) + self.position_embedding(
            positions
        )
        query, key, value = self.qkv(hidden).chunk(3, dim=-1)
        scores = query @ key.transpose(0, 1) / math.sqrt(hidden.shape[-1])
        mask = torch.triu(
            torch.ones_like(scores, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(mask, -torch.inf)
        attended = torch.softmax(scores, dim=-1) @ value
        return self.decoder((hidden + self.output(attended))[-1])


class _TinyMeanPooledPrefix(nn.Module):
    """Distinct no-latent mean-pooled autoregressive control."""

    def __init__(self, vocabulary: VocabularyIdentity) -> None:
        super().__init__()
        self.vocabulary = vocabulary
        self.embedding = nn.Embedding(
            vocabulary.size, 4, dtype=torch.float64
        )
        self.decoder = nn.Linear(4, vocabulary.size, dtype=torch.float64)

    def forward(self, prefix: CausalPrefix) -> torch.Tensor:
        prefix.__post_init__()
        if prefix.vocabulary != self.vocabulary:
            raise ValueError("mean-pooled model received a foreign vocabulary")
        return self.decoder(self.embedding(prefix.token_ids).mean(dim=0))


class _ExactAutoregressivePredictor:
    """Typed exact predictor backed by one actual target-blind forward model."""

    def __init__(
        self,
        *,
        model: _TinyCausalDecoder | _TinyMeanPooledPrefix,
        path: str,
    ) -> None:
        self.model = model
        self.vocabulary = model.vocabulary
        self.estimator_spec = EstimatorSpec.create(
            kind="deterministic_exact",
            particle_count=None,
            resampling="none",
        )
        self.estimator_identity = EstimatorIdentity.from_spec(
            self.estimator_spec
        )
        self.predictor_config_sha256 = _digest(
            f"exact-predictor:{path}"
        )
        self.data_safety_sha256 = _digest("generated-validation-only")
        self.vocabulary_sha256 = vocabulary_identity_sha256(
            self.vocabulary
        )
        self.model_family_sha256 = _digest(path)
        self.model_state_sha256 = canonical_model_state_sha256(model)
        self.proposal_identity_sha256 = owned_sha256(
            "vfe4.wt103.synthetic-exact-proposal.v1",
            {
                "path": path,
                "model_state_sha256": self.model_state_sha256,
            },
        )

    def next_token_log_probs(
        self,
        prefix_tokens: CausalPrefix,
        estimator_rng: EstimatorStream,
        cache: PrefixCache | None = None,
    ) -> PriorPrediction:
        del cache
        prefix_tokens.__post_init__()
        estimator_rng.__post_init__()
        if (
            estimator_rng.estimator_identity_sha256
            != self.estimator_identity.identity_sha256
        ):
            raise ValueError("exact predictor received a foreign stream")
        log_probs = torch.log_softmax(
            self.model(prefix_tokens),
            dim=0,
        )
        population = ProposalPopulation.create(
            {"exact_hidden": torch.zeros((1, 1), dtype=torch.float64)}
        )
        step = ProposalStep.create(
            position=prefix_tokens.receiver_t,
            population=population,
            emission_log_probs=log_probs.unsqueeze(0),
            counter_consumption=(),
            proposal_identity_sha256=self.proposal_identity_sha256,
        )
        pending = PendingPrediction.create(
            prefix_sha256=prefix_tokens.prefix_sha256,
            step=step,
            parent_log_weights=torch.zeros(1, dtype=torch.float64),
            prediction_log_probs=log_probs,
        )
        key = PrefixCacheKey.create(
            prefix=prefix_tokens,
            vocabulary_sha256=self.vocabulary_sha256,
            predictor_config_sha256=self.predictor_config_sha256,
            model_family_sha256=self.model_family_sha256,
            model_state_sha256=self.model_state_sha256,
            proposal_identity_sha256=self.proposal_identity_sha256,
            estimator_semantic_sha256=(
                self.estimator_identity.semantic_sha256
            ),
            estimator_artifact_bytes_sha256=(
                self.estimator_identity.artifact_bytes_sha256
            ),
            estimator_stream_sha256=estimator_rng.stream_sha256,
            data_safety_sha256=self.data_safety_sha256,
        )
        assimilations = tuple(
            AssimilationRecord.create(
                position=position,
                observed_token=token_id,
                incremental_log_normalizer=0.0,
                ess=1.0,
                ancestors=(),
                resampling_consumption=None,
            )
            for position, token_id in enumerate(
                (
                    int(value)
                    for value in prefix_tokens.token_ids.tolist()
                ),
                start=1,
            )
        )
        exact_cache = PrefixCache.create(
            key=key,
            filtered_population=population,
            filtered_log_weights=torch.zeros(1, dtype=torch.float64),
            cumulative_log_normalizer=0.0,
            pending=pending,
            assimilations=assimilations,
            counter_consumption=(),
        )
        record = EstimatorRecord.from_cache(
            stream=estimator_rng,
            cache=exact_cache,
        )
        return PriorPrediction.create(
            vocabulary=self.vocabulary,
            log_probs=log_probs,
            cache=exact_cache,
            estimator_record=record,
        )


def _tensor_sha256(value: torch.Tensor) -> str:
    raw = bytes(
        value.detach()
        .to(device="cpu")
        .contiguous()
        .view(torch.uint8)
        .reshape(-1)
        .tolist()
    )
    return hashlib.sha256(raw).hexdigest()


def _generated_evaluation_batches(
    *,
    fixture_root: Path,
    backend: DurabilityBackend,
) -> tuple[WT103EvaluationBatches, VocabularyIdentity]:
    adapter = _TinyTokenizerAdapter()
    contract = SyntheticTokenizerFixtureContract.create(
        distribution_name=adapter.distribution_name,
        distribution_version=adapter.distribution_version,
        encoding_name=adapter.encoding_name,
        vocabulary_size=adapter.vocabulary_size,
        special_tokens=adapter.special_tokens,
        regex_pattern_sha256=adapter.regex_pattern_sha256,
        mergeable_ranks_sha256=adapter.mergeable_ranks_sha256,
        ordinary_encoding_policy=adapter.ordinary_encoding_policy,
        golden_vectors=(
            ("ascii", "abc", (0, 1, 2)),
            ("unicode", "cab", (2, 0, 1)),
            ("newlines", "bca", (1, 2, 0)),
        ),
    )
    tokenizer = build_synthetic_fixture_tokenizer_spec(
        contract,
        adapter,
    )
    raw = b"abca"
    record = encode_fixture_split_record(
        split="validation",
        raw_bytes=raw,
        raw_parent_sha256=hashlib.sha256(raw).hexdigest(),
        spec=tokenizer,
        fixture_contract=contract,
        adapter=adapter,
        cache_root=fixture_root / "token-cache",
        durability_backend=backend,
        available_disk_bytes=20 * 1024**3,
        available_host_ram_bytes=20 * 1024**3,
        smoke_token_bytes_per_raw_byte=4.0,
        smoke_ram_bytes_per_raw_byte=2.0,
    )
    capability = issue_fixture_split_capability(
        allowed_splits=("validation",),
        cache_identities=(record,),
    )
    windows = materialize_causal_window_set(
        cache_record=record,
        tokenizer_spec=tokenizer,
        cache_root=fixture_root / "token-cache",
        split_capability=capability,
        artifact_root=fixture_root / "window-artifacts",
        durability_backend=backend,
    )
    evaluation = WT103EvaluationBatches.create(
        windows=windows,
        schedule=build_evaluation_schedule(windows),
    )
    vocabulary = VocabularyIdentity(
        "task12-generated-three-token-vocabulary-v1",
        3,
        tokenizer.spec_sha256,
    )
    return evaluation, vocabulary


def _language_structure() -> H6LanguageStructure:
    dag = CausalDag.create(
        node_labels=(0, 1, 2, 3, 4),
        rows=tuple(
            CausalDagRow(receiver_t, tuple(range(receiver_t)))
            for receiver_t in range(1, 5)
        ),
    )
    return H6LanguageStructure.create(
        base=ZeroDimensionalBase.create(),
        dag=dag,
        receiver_labels=(1, 2, 3, 4),
    )


def _language_model(
    *,
    prior_variant: str,
    vocabulary: VocabularyIdentity,
) -> LanguageGenerativeModel:
    structure = _language_structure()
    family = _digest(f"tiny-language-generative:{prior_variant}")
    if prior_variant == "fixed":
        rows = tuple(
            torch.linspace(
                -0.2,
                0.2,
                receiver_t,
                dtype=torch.float64,
            )
            for receiver_t in range(1, 5)
        )
        prior = FixedSourcePrior(
            structure=structure,
            vocabulary=vocabulary,
            fixture_sha256=_digest("tiny-language-fixture"),
            predictor_config_sha256=_digest("tiny-language-predictor"),
            model_family_sha256=family,
            state_logits=rows,
            model_logits=tuple(row.flip(0) for row in rows),
        )
    elif prior_variant == "parent_specific_pooled_prefix":
        prior = ParentSpecificPooledPrefixSourcePrior(
            structure=structure,
            vocabulary=vocabulary,
            fixture_sha256=_digest("tiny-language-fixture"),
            predictor_config_sha256=_digest("tiny-language-predictor"),
            model_family_sha256=family,
            latent_dim=2,
            context_dim=2,
        )
    else:
        raise ValueError("synthetic language model prior is unsupported")
    model = LanguageGenerativeModel(
        structure=structure,
        vocabulary=vocabulary,
        model_family_sha256=family,
        latent_dim=2,
        source_prior=prior,
    )
    with torch.no_grad():
        model.initial_log_scale.fill_(-0.4)
        model.model_transition_log_scale.fill_(-0.7)
        model.state_transition_log_scale.fill_(-0.6)
        model.emission_state_weight.copy_(
            torch.tensor(
                ((0.25, -0.1), (-0.15, 0.2), (0.05, 0.05)),
                dtype=torch.float64,
            )
        )
        model.emission_model_weight.copy_(
            torch.tensor(
                ((0.1, 0.05), (0.0, -0.1), (-0.1, 0.05)),
                dtype=torch.float64,
            )
        )
        model.emission_bias.copy_(
            torch.tensor((0.08, -0.03, -0.05), dtype=torch.float64)
        )
    return model


def _arm_science_evidence(
    *,
    index: int,
    spec: WT103ArmSpec,
    evaluation: WT103EvaluationBatches,
    vocabulary: VocabularyIdentity,
) -> _ArmScienceEvidence:
    protocol = EstimatorProtocol.create()
    source_entropy = 0.0
    source_support_size = 0
    if spec.scorer_kind == "exact_autoregressive":
        if index == 0:
            model: _TinyCausalDecoder | _TinyMeanPooledPrefix = (
                _TinyCausalDecoder(vocabulary)
            )
            forward_path = "wt103_a0_decoder_cross_entropy"
        elif index == 4:
            model = _TinyMeanPooledPrefix(vocabulary)
            forward_path = "mean_pooled_prefix_cross_entropy"
        else:
            raise RuntimeError("exact scorer is bound to the wrong arm")
        predictor = _ExactAutoregressivePredictor(
            model=model,
            path=forward_path,
        )
        probe_prefix = CausalPrefix.create(
            receiver_t=4,
            vocabulary=vocabulary,
            token_ids=torch.tensor((0, 1, 2), dtype=torch.int64),
        )
        forward_evidence_sha256 = _tensor_sha256(
            torch.log_softmax(model(probe_prefix), dim=0)
        )
        source_factor_sha256 = _digest("source-factor-absent")
        logical_stream_id = None
    else:
        language = _language_model(
            prior_variant=spec.prior_variant,
            vocabulary=vocabulary,
        )
        proposal = LanguageGenerativeProposalAdapter(language)
        estimator_spec = EstimatorSpec.create(
            kind="weighted_smc",
            particle_count=128,
            resampling="systematic_ess_half",
        )
        estimator_identity = EstimatorIdentity.from_spec(estimator_spec)
        predictor = BootstrapSmcPredictor(
            proposal=proposal,
            estimator_spec=estimator_spec,
            estimator_identity=estimator_identity,
            predictor_config_sha256=_digest("tiny-language-predictor"),
            data_safety_sha256=_digest("generated-validation-only"),
        )
        prefix = CausalPrefix.create(
            receiver_t=3,
            vocabulary=vocabulary,
            token_ids=torch.tensor((0, 1), dtype=torch.int64),
        )
        earlier = torch.tensor(
            ((0.0, 0.1), (0.2, -0.1), (0.3, 0.05)),
            dtype=torch.float64,
        )
        if type(language.source_prior) is FixedSourcePrior:
            source_factor = language.state_source_log_probs(receiver_t=3)
        elif type(language.source_prior) is ParentSpecificPooledPrefixSourcePrior:
            source_factor = language.state_source_log_probs(
                receiver_t=3,
                prefix=prefix,
                earlier_latents=earlier,
            )
        else:
            raise RuntimeError("latent smoke omitted its source prior")
        source_factor_sha256 = source_factor.factor_identity_sha256
        source_values = source_factor.log_probs.value()
        finite = torch.isfinite(source_values)
        probabilities = torch.exp(source_values[finite])
        source_entropy = float(
            -torch.sum(probabilities * source_values[finite]).item()
        )
        source_support_size = int(torch.sum(finite).item())
        emission = language.emission_log_probs(
            receiver_t=3,
            current_state=torch.tensor((0.2, -0.1), dtype=torch.float64),
            current_model=torch.tensor((-0.3, 0.4), dtype=torch.float64),
        )
        initial = language.initial_log_prob(
            initial_latents=torch.zeros((2, 2), dtype=torch.float64)
        )
        forward_evidence_sha256 = owned_sha256(
            "vfe4.wt103.synthetic-language-forward.v1",
            (
                initial.factor_identity_sha256,
                source_factor.factor_identity_sha256,
                emission.factor_identity_sha256,
                spec.training_objective,
            ),
        )
        forward_path = (
            "language_generative_emission_only_non_elbo"
            if spec.training_objective
            == "emission_only_ablation_non_elbo"
            else "language_generative_complete_elbo"
        )
        logical_stream_id = 0
    stream_seed = wt103_estimator_stream_seed(
        split="validation",
        estimator_protocol_sha256=protocol.protocol_sha256,
        logical_stream_id=logical_stream_id,
    )
    stream = EstimatorStream.create(
        stream_seed=stream_seed,
        estimator_identity=predictor.estimator_identity,
    )
    binding = WT103EstimatorStreamBinding.create(
        split="validation",
        logical_stream_id=logical_stream_id,
        estimator_protocol=protocol,
        stream=stream,
    )
    bound = bind_wt103_prior_predictor(predictor, binding)
    totals = score_prior_nll(bound, evaluation, stream)
    score_trace = wt103_score_trace(totals)
    trace = SyntheticArmExecutionTrace.create(
        forward_path=forward_path,
        prior_path=spec.prior_variant,
        scorer_path=spec.scorer_kind,
        evaluator_path="score_prior_nll",
        counted_targets=totals.counted_targets,
        forward_evidence_sha256=forward_evidence_sha256,
        source_factor_sha256=source_factor_sha256,
        score_trace_sha256=score_trace.trace_sha256,
        nll_totals_sha256=totals.totals_sha256,
    )
    return _ArmScienceEvidence(
        trace=trace,
        totals=totals,
        source_entropy=source_entropy,
        source_support_size=source_support_size,
    )


@dataclass(frozen=True, slots=True)
class _SyntheticBatch:
    context: torch.Tensor
    target: torch.Tensor

    def __post_init__(self) -> None:
        if any(
            type(value) is not torch.Tensor
            or value.device.type != "cpu"
            or value.dtype is not torch.float64
            or value.numel() != 1
            for value in (self.context, self.target)
        ):
            raise ValueError("synthetic batch must contain two CPU float64 scalars")


class _SyntheticNoLatentModel(nn.Module):
    def __init__(self, value: float = 0.25) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value, dtype=torch.float64))


class _SyntheticModelBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.primary = nn.Parameter(torch.tensor(0.20, dtype=torch.float64))
        self.latent = nn.Parameter(torch.tensor(0.30, dtype=torch.float64))
        self.source = nn.Parameter(torch.tensor(0.40, dtype=torch.float64))


class _SyntheticRecognition(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(0.50, dtype=torch.float64))


class _SyntheticLatentContainer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _SyntheticModelBlock()
        self.recognition = _SyntheticRecognition()


def _latent_factory_components() -> WT103ArmRuntimeComponents:
    container = _SyntheticLatentContainer()
    return WT103ArmRuntimeComponents.create(
        model=container,
        model_parameter_names=("model.primary",),
        latent_parameter_names=("model.latent",),
        source_parameter_names=("model.source",),
        frame_parameter_names=(),
        recognition_parameter_names=("recognition.value",),
        optimizer_bindings=(
            OptimizerParameterBinding(
                optimizer_id="recognition_adamw",
                parameter_names=("recognition.value",),
            ),
            OptimizerParameterBinding(
                optimizer_id="model_adamw",
                parameter_names=(
                    "model.primary",
                    "model.latent",
                    "model.source",
                ),
            ),
        ),
        filler_parameter_names=(),
        dormant_parameter_names=(),
    )


def _nolatent_factory_components() -> WT103ArmRuntimeComponents:
    model = _SyntheticNoLatentModel()
    return WT103ArmRuntimeComponents.create(
        model=model,
        model_parameter_names=("value",),
        latent_parameter_names=(),
        source_parameter_names=(),
        frame_parameter_names=(),
        recognition_parameter_names=(),
        optimizer_bindings=(
            OptimizerParameterBinding(
                optimizer_id="model_adamw",
                parameter_names=("value",),
            ),
        ),
        filler_parameter_names=(),
        dormant_parameter_names=(),
    )


def _a0_factory_inputs(config: TrainingConfig) -> A0FactoryInputs:
    profile = config.profile
    workload = A0FlopWorkload(
        batch_size=1,
        sequence_length=2,
        vocabulary_size=profile.vocabulary_size,
        hidden_width=20,
        parameter_count=2_068_197,
        decoder_chunk_size=2,
        optimizer_steps=1,
        validation_batches=1,
    )
    ledger = reconstruct_a0_flops(workload)
    matching = audit_arm_matching(
        profile=profile,
        endpoint_inventory=config.endpoint_inventory,
        primary_parameter_count=2_068_197,
        primary_semantic_train_flops=ledger.semantic_train_flops,
        workload_template=workload,
        optimizer_access_exact=True,
    )
    if (
        matching.status is not GateStatus.PASS
        or matching.selected_hidden_width != 20
    ):
        raise RuntimeError("synthetic smoke could not bind the exact A0 row")
    meta_model = WT103A0Model(
        vocabulary_size=profile.vocabulary_size,
        positional_capacity=profile.sequence_length,
        hidden_width=20,
        attention_heads=2,
        layer_norm_epsilon=1.0e-5,
        device=torch.device("meta"),
        dtype=torch.float32,
    )
    inventory = reconstruct_a0_parameters(
        meta_model,
        vocabulary_size=profile.vocabulary_size,
        positional_capacity=profile.sequence_length,
        hidden_width=20,
    )
    formula = build_a0_formula_record(inventory=inventory, ledger=ledger)
    architecture = build_a0_architecture_profile(
        hidden_width=20,
        formula=formula,
        source_lock_scope="candidate_unverified",
        pytorch_version="unresolved_until_task13_source_lock",
        sdpa_api_sha256="unresolved_until_task13_source_lock",
        flash_backend_sha256="unresolved_until_task13_source_lock",
    )
    return A0FactoryInputs(
        architecture=architecture,
        formula=formula,
        flop_ledger=ledger,
        matching=matching,
        device=torch.device("meta"),
        dtype=torch.float32,
    )


def _build_exact_factories(
    config: TrainingConfig,
) -> tuple[BuiltWT103Arm, ...]:
    arms = config.endpoint_inventory.arms
    a0_inputs = _a0_factory_inputs(config)
    builds: list[BuiltWT103Arm] = []
    for index, spec in enumerate(arms):
        runtime = None
        if index in (1, 2, 3):
            runtime = _latent_factory_components()
        elif index == 4:
            runtime = _nolatent_factory_components()
        builds.append(
            build_wt103_arm(
                spec=spec,
                profile=config.profile,
                a0_inputs=a0_inputs if index == 0 else None,
                runtime=runtime,
                execution_scope=_AUTHORITY,
            )
        )
    result = tuple(builds)
    WT103FactorySetIdentity.create(result).__post_init__()
    return result


def _adamw(module: nn.Module) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        module.parameters(),
        lr=0.01,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.0,
        amsgrad=False,
        foreach=False,
        fused=False,
    )


def _sum_parameters(module: nn.Module) -> torch.Tensor:
    values = tuple(module.parameters())
    if not values:
        raise RuntimeError("synthetic module parameter inventory is empty")
    total = values[0].sum()
    for value in values[1:]:
        total = total + value.sum()
    return total


def _make_compute_terms(
    *,
    spec: WT103ArmSpec,
    model: nn.Module,
    recognition: nn.Module | None,
    science: _ArmScienceEvidence,
):
    def compute(
        phase: str,
        batch_value: object,
        snapshot: RecognitionSnapshot | None,
    ) -> ForwardTerms:
        if type(batch_value) is not _SyntheticBatch:
            raise TypeError("training smoke accepts only _SyntheticBatch")
        batch_value.__post_init__()
        model_total = _sum_parameters(model)
        if not spec.latent_enabled:
            if phase != "model_ce_adam_proposal" or snapshot is not None:
                raise RuntimeError("nonlatent smoke phase is not exact")
            coefficient = (
                0.125
                if science.trace.forward_path
                == "wt103_a0_decoder_cross_entropy"
                else 0.25
            )
            loss = (
                model_total
                + coefficient * batch_value.context
                - batch_value.target
                + 0.001 * science.totals.nll_per_token
            ).square()
            return ForwardTerms.cross_entropy(
                value=loss,
                counted_targets=1,
            )

        if recognition is None:
            raise RuntimeError("latent smoke omitted recognition state")
        if phase == "recognition_adam_proposal":
            if snapshot is not None:
                raise RuntimeError("recognition proposal cannot consume snapshot")
            recognition_total = _sum_parameters(recognition)
        elif phase == "model_adam_proposal":
            if snapshot is None:
                raise RuntimeError("model proposal requires detached snapshot")
            snapshot.assert_nonaliasing(recognition)
            recognition_total = snapshot.tensor("value").reshape(())
        else:
            raise RuntimeError("latent smoke phase is not exact")
        residual = (
            model_total
            + recognition_total
            + 0.125 * batch_value.context
            - batch_value.target
            + 0.001 * science.totals.nll_per_token
            + 0.001 * science.source_entropy
        )
        emission = -residual.square()
        zero = (model_total + recognition_total).square() * 0.0
        if spec.training_objective == "emission_only_ablation_non_elbo":
            return ForwardTerms.emission_only(
                expected_log_emission=(emission,),
                counted_targets=1,
            )
        if spec.training_objective != "complete_elbo":
            raise RuntimeError("latent smoke objective is not exact")
        model_source_cross_entropy = (
            0.005 * recognition_total.square()
        )
        state_source_cross_entropy = (
            0.003 * recognition_total.square()
        )
        continuous_recognition_entropy = 0.001 * recognition_total
        return ForwardTerms.complete_elbo(
            expected_log_emission=(emission,),
            initial_model_cross_entropy=0.01 * model_total.square(),
            initial_state_cross_entropy=(
                0.01 * recognition_total.square()
            ),
            model_source_cross_entropy=(model_source_cross_entropy,),
            model_transition_cross_entropy=(
                0.004 * recognition_total.square(),
            ),
            state_source_cross_entropy=(state_source_cross_entropy,),
            state_transition_cross_entropy=(
                0.002 * recognition_total.square(),
            ),
            model_source_kl=(model_source_cross_entropy,),
            state_source_kl=(state_source_cross_entropy,),
            continuous_recognition_entropy=(
                continuous_recognition_entropy
            ),
            conditional_source_entropy_estimate=zero,
            joint_recognition_entropy_estimate=(
                continuous_recognition_entropy
            ),
            estimator_error_bound=None,
            counted_targets=1,
        )

    return compute


def _engine_runtime(
    spec: WT103ArmSpec,
    *,
    build: BuiltWT103Arm | None,
    science: _ArmScienceEvidence,
) -> ArmExecutionRuntime:
    if spec.latent_enabled:
        if build is None:
            container = _SyntheticLatentContainer()
        else:
            candidate = build.runtime.model
            if type(candidate) is not _SyntheticLatentContainer:
                raise RuntimeError("latent factory runtime type is not exact")
            container = candidate
        model: nn.Module = container.model
        recognition: nn.Module | None = container.recognition
    else:
        if build is not None and type(build.runtime.model) is _SyntheticNoLatentModel:
            model = build.runtime.model
        else:
            model = _SyntheticNoLatentModel()
        recognition = None
    return ArmExecutionRuntime(
        arm_spec=spec,
        model=model,
        recognition=recognition,
        model_optimizer=_adamw(model),
        recognition_optimizer=(
            None if recognition is None else _adamw(recognition)
        ),
        model_scheduler=None,
        recognition_scheduler=None,
        grad_scaler=None,
        compute_terms=_make_compute_terms(
            spec=spec,
            model=model,
            recognition=recognition,
            science=science,
        ),
        support_validator=lambda: True,
        spd_validator=lambda: True,
        damping_observer=lambda: False,
        projection_observer=lambda: False,
        state_participants=(),
        gradient_clip_norm=1.0,
    )


def _target_blind_prediction(
    model: nn.Module,
    batch: _SyntheticBatch,
) -> torch.Tensor:
    return (
        _sum_parameters(model).detach()
        + 0.125 * batch.context.detach()
    ).reshape(1)


def _next_prediction_fixture(model: nn.Module) -> tuple[torch.Tensor, ...]:
    return (
        _target_blind_prediction(
            model,
            _SyntheticBatch(
                context=torch.tensor(0.75, dtype=torch.float64),
                target=torch.tensor(1.25, dtype=torch.float64),
            ),
        ),
        _target_blind_prediction(
            model,
            _SyntheticBatch(
                context=torch.tensor(1.25, dtype=torch.float64),
                target=torch.tensor(-9.0, dtype=torch.float64),
            ),
        ),
    )


def _objective_value(result: StepResult) -> float:
    if not result.objective_terms:
        raise RuntimeError("accepted synthetic step omitted objective diagnostics")
    if result.objective_kind == "complete_elbo":
        if result.complete_elbo_value is None:
            raise RuntimeError("complete smoke step omitted per-token ELBO")
        return float(result.complete_elbo_value)
    if result.objective_kind == "emission_only_ablation_non_elbo":
        numerator = result.objective_terms["emission_only_non_elbo"]
    else:
        numerator = result.objective_terms["cross_entropy_value"]
    if result.counted_targets is None:
        raise RuntimeError("synthetic objective omitted counted targets")
    return float(numerator / result.counted_targets)


def _metric_number(
    name: str,
    result: StepResult,
    science: _ArmScienceEvidence,
) -> float:
    terms = result.objective_terms or {}
    objective = _objective_value(result)
    counted = result.counted_targets
    if type(counted) is not int or counted <= 0:
        raise RuntimeError("synthetic metric step omitted counted targets")

    def per_token(name: str, fallback: float) -> float:
        return float(terms.get(name, fallback * counted)) / counted

    first_evidence = result.proposal_evidence[0]
    last_evidence = result.proposal_evidence[-1]
    first_control = result.update_controls[0]
    aliases = {
        "train_cross_entropy": per_token(
            "cross_entropy_value", objective
        ),
        "complete_elbo": (
            result.complete_elbo_value
            if result.complete_elbo_value is not None
            else objective
        ),
        "expected_log_emission": per_token(
            "expected_log_emission[0]", objective
        ),
        "initial_model_cross_entropy": per_token(
            "initial_model_cross_entropy", 0.0
        ),
        "initial_state_cross_entropy": per_token(
            "initial_state_cross_entropy", 0.0
        ),
        "model_source_cross_entropy": per_token(
            "model_source_cross_entropy[0]", 0.0
        ),
        "model_source_kl": per_token("model_source_kl[0]", 0.0),
        "model_transition_cross_entropy": per_token(
            "model_transition_cross_entropy[0]", 0.0
        ),
        "state_source_cross_entropy": per_token(
            "state_source_cross_entropy[0]", 0.0
        ),
        "state_source_kl": per_token("state_source_kl[0]", 0.0),
        "state_transition_cross_entropy": per_token(
            "state_transition_cross_entropy[0]", 0.0
        ),
        "continuous_recognition_entropy": per_token(
            "continuous_recognition_entropy", 0.0
        ),
        "conditional_source_entropy_estimate": per_token(
            "conditional_source_entropy_estimate", 0.0
        ),
        "joint_recognition_entropy_estimate": per_token(
            "joint_recognition_entropy_estimate", 0.0
        ),
        "estimator_error_bound": per_token(
            "estimator_error_bound", 0.0
        ),
        "emission_only_non_elbo": per_token(
            "emission_only_non_elbo", objective
        ),
        "prior_nll_sum": science.totals.summed_nll,
        "prior_nll_per_token": science.totals.nll_per_token,
        "perplexity": science.totals.perplexity,
        "estimator_stream": float(
            science.totals.estimator_stream_id or 0
        ),
        "particle_count": float(science.totals.particle_count or 0),
        "cache_audit_passed": 1.0,
        "source_entropy": science.source_entropy,
        "source_support_size": float(science.source_support_size),
        "effective_source_count": math.exp(science.source_entropy),
        "accepted_proposals": float(len(result.updates)),
        "rejected_proposals": 0.0,
        "acceptance_rate": 1.0,
        "damping_events": 0.0,
        "objective_before": float(first_evidence.objective_before_value or 0.0),
        "objective_after": float(last_evidence.objective_after_value or 0.0),
        "snapshot_identity_present": float(result.snapshot_sha256 is not None),
        "learning_rate": first_control.learning_rate,
        "scheduler_ordinal": float(first_control.scheduler_ordinal),
        "gradient_pre_clip_l2": float(first_control.pre_clip_norm or 0.0),
        "gradient_post_clip_l2": float(first_control.post_clip_norm or 0.0),
        "minimum_cholesky_pivot": 1.0,
        "failed_pivots": 0.0,
        "spd_projections": 0.0,
        "condition_estimate": 1.0,
        "solve_residual": 0.0,
        "gradient_l2": float(first_control.post_clip_norm or 0.0),
        "gradient_inf": float(
            first_control.post_clip_inf_norm or 0.0
        ),
        "counted_targets": float(result.counted_targets or 0),
        "tokens_per_second": 1.0,
        "data_wait_seconds": 0.0,
        "forward_seconds": 0.0,
        "inference_seconds": 0.0,
        "backward_seconds": 0.0,
        "update_seconds": 0.0,
        "evaluation_seconds": 0.0,
        "checkpoint_seconds": 0.0,
        "wall_seconds": 0.0,
        "process_rss_bytes": 0.0,
        "process_hwm_bytes": 0.0,
        "cuda_allocated_bytes": 0.0,
        "cuda_reserved_bytes": 0.0,
        "cuda_peak_allocated_bytes": 0.0,
        "cuda_peak_reserved_bytes": 0.0,
        "allocation_retries": 0.0,
        "oom_count": 0.0,
    }
    value = float(aliases[name])
    if not math.isfinite(value):
        raise RuntimeError(f"synthetic metric {name!r} is nonfinite")
    return value


def _proposal_objective_numerator(
    result: StepResult,
    *,
    before: bool,
) -> tuple[float, int]:
    evidence = (
        result.proposal_evidence[0]
        if before
        else result.proposal_evidence[-1]
    )
    rows = (
        evidence.objective_before_terms
        if before
        else evidence.objective_after_terms
    )
    if rows is None or evidence.counted_targets is None:
        raise RuntimeError("synthetic proposal omitted objective evidence")
    values = dict(rows)
    if result.objective_kind == "cross_entropy":
        numerator = -values["cross_entropy_value"]
    elif result.objective_kind == "complete_elbo":
        numerator = values["complete_elbo_numerator"]
    elif result.objective_kind == "emission_only_ablation_non_elbo":
        numerator = values["emission_only_non_elbo"]
    else:  # pragma: no cover - StepResult owns the closed inventory
        raise RuntimeError("synthetic proposal objective kind is unknown")
    return float(numerator), evidence.counted_targets


def _synthetic_metric_components(
    *,
    name: str,
    result: StepResult,
    science: _ArmScienceEvidence,
    value: float,
) -> tuple[float | None, int | None, float]:
    semantic = WT103_METRIC_SEMANTIC_BY_NAME[name]
    if semantic == "scalar":
        return None, None, value
    if name in {"prior_nll_per_token", "perplexity"}:
        numerator = science.totals.summed_nll
        denominator = science.totals.counted_targets
    elif name in {"source_entropy", "effective_source_count"}:
        numerator = science.source_entropy
        denominator = 1
    elif name == "source_support_size":
        numerator = float(science.source_support_size)
        denominator = 1
    elif name == "acceptance_rate":
        denominator = len(result.updates)
        numerator = float(sum(update.accepted for update in result.updates))
    elif name in {"objective_before", "objective_after"}:
        numerator, denominator = _proposal_objective_numerator(
            result,
            before=name == "objective_before",
        )
    elif name == "tokens_per_second":
        if result.counted_targets is None:
            raise RuntimeError(
                "synthetic throughput omitted counted targets"
            )
        numerator = float(result.counted_targets)
        denominator = result.counted_targets * 1_000_000_000
    else:
        raise RuntimeError(
            f"synthetic derived metric {name!r} omitted raw components"
        )
    if semantic == "ratio":
        derived = numerator / denominator
    elif semantic == "exp_ratio":
        derived = math.exp(numerator / denominator)
    elif semantic == "tokens_per_second":
        derived = numerator / (denominator / 1_000_000_000.0)
    else:  # pragma: no cover - guarded by the semantic inventory above
        raise RuntimeError(f"unknown synthetic metric semantic {semantic!r}")
    if value != derived:
        raise RuntimeError(
            f"synthetic metric {name!r} changed its raw derivation"
        )
    return numerator, denominator, derived


class _MetricSink:
    def __init__(
        self,
        *,
        arm_spec: WT103ArmSpec,
        runtime: ArmExecutionRuntime,
        science: _ArmScienceEvidence,
        batch: _SyntheticBatch,
        run_id: str,
        log_path: Path,
        backend: DurabilityBackend,
        ordinal_start: int = 0,
        previous_record_sha256: str = _ZERO_SHA256,
        step_offset: int = 0,
    ) -> None:
        self.arm_spec = arm_spec
        self.runtime = runtime
        self.science = science
        self.batch = batch
        self.run_id = run_id
        self.log_path = log_path
        self.backend = backend
        self.next_ordinal = ordinal_start
        self.previous_record_sha256 = previous_record_sha256
        self.step_offset = step_offset
        self.validation_count = 0
        self.failure_count = 0

    def record_step(
        self,
        *,
        step_index: int,
        cumulative_counted_targets: int,
        result: StepResult,
    ) -> None:
        del cumulative_counted_targets
        if type(result) is not StepResult or not result.accepted:
            raise RuntimeError("synthetic metric sink accepts only exact updates")
        if result.objective_terms is None or result.counted_targets is None:
            raise RuntimeError("synthetic metric step omitted objective evidence")
        projected = project_objective_metrics(
            objective_kind=result.objective_kind,
            objective_terms=result.objective_terms,
            complete_elbo_numerator=result.complete_elbo_numerator,
            complete_elbo_value=result.complete_elbo_value,
            counted_targets=result.counted_targets,
        )
        values = []
        for name in WT103_REQUIRED_METRIC_FAMILIES:
            applicable, reason = metric_family_applicability(
                self.arm_spec,
                name,
            )
            if (
                name == "estimator_error_bound"
                and self.arm_spec.training_objective == "complete_elbo"
            ):
                values.append(
                    not_applicable_metric(
                        name=name,
                        reason=WT103_UNAVAILABLE_ESTIMATOR_BOUND_REASON,
                        units=metric_family_units(name),
                    )
                )
                continue
            if applicable:
                if name in projected:
                    projection = projected[name]
                    numerator = projection.numerator
                    denominator = projection.denominator
                    value = projection.value
                    measurement_reason = (
                        WT103_SOURCE_KL_DIAGNOSTIC_REASON
                        if name in {"model_source_kl", "state_source_kl"}
                        else "generated_fixture_objective_projection"
                    )
                else:
                    value = _metric_number(name, result, self.science)
                    numerator, denominator, value = (
                        _synthetic_metric_components(
                            name=name,
                            result=result,
                            science=self.science,
                            value=value,
                        )
                    )
                    measurement_reason = "generated_fixture_measurement"
                values.append(
                    applicable_metric(
                        name=name,
                        numerator=numerator,
                        denominator=denominator,
                        value=value,
                        units=metric_family_units(name),
                        reason=measurement_reason,
                    )
                )
            else:
                values.append(
                    not_applicable_metric(
                        name=name,
                        reason=reason,
                        units=metric_family_units(name),
                    )
                )
        ordinal = self.next_ordinal
        record = create_metric_record(
            ordinal=ordinal,
            utc_timestamp=(
                f"2026-07-28T00:00:{ordinal:02d}.000000Z"
            ),
            monotonic_ns=ordinal,
            run_id=self.run_id,
            arm_id=self.arm_spec.arm_id,
            seed_id=_SEED_ID,
            phase=result.phase_order[-1],
            split="train",
            step=self.step_offset + step_index,
            pass_index=0,
            previous_record_sha256=self.previous_record_sha256,
            values=tuple(values),
        )
        append_metric(
            self.log_path,
            record,
            durability_backend=self.backend,
        )
        self.previous_record_sha256 = record.record_sha256
        self.next_ordinal += 1

    def validate_target_blind(
        self,
        *,
        step_index: int,
        cumulative_counted_targets: int,
    ) -> None:
        if step_index != 1 or cumulative_counted_targets != 1:
            raise RuntimeError("synthetic validation boundary is not exact")
        changed_target = _SyntheticBatch(
            context=self.batch.context.clone(),
            target=self.batch.target + 7.0,
        )
        before = _target_blind_prediction(self.runtime.model, self.batch)
        after = _target_blind_prediction(self.runtime.model, changed_target)
        if not torch.equal(before, after):
            raise RuntimeError("synthetic validation predictor accessed target")
        self.validation_count += 1

    def record_terminal_failure(
        self,
        *,
        step_index: int,
        cumulative_counted_targets: int,
        result: StepResult | None,
        exception: Exception | None,
    ) -> None:
        del step_index, cumulative_counted_targets, result, exception
        self.failure_count += 1


def _platform_backend() -> DurabilityBackend:
    return (
        WindowsDurabilityBackend()
        if os.name == "nt"
        else PosixDurabilityBackend()
    )


def _artifact_record(path: Path, *, relative_path: str) -> ArtifactIntegrityRecord:
    payload = path.read_bytes()
    return ArtifactIntegrityRecord.create(
        kind="file",
        relative_path=relative_path,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _checkpoint_contract(
    *,
    config: TrainingConfig,
    plan: ExperimentPlanIdentity,
    spec: WT103ArmSpec,
    logical_key: str,
    checkpoint_role: Literal["resume_only", "terminal_scoring"],
    training_complete: bool,
    objective_sha256: str,
    cache_fixture_sha256: str,
    environment_sha256: str,
) -> ResumeContract:
    hashes = {
        "model_schema_sha256": _digest("scalar-model-schema"),
        "recognition_schema_sha256": _digest("scalar-recognition-schema"),
        "optimizer_schema_sha256": _digest("adamw-schema"),
        "scheduler_schema_sha256": _digest("no-scheduler-schema"),
        "amp_schema_sha256": _digest("no-amp-schema"),
        "rng_schema_sha256": _digest("cpu-rng-schema"),
        "estimator_schema_sha256": config.endpoint_inventory.estimator_protocol_sha256,
        "cursor_schema_sha256": _digest("one-scalar-batch-cursor"),
        "metric_schema_sha256": _digest(config.profile.schemas.metric_schema),
        "update_trace_schema_sha256": _digest("engine-update-trace"),
        "precision_profile_sha256": owned_sha256(
            "vfe4.wt103.synthetic-smoke-precision.v1",
            config.profile.precision,
        ),
        "dependency_lock_sha256": _digest("no-external-dependencies-opened"),
        "source_sha256": cache_fixture_sha256,
        "tokenizer_sha256": _digest("no-tokenizer-imported"),
        "data_sha256": cache_fixture_sha256,
        "window_sha256": _digest("generated-scalar-window"),
        "permutation_sha256": _digest("identity-permutation"),
        "evidence_sha256": _digest("nonproduction-smoke-evidence"),
        "environment_sha256": environment_sha256,
    }
    return ResumeContract.create(
        logical_key=logical_key,
        checkpoint_role=checkpoint_role,
        training_complete=training_complete,
        arm_spec_sha256=spec.arm_spec_sha256,
        experiment_plan_sha256=plan.plan.experiment_plan_sha256,
        config_sha256=config.config_sha256,
        objective_sha256=objective_sha256,
        maximum_checkpoint_bytes=2 * 1024 * 1024,
        maximum_tensor_bytes=256 * 1024,
        maximum_total_tensor_bytes=512 * 1024,
        maximum_tensor_count=128,
        maximum_container_items=8192,
        maximum_recursion_depth=24,
        **hashes,
    )


def _plain_state_dict(module: nn.Module) -> dict[str, object]:
    return {
        name: value.detach().to(device="cpu").contiguous().clone()
        for name, value in module.state_dict().items()
    }


def _scientific_state(
    *,
    runtime: ArmExecutionRuntime,
    metric_head: str,
    successful_updates: int,
    counted_targets: int,
    batch_index: int,
) -> dict[str, object]:
    recognition_state = (
        None
        if runtime.recognition is None
        else _plain_state_dict(runtime.recognition)
    )
    numpy_state = np.random.get_state()
    return {
        "model_state": _plain_state_dict(runtime.model),
        "recognition_state": recognition_state,
        "optimizer_state": {
            "model": runtime.model_optimizer.state_dict(),
            "recognition": (
                None
                if runtime.recognition_optimizer is None
                else runtime.recognition_optimizer.state_dict()
            ),
        },
        "scheduler_state": {
            "model": {
                "kind": "none",
                "ordinal": runtime.update_counter,
            },
            "recognition": {
                "kind": "none",
                "ordinal": runtime.update_counter,
            },
        },
        "amp_scaler_state": None,
        "rng_state": {
            "python": random.getstate(),
            "numpy": (
                str(numpy_state[0]),
                torch.from_numpy(
                    numpy_state[1].astype(np.int64, copy=True)
                ),
                int(numpy_state[2]),
                int(numpy_state[3]),
                float(numpy_state[4]),
            ),
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": (),
        },
        "estimator_state": {
            "stream_counters": {
                "validation": 0,
                "test": 0,
            },
            "particle_level": 0,
        },
        "data_cursor_state": {
            "pass_index": 0,
            "batch_index": batch_index,
            "next_window_ids": (batch_index + 1,),
            "permutation_bytes": torch.tensor(
                [0],
                dtype=torch.uint8,
            ),
            "permutation_sha256": _digest("identity-permutation"),
        },
        "update_trace_state": {
            "global_step": batch_index,
            "successful_updates": successful_updates,
            "rejected_updates": 0,
            "counted_targets": counted_targets,
        },
        "metric_state": {
            "next_ordinal": batch_index,
            "hash_chain_head": metric_head,
            "nll_numerator": float(counted_targets),
            "nll_denominator": counted_targets,
            "failure_ledger_head": _ZERO_SHA256,
        },
        "next_prediction_fixture": _next_prediction_fixture(runtime.model),
    }


def _assert_state_equal(left: object, right: object) -> None:
    if type(left) is not type(right):
        raise RuntimeError("restored scientific state changed value types")
    if type(left) is torch.Tensor:
        if not torch.equal(left, right):  # type: ignore[arg-type]
            raise RuntimeError("restored scientific tensor differs")
        return
    if type(left) is dict:
        if set(left) != set(right):  # type: ignore[arg-type]
            raise RuntimeError("restored scientific mapping keys differ")
        for key, value in left.items():
            _assert_state_equal(value, right[key])  # type: ignore[index]
        return
    if type(left) in (tuple, list):
        if len(left) != len(right):  # type: ignore[arg-type]
            raise RuntimeError("restored scientific sequence length differs")
        for left_item, right_item in zip(left, right, strict=True):  # type: ignore[arg-type]
            _assert_state_equal(left_item, right_item)
        return
    if left != right:
        raise RuntimeError("restored scientific primitive differs")


def _repeatable_scientific_state(
    *,
    runtime: ArmExecutionRuntime,
    metric_head: str,
    successful_updates: int,
    counted_targets: int,
    batch_index: int,
) -> dict[str, object]:
    first = _scientific_state(
        runtime=runtime,
        metric_head=metric_head,
        successful_updates=successful_updates,
        counted_targets=counted_targets,
        batch_index=batch_index,
    )
    second = _scientific_state(
        runtime=runtime,
        metric_head=metric_head,
        successful_updates=successful_updates,
        counted_targets=counted_targets,
        batch_index=batch_index,
    )
    _assert_state_equal(first, second)
    if scientific_state_sha256(first) != scientific_state_sha256(second):
        raise RuntimeError("synthetic scientific projection is not repeatable")
    return first


class _FreshSyntheticTarget:
    def __init__(self, contract: ResumeContract) -> None:
        self.checkpoint_contract_sha256 = contract.contract_sha256
        self.restored_state: dict[str, object] | None = None

    def is_fresh_checkpoint_target(self) -> bool:
        return self.restored_state is None

    def validate_checkpoint_state(self, state: dict[str, object]) -> None:
        if self.restored_state is not None or set(state) != {
            "model_state",
            "recognition_state",
            "optimizer_state",
            "scheduler_state",
            "amp_scaler_state",
            "rng_state",
            "estimator_state",
            "data_cursor_state",
            "update_trace_state",
            "metric_state",
            "next_prediction_fixture",
        }:
            raise ValueError("fresh synthetic target rejected state")

    def restore_checkpoint_state(self, state: dict[str, object]) -> None:
        if self.restored_state is not None:
            raise ValueError("fresh synthetic target was already restored")
        self.restored_state = state


def _runtime_from_restored_state(
    *,
    spec: WT103ArmSpec,
    state: dict[str, object],
    science: _ArmScienceEvidence,
) -> ArmExecutionRuntime:
    runtime = _engine_runtime(spec, build=None, science=science)
    model_state = state["model_state"]
    recognition_state = state["recognition_state"]
    optimizer_state = state["optimizer_state"]
    if type(model_state) is not dict or type(optimizer_state) is not dict:
        raise RuntimeError("restored runtime state mappings are invalid")
    runtime.model.load_state_dict(model_state)  # type: ignore[arg-type]
    runtime.model_optimizer.load_state_dict(optimizer_state["model"])  # type: ignore[arg-type]
    if runtime.recognition is None:
        if recognition_state is not None:
            raise RuntimeError("nonlatent restore fabricated recognition state")
    else:
        if (
            type(recognition_state) is not dict
            or runtime.recognition_optimizer is None
        ):
            raise RuntimeError("latent restore omitted recognition state")
        runtime.recognition.load_state_dict(recognition_state)  # type: ignore[arg-type]
        runtime.recognition_optimizer.load_state_dict(
            optimizer_state["recognition"]  # type: ignore[arg-type]
        )
    trace = state["update_trace_state"]
    if type(trace) is not dict or type(trace["successful_updates"]) is not int:
        raise RuntimeError("restored update trace is invalid")
    runtime.update_counter = trace["successful_updates"]
    rng_state = state["rng_state"]
    cursor_state = state["data_cursor_state"]
    metric_state = state["metric_state"]
    if (
        type(rng_state) is not dict
        or type(cursor_state) is not dict
        or type(metric_state) is not dict
        or type(rng_state["python"]) is not tuple
        or type(rng_state["numpy"]) is not tuple
        or len(rng_state["numpy"]) != 5
        or type(rng_state["torch_cpu"]) is not torch.Tensor
        or rng_state["torch_cuda"] != ()
        or type(cursor_state["batch_index"]) is not int
        or type(metric_state["next_ordinal"]) is not int
        or metric_state["next_ordinal"] != cursor_state["batch_index"]
        or type(metric_state["hash_chain_head"]) is not str
    ):
        raise RuntimeError("restored RNG/cursor/metric state is invalid")
    numpy_state = rng_state["numpy"]
    numpy_vector = numpy_state[1]
    if (
        type(numpy_vector) is not torch.Tensor
        or numpy_vector.device.type != "cpu"
        or numpy_vector.dtype is not torch.int64
    ):
        raise RuntimeError("restored NumPy RNG vector is invalid")
    random.setstate(rng_state["python"])  # type: ignore[arg-type]
    np.random.set_state(
        (
            numpy_state[0],
            numpy_vector.numpy().astype(np.uint32, copy=True),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        )
    )
    torch.set_rng_state(rng_state["torch_cpu"])
    return runtime


def _publish_cache_fixture(
    *,
    cache_root: Path,
    smoke_run_id: str,
    backend: DurabilityBackend,
) -> tuple[Path, str]:
    cache_root.mkdir(parents=True, exist_ok=True)
    probe = backend.probe(cache_root)
    if probe.status != "pass":
        raise RuntimeError(
            "synthetic cache durability probe did not pass: "
            + ",".join(probe.obligations)
        )
    fixture_root = cache_root / smoke_run_id
    fixture_root.mkdir()
    payload = canonical_json_bytes_generic(
        {
            "schema_version": "wt103-generated-smoke-fixture-v1",
            "authority": _AUTHORITY,
            "dataset": "generated_scalars_only",
            "tokenizer_imported": False,
            "network_accessed": False,
            "heldout_test_opened": False,
            "context": [0.75],
            "target": [1.25],
        }
    )
    fixture_path = fixture_root / "synthetic-fixture.json"
    identity = backend.create_exclusive(fixture_path, payload)
    if identity.sha256 != hashlib.sha256(payload).hexdigest():
        raise RuntimeError("synthetic cache fixture identity differs")
    return fixture_path, identity.sha256


def _experiment_plan(
    *,
    config: TrainingConfig,
    smoke_run_id: str,
    factory_set: WT103FactorySetIdentity,
    cache_fixture_sha256: str,
) -> ExperimentPlan:
    objective_sha256 = owned_sha256(
        "vfe4.wt103.synthetic-smoke-objectives.v1",
        tuple(
            (arm.arm_spec_sha256, arm.training_objective, arm.update_phases)
            for arm in config.endpoint_inventory.arms
        ),
    )
    return ExperimentPlan.create(
        experiment_id=f"{smoke_run_id}-experiment",
        endpoint_inventory=config.endpoint_inventory,
        git_head=_digest("nonproduction-uncommitted-revision")[:40],
        dirty_digest=_digest("nonproduction-uncommitted-dirty-state"),
        config_sha256=config.config_sha256,
        source_record_sha256=cache_fixture_sha256,
        tokenizer_spec_sha256=_digest("no-tokenizer-imported"),
        token_cache_set_sha256=cache_fixture_sha256,
        window_manifest_sha256s=(
            _digest("generated-train-window"),
            _digest("generated-validation-window"),
            _digest("sealed-test-window-never-opened"),
        ),
        schedule_set_sha256=_digest("one-generated-update-one-boundary"),
        factory_set_sha256=factory_set.factory_set_sha256,
        objective_sha256=objective_sha256,
        checkpoint_schema_sha256=_digest(
            config.profile.schemas.checkpoint_schema
        ),
        resource_forecast_sha256=_digest(
            "one-thread-cpu-scalar-no-cuda"
        ),
        expected_run_artifact_paths=_RUN_ARTIFACT_PATHS,
        expected_group_artifact_paths=("result-table.json",),
    )


def _uninterrupted_oracle(
    *,
    spec: WT103ArmSpec,
    science: _ArmScienceEvidence,
    batch: _SyntheticBatch,
    run_id: str,
    log_path: Path,
    backend: DurabilityBackend,
) -> tuple[dict[str, object], tuple[torch.Tensor, ...], str]:
    """Run the two-update reference without checkpoint serialization."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    runtime = _engine_runtime(spec, build=None, science=science)
    sink = _MetricSink(
        arm_spec=spec,
        runtime=runtime,
        science=science,
        batch=batch,
        run_id=run_id,
        log_path=log_path,
        backend=backend,
    )
    first = train_attempt(
        runtime,
        batches=(batch,),
        validation_step_boundaries=(1,),
        event_sink=sink,
    )
    if (
        not all(step.accepted for step in first.steps)
        or first.completed_validation_step_boundaries != (1,)
    ):
        raise RuntimeError("uninterrupted oracle first update failed")
    sink.step_offset = 1
    second = train_attempt(
        runtime,
        batches=(batch,),
        validation_step_boundaries=(1,),
        event_sink=sink,
    )
    if (
        not all(step.accepted for step in second.steps)
        or second.completed_validation_step_boundaries != (1,)
        or sink.validation_count != 2
        or sink.failure_count
    ):
        raise RuntimeError("uninterrupted oracle second update failed")
    state = _repeatable_scientific_state(
        runtime=runtime,
        metric_head=sink.previous_record_sha256,
        successful_updates=runtime.update_counter,
        counted_targets=2,
        batch_index=2,
    )
    return (
        state,
        _next_prediction_fixture(runtime.model),
        hashlib.sha256(log_path.read_bytes()).hexdigest(),
    )


def _run_one_arm(
    *,
    index: int,
    config: TrainingConfig,
    build: BuiltWT103Arm,
    experiment_root: Path,
    plan: ExperimentPlanIdentity,
    backend: DurabilityBackend,
    smoke_run_id: str,
    cache_fixture_sha256: str,
    environment_sha256: str,
    science: _ArmScienceEvidence,
    oracle_root: Path,
) -> tuple[
    SyntheticSmokeArmResult,
    WT103CheckpointIdentity | None,
    str | None,
    str | None,
    bool,
    _ResumeOracleEvidence | None,
    RunManifestIdentity,
]:
    spec = build.record.spec
    if scorer_dispatch(build) != spec.scorer_kind:
        raise RuntimeError("factory scorer dispatch changed arm semantics")
    run_id = f"{smoke_run_id}-arm-{index}"
    reserved = reserve_run(
        experiment_root,
        run_id,
        run_role="confirmation",
        started_utc=f"2026-07-28T00:1{index}:00Z",
        plan=plan,
        backend=backend,
    )
    batch = _SyntheticBatch(
        context=torch.tensor(0.75, dtype=torch.float64),
        target=torch.tensor(1.25, dtype=torch.float64),
    )
    runtime = _engine_runtime(
        spec,
        build=build if index else None,
        science=science,
    )
    metric_path = reserved.inprogress_path / "metrics.jsonl"
    sink = _MetricSink(
        arm_spec=spec,
        runtime=runtime,
        science=science,
        batch=batch,
        run_id=run_id,
        log_path=metric_path,
        backend=backend,
    )
    uninterrupted_state: dict[str, object] | None = None
    uninterrupted_predictions: tuple[torch.Tensor, ...] | None = None
    uninterrupted_metric_sha256: str | None = None
    if index == _RESUME_ARM_INDEX:
        (
            uninterrupted_state,
            uninterrupted_predictions,
            uninterrupted_metric_sha256,
        ) = _uninterrupted_oracle(
            spec=spec,
            science=science,
            batch=batch,
            run_id=run_id,
            log_path=oracle_root / "uninterrupted-metrics.jsonl",
            backend=backend,
        )
    attempt = train_attempt(
        runtime,
        batches=(batch,),
        validation_step_boundaries=(1,),
        event_sink=sink,
    )
    if (
        not all(step.accepted for step in attempt.steps)
        or attempt.completed_validation_step_boundaries != (1,)
        or sink.validation_count != 1
        or sink.failure_count
    ):
        raise RuntimeError("synthetic arm did not complete its exact boundary")

    objective_sha256 = owned_sha256(
        "vfe4.wt103.synthetic-smoke-objective.v1",
        {
            "arm_spec_sha256": spec.arm_spec_sha256,
            "training_objective": spec.training_objective,
            "update_phases": spec.update_phases,
        },
    )
    resume_identity: WT103CheckpointIdentity | None = None
    resume_artifact_record: ArtifactIntegrityRecord | None = None
    resume_before: str | None = None
    resume_after: str | None = None
    resume_next_predictions_equal = False
    if index == _RESUME_ARM_INDEX:
        before_state = _repeatable_scientific_state(
            runtime=runtime,
            metric_head=sink.previous_record_sha256,
            successful_updates=len(attempt.steps[0].updates),
            counted_targets=attempt.cumulative_counted_targets,
            batch_index=1,
        )
        resume_contract = _checkpoint_contract(
            config=config,
            plan=plan,
            spec=spec,
            logical_key=f"resume/{spec.arm_id}/seed={_SEED_ID}/step=1",
            checkpoint_role="resume_only",
            training_complete=False,
            objective_sha256=objective_sha256,
            cache_fixture_sha256=cache_fixture_sha256,
            environment_sha256=environment_sha256,
        )
        resume_path = reserved.inprogress_path / "resume-only.pt"
        resume_identity = save_checkpoint(
            resume_path,
            contract=resume_contract,
            scientific_state=before_state,
            durability_backend=backend,
            operational_metadata={
                "process_id": 0,
                "utc_timestamp": "2026-07-28T00:20:00Z",
                "monotonic_seconds": 1.0,
                "elapsed_seconds": 1.0,
                "path_hint": "nonproduction-resume-only",
                "write_ordinal": 1,
            },
        )
        resume_artifact_record = _artifact_record(
            resume_path,
            relative_path="resume-only.pt",
        )
        target = _FreshSyntheticTarget(resume_contract)
        loaded = load_checkpoint(
            resume_path,
            expected_identity=resume_identity,
            expected_contract=resume_contract,
            fresh_target=target,
        )
        if target.restored_state is None:
            raise RuntimeError("resume target did not restore scientific state")
        _assert_state_equal(before_state, target.restored_state)
        resume_before = resume_identity.scientific_state_sha256
        resume_after = scientific_state_sha256(target.restored_state)
        if (
            loaded.identity.checkpoint_identity_sha256
            != resume_identity.checkpoint_identity_sha256
            or resume_before != resume_after
        ):
            raise RuntimeError("resume checkpoint identity or state changed")
        before_predictions = before_state["next_prediction_fixture"]
        after_predictions = target.restored_state["next_prediction_fixture"]
        _assert_state_equal(before_predictions, after_predictions)

        # Simulate the original process terminating at this authenticated
        # boundary: a real process exit releases the OS execution lease.
        release_run_execution_lease(reserved)
        reserved = reserve_run(
            experiment_root,
            run_id,
            run_role="confirmation",
            started_utc=None,
            plan=plan,
            backend=backend,
            mode="resume",
            resume_lineage=ResumeLineageEvent.create(
                parent_checkpoint=resume_identity,
                environment_sha256=environment_sha256,
                cursor_sha256=owned_sha256(
                    "vfe4.wt103.synthetic-smoke-cursor.v1",
                    {
                        "arm_spec_sha256": spec.arm_spec_sha256,
                        "pass_index": 0,
                        "batch_index": 1,
                        "next_window_ids": (2,),
                        "permutation_sha256": _digest(
                            "identity-permutation"
                        ),
                    },
                ),
                reason="bounded synthetic interruption and exact continuation",
                resumed_utc="2026-07-28T00:21:00Z",
            ),
        )
        runtime = _runtime_from_restored_state(
            spec=spec,
            state=target.restored_state,
            science=science,
        )
        _assert_state_equal(
            before_predictions,
            _next_prediction_fixture(runtime.model),
        )
        resume_next_predictions_equal = True
        restored_cursor = target.restored_state["data_cursor_state"]
        restored_metric = target.restored_state["metric_state"]
        if type(restored_cursor) is not dict or type(restored_metric) is not dict:
            raise RuntimeError("restored continuation state is invalid")
        continued_sink = _MetricSink(
            arm_spec=spec,
            runtime=runtime,
            science=science,
            batch=batch,
            run_id=run_id,
            log_path=metric_path,
            backend=backend,
            ordinal_start=restored_metric["next_ordinal"],  # type: ignore[arg-type]
            previous_record_sha256=restored_metric[  # type: ignore[arg-type]
                "hash_chain_head"
            ],
            step_offset=restored_cursor["batch_index"],  # type: ignore[arg-type]
        )
        consume_resume_execution_retry(reserved, backend=backend)
        continued = train_attempt(
            runtime,
            batches=(batch,),
            validation_step_boundaries=(1,),
            event_sink=continued_sink,
        )
        if (
            not all(step.accepted for step in continued.steps)
            or continued.completed_validation_step_boundaries != (1,)
            or continued_sink.validation_count != 1
            or continued_sink.failure_count
        ):
            raise RuntimeError("restored arm did not continue exactly")
        attempt = continued
        sink = continued_sink

    records = validate_metric_log(metric_path)
    validate_required_metric_families(records, arm_spec=spec)
    csv_path = reserved.inprogress_path / "metrics.csv"
    csv_payload = export_metrics_csv(
        log_path=metric_path,
        output_path=csv_path,
        durability_backend=backend,
    )
    if csv_payload != csv_path.read_bytes():
        raise RuntimeError("canonical metrics CSV failed reopen")
    authority_path = (
        reserved.inprogress_path / "nonproduction-authority.json"
    )
    authority_payload = canonical_json_bytes_generic(
        {
            "schema_version": "wt103-smoke-authority-v1",
            "authority": _AUTHORITY,
            "arm_id": spec.arm_id,
            "arm_spec_sha256": spec.arm_spec_sha256,
            "build_sha256": build.record.build_sha256,
            "production_readiness_eligible": False,
            "heldout_test_opened": False,
        }
    )
    backend.create_exclusive(authority_path, authority_payload)

    final_state = _repeatable_scientific_state(
        runtime=runtime,
        metric_head=sink.previous_record_sha256,
        successful_updates=runtime.update_counter,
        counted_targets=(
            2
            if index == _RESUME_ARM_INDEX
            else attempt.cumulative_counted_targets
        ),
        batch_index=(2 if index == _RESUME_ARM_INDEX else 1),
    )
    resume_oracle: _ResumeOracleEvidence | None = None
    if index == _RESUME_ARM_INDEX:
        if (
            uninterrupted_state is None
            or uninterrupted_predictions is None
            or uninterrupted_metric_sha256 is None
        ):
            raise RuntimeError("resume arm omitted uninterrupted oracle state")
        _assert_state_equal(uninterrupted_state, final_state)
        resumed_predictions = _next_prediction_fixture(runtime.model)
        _assert_state_equal(
            uninterrupted_predictions,
            resumed_predictions,
        )
        resumed_metric_sha256 = hashlib.sha256(
            metric_path.read_bytes()
        ).hexdigest()
        resume_oracle = _ResumeOracleEvidence(
            uninterrupted_terminal_scientific_state_sha256=(
                scientific_state_sha256(uninterrupted_state)
            ),
            resumed_terminal_scientific_state_sha256=(
                scientific_state_sha256(final_state)
            ),
            uninterrupted_metrics_jsonl_sha256=(
                uninterrupted_metric_sha256
            ),
            resumed_metrics_jsonl_sha256=resumed_metric_sha256,
            uninterrupted_next_predictions_equal=True,
        )
    terminal_key = plan.plan.terminal_checkpoint_keys[
        index * len(config.profile.statistics.confirmatory_seed_ids)
    ]
    terminal_contract = _checkpoint_contract(
        config=config,
        plan=plan,
        spec=spec,
        logical_key=terminal_key,
        checkpoint_role="terminal_scoring",
        training_complete=True,
        objective_sha256=objective_sha256,
        cache_fixture_sha256=cache_fixture_sha256,
        environment_sha256=environment_sha256,
    )
    terminal_path = reserved.inprogress_path / "terminal-scoring.pt"
    terminal_identity = save_checkpoint(
        terminal_path,
        contract=terminal_contract,
        scientific_state=final_state,
        durability_backend=backend,
        operational_metadata={
            "process_id": 0,
            "utc_timestamp": f"2026-07-28T00:2{index}:00Z",
            "monotonic_seconds": 2.0,
            "elapsed_seconds": 2.0,
            "path_hint": "nonproduction-terminal-scoring",
            "write_ordinal": 2,
        },
    )
    terminal_artifact_record = _artifact_record(
        terminal_path,
        relative_path="terminal-scoring.pt",
    )
    require_terminal_scoring(terminal_identity, purpose="endpoint")
    artifact_records = tuple(
        sorted(
            (
                _artifact_record(
                    csv_path,
                    relative_path="metrics.csv",
                ),
                _artifact_record(
                    metric_path,
                    relative_path="metrics.jsonl",
                ),
                _artifact_record(
                    authority_path,
                    relative_path="nonproduction-authority.json",
                ),
            ),
            key=lambda item: item.relative_path,
        )
    )
    manifest = finalize_run(
        reserved,
        disposition="success",
        checkpoints=(
            (() if resume_identity is None else (resume_identity,))
            + (terminal_identity,)
        ),
        checkpoint_artifact_records=(
            (
                ()
                if resume_artifact_record is None
                else (resume_artifact_record,)
            )
            + (terminal_artifact_record,)
        ),
        artifact_records=artifact_records,
        environment_sha256=environment_sha256,
        provenance_sha256=_digest(
            f"{_AUTHORITY}:{spec.arm_spec_sha256}:{build.record.build_sha256}"
        ),
        ended_utc=f"2026-07-28T00:3{index}:00Z",
        monotonic_duration_seconds=0.0,
        failure_record_sha256=None,
        backend=backend,
    )
    final_metric_path = manifest.run_path / "metrics.jsonl"
    final_csv_path = manifest.run_path / "metrics.csv"
    result = SyntheticSmokeArmResult.create(
        arm_id=spec.arm_id,
        arm_spec_sha256=spec.arm_spec_sha256,
        constructor_id=build.record.constructor_id,
        build_sha256=build.record.build_sha256,
        scorer_kind=spec.scorer_kind,
        execution_trace=science.trace,
        update_phase_order=spec.update_phases,
        accepted_update=True,
        validation_completed=True,
        terminal_checkpoint_role="terminal_scoring",
        terminal_checkpoint_identity_sha256=(
            terminal_identity.checkpoint_identity_sha256
        ),
        terminal_scientific_state_sha256=(
            terminal_identity.scientific_state_sha256
        ),
        metrics_jsonl_path=str(final_metric_path),
        metrics_jsonl_sha256=hashlib.sha256(
            final_metric_path.read_bytes()
        ).hexdigest(),
        metrics_csv_path=str(final_csv_path),
        metrics_csv_sha256=hashlib.sha256(
            final_csv_path.read_bytes()
        ).hexdigest(),
        run_manifest_path=str(
            manifest.run_path / "run-manifest.json"
        ),
        run_manifest_sha256=manifest.manifest_sha256,
        resume_exercised=index == _RESUME_ARM_INDEX,
    )
    return (
        result,
        resume_identity,
        resume_before,
        resume_after,
        resume_next_predictions_equal,
        resume_oracle,
        manifest,
    )


def _run_wt103_synthetic_smoke_in_process(
    *,
    config: TrainingConfig,
    cache_root: Path,
    run_root: Path,
    smoke_run_id: str,
    parent_process_id: int,
) -> WT103SyntheticSmokeResult:
    """Run the generated-data-only smoke inside its isolated worker."""

    if type(config) is not TrainingConfig:
        raise TypeError("config must be an exact TrainingConfig")
    if (
        config.operation != "synthetic_smoke"
        or config.synthetic_authority != _AUTHORITY
    ):
        raise ValueError("synthetic smoke requires its exact nonproduction mode")
    if (
        not isinstance(cache_root, Path)
        or not isinstance(run_root, Path)
        or not cache_root.is_absolute()
        or not run_root.is_absolute()
        or cache_root == run_root
        or "v3_transformer"
        in f"{cache_root}{run_root}".casefold()
    ):
        raise ValueError("synthetic smoke roots must be distinct absolute VFE4 paths")
    if (
        type(smoke_run_id) is not str
        or not smoke_run_id
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-_"
            for character in smoke_run_id
        )
    ):
        raise ValueError("smoke_run_id is not a portable lowercase component")
    if (
        type(parent_process_id) is not int
        or parent_process_id <= 0
        or parent_process_id == os.getpid()
    ):
        raise ValueError("smoke parent process identity is invalid")
    if (
        torch.get_num_threads() != 1
        or torch.get_num_interop_threads() != 1
        or tuple(
            (name, os.environ.get(name))
            for name in _THREAD_ENVIRONMENT_NAMES
        )
        != tuple((name, "1") for name in _THREAD_ENVIRONMENT_NAMES)
        or os.environ.get("CUDA_VISIBLE_DEVICES") != "-1"
    ):
        raise RuntimeError("isolated smoke worker limits are not active")

    cuda_initialized_on_entry = torch.cuda.is_initialized()
    cuda_available = torch.cuda.is_available()
    if cuda_initialized_on_entry or cuda_available:
        raise RuntimeError("isolated smoke worker unexpectedly exposed CUDA")
    random.seed(_SEED_ID, version=2)
    np.random.seed(_SEED_ID % (2**32))
    fixed_cpu_generator = torch.Generator(device="cpu")
    fixed_cpu_generator.manual_seed(_SEED_ID)
    torch.set_rng_state(fixed_cpu_generator.get_state())
    try:
        backend = _platform_backend()
        fixture_path, fixture_sha256 = _publish_cache_fixture(
            cache_root=cache_root,
            smoke_run_id=smoke_run_id,
            backend=backend,
        )
        evaluation, vocabulary = _generated_evaluation_batches(
            fixture_root=fixture_path.parent,
            backend=backend,
        )
        run_root.mkdir(parents=True, exist_ok=True)
        run_durability = backend.probe(run_root)
        if run_durability.status != "pass":
            raise RuntimeError(
                "synthetic run durability probe did not pass: "
                + ",".join(run_durability.obligations)
            )
        builds = _build_exact_factories(config)
        factory_set = WT103FactorySetIdentity.create(builds)
        plan = _experiment_plan(
            config=config,
            smoke_run_id=smoke_run_id,
            factory_set=factory_set,
            cache_fixture_sha256=fixture_sha256,
        )
        experiment_root = run_root / smoke_run_id
        plan_identity = publish_experiment_plan(
            experiment_root,
            plan,
            backend=backend,
        )
        environment_sha256 = owned_sha256(
            "vfe4.wt103.synthetic-smoke-environment.v1",
            {
                "authority": _AUTHORITY,
                "device": "cpu",
                "dtype": "float64",
                "torch_threads": torch.get_num_threads(),
                "cache_fixture_sha256": fixture_sha256,
                "run_durability_sha256": run_durability.identity_sha256,
                "cuda_initialized_on_entry": cuda_initialized_on_entry,
                "cuda_initialized_by_smoke": False,
                "production_tokenizer_imported": False,
                "generated_tokenizer_spec_sha256": (
                    vocabulary.tokenizer_spec_sha256
                ),
            },
        )
        results: list[SyntheticSmokeArmResult] = []
        manifests: list[RunManifestIdentity] = []
        resume_identity: WT103CheckpointIdentity | None = None
        resume_before: str | None = None
        resume_after: str | None = None
        resume_next_predictions_equal = False
        resume_oracle: _ResumeOracleEvidence | None = None
        for index, build in enumerate(builds):
            science = _arm_science_evidence(
                index=index,
                spec=build.record.spec,
                evaluation=evaluation,
                vocabulary=vocabulary,
            )
            (
                arm_result,
                arm_resume_identity,
                arm_resume_before,
                arm_resume_after,
                arm_predictions_equal,
                arm_resume_oracle,
                manifest,
            ) = _run_one_arm(
                index=index,
                config=config,
                build=build,
                experiment_root=experiment_root,
                plan=plan_identity,
                backend=backend,
                smoke_run_id=smoke_run_id,
                cache_fixture_sha256=fixture_sha256,
                environment_sha256=environment_sha256,
                science=science,
                oracle_root=fixture_path.parent / "resume-oracle",
            )
            results.append(arm_result)
            manifests.append(manifest)
            if arm_resume_identity is not None:
                if resume_identity is not None:
                    raise RuntimeError("more than one arm exercised resume")
                resume_identity = arm_resume_identity
                resume_before = arm_resume_before
                resume_after = arm_resume_after
                resume_next_predictions_equal = arm_predictions_equal
                resume_oracle = arm_resume_oracle
        if (
            resume_identity is None
            or resume_before is None
            or resume_after is None
            or not resume_next_predictions_equal
            or resume_oracle is None
        ):
            raise RuntimeError("synthetic smoke did not close resume evidence")
        experiment_index = publish_experiment_index(
            experiment_root,
            plan=plan_identity,
            run_manifests=tuple(manifests),
            stage="pretest",
            artifact_records=(),
            backend=backend,
        )
        cuda_initialized_on_exit = torch.cuda.is_initialized()
        if cuda_initialized_on_exit:
            raise RuntimeError("synthetic smoke unexpectedly initialized CUDA")
        runtime_observation = SyntheticRuntimeObservation.create(
            parent_process_id=parent_process_id,
            worker_process_id=os.getpid(),
            intraop_threads=1,
            interop_threads=1,
            thread_environment=tuple(
                (name, os.environ[name])
                for name in _THREAD_ENVIRONMENT_NAMES
            ),
            cuda_visible_devices="-1",
            cuda_available=False,
            cuda_initialized_on_entry=False,
            cuda_initialized_on_exit=False,
        )
        return WT103SyntheticSmokeResult.create(
            smoke_run_id=smoke_run_id,
            config_sha256=config.config_sha256,
            cache_fixture_path=str(fixture_path),
            cache_fixture_sha256=fixture_sha256,
            factory_set_sha256=factory_set.factory_set_sha256,
            arm_results=tuple(results),
            resume_checkpoint_role="resume_only",
            resume_checkpoint_identity_sha256=(
                resume_identity.checkpoint_identity_sha256
            ),
            resume_identity_before_sha256=(
                resume_identity.checkpoint_identity_sha256
            ),
            resume_identity_after_sha256=(
                resume_identity.checkpoint_identity_sha256
            ),
            resume_scientific_state_before_sha256=resume_before,
            resume_scientific_state_after_sha256=resume_after,
            resume_next_predictions_equal=True,
            resume_oracle_passed=True,
            resume_uninterrupted_terminal_scientific_state_sha256=(
                resume_oracle.uninterrupted_terminal_scientific_state_sha256
            ),
            resume_resumed_terminal_scientific_state_sha256=(
                resume_oracle.resumed_terminal_scientific_state_sha256
            ),
            resume_uninterrupted_metrics_jsonl_sha256=(
                resume_oracle.uninterrupted_metrics_jsonl_sha256
            ),
            resume_resumed_metrics_jsonl_sha256=(
                resume_oracle.resumed_metrics_jsonl_sha256
            ),
            resume_uninterrupted_next_predictions_equal=True,
            runtime_observation=runtime_observation,
            experiment_plan_path=str(plan_identity.plan_path),
            experiment_plan_sha256=plan.experiment_plan_sha256,
            experiment_index_path=str(experiment_index.index_path),
            experiment_index_sha256=experiment_index.identity_sha256,
            experiment_index_stage="pretest",
        )
    finally:
        if torch.cuda.is_initialized():
            raise RuntimeError(
                "synthetic smoke unexpectedly initialized CUDA"
            )


def run_wt103_synthetic_smoke(
    *,
    config: TrainingConfig,
    cache_root: Path,
    run_root: Path,
    smoke_run_id: str,
) -> WT103SyntheticSmokeResult:
    """Run the bounded smoke in a fresh one-thread, CUDA-hidden process."""

    if type(config) is not TrainingConfig:
        raise TypeError("config must be an exact TrainingConfig")
    for root, name in (
        (cache_root, "cache_root"),
        (run_root, "run_root"),
    ):
        if not isinstance(root, Path) or not root.is_absolute():
            raise ValueError(f"{name} must be an absolute Path")
    request = pickle.dumps(
        {
            "config": config,
            "cache_root": cache_root,
            "run_root": run_root,
            "smoke_run_id": smoke_run_id,
            "parent_process_id": os.getpid(),
        },
        protocol=5,
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    for name in _THREAD_ENVIRONMENT_NAMES:
        environment[name] = "1"
    completed = subprocess.run(
        [sys.executable, "-m", "vfe4.training.smoke_worker"],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        input=request,
        capture_output=True,
        check=False,
        timeout=300,
    )
    try:
        response = pickle.loads(completed.stdout)
    except Exception as exc:
        stderr = completed.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(
            "isolated smoke worker returned no typed response: "
            f"exit={completed.returncode}; stderr={stderr}"
        ) from exc
    if (
        type(response) is not tuple
        or len(response) not in (2, 4)
        or type(response[0]) is not bool
    ):
        raise RuntimeError("isolated smoke worker response is malformed")
    if response[0] is not True:
        raise RuntimeError(
            "isolated smoke worker failed: "
            f"{response[1]}: {response[2]}\n{response[3]}"
        )
    result = response[1]
    if type(result) is not WT103SyntheticSmokeResult:
        raise RuntimeError("isolated smoke worker returned a foreign result")
    result.__post_init__()
    if (
        result.runtime_observation.parent_process_id != os.getpid()
        or result.runtime_observation.worker_process_id
        == result.runtime_observation.parent_process_id
    ):
        raise RuntimeError("isolated smoke process identity is stale")
    return result


__all__ = [
    "SyntheticArmExecutionTrace",
    "SyntheticRuntimeObservation",
    "SyntheticSmokeArmResult",
    "WT103SyntheticSmokeResult",
    "run_wt103_synthetic_smoke",
]
