"""Target-blind CPU validation for canonical H6-Prediction v3 checkpoints."""

from __future__ import annotations

import hashlib
import math

import torch
from torch import Tensor, nn

from vfe4.artifacts.h6_prediction_v3 import (
    H6ValidationRecordV3,
    _create_h6_validation_record_v3,
    _validate_planned_checkpoint_v3,
)
from vfe4.data.access import (
    _consume_h6_validation_capability_v3,
)
from vfe4.data.windows import CausalPrefix
from vfe4.predictive.proposal import EstimatorStream
from vfe4.training.arms import (
    _predictive_boundary,
    build_arm_model,
)
from vfe4.training.checkpoint_v3 import H6CheckpointV3, H6TensorRecordV3
from vfe4.training.h6_experiment_v3 import (
    H6ExperimentPlanV3,
    H6PlannedAttemptV3,
    model_factory_sha256_v3,
)


def _model_records(
    checkpoint: H6CheckpointV3,
) -> tuple[H6TensorRecordV3, ...]:
    records = tuple(
        record
        for record in checkpoint.module_tensors
        if record.name.startswith("model.")
    )
    if not records or any(record.name == "model." for record in records):
        raise ValueError("checkpoint model tensor inventory is missing")
    return records


def _fresh_cpu_model(
    checkpoint: H6CheckpointV3,
    *,
    plan: H6ExperimentPlanV3,
    planned_attempt: H6PlannedAttemptV3,
) -> nn.Module:
    config = next(
        (
            item
            for item in plan.endpoint_configs
            if item.config_id == planned_attempt.endpoint_config_id
        ),
        None,
    )
    if config is None or config.config_sha256 != (
        planned_attempt.endpoint_config_sha256
    ):
        raise ValueError("planned endpoint configuration is not in the plan")
    if planned_attempt.attempt_spec.model_factory_sha256 != (
        model_factory_sha256_v3(config)
    ):
        raise ValueError("planned model-factory identity is stale")
    model = build_arm_model(config)
    model.to(device=torch.device("cpu"), dtype=torch.float64)
    state = model.state_dict()
    records = _model_records(checkpoint)
    decoded = {
        record.name.removeprefix("model."): record.decode_cpu()
        for record in records
    }
    if tuple(sorted(decoded)) != tuple(sorted(state)):
        raise ValueError("checkpoint model tensor inventory is incomplete")
    parameter_names = {name for name, _ in model.named_parameters()}
    buffer_names = {name for name, _ in model.named_buffers()}
    for record in records:
        name = record.name.removeprefix("model.")
        expected_role = (
            "module_parameter" if name in parameter_names else "module_buffer"
        )
        if name not in parameter_names | buffer_names or record.role != expected_role:
            raise ValueError("checkpoint model tensor role does not match the model")
        tensor = decoded[name]
        if tensor.device.type != "cpu" or (
            tensor.is_floating_point() and tensor.dtype is not torch.float64
        ):
            raise ValueError("checkpoint model state is not CPU float64")
    model.load_state_dict(decoded, strict=True)
    aliases: dict[tuple[str, int | None, int], str] = {}
    for name, observed in model.state_dict().items():
        checkpoint_record = next(
            record for record in records if record.name == f"model.{name}"
        )
        reproduced = H6TensorRecordV3.capture(
            role=checkpoint_record.role,
            name=checkpoint_record.name,
            tensor=observed,
            aliases=aliases,
        )
        if (
            reproduced.manifest_payload()
            != checkpoint_record.manifest_payload()
            or reproduced.raw_bytes() != checkpoint_record.raw_bytes()
        ):
            raise ValueError("fresh CPU model does not reproduce checkpoint bytes")
    for parameter in model.parameters():
        if parameter.device.type != "cpu" or (
            parameter.is_floating_point() and parameter.dtype is not torch.float64
        ):
            raise ValueError("fresh scoring model is not CPU float64")
        parameter.requires_grad_(False)
    for buffer in model.buffers():
        if buffer.device.type != "cpu" or (
            buffer.is_floating_point() and buffer.dtype is not torch.float64
        ):
            raise ValueError("fresh scoring buffers are not CPU float64")
    model.eval()
    return model


def _validate_prior_log_probs(
    value: object,
    *,
    vocabulary_size: int,
) -> Tensor:
    if (
        type(value) is not Tensor
        or value.device.type != "cpu"
        or value.dtype is not torch.float64
        or value.ndim != 1
        or tuple(value.shape) != (vocabulary_size,)
    ):
        raise ValueError(
            "canonical target-blind prior did not return CPU float64 log probabilities"
        )
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError("target-blind prior log probabilities must be finite")
    if not torch.isclose(
        torch.logsumexp(value, dim=0),
        torch.tensor(0.0, dtype=torch.float64),
        atol=1.0e-10,
        rtol=0.0,
    ):
        raise ValueError("target-blind prior log probabilities are not normalized")
    return value


def _canonical_prior_log_probs(
    *,
    model: nn.Module,
    plan: H6ExperimentPlanV3,
    planned_attempt: H6PlannedAttemptV3,
    prefix: CausalPrefix,
) -> Tensor:
    config = next(
        item
        for item in plan.endpoint_configs
        if item.config_id == planned_attempt.endpoint_config_id
    )
    from vfe4.training.arms import _model_family_sha256

    _, predictor = _predictive_boundary(
        config=config,
        model=model,  # type: ignore[arg-type]
        model_family_sha256=_model_family_sha256(config),
    )
    stream = EstimatorStream.create(
        stream_seed=planned_attempt.training_seed,
        estimator_identity=predictor.estimator_identity,
    )
    prediction = predictor.next_token_log_probs(prefix, stream)
    prediction.__post_init__()
    return prediction.log_probs.value()


def score_h6_validation_checkpoint_v3(
    *,
    capability: object,
    checkpoint: H6CheckpointV3,
    planned_attempt: H6PlannedAttemptV3,
    plan: H6ExperimentPlanV3,
) -> H6ValidationRecordV3:
    """Score one planned tuning checkpoint through its canonical prior only."""

    authorized = _consume_h6_validation_capability_v3(
        capability,
        plan=plan,
    )
    raw_checkpoint = _validate_planned_checkpoint_v3(
        checkpoint=checkpoint,
        planned_attempt=planned_attempt,
        plan=plan,
        stage="tuning",
    )
    if planned_attempt.tuning_cell is None:
        raise ValueError("validation scoring requires a planned tuning cell")
    for name, expected in (
        ("readiness_sha256", plan.readiness_sha256),
        ("experiment_config_sha256", plan.experiment_config_sha256),
        ("plan_sha256", plan.plan_sha256),
        ("matching_set_sha256", plan.matching_set_sha256),
        (
            "data_identity_sha256",
            planned_attempt.attempt_spec.data_identity_sha256,
        ),
        (
            "runtime_identity_sha256",
            plan.training_schedule.runtime_identity_sha256,
        ),
    ):
        if getattr(authorized, name, None) != expected:
            raise ValueError("validation capability plan authority drift")
    model = _fresh_cpu_model(
        checkpoint,
        plan=plan,
        planned_attempt=planned_attempt,
    )

    losses: list[float] = []
    # FrozenTensorSnapshot records storage version counters, which inference
    # tensors deliberately omit. Gradient suppression is sufficient here.
    with torch.no_grad():
        for window_index, real_count in enumerate(
            authorized.windows.real_target_counts
        ):
            for target_index in range(real_count):
                prefix = authorized.windows.causal_prefix(
                    window_index=window_index,
                    receiver_t=target_index + 1,
                    vocabulary=authorized.vocabulary,
                )
                log_probs = _validate_prior_log_probs(
                    _canonical_prior_log_probs(
                        model=model,
                        plan=plan,
                        planned_attempt=planned_attempt,
                        prefix=prefix,
                    ),
                    vocabulary_size=authorized.vocabulary.size,
                )
                # The canonical predictor returns before the scorer reads a target.
                target = authorized.windows.targets[window_index][target_index]
                loss = -float(log_probs[target].item())
                if not math.isfinite(loss) or loss < 0.0:
                    raise ValueError("validation prior NLL is not finite/nonnegative")
                losses.append(loss)
    if len(losses) != authorized.windows.counted_target_total:
        raise ValueError("validation target accounting drift")
    return _create_h6_validation_record_v3(
        experiment_config_sha256=plan.experiment_config_sha256,
        plan_sha256=plan.plan_sha256,
        endpoint_config_id=planned_attempt.endpoint_config_id,
        endpoint_config_sha256=planned_attempt.endpoint_config_sha256,
        tuning_cell=planned_attempt.tuning_cell,
        training_seed=planned_attempt.training_seed,
        attempt_spec_sha256=planned_attempt.attempt_spec.attempt_spec_sha256,
        checkpoint_sha256=checkpoint.checkpoint_sha256,
        checkpoint_bytes_sha256=hashlib.sha256(raw_checkpoint).hexdigest(),
        readiness_sha256=plan.readiness_sha256,
        matching_set_sha256=plan.matching_set_sha256,
        data_identity_sha256=planned_attempt.attempt_spec.data_identity_sha256,
        runtime_identity_sha256=plan.training_schedule.runtime_identity_sha256,
        counted_target_total=len(losses),
        total_prior_nll=math.fsum(losses),
    )


__all__ = ["score_h6_validation_checkpoint_v3"]
