"""Click-to-run H6-Prediction v3 experiment launcher.

Edit ``CONFIG`` in this file, enable exactly one operation, provide that
operation's exact authorization phrase, and click Run. Importing this module
or running it with every operation disabled performs no repository, artifact,
data, model, CUDA, training, validation, or held-out-scoring work.
"""

from __future__ import annotations

import hashlib
import hmac
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


def _scientific_config_template() -> dict[str, object]:
    """Return one complete editable-shape v3 scientific configuration."""

    endpoint_phases = [
        {
            "endpoint_config_sha256": digest,
            "latent_enabled": latent_enabled,
            "phases": (
                [
                    "recognition_adamw",
                    "immutable_detached_snapshot",
                    "model_adamw",
                ]
                if latent_enabled
                else ["model_ce_adamw"]
            ),
            "recognition_updates_per_batch": 1 if latent_enabled else 0,
            "model_updates_per_batch": 1,
            "no_op_phases": 0,
        }
        for digest, latent_enabled in (
            (
                "2340c403c5e6f4ac25dea2d29db16c80813a1d9ef3d49e632984d84421f1bc9b",
                False,
            ),
            (
                "2cc89da81814e64a1e631b7a35bb0b6a82bd188a71ca59784c6511ad4084f801",
                True,
            ),
            (
                "5108eb0282bdcc928e09753121138ffa9cd4222e7220211996ec5382f88a9d6a",
                True,
            ),
            (
                "93b1d810ed5241f2ade9c4f8962e5430fd1c17b0883fe7253dad803fb49608f0",
                True,
            ),
            (
                "ad720e158315419e1d8bb8144f33ddffdaba87f1d0b78ad707613c08163282f4",
                True,
            ),
            (
                "1614dac8f7b46cd6da02cdca33774e7e64ded31df21f84f113ab199778886a71",
                True,
            ),
            (
                "06c48a1125cc7a1ba6f5e47b5e5214b288c03d227b31429cc96bdc4b4e805334",
                True,
            ),
            (
                "7b02376ea173b9c7fb2c65db220e22e41126a8bed555a09cb128ff9c8ef8a408",
                True,
            ),
            (
                "4140525a58759b47ded2e48b8b85884b58aa8aa67a44848510fa222322155ea7",
                True,
            ),
            (
                "5474a8cb0011b2fbdd6f21e123622f29abf0bb314812adc39ef008778b764f89",
                True,
            ),
            (
                "74cc2284f8ed0d809fba478ad1df45f955a74fa1211d085055c3f387fb3106d9",
                False,
            ),
            (
                "271f46382e230e60e31dcb00d160094e12b37cd43e7fcf47e6e0f923ed7a5f72",
                True,
            ),
        )
    ]
    estimator = {
        "schema_version": "h6-recognition-estimator-v3",
        "evaluation_method": "reparameterized_mc",
        "continuous_base_samples_per_receiver_per_example_per_phase": 1,
        "categorical_evaluation": "exact_support_sum",
        "component_sampling": "common_random_numbers_per_receiver",
        "gaussian_entropy": "analytic",
        "estimator_sha256": (
            "7d5ec553f30a9b66e6f5f621b26511cb2629aca48f27d58e2c3d647c8962df19"
        ),
    }
    runtime = {
        "schema_version": "h6-prediction-runtime-v3",
        "python_executable": "C:/anaconda/python.exe",
        "python_version": "3.13.5",
        "torch_full_version": "2.10.0.dev20251210+cu128",
        "cuda_runtime_version": "12.8",
        "training_device": "cuda:0",
        "training_dtype": "float64",
        "validation_device": "cpu",
        "heldout_scoring_device": "cpu",
        "scoring_dtype": "float64",
        "cuda_device_name": "NVIDIA GeForce RTX 5090",
        "cuda_compute_capability": [12, 0],
        "deterministic_policy_sha256": (
            "de6d7ad87b9389a2d2ea20fa98e6ae17ab162269b656b74cecb01b3e7c5b31c7"
        ),
        "runtime_identity_sha256": (
            "6757ec16320253535100e6aca9f25ded64688cc382234c832ec95b9e4d11da42"
        ),
    }
    return {
        "schema_version": "h6-prediction-config-v3",
        "operation": "H6-Prediction",
        "source": {
            "git_head": "1111111111111111111111111111111111111111",
            "dirty_digest": "2" * 64,
            "source_sha256": "5" * 64,
        },
        "data": {
            "schema_version": "h6-data-config-v1",
            "source_url": (
                "https://s3.amazonaws.com/research.metamind.io/"
                "wikitext/wikitext-2-raw-v1.zip"
            ),
            "max_archive_bytes": 16_777_216,
            "member_paths": [
                "wikitext-2-raw/",
                "wikitext-2-raw/wiki.train.raw",
                "wikitext-2-raw/wiki.valid.raw",
                "wikitext-2-raw/wiki.test.raw",
            ],
            "allowed_compression_methods": [0, 8],
            "max_member_bytes": 16_777_216,
            "max_total_uncompressed_bytes": 33_554_432,
            "max_compression_ratio": 100,
            "observed_archive": None,
        },
        "prerequisites": {
            "correctness_manifests": {
                "H1": "6" * 64,
                "H2": "7" * 64,
                "H3": "8" * 64,
                "H5": "9" * 64,
            },
            "h1_prefix_prior_manifest_sha256": "a" * 64,
            "h1_prefix_prior_generative_factor_schema_sha256": (
                "0ab33d1cc790711eee82c598bb853d46ab52662eb31e9433e973978e77d9e375"
            ),
            "smc_validation_manifest_sha256": "b" * 64,
            "prefix_certificate_set_sha256": "c" * 64,
            "a0_direct_exact_prefix_certificate_sha256": "1" * 64,
        },
        "h5_update_binding_sha256": "d" * 64,
        "training_schedule": {
            "schedule_schema": "h6-training-schedule-v3",
            "outer": {
                "schedule_schema": "h6-outer-schedule-v1",
                "optimizer_class": "AdamW",
                "optimizer_policy_sha256": (
                    "67b498399b293d4f267cb7ffbe5f0e329ac0025adaaa5f86869588ad720f5ce8"
                ),
                "model_updates_per_batch": 1,
                "validation_twentieths_per_pass": 20,
                "full_passes": 2,
            },
            "endpoint_phases": endpoint_phases,
            "recognition_estimator_sha256": estimator["estimator_sha256"],
            "runtime_identity_sha256": runtime["runtime_identity_sha256"],
            "training_noise_domain": "vfe4.h6.training-rmc-normal.v1",
            "counter_mapping_sha256": (
                "eacc87f6fae59aaa9f1ea5b95211018ed5f8976bc2a83c2e7aad000e2517e91a"
            ),
            "phase_ownership_sha256": (
                "2a277140b0c9cf4bf07820ea1948cd4a289d30392cce2f9e9f81f190c83d28f4"
            ),
            "checkpoint_codec_sha256": (
                "522e0803aab5da303afac7102829daf477074dbea573161555eddaccfdab284b"
            ),
        },
        "critical_values_sha256": (
            "a127a8b4776f0de17a69d47d4d53229fd70eb73d59311775ea788399e6168f73"
        ),
        "endpoint_smc_protocol": {
            "protocol_schema": "h6-endpoint-smc-v1",
            "particle_counts": [128, 256, 512, 1024],
            "replicate_count": 64,
            "registry_root_seed": 2026072198,
            "common_stream_domain": "h6-wt2-endpoint-mc-v1",
            "simultaneous_interval_count": 352,
            "familywise_alpha": 0.01,
            "critical_value_df63": 4.514490453537714,
            "remainder_contraction": 0.75,
        },
        "smc_bias_semantics_sha256": (
            "f3ce5b0b771f2ef1ca1485e395ea73fbb4d019e59c3772d30e8b2e360ee51950"
        ),
        "attribution_matrix_sha256": (
            "cdaf23b181c9be3d10fcbd892d3790ec5a2f21fb0d49bcd840a066ad85e3bd4a"
        ),
        "matching_policy_schema": "h6-amended-matching-policy-v3",
        "matching_policy_sha256": (
            "a552f19df459905ce70ba63170488382ec374534e368e4d9d6c1afdb11d24ee7"
        ),
        "matching_set_schema": "h6-amended-matching-set-v3",
        "matching_set_sha256": (
            "bfcfa601febd06117e6cb1575cf47b689419337d64dc46bb15b9379d1659b192"
        ),
        "objective_gate": {
            "schema_version": "h6-objective-gate-v1",
            "complete_arm_id": (
                "h6-a5-structured-parent-specific-prefix-exact-"
                "complete-latent-smoothing-v2"
            ),
            "emission_arm_id": (
                "h6-a5-structured-parent-specific-prefix-exact-"
                "emission-latent-smoothing-v2"
            ),
            "orientation": "nll_complete_minus_nll_emission",
            "delta_obj": 0.01005033585350145,
            "opening_policy": "single_all_or_none",
            "evaluation_order": "OBJECTIVE_then_PRIMARY",
            "spec_sha256": (
                "89cdf6c370baa4abd594bab08adb45a7f1c099a19cf6fabf4ccc26c1641dea4d"
            ),
        },
        "data_identity_sha256": "e" * 64,
        "access_policy_sha256": "f" * 64,
        "recognition_contract": {
            "trajectory_schema": "h6-language-recognition-trajectory-v3",
            "categorical_posterior_schema": (
                "h6-categorical-source-posterior-v3"
            ),
            "terminal_mixture_schema": "h6-terminal-source-mixture-v1",
            "estimator": estimator,
        },
        "runtime": runtime,
        "counter_mapping_sha256": (
            "eacc87f6fae59aaa9f1ea5b95211018ed5f8976bc2a83c2e7aad000e2517e91a"
        ),
        "phase_ownership_sha256": (
            "2a277140b0c9cf4bf07820ea1948cd4a289d30392cce2f9e9f81f190c83d28f4"
        ),
        "checkpoint_codec_sha256": (
            "522e0803aab5da303afac7102829daf477074dbea573161555eddaccfdab284b"
        ),
        "scoring_inventory_sha256": (
            "dbc4a85f4a2e6b544d3591d50dcc930ee09d2d2d9b526517ffd7479c6503687e"
        ),
        "expected_test_row_count": 4104,
        "artifact_root": "C:/tmp/vfe4-h6-prediction-v3",
    }


CONFIG: dict[str, object] = {
    "launcher_schema": "vfe4-train-click-run-v3",
    "operations": {
        "prediction_readiness": {
            "enabled": False,
            "authorization": None,
            "config": (
                _operation_config := {
                    "scientific_config": _scientific_config_template(),
                    "correctness_artifact_roots": {
                        "H1": "C:/tmp/vfe4-h1/CURRENT",
                        "H2": "C:/tmp/vfe4-h2/CURRENT",
                        "H3": "C:/tmp/vfe4-h3/CURRENT",
                        "H5": "C:/tmp/vfe4-h5/CURRENT",
                    },
                    "h1_prefix_prior_artifact_root": (
                        "C:/tmp/vfe4-h1-prefix-prior/CURRENT"
                    ),
                    "smc_accuracy_artifact_root": (
                        "C:/tmp/vfe4-h6-smc-accuracy/CURRENT"
                    ),
                    "h6_prefix_artifact_root": (
                        "C:/tmp/vfe4-h6-prefix-v3/CURRENT"
                    ),
                    "h6_prefix_manifest_sha256": "0" * 64,
                    "h6_prefix_junit_sha256": "0" * 64,
                    "blinded_store_manifest_path": (
                        "C:/tmp/vfe4-h6-data-v3/"
                        "authenticated_blinded_store_v3.json"
                    ),
                    "blinded_store_artifact_root": "C:/tmp/vfe4-h6-data-v3",
                    "authorities_run_root": "C:/tmp/vfe4-h6-prediction-v3",
                    "authorities_run_name": "AUTHORITIES",
                    "authorities_directory": (
                        "C:/tmp/vfe4-h6-prediction-v3/AUTHORITIES"
                    ),
                    "planned_attempt_sha256": "0" * 64,
                    "checkpoint_path": (
                        "C:/tmp/vfe4-h6-prediction-v3/checkpoints/selected.h6v3"
                    ),
                    "maximum_checkpoint_bytes": 1_073_741_824,
                    "validation_bundle_directory": (
                        "C:/tmp/vfe4-h6-prediction-v3/VALIDATION"
                    ),
                    "transaction_pointer_root": (
                        "C:/tmp/vfe4-h6-prediction-v3/POINTERS"
                    ),
                    "transaction_pointer_name": "current",
                }
            ),
        },
        "plan": {
            "enabled": False,
            "authorization": None,
            "config": dict(_operation_config),
        },
        "train": {
            "enabled": False,
            "authorization": None,
            "config": dict(_operation_config),
        },
        "score_validation": {
            "enabled": False,
            "authorization": None,
            "config": dict(_operation_config),
        },
        "score_test_transaction": {
            "enabled": False,
            "authorization": None,
            "config": dict(_operation_config),
        },
    },
}
del _operation_config


_AUTHORIZATION_PHRASES = {
    "prediction_readiness": "AUTHORIZE_VFE4_H6_PREDICTION_READINESS_V1",
    "plan": "AUTHORIZE_VFE4_H6_EXPERIMENT_PLAN_V1",
    "train": "AUTHORIZE_VFE4_H6_TRAINING_V1",
    "score_validation": "AUTHORIZE_VFE4_H6_VALIDATION_SCORING_V1",
    "score_test_transaction": (
        "AUTHORIZE_VFE4_H6_ONE_TIME_TEST_TRANSACTION_V1"
    ),
}
_OPERATION_NAMES = tuple(_AUTHORIZATION_PHRASES)
_REPO_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class TrainLauncherResult:
    launcher_schema: Literal["vfe4-train-click-run-v3"]
    operation: str | None
    status: Literal["IDLE", "COMPLETED"]
    _payload: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.launcher_schema != "vfe4-train-click-run-v3":
            raise ValueError("unsupported train launcher schema")
        if self.status not in ("IDLE", "COMPLETED"):
            raise ValueError("train launcher status must be IDLE or COMPLETED")
        if self.status == "IDLE":
            if self.operation is not None or self._payload is not None:
                raise ValueError("an idle launcher cannot retain an operation")
        elif self.operation not in _OPERATION_NAMES or self._payload is None:
            raise ValueError("a completed launcher result requires a payload")


def _mapping(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(
        type(key) is not str for key in value
    ):
        raise ValueError(f"{location} must be a string-keyed mapping")
    return value


def _selected_operation(
    config: Mapping[str, object],
) -> tuple[str, Mapping[str, object], str] | None:
    if set(config) != {"launcher_schema", "operations"}:
        raise ValueError("train CONFIG has unknown or missing root keys")
    if config["launcher_schema"] != "vfe4-train-click-run-v3":
        raise ValueError("train CONFIG launcher_schema is unsupported")
    operations = _mapping(config["operations"], "operations")
    if tuple(operations) != _OPERATION_NAMES:
        raise ValueError("train CONFIG operations are incomplete or reordered")
    enabled: list[tuple[str, Mapping[str, object]]] = []
    for name in _OPERATION_NAMES:
        entry = _mapping(operations[name], f"operations.{name}")
        if set(entry) != {"enabled", "authorization", "config"}:
            raise ValueError(f"operations.{name} has unknown or missing keys")
        if type(entry["enabled"]) is not bool:
            raise ValueError(f"operations.{name}.enabled must be boolean")
        _mapping(entry["config"], f"operations.{name}.config")
        if entry["enabled"]:
            enabled.append((name, entry))
    if not enabled:
        return None
    if len(enabled) != 1:
        raise ValueError("enable exactly one train operation")
    name, entry = enabled[0]
    authorization = entry["authorization"]
    if type(authorization) is not str or not hmac.compare_digest(
        authorization,
        _AUTHORIZATION_PHRASES[name],
    ):
        raise PermissionError(
            f"operations.{name}.authorization does not equal its explicit phrase"
        )
    return (
        name,
        _mapping(entry["config"], f"operations.{name}.config"),
        authorization,
    )


def _run_operation(
    operation: str,
    raw: Mapping[str, object],
    authorization: str,
) -> object:
    from vfe4.config import resolve_h6_prediction_v3_config

    scientific = _mapping(raw.get("scientific_config"), "scientific_config")
    resolved = resolve_h6_prediction_v3_config(
        scientific,
        repo_root=_REPO_ROOT,
    )
    runtime = None
    if operation == "train":
        from vfe4.training.h6_runtime_v3 import (
            configure_installed_runtime_v3,
        )

        runtime = configure_installed_runtime_v3(
            expected_identity=resolved.runtime,
        )
    if operation == "score_test_transaction":
        from vfe4.training.h6_experiment_v3 import (
            prepare_h6_test_transaction_v3,
        )
        from vfe4.training.h6_test_transaction_v3 import (
            execute_h6_test_transaction_v3,
        )

        transaction = prepare_h6_test_transaction_v3(
            config=resolved,
            operation_config=raw,
            authorization_sha256=hashlib.sha256(
                authorization.encode("ascii")
            ).hexdigest(),
        )
        return execute_h6_test_transaction_v3(**transaction)

    from vfe4.training.h6_experiment_v3 import run_h6_experiment_v3

    return run_h6_experiment_v3(
        operation=operation,
        config=resolved,
        runtime=runtime,
        operation_config=raw,
        authorization_sha256=hashlib.sha256(
            authorization.encode("ascii")
        ).hexdigest(),
    )


def main(config: Mapping[str, object] = CONFIG) -> TrainLauncherResult:
    selected = _selected_operation(_mapping(config, "CONFIG"))
    if selected is None:
        return TrainLauncherResult(
            "vfe4-train-click-run-v3",
            None,
            "IDLE",
        )
    operation, raw, authorization = selected
    payload = _run_operation(operation, raw, authorization)
    return TrainLauncherResult(
        "vfe4-train-click-run-v3",
        operation,
        "COMPLETED",
        payload,
    )


def _script_main() -> int:
    try:
        result = main()
    except (OSError, PermissionError, RuntimeError, TypeError, ValueError) as exc:
        print(f"VFE4 train operation unavailable: {exc}", file=sys.stderr)
        return 2
    if result.status == "IDLE":
        print("VFE4 train launcher is idle; enable exactly one CONFIG operation.")
    else:
        print(f"VFE4 train operation completed: {result.operation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_script_main())
