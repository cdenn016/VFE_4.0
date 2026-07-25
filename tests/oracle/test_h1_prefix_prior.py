from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path

import pytest

from verification.h1_prefix_prior_gate import (
    EXPECTED_GENERATIVE_FACTOR_SCHEMA_SHA256,
    EXPECTED_H1_PREFIX_PRIOR_FIXTURE_SHA256,
    EXPECTED_H1_PREFIX_PRIOR_V2_FIXTURE_SHA256,
    EXPECTED_PARENT_SPECIFIC_GENERATIVE_FACTOR_SCHEMA_SHA256,
    H1_PREFIX_PRIOR_CONVERGENCE_NAMES,
    H1_PREFIX_PRIOR_INVARIANT_NAMES,
    H1_PREFIX_PRIOR_MEASUREMENT_NAMES,
    evaluate_h1_prefix_prior,
    evaluate_parent_specific_h1_prefix_prior,
    evaluate_parent_specific_h1_prefix_prior_artifact,
    h1_prefix_prior_artifact_payloads,
    parent_specific_h1_prefix_prior_artifact_payloads,
    run_h1_prefix_prior,
    run_parent_specific_h1_prefix_prior,
)
from verification.h7_gate import _validate_h1_prefix_prior_v2_payloads
from verification.numpy_oracles.h1_prefix_prior import (
    load_h1_prefix_prior_fixture,
    load_parent_specific_h1_prefix_prior_fixture,
    parent_specific_prefix_prior_probabilities,
    prefix_prior_probabilities,
)
from vfe4.artifacts import (
    canonical_json_bytes,
    project_h1_prefix_prior_config,
    project_h1_prefix_prior_v2_config,
    source_candidate_sha256,
)
from vfe4.config import (
    H1PrefixPriorResolvedConfig,
    H1PrefixPriorV2ResolvedConfig,
    resolve_h1_prefix_prior_config,
    resolve_h1_prefix_prior_v2_config,
)
from vfe4.types.h6 import H1PrefixPriorArtifactRef
from vfe4.types.results import GateStatus, H1PrefixPriorGateResult


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = (
    REPO_ROOT / "vfe4" / "validation" / "fixtures" / "h1_prefix_prior_v1.json"
)
BASE_FIXTURE_PATH = REPO_ROOT / "vfe4" / "validation" / "fixtures" / "h1_v1.json"
V2_FIXTURE_PATH = (
    REPO_ROOT / "vfe4" / "validation" / "fixtures" / "h1_prefix_prior_v2.json"
)
ORACLE_PATH = REPO_ROOT / "verification" / "numpy_oracles" / "h1_prefix_prior.py"


def _sha(character: str) -> str:
    return character * 64


def _raw_config(run_root: Path) -> dict[str, object]:
    return {
        "schema_version": "h1-prefix-prior-config-v1",
        "operation": "H1-Prefix-Prior",
        "source": {
            "git_head": "1" * 40,
            "dirty_digest": _sha("2"),
            "source_sha256": _sha("3"),
        },
        "fixture": {
            "fixture_id": "h1-prefix-prior-v1",
            "fixture_sha256": EXPECTED_H1_PREFIX_PRIOR_FIXTURE_SHA256,
            "base_fixture_sha256": (
                "388e38cc8c16d8b5e2c61919c1e712a134d88fb0bbd8ec1f2939b9859c9a583b"
            ),
        },
        "generative_factor_schema_sha256": (
            EXPECTED_GENERATIVE_FACTOR_SCHEMA_SHA256
        ),
        "model": {
            "horizon": 2,
            "d_z": 1,
            "d_m": 1,
            "vocabulary_size": 3,
            "state_parent_sets": [[0], [0, 1]],
            "model_parent_sets": [[0], [0, 1]],
            "latent_projection_policy": "exact_zero",
            "prefix_policy": "strictly_prior_tokens",
        },
        "quadrature": {
            "order": 21,
            "convergence_check_order": 17,
            "maximum_convergence_estimate": 1e-9,
        },
        "artifact_root": str(run_root),
    }


def _raw_v2_config(run_root: Path) -> dict[str, object]:
    git_head = "1" * 40
    dirty_digest = _sha("2")
    return {
        "schema_version": "h1-prefix-prior-config-v2",
        "operation": "H1-Prefix-Prior",
        "source": {
            "git_head": git_head,
            "dirty_digest": dirty_digest,
            "source_sha256": source_candidate_sha256(
                git_head_value=git_head,
                dirty_digest_value=dirty_digest,
            ),
        },
        "fixture": {
            "fixture_id": "h1-prefix-prior-scorer-v2",
            "fixture_sha256": EXPECTED_H1_PREFIX_PRIOR_V2_FIXTURE_SHA256,
            "base_fixture_sha256": (
                "388e38cc8c16d8b5e2c61919c1e712a134d88fb0bbd8ec1f2939b9859c9a583b"
            ),
        },
        "generative_factor_schema_sha256": (
            EXPECTED_PARENT_SPECIFIC_GENERATIVE_FACTOR_SCHEMA_SHA256
        ),
        "scorer_schema": "parent-specific-pooled-prefix-bilinear-v1",
        "model": {
            "horizon": 2,
            "d_z": 1,
            "d_m": 1,
            "vocabulary_size": 3,
            "state_parent_sets": [[0], [0, 1]],
            "model_parent_sets": [[0], [0, 1]],
            "latent_projection_policy": "nonzero_bank_projections",
            "parent_history_policy": "active_swapped_distinct_nonzero",
            "prefix_policy": "strictly_prior_tokens",
        },
        "quadrature": {
            "order": 21,
            "convergence_check_order": 17,
            "maximum_convergence_estimate": 1e-9,
        },
        "artifact_root": str(run_root),
    }


def _manifest(payload_bytes: dict[str, bytes]) -> bytes:
    return "".join(
        f"{hashlib.sha256(content).hexdigest()}  {name}\n"
        for name, content in sorted(payload_bytes.items())
    ).encode("ascii")


def test_fixture_is_target_blind_normalized_and_numpy_independent() -> None:
    fixture_bytes = FIXTURE_PATH.read_bytes()
    assert hashlib.sha256(fixture_bytes).hexdigest() == (
        "b6638ea3b64c7fd68882cbaced914e4d17d2cd03c8b6b8a939fd575a1b9f43f1"
    )
    assert EXPECTED_H1_PREFIX_PRIOR_FIXTURE_SHA256 == hashlib.sha256(
        fixture_bytes
    ).hexdigest()

    fixture = load_h1_prefix_prior_fixture(FIXTURE_PATH)
    first = prefix_prior_probabilities(fixture, case_id="prefix-token-0")
    second = prefix_prior_probabilities(fixture, case_id="prefix-token-2")
    current_target = prefix_prior_probabilities(fixture, use_current_target=True)

    assert fixture.structure == (2, 1, 1, 3)
    assert fixture.state_parent_sets == ((0,), (0, 1))
    assert fixture.model_parent_sets == ((0,), (0, 1))
    assert fixture.state_latent_projection == ((0.0,),)
    assert fixture.model_latent_projection == ((0.0,),)
    assert fixture.generative_factor_schema_sha256 == (
        "f38a83b80e046e1d4115a9eca2ccc3afe080fd6b0352fcef399afaf30bea6816"
    )
    assert fixture.generative_factor_schema_sha256 == (
        EXPECTED_GENERATIVE_FACTOR_SCHEMA_SHA256
    )
    for record in (first, second, current_target):
        assert math.fsum(record.state_probabilities) == pytest.approx(1.0, abs=1e-15)
        assert math.fsum(record.model_probabilities) == pytest.approx(1.0, abs=1e-15)
        assert all(value > 0.0 for value in record.state_probabilities)
        assert all(value > 0.0 for value in record.model_probabilities)
    assert first.state_probabilities != second.state_probabilities
    assert first.model_probabilities != second.model_probabilities
    assert current_target.prefix_token_ids == (fixture.current_target_token_id,)
    assert current_target.state_probabilities != first.state_probabilities
    assert current_target.model_probabilities != first.model_probabilities

    tree = ast.parse(ORACLE_PATH.read_text(encoding="utf-8"))
    imported_roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    assert "vfe4" not in imported_roots


def test_parent_specific_joint_is_normalized_and_target_blind() -> None:
    fixture_bytes = V2_FIXTURE_PATH.read_bytes()
    assert hashlib.sha256(fixture_bytes).hexdigest() == (
        EXPECTED_H1_PREFIX_PRIOR_V2_FIXTURE_SHA256
    )
    fixture = load_parent_specific_h1_prefix_prior_fixture(V2_FIXTURE_PATH)
    active = parent_specific_prefix_prior_probabilities(
        fixture, history_id="active"
    )
    swapped = parent_specific_prefix_prior_probabilities(
        fixture, history_id="swapped"
    )

    assert fixture.structure == (2, 1, 1, 3)
    assert fixture.fixed_prefix_token_ids == (2,)
    assert fixture.state_latent_projection != ((0.0,),)
    assert fixture.model_latent_projection != ((0.0,),)
    assert fixture.active_state_latents == tuple(reversed(fixture.swapped_state_latents))
    assert fixture.active_model_latents == tuple(reversed(fixture.swapped_model_latents))
    for record in (active, swapped):
        assert math.fsum(record.state_probabilities) == pytest.approx(
            1.0, abs=1e-15
        )
        assert math.fsum(record.model_probabilities) == pytest.approx(
            1.0, abs=1e-15
        )
        assert all(value > 0.0 for value in record.state_probabilities)
        assert all(value > 0.0 for value in record.model_probabilities)
    assert swapped.state_probabilities == pytest.approx(
        tuple(reversed(active.state_probabilities)), abs=1e-15
    )
    assert swapped.model_probabilities == pytest.approx(
        tuple(reversed(active.model_probabilities)), abs=1e-15
    )

    evaluation = evaluate_parent_specific_h1_prefix_prior(
        fixture_bytes,
        base_fixture_bytes=BASE_FIXTURE_PATH.read_bytes(),
    )
    assert evaluation.status is GateStatus.PASS
    assert all(item.passed for item in evaluation.invariants)
    assert tuple(evaluation.production_priors) == (
        "active",
        "swapped",
        "target_suffix_a",
        "target_suffix_b",
    )
    for history_id in ("active", "swapped"):
        production_prior = evaluation.production_priors[history_id]
        independent_prior = evaluation.independent_priors[history_id]
        assert production_prior.state_probabilities == pytest.approx(
            independent_prior.state_probabilities, abs=1e-15
        )
        assert production_prior.model_probabilities == pytest.approx(
            independent_prior.model_probabilities, abs=1e-15
        )
        production_h1 = evaluation.production_objectives[history_id]
        independent_h1 = evaluation.independent_objectives[history_id]
        assert production_h1.monolithic.value == pytest.approx(
            production_h1.local.complete_elbo,
            abs=production_h1.monolithic.numerical_allowance.total
            + production_h1.local.allowances.complete_elbo.total,
        )
        assert production_h1.monolithic.value == pytest.approx(
            independent_h1.identity.elbo_from_identity,
            abs=production_h1.monolithic.numerical_allowance.total
            + independent_h1.identity.identity_allowance.total,
        )
    control_a = evaluation.production_priors["target_suffix_a"]
    control_b = evaluation.production_priors["target_suffix_b"]
    assert control_a.state_factor_identity_sha256 == (
        control_b.state_factor_identity_sha256
    )
    assert control_a.model_factor_identity_sha256 == (
        control_b.model_factor_identity_sha256
    )
    assert control_a.state_probabilities == control_b.state_probabilities
    assert control_a.model_probabilities == control_b.model_probabilities
    assert (
        evaluation.production_objectives["active"].monolithic.value
        != evaluation.production_objectives["swapped"].monolithic.value
    )

    schema = dict(fixture.generative_factor_schema)
    assert schema["prior_variant"] == "parent_specific_pooled_prefix"
    assert schema["scorer_schema"] == (
        "parent-specific-pooled-prefix-bilinear-v1"
    )
    assert "prefix_conditioned" not in json.dumps(schema, sort_keys=True)


def test_parent_specific_scorer_v2_config_publishes_typed_artifact_bytes(
    tmp_path: Path,
) -> None:
    raw = _raw_v2_config(tmp_path / "runs")
    config = resolve_h1_prefix_prior_v2_config(raw, repo_root=REPO_ROOT)
    assert type(config) is H1PrefixPriorV2ResolvedConfig
    assert config.generative_factor_schema_sha256 == (
        EXPECTED_PARENT_SPECIFIC_GENERATIVE_FACTOR_SCHEMA_SHA256
    )
    with pytest.raises(ValueError, match="not an H1 Prefix Prior configuration"):
        resolve_h1_prefix_prior_config(raw, repo_root=REPO_ROOT)

    evaluation = evaluate_parent_specific_h1_prefix_prior_artifact(
        config,
        fixture_bytes=V2_FIXTURE_PATH.read_bytes(),
        base_fixture_bytes=BASE_FIXTURE_PATH.read_bytes(),
        junit_sha256="c" * 64,
    )
    assert evaluation.result.status is GateStatus.PASS
    payloads = parent_specific_h1_prefix_prior_artifact_payloads(
        config,
        evaluation,
    )
    assert tuple(payloads) == (
        "config.json",
        "schemas/generative_factor.json",
        "validation/h1_prefix_prior.json",
    )
    assert (
        payloads["validation/h1_prefix_prior.json"]["junit_sha256"]
        == "c" * 64
    )
    _validate_h1_prefix_prior_v2_payloads(
        payloads,
        repo_root=REPO_ROOT,
        git_head=config.source.git_head,
        dirty_digest=config.source.dirty_digest,
        junit_sha256="c" * 64,
    )
    for field, invalid in (
        ("schema_version", "h1-prefix-prior-validation-v2"),
        ("junit_sha256", "d" * 64),
        ("source_sha256", "d" * 64),
        ("scorer_schema", "pooled-history-legacy-v1"),
    ):
        mutated = dict(payloads)
        mutated_validation = dict(
            payloads["validation/h1_prefix_prior.json"]
        )
        mutated_validation[field] = invalid
        mutated["validation/h1_prefix_prior.json"] = mutated_validation
        with pytest.raises(ValueError, match="scorer-v2"):
            _validate_h1_prefix_prior_v2_payloads(
                mutated,
                repo_root=REPO_ROOT,
                git_head=config.source.git_head,
                dirty_digest=config.source.dirty_digest,
                junit_sha256="c" * 64,
            )
    payload_bytes = {
        name: canonical_json_bytes(payload)
        for name, payload in payloads.items()
    }
    reference = H1PrefixPriorArtifactRef.from_bytes(
        artifact_path=Path("validation/h1_prefix_prior.json"),
        manifest_bytes=_manifest(payload_bytes),
        git_head=config.source.git_head,
        dirty_digest=config.source.dirty_digest,
        generative_factor_schema_bytes=payload_bytes[
            "schemas/generative_factor.json"
        ],
        config_bytes=payload_bytes["config.json"],
        validation_payload_bytes=payload_bytes[
            "validation/h1_prefix_prior.json"
        ],
    )
    assert reference.status is GateStatus.PASS
    assert reference.generative_factor_schema_sha256 == (
        config.generative_factor_schema_sha256
    )
    assert reference.generative_factor_schema_sha256 != (
        EXPECTED_GENERATIVE_FACTOR_SCHEMA_SHA256
    )
    with pytest.raises(
        ValueError,
        match="generative_factor_schema_sha256",
    ):
        H1PrefixPriorArtifactRef.from_bytes(
            artifact_path=Path("validation/h1_prefix_prior.json"),
            manifest_bytes=_manifest(payload_bytes),
            git_head=config.source.git_head,
            dirty_digest=config.source.dirty_digest,
            generative_factor_schema_bytes=canonical_json_bytes(
                {"schema_version": "legacy-v1"}
            ),
            config_bytes=payload_bytes["config.json"],
            validation_payload_bytes=payload_bytes[
                "validation/h1_prefix_prior.json"
            ],
        )
    assert callable(run_parent_specific_h1_prefix_prior)


def test_versioned_h1_projectors_reject_cross_version_dispatch(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="schema_version"):
        project_h1_prefix_prior_v2_config(
            _raw_config(tmp_path / "legacy-runs")
        )
    with pytest.raises(ValueError, match="schema_version"):
        project_h1_prefix_prior_config(
            _raw_v2_config(tmp_path / "scorer-v2-runs")
        )


def test_gate_compares_all_h1_paths_and_builds_typed_artifact_bytes(
    tmp_path: Path,
) -> None:
    config = resolve_h1_prefix_prior_config(
        _raw_config(tmp_path / "runs"),
        repo_root=REPO_ROOT,
    )
    assert type(config) is H1PrefixPriorResolvedConfig
    assert config.config_sha256 == hashlib.sha256(
        config.canonical_json.encode("utf-8")
    ).hexdigest()
    assert tuple(json.loads(config.canonical_json)) == (
        "artifact_root",
        "fixture",
        "generative_factor_schema_sha256",
        "model",
        "operation",
        "quadrature",
        "schema_version",
        "source",
    )

    evaluation = evaluate_h1_prefix_prior(
        config,
        fixture_bytes=FIXTURE_PATH.read_bytes(),
        base_fixture_bytes=BASE_FIXTURE_PATH.read_bytes(),
    )
    result = evaluation.result
    assert type(result) is H1PrefixPriorGateResult
    assert result.gate == "H1-Prefix-Prior"
    assert result.status is GateStatus.PASS
    assert tuple(result.measurements) == H1_PREFIX_PRIOR_MEASUREMENT_NAMES
    assert tuple(item.name for item in result.invariants) == (
        H1_PREFIX_PRIOR_INVARIANT_NAMES
    )
    convergence_invariants = {
        item.name: item
        for item in result.invariants
        if item.name.startswith("convergence.")
    }
    assert tuple(convergence_invariants) == tuple(
        f"convergence.{name}" for name in H1_PREFIX_PRIOR_CONVERGENCE_NAMES
    )
    assert all(item.passed for item in convergence_invariants.values())
    assert all(
        item.limit == config.maximum_convergence_estimate
        for item in convergence_invariants.values()
    )
    assert result.measurements["monolithic_elbo"] == pytest.approx(
        result.measurements["local_elbo"],
        abs=result.calibrated_allowance,
    )
    assert result.measurements["monolithic_elbo"] == pytest.approx(
        result.measurements["evidence_minus_posterior_kl"],
        abs=result.calibrated_allowance,
    )
    assert evaluation.validation_payload["negative_controls"][
        "current_target_as_prefix"
    ]["passed"] is True
    assert evaluation.validation_payload["fixture_sha256"] == (
        EXPECTED_H1_PREFIX_PRIOR_FIXTURE_SHA256
    )
    assert evaluation.validation_payload["generative_factor_schema_sha256"] == (
        EXPECTED_GENERATIVE_FACTOR_SCHEMA_SHA256
    )
    computation = evaluation.validation_payload["computation"]
    assert isinstance(computation, dict)
    convergence = computation["convergence_estimates"]
    assert isinstance(convergence, dict)
    assert tuple(convergence) == H1_PREFIX_PRIOR_CONVERGENCE_NAMES
    source_priors = computation["source_priors"]
    assert isinstance(source_priors, dict)
    for case in ("active", "alternate"):
        production_prior = source_priors[case]
        independent_prior = source_priors[f"independent_{case}"]
        assert isinstance(production_prior, dict)
        assert isinstance(independent_prior, dict)
        assert production_prior["prefix_token_ids"] == (
            independent_prior["prefix_token_ids"]
        )
        assert production_prior["state_probabilities"] == pytest.approx(
            independent_prior["state_probabilities"],
            abs=256.0 * float.fromhex("0x1.0000000000000p-52"),
        )
        assert production_prior["model_probabilities"] == pytest.approx(
            independent_prior["model_probabilities"],
            abs=256.0 * float.fromhex("0x1.0000000000000p-52"),
        )

    payloads = h1_prefix_prior_artifact_payloads(config, evaluation)
    assert tuple(payloads) == (
        "config.json",
        "schemas/generative_factor.json",
        "validation/h1_prefix_prior.json",
    )
    payload_bytes = {
        name: canonical_json_bytes(payload) for name, payload in payloads.items()
    }
    validation = json.loads(
        payload_bytes["validation/h1_prefix_prior.json"].decode("utf-8")
    )
    assert validation["gate"] == "H1-Prefix-Prior"
    assert validation["status"] == "pass"
    assert validation["obligations"] == []
    assert validation["config_sha256"] == hashlib.sha256(
        payload_bytes["config.json"]
    ).hexdigest()
    assert validation["generative_factor_schema_sha256"] == hashlib.sha256(
        payload_bytes["schemas/generative_factor.json"]
    ).hexdigest()

    reference = H1PrefixPriorArtifactRef.from_bytes(
        artifact_path=Path("validation/h1_prefix_prior.json"),
        manifest_bytes=_manifest(payload_bytes),
        git_head=config.source.git_head,
        dirty_digest=config.source.dirty_digest,
        generative_factor_schema_bytes=payload_bytes[
            "schemas/generative_factor.json"
        ],
        config_bytes=payload_bytes["config.json"],
        validation_payload_bytes=payload_bytes[
            "validation/h1_prefix_prior.json"
        ],
    )
    assert reference.status is GateStatus.PASS
    assert callable(run_h1_prefix_prior)
