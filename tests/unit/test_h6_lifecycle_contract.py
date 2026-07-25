from __future__ import annotations

import copy
import hashlib
import inspect
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from verification.projected_runner import run_projected_current_candidate
from vfe4.artifacts import (
    ArtifactPublicationError,
    CandidateArtifactReference,
    ProjectedCurrentCandidateConfig,
    canonical_json_bytes,
    project_h1_prefix_prior_config,
    project_h6_prefix_config,
)
from vfe4.artifacts.h6 import (
    run_projected_current_candidate as production_run_projected_current_candidate,
)
from vfe4.types.h6 import (
    ArmConfig,
    ArmId,
    CapacityAllocation,
    CausalDag,
    CausalDagRow,
    EstimatorSpec,
    H6LanguageStructure,
    H6PrefixProfilePair,
    VocabularyIdentity,
    ZeroDimensionalBase,
)


def _h6_prefix_raw(tmp_path: Path) -> dict[str, object]:
    source = {
        "git_head": "1" * 40,
        "dirty_digest": "2" * 64,
        "source_sha256": "3" * 64,
    }
    structures = tuple(
        H6LanguageStructure.create(
            base=ZeroDimensionalBase.create(),
            dag=CausalDag.create(
                node_labels=tuple(range(horizon + 1)),
                rows=tuple(
                    CausalDagRow(receiver, tuple(range(receiver)))
                    for receiver in range(1, horizon + 1)
                ),
            ),
            receiver_labels=tuple(range(1, horizon + 1)),
        )
        for horizon in (4, 32)
    )
    configs = tuple(
        ArmConfig.create(
            arm=ArmId.A0,
            config_id="h6-a0-transformer-v2",
            vocabulary=VocabularyIdentity(
                vocabulary_id,
                size,
                "4" * 64,
            ),
            horizon=horizon,
            latent_enabled=False,
            state_channel_enabled=False,
            model_channel_enabled=False,
            source_mode="absent",
            map_mode="absent",
            recognition_family="absent",
            recognition_conditioning="absent",
            prior_variant="absent",
            mixture_mode="absent",
            objective_kind="cross_entropy",
            capacity_allocation=CapacityAllocation.create(
                emission_width=48 if size == 3 else 52,
                latent_width=None,
                recognition_width=None,
            ),
        )
        for vocabulary_id, size, horizon in (
            ("h6-prefix-small-v1", 3, 4),
            ("wikitext-2-byte-v1", 258, 32),
        )
    )
    estimator = EstimatorSpec.create(
        kind="weighted_smc",
        particle_count=4,
        resampling="systematic_ess_half",
    )
    model_hashes = tuple(
        hashlib.sha256(
            b"vfe4.h6.arm-model-family.v1\x00"
            + canonical_json_bytes(
                {
                    "config_sha256": config.config_sha256,
                    "factory": "build_a0@h6-arm-v2",
                }
            )
        ).hexdigest()
        for config in configs
    )
    data_safety_sha256 = hashlib.sha256(
        b"VFE4-H6-TARGET-FREE-PREDICTIVE-BOUNDARY-V1"
    ).hexdigest()
    profile = H6PrefixProfilePair.create(
        profile_id="h6-a0-focused-v1",
        small_arm_config=configs[0],
        production_arm_config=configs[1],
        estimator=estimator,
        small_structure=structures[0],
        production_structure=structures[1],
        data_safety_sha256=data_safety_sha256,
        small_model_family_sha256=model_hashes[0],
        production_model_family_sha256=model_hashes[1],
    )

    def raw_arm(config: ArmConfig) -> dict[str, object]:
        payload = config.canonical_payload()
        payload.pop("capacity_allocation_sha256")
        return payload

    def raw_structure(structure: H6LanguageStructure) -> dict[str, object]:
        return {
            "base": {"base_id": "C0", "points": ["*"], "dimension": 0},
            "dag": {
                "labeling": "zero_based",
                "node_labels": list(structure.dag.node_labels),
                "rows": [
                    {
                        "receiver_t": row.receiver_t,
                        "parents": list(row.parents),
                    }
                    for row in structure.dag.rows
                ],
            },
            "receiver_labels": list(structure.receiver_labels),
        }

    return {
        "schema_version": "h6-prefix-config-v1",
        "operation": "H6-Prefix",
        "source": source,
        "execution_mode": "focused_subset",
        "profiles": [
            {
                "profile_id": profile.profile_id,
                "small_arm_config": raw_arm(configs[0]),
                "production_arm_config": raw_arm(configs[1]),
                "estimator": {
                    "schema_version": estimator.schema_version,
                    "kind": estimator.kind,
                    "particle_count": estimator.particle_count,
                    "resampling": estimator.resampling,
                    "dtype": estimator.dtype,
                    "device": estimator.device,
                },
                "small_structure": raw_structure(structures[0]),
                "production_structure": raw_structure(structures[1]),
                "data_safety_sha256": data_safety_sha256,
                "small_model_family_sha256": model_hashes[0],
                "production_model_family_sha256": model_hashes[1],
                "profile_pair_sha256": profile.profile_pair_sha256,
            }
        ],
        "authorization_sha256": None,
        "artifact_root": str(tmp_path / "h6"),
    }


def test_h6_lifecycle_adapters_match_the_frozen_h7_h8_consumer_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    source = {
        "git_head": "1" * 40,
        "dirty_digest": "2" * 64,
        "source_sha256": "3" * 64,
    }
    h1_raw: dict[str, object] = {
        "schema_version": "h1-prefix-prior-config-v1",
        "operation": "H1-Prefix-Prior",
        "source": source,
        "artifact_root": str(tmp_path / "h1"),
        "owned_sequence": ["h1"],
    }
    h6_raw = _h6_prefix_raw(tmp_path)

    def resolved_h1(raw: object, *, repo_root: Path) -> SimpleNamespace:
        assert repo_root == Path(__file__).resolve().parents[2]
        assert isinstance(raw, dict)
        canonical = canonical_json_bytes(raw).decode("utf-8")
        source_value = raw["source"]
        assert isinstance(source_value, dict)
        return SimpleNamespace(
            operation=raw["operation"],
            canonical_json=canonical,
            config_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            source=SimpleNamespace(
                git_head=source_value["git_head"],
                dirty_digest=source_value["dirty_digest"],
                source_sha256=source_value["source_sha256"],
            ),
        )

    import vfe4.config as config_module

    monkeypatch.setattr(
        config_module,
        "resolve_h1_prefix_prior_config",
        resolved_h1,
    )

    h1_root = {
        "schema_version": "vfe4-click-run-v1",
        "operations": {
            "h1_prefix_prior": {
                "enabled": False,
                "authorization": None,
                "config": h1_raw,
            }
        },
    }
    h6_root = {
        "schema_version": "vfe4-click-run-v1",
        "operations": {
            "h6_prefix": {
                "enabled": False,
                "authorization": None,
                "config": h6_raw,
            }
        },
    }
    before_h1 = copy.deepcopy(h1_root)
    before_h6 = copy.deepcopy(h6_root)
    projected_h1 = project_h1_prefix_prior_config(h1_root)
    projected_h6 = project_h6_prefix_config(h6_root)
    direct_h1 = project_h1_prefix_prior_config(h1_raw)
    direct_h6 = project_h6_prefix_config(h6_raw)

    assert h1_root == before_h1
    assert h6_root == before_h6
    assert projected_h1 == direct_h1
    assert projected_h6 == direct_h6

    from verification import h6_prefix_gate as real_prefix_gate
    from vfe4.config import resolve_h6_prefix_config

    resolved_h6 = resolve_h6_prefix_config(h6_raw, repo_root=repo_root)
    effects: list[str] = []
    monkeypatch.setattr(
        real_prefix_gate,
        "current_source_identity",
        lambda repo_root, run_root: ("f" * 40, "e" * 64, "d" * 64),
    )

    def forbidden_effect(*args: object, **kwargs: object) -> object:
        effects.append("effect")
        raise AssertionError("runner reached a workload before source preflight")

    monkeypatch.setattr(
        real_prefix_gate,
        "load_frozen_validation_perturbations",
        forbidden_effect,
    )
    monkeypatch.setattr(real_prefix_gate, "_small_cases", forbidden_effect)
    monkeypatch.setattr(real_prefix_gate, "build_arm", forbidden_effect)
    with pytest.raises(ValueError, match="stale for the live candidate"):
        real_prefix_gate.run_h6_prefix(
            config=resolved_h6,
            junit_sha256=None,
        )
    assert effects == []

    h1_raw["owned_sequence"].append("mutated")  # type: ignore[union-attr]
    h6_profiles = h6_raw["profiles"]
    assert isinstance(h6_profiles, list)
    h6_profile = h6_profiles[0]
    assert isinstance(h6_profile, dict)
    original_profile_id = h6_profile["profile_id"]
    h6_profile["profile_id"] = "mutated-after-projection"
    assert projected_h1.raw_config["owned_sequence"] == ("h1",)
    projected_profiles = projected_h6.raw_config["profiles"]
    assert isinstance(projected_profiles, tuple)
    assert projected_profiles[0]["profile_id"] == original_profile_id
    with pytest.raises(ValueError, match="direct operation|operations"):
        project_h6_prefix_config({"h6_prefix": h6_raw})

    assert tuple(ProjectedCurrentCandidateConfig.__dataclass_fields__) == (
        "operation",
        "raw_config",
        "canonical_sha256",
    )
    assert tuple(CandidateArtifactReference.__dataclass_fields__) == (
        "artifact_path",
        "git_head",
        "dirty_digest",
        "manifest_sha256",
        "payload_hashes",
    )
    assert tuple(inspect.signature(project_h1_prefix_prior_config).parameters) == (
        "raw_config",
    )
    assert tuple(inspect.signature(project_h6_prefix_config).parameters) == (
        "raw_config",
    )
    assert (
        inspect.signature(project_h1_prefix_prior_config)
        .parameters["raw_config"]
        .annotation
        == "Mapping[str, object]"
    )
    assert (
        inspect.signature(project_h6_prefix_config)
        .parameters["raw_config"]
        .annotation
        == "Mapping[str, object]"
    )
    assert (
        inspect.signature(project_h1_prefix_prior_config).return_annotation
        == "ProjectedCurrentCandidateConfig"
    )
    assert (
        inspect.signature(project_h6_prefix_config).return_annotation
        == "ProjectedCurrentCandidateConfig"
    )
    runner_signature = inspect.signature(run_projected_current_candidate)
    assert runner_signature == inspect.signature(
        production_run_projected_current_candidate
    )
    assert tuple(runner_signature.parameters) == (
        "config",
        "junit_sha256",
        "predecessor_refs",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in runner_signature.parameters.values()
    )
    assert (
        runner_signature.parameters["config"].annotation
        == "ProjectedCurrentCandidateConfig"
    )
    assert runner_signature.parameters["junit_sha256"].annotation == "str | None"
    assert (
        runner_signature.parameters["predecessor_refs"].annotation
        == "Mapping[str, CandidateArtifactReference]"
    )
    assert runner_signature.return_annotation == "CandidateArtifactReference"

    calls: list[tuple[str, str | None]] = []
    h1_directory = tmp_path / "published-h1"
    h6_directory = tmp_path / "published-h6"
    reuse_h1_directory = False

    def publish_fake(
        directory: Path,
        config: object,
        payloads: dict[str, object],
    ) -> None:
        directory.mkdir(parents=True)
        encoded = {"config.json": config.canonical_json.encode("utf-8")}
        encoded.update(
            {
                name: canonical_json_bytes(value)
                for name, value in payloads.items()
            }
        )
        for name, content in encoded.items():
            path = directory / Path(*name.split("/"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        manifest = "".join(
            f"{hashlib.sha256(content).hexdigest()}  {name}\n"
            for name, content in sorted(encoded.items())
        ).encode("ascii")
        (directory / "manifest.sha256").write_bytes(manifest)

    def run_h1(config: SimpleNamespace) -> tuple[SimpleNamespace, Path]:
        nonlocal reuse_h1_directory
        calls.append(("H1-Prefix-Prior", None))
        if not reuse_h1_directory:
            publish_fake(
                h1_directory,
                config,
                {
                    "schemas/generative_factor.json": {"schema": "test"},
                    "validation/h1_prefix_prior.json": {
                        "gate": "H1-Prefix-Prior",
                        "status": "pass",
                        "git_head": config.source.git_head,
                        "dirty_digest": config.source.dirty_digest,
                        "config_sha256": config.config_sha256,
                        "obligations": [],
                    },
                },
            )
        return SimpleNamespace(
            gate="H1-Prefix-Prior",
            status=SimpleNamespace(value="pass"),
            obligations=(),
        ), h1_directory

    def run_h6(
        *,
        config: object,
        junit_sha256: str | None,
    ) -> tuple[SimpleNamespace, Path]:
        calls.append(("H6-Prefix", junit_sha256))
        publish_fake(
            h6_directory,
            config,
            {
                "certificates/prefix_set.json": {
                    "prefix_certificate_set_sha256": "5" * 64,
                    "certificates": [],
                },
                "environment.json": {"kind": "test"},
                "provenance.json": {
                    "git_head": config.source.git_head,
                    "dirty_digest": config.source.dirty_digest,
                    "source_sha256": config.source.source_sha256,
                    "junit_sha256": junit_sha256,
                },
                "validation/h6_prefix.json": {
                    "gate": "H6-Prefix",
                    "status": "inconclusive",
                    "validation_payload_sha256": "4" * 64,
                    "prefix_certificate_set_sha256": "5" * 64,
                    "obligations": ["deferred synthetic execution"],
                },
            },
        )
        return SimpleNamespace(
            gate="H6-Prefix",
            status=SimpleNamespace(value="inconclusive"),
            validation_payload_sha256="4" * 64,
            prefix_certificate_set_sha256="5" * 64,
            obligations=("deferred synthetic execution",),
        ), h6_directory

    h1_module = ModuleType("verification.h1_prefix_prior_gate")
    h1_module.run_h1_prefix_prior = run_h1  # type: ignore[attr-defined]
    h6_module = ModuleType("verification.h6_prefix_gate")
    h6_module.run_h6_prefix = run_h6  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, h1_module.__name__, h1_module)
    monkeypatch.setitem(sys.modules, h6_module.__name__, h6_module)

    import vfe4.artifacts.h6 as h6_artifacts

    monkeypatch.setattr(
        h6_artifacts,
        "_PROJECTED_CURRENT_CANDIDATE_RUNNER",
        None,
    )
    with pytest.raises(
        ArtifactPublicationError,
        match="no eligible projected current-candidate runner",
    ):
        production_run_projected_current_candidate(
            config=projected_h1,
            junit_sha256=None,
            predecessor_refs={},
        )

    h1_reference = run_projected_current_candidate(
        config=projected_h1,
        junit_sha256=None,
        predecessor_refs={},
    )
    junit_sha256 = "a" * 64
    h6_reference = run_projected_current_candidate(
        config=projected_h6,
        junit_sha256=junit_sha256,
        predecessor_refs={},
    )
    assert calls == [
        ("H1-Prefix-Prior", None),
        ("H6-Prefix", junit_sha256),
    ]
    assert tuple(h1_reference.payload_hashes) == (
        "config.json",
        "schemas/generative_factor.json",
        "validation/h1_prefix_prior.json",
    )
    assert tuple(h6_reference.payload_hashes) == (
        "certificates/prefix_set.json",
        "config.json",
        "environment.json",
        "provenance.json",
        "validation/h6_prefix.json",
    )
    with pytest.raises(ValueError, match="predecessor"):
        run_projected_current_candidate(
            config=projected_h6,
            junit_sha256=junit_sha256,
            predecessor_refs={"H1": h1_reference},
        )
    with pytest.raises(ValueError, match="JUnit"):
        run_projected_current_candidate(
            config=projected_h6,
            junit_sha256="A" * 64,
            predecessor_refs={},
        )

    (h1_directory / "validation" / "h1_prefix_prior.json").write_text(
        json.dumps({"gate": "H1-Prefix-Prior", "status": "fail"}),
        encoding="utf-8",
    )
    reuse_h1_directory = True
    with pytest.raises(ArtifactPublicationError, match="hash|manifest"):
        run_projected_current_candidate(
            config=projected_h1,
            junit_sha256=None,
            predecessor_refs={},
        )

    h7_text = (
        repo_root
        / "docs"
        / "superpowers"
        / "plans"
        / "2026-07-21-vfe4-h7-frame-covariance.md"
    ).read_text(encoding="utf-8")
    h8_text = (
        repo_root
        / "docs"
        / "superpowers"
        / "plans"
        / "2026-07-21-vfe4-h8-sparse-scale.md"
    ).read_text(encoding="utf-8")
    combined = h7_text + "\n" + h8_text
    assert "project_h6_prefix_config(CONFIG)" in h7_text
    assert (
        "run_projected_current_candidate(*, config, junit_sha256, "
        "predecessor_refs) -> CandidateArtifactReference"
    ) in h7_text
    assert "H8's pure pinned-schema `project_h7_compatibility_config`" in h8_text
    assert "predecessor_refs={}" in h7_text
    assert "predecessor_refs={}" in h8_text
    for stale in (
        "AtomicArtifactRef",
        "project_h6_prefix_config(CONFIG,current_refs)",
        "project_h6_prefix_config(CONFIG,current_h1_h5_refs)",
        "predecessor_refs=current_h1_h5_refs",
        "run_projected_current_candidate(config,junit_sha256,predecessor_refs)",
    ):
        assert stale not in combined
