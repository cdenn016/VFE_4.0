from __future__ import annotations

import hashlib
import json
import shutil
import struct
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


_FIXTURE_DOMAIN = b"VFE4-H6-VALIDATION-SAFETY-FIXTURE-V1\x00"
_FIXTURE_ROW = struct.Struct("<QH33H")
_TOKEN_SHA256 = "b" * 64
_FIXTURE_SHA256 = "219385f5a5e92aaba59c1158a61327004a6a9283875149da61919f7c4c13a7f9"
_VOCABULARY_SHA256 = (
    "5aea771bc9b54b0e6ad0ce9b5cddbd6d32e89a4201e4f9cd11bb00bf8713dd68"
)
_TOKENIZER_SHA256 = (
    "1c924ca10bed173c8aaa0e2cb6389df02524269d6405bb1339aa3903834689d4"
)
_CANDIDATE_BYTES = b'{"synthetic_complete_candidate":true}'


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _synthetic_fixture_bytes() -> bytes:
    raw = bytearray(
        _FIXTURE_DOMAIN + bytes.fromhex(_TOKEN_SHA256) + struct.pack("<I", 4096)
    )
    for index in range(4096):
        raw += _FIXTURE_ROW.pack(
            index * 32,
            index % 32 + 1,
            *((index + offset) % 258 for offset in range(33)),
        )
    assert len(raw) == 311_369
    return bytes(raw)


def _fixture_reference(root: Path, *, raw: bytes | None = None):
    from vfe4.h6_validation_fixture import ValidationSafetyFixtureReference

    root.mkdir(parents=True)
    payload_path = root / "validation_safety_fixture.bin"
    if raw is None:
        raw = b"not-opened-by-this-test"
        fixture_sha256 = _FIXTURE_SHA256
    else:
        fixture_sha256 = hashlib.sha256(raw).hexdigest()
    payload_path.write_bytes(raw)
    manifest_identity = hashlib.sha256(b"synthetic fixture manifest").hexdigest()
    (root / "manifest.sha256").write_bytes(
        (manifest_identity + "\n").encode("ascii")
    )
    return ValidationSafetyFixtureReference.create(
        local_payload_path=payload_path,
        binary_directory_manifest_sha256=manifest_identity,
        data_identity_sha256="1" * 64,
        access_policy_sha256="2" * 64,
        validation_token_sha256=_TOKEN_SHA256,
        fixture_raw_sha256=fixture_sha256,
        fixture_raw_length=311_369,
        row_count=4096,
    )


class _Vocabulary:
    vocabulary_id = "wikitext-2-byte-v1"
    size = 258
    tokenizer_spec_sha256 = _TOKENIZER_SHA256
    vocabulary_sha256 = _VOCABULARY_SHA256

    @classmethod
    def create(cls, **kwargs: object) -> "_Vocabulary":
        assert kwargs == {
            "vocabulary_id": cls.vocabulary_id,
            "size": cls.size,
            "tokenizer_spec_sha256": cls.tokenizer_spec_sha256,
        }
        return cls()


class _CompleteCandidate:
    def __init__(
        self,
        *,
        candidate_bytes: bytes = _CANDIDATE_BYTES,
        fixture_sha256: str,
        token_sha256: str = _TOKEN_SHA256,
    ) -> None:
        self.canonical_bytes = candidate_bytes
        self.raw_sha256 = hashlib.sha256(candidate_bytes).hexdigest()
        self.schema_version = "h6-validation-perturbations-v1"
        self.generator_version = "h6-validation-perturbations-v1"
        self.seed = 2026072197
        self.vocabulary = _Vocabulary()
        self.validation_token_sha256 = token_sha256
        self.validation_safety_fixture_sha256 = fixture_sha256
        self.full_count = 4096
        self.materialized_count = 4096
        self.records = tuple(
            SimpleNamespace(case_index=index) for index in range(4096)
        )
        self.manifest_sha256 = hashlib.sha256(
            b"synthetic complete candidate inner manifest"
        ).hexdigest()


class _Oracle:
    IndependentVocabularyIdentity = _Vocabulary

    def __init__(self, fixture_sha256: str) -> None:
        self.fixture_sha256 = fixture_sha256
        self.loads: list[dict[str, object]] = []
        self.generations: list[dict[str, object]] = []

    def generate_frozen_validation_perturbations(
        self, fixture_bytes: bytes, **kwargs: object
    ) -> _CompleteCandidate:
        self.generations.append({"fixture_bytes": fixture_bytes, **kwargs})
        assert "record_indices" not in kwargs
        assert "max_cases" not in kwargs
        return _CompleteCandidate(fixture_sha256=self.fixture_sha256)

    def load_frozen_validation_perturbations(
        self, candidate_bytes: bytes, **kwargs: object
    ) -> _CompleteCandidate:
        self.loads.append({"candidate_bytes": candidate_bytes, **kwargs})
        assert candidate_bytes == _CANDIDATE_BYTES
        return _CompleteCandidate(fixture_sha256=self.fixture_sha256)


def _git(*arguments: str, cwd: Path) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _config(module: object, tmp_path: Path, reference: object):
    source = module.FixtureBuildSourceIdentity.create(
        git_head="a" * 40,
        dirty_digest="b" * 64,
    )
    return module.H6ValidationPerturbationBuildResolvedConfig.create(
        source=source,
        fixture_reference=reference,
        artifact_root=tmp_path / "artifacts",
    )


def _rewrite_manifest_and_name(root: Path) -> Path:
    names = (
        "config.json",
        "provenance.json",
        "validation/h6_validation_perturbations_v1.json",
    )
    manifest = b"".join(
        (
            hashlib.sha256((root / name).read_bytes()).hexdigest()
            + "  "
            + name
            + "\n"
        ).encode("ascii")
        for name in names
    )
    (root / "manifest.sha256").write_bytes(manifest)
    renamed = root.with_name(
        "h6-validation-perturbation-candidate-"
        + hashlib.sha256(manifest).hexdigest()
    )
    root.rename(renamed)
    return renamed


def test_candidate_requires_complete_cross_bound_oracle_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = __import__(
        "verification.h6_validation_candidate", fromlist=["unused"]
    )
    fixture_raw = _synthetic_fixture_bytes()
    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", cwd=repo)
    reference = _fixture_reference(repo / "fixture", raw=fixture_raw)
    _git("add", ".", cwd=repo)
    _git(
        "-c",
        "user.name=VFE4 Test",
        "-c",
        "user.email=vfe4@example.invalid",
        "commit",
        "-m",
        "synthetic fixture",
        cwd=repo,
    )
    artifact_root = repo / "candidate-artifacts"
    source = module.capture_fixture_build_source_identity(repo, artifact_root)
    config = module.H6ValidationPerturbationBuildResolvedConfig.create(
        source=source,
        fixture_reference=reference,
        artifact_root=artifact_root,
    )
    oracle = _Oracle(reference.fixture_raw_sha256)
    monkeypatch.setattr(module, "_load_oracle_module", lambda: oracle)

    built = module.build_h6_validation_perturbation_candidate(
        config, repo_root=repo
    )

    assert built.materialized_count == built.full_count == 4096
    assert len(oracle.generations) == 1
    assert len(oracle.loads) >= 2
    assert all(call["require_complete"] is True for call in oracle.loads)
    assert all(
        call["expected_raw_sha256"]
        == hashlib.sha256(_CANDIDATE_BYTES).hexdigest()
        and call["expected_vocabulary_sha256"] == _VOCABULARY_SHA256
        and call["expected_validation_token_sha256"] == _TOKEN_SHA256
        and call["expected_validation_safety_fixture_sha256"]
        == reference.fixture_raw_sha256
        for call in oracle.loads
    )
    assert "record_indices" not in oracle.generations[0]
    assert "max_cases" not in oracle.generations[0]

    placeholder = (
        tmp_path / "tracked-two-record-placeholder.json"
    )
    placeholder.write_bytes(
        (
            Path(__file__).parents[2]
            / "vfe4"
            / "validation"
            / "fixtures"
            / "h6_validation_perturbations_v1.json"
        ).read_bytes()
    )
    placeholder_reference = _fixture_reference(
        tmp_path / "placeholder-fixture"
    )
    placeholder_config = _config(module, tmp_path / "placeholder", placeholder_reference)
    real_oracle = __import__(
        "verification.numpy_oracles.h6_prefix", fromlist=["unused"]
    )
    with pytest.raises(ValueError, match="complete|4,096|4096"):
        module.validate_h6_validation_candidate_bytes(
            placeholder.read_bytes(),
            placeholder_config,
            _oracle_module=real_oracle,
        )
    assert not (tmp_path / "placeholder" / "artifacts").exists()


def test_candidate_publication_is_exact_no_replace_and_strictly_loadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = __import__(
        "verification.h6_validation_candidate", fromlist=["unused"]
    )
    reference = _fixture_reference(tmp_path / "fixture")
    config = _config(module, tmp_path, reference)
    oracle = _Oracle(reference.fixture_raw_sha256)
    candidate = module.validate_h6_validation_candidate_bytes(
        _CANDIDATE_BYTES,
        config,
        _oracle_module=oracle,
    )
    monkeypatch.setattr(
        module.os,
        "replace",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("candidate publication must never use os.replace")
        ),
    )

    artifact = module.publish_h6_validation_perturbation_candidate(
        config,
        candidate,
        _oracle_module=oracle,
    )

    expected_names = {
        "config.json",
        "manifest.sha256",
        "provenance.json",
        "validation",
        "validation/h6_validation_perturbations_v1.json",
    }
    observed_names = {
        path.relative_to(artifact.local_artifact_path).as_posix()
        for path in artifact.local_artifact_path.rglob("*")
    }
    assert observed_names == expected_names
    manifest = (artifact.local_artifact_path / "manifest.sha256").read_bytes()
    expected_order = (
        "config.json",
        "provenance.json",
        "validation/h6_validation_perturbations_v1.json",
    )
    assert tuple(
        line.decode("ascii").split("  ", 1)[1]
        for line in manifest.splitlines()
    ) == expected_order
    assert artifact.directory_manifest_sha256 == hashlib.sha256(manifest).hexdigest()
    assert artifact.local_artifact_path.name == (
        "h6-validation-perturbation-candidate-"
        + artifact.directory_manifest_sha256
    )
    assert (
        artifact.local_artifact_path
        / "validation"
        / "h6_validation_perturbations_v1.json"
    ).read_bytes() == _CANDIDATE_BYTES
    assert (
        module.load_h6_validation_perturbation_artifact(
            artifact.local_artifact_path,
            _oracle_module=oracle,
        )
        == artifact
    )
    with pytest.raises(module.H6ValidationCandidateError, match="already exists"):
        module.publish_h6_validation_perturbation_candidate(
            config,
            candidate,
            _oracle_module=oracle,
        )

    mutations = ("extra", "manifest-order", "duplicate-config", "cross-binding")
    for mutation in mutations:
        mutation_parent = tmp_path / f"mutation-{mutation}"
        mutation_parent.mkdir()
        copied = mutation_parent / artifact.local_artifact_path.name
        shutil.copytree(artifact.local_artifact_path, copied)
        if mutation == "extra":
            (copied / "extra.json").write_bytes(b"{}")
        elif mutation == "manifest-order":
            lines = (copied / "manifest.sha256").read_bytes().splitlines(
                keepends=True
            )
            (copied / "manifest.sha256").write_bytes(b"".join(reversed(lines)))
        elif mutation == "duplicate-config":
            original = (copied / "config.json").read_bytes()
            assert original.startswith(b"{")
            (copied / "config.json").write_bytes(
                b'{"operation":"duplicate",' + original[1:]
            )
            old = copied
            copied = old.with_name("temporary")
            old.rename(copied)
            copied = _rewrite_manifest_and_name(copied)
        else:
            provenance_path = copied / "provenance.json"
            provenance = json.loads(provenance_path.read_bytes())
            provenance["config_sha256"] = "f" * 64
            provenance_path.write_bytes(_canonical(provenance))
            old = copied
            copied = old.with_name("temporary")
            old.rename(copied)
            copied = _rewrite_manifest_and_name(copied)
        with pytest.raises(
            module.H6ValidationCandidateError,
            match="inventory|manifest|duplicate|canonical|cross|config",
        ):
            module.load_h6_validation_perturbation_artifact(
                copied,
                _oracle_module=oracle,
            )
