"""Stdlib-only producer and strict loader for the full H6 candidate set."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import stat
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import ModuleType, SimpleNamespace
from typing import Final, Literal, Mapping

from vfe4.h6_validation_fixture import (
    H6ValidationPerturbationArtifactReference,
    ValidationSafetyFixtureReference,
    read_validation_safety_fixture_payload,
)


_CONFIG_SCHEMA: Final = "h6-validation-perturbation-build-config-v1"
_OPERATION: Final = "H6-Validation-Perturbations"
_PERTURBATION_SCHEMA: Final = "h6-validation-perturbations-v1"
_SEED: Final = 2026072197
_FULL_COUNT: Final = 4096
_VOCABULARY_ID: Final = "wikitext-2-byte-v1"
_VOCABULARY_SIZE: Final = 258
_TOKENIZER_SPEC_SHA256: Final = (
    "1c924ca10bed173c8aaa0e2cb6389df02524269d6405bb1339aa3903834689d4"
)
_VOCABULARY_SHA256: Final = (
    "5aea771bc9b54b0e6ad0ce9b5cddbd6d32e89a4201e4f9cd11bb00bf8713dd68"
)
_AUTHORIZATION_SHA256: Final = (
    "6a2c61ad2f1ad7fdeb798dee8be231b6ff1393290ad77ac0bd262f2d49da88ae"
)
_SOURCE_DOMAIN: Final = b"VFE4-H6-SOURCE-CANDIDATE-V1\x00"
_REFERENCE_DOMAIN: Final = (
    b"vfe4.h6.validation-perturbation-artifact-reference.v1\x00"
)
_STATUS: Final = "CANDIDATE"
_NONCLAIM: Final = (
    "not Prefix evidence, certificate, readiness, or predictive result"
)
_PAYLOAD_NAMES: Final = (
    "config.json",
    "provenance.json",
    "validation/h6_validation_perturbations_v1.json",
)
_MANIFEST_NAME: Final = "manifest.sha256"
_DIRECTORY_PREFIX: Final = "h6-validation-perturbation-candidate-"
_LOWER_HEX: Final = frozenset("0123456789abcdef")
_WINDOWS_REPARSE_POINT: Final = 0x400
_STDLIB_ORACLE_MODULE_NAME: Final = "_vfe4_h6_prefix_stdlib_oracle_v1"
_STDLIB_ORACLE_RELATIVE_PATH: Final = "numpy_oracles/h6_prefix.py"


class H6ValidationCandidateError(ValueError):
    """The candidate could not be validated, published, or loaded safely."""


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise H6ValidationCandidateError(
            f"canonical JSON serialization failed: {exc}"
        ) from exc


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise H6ValidationCandidateError(
            f"{name} must be lowercase SHA-256 hex"
        )
    return value


def _require_git_head(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 40
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise H6ValidationCandidateError(
            "git_head must be exact lowercase SHA-1 hex"
        )
    return value


def _is_redirect(path: Path, metadata: os.stat_result) -> bool:
    return (
        stat.S_ISLNK(metadata.st_mode)
        or bool(
            getattr(metadata, "st_file_attributes", 0)
            & _WINDOWS_REPARSE_POINT
        )
    )


def _io_path(path: Path) -> str:
    """Return an exact Win32 extended path for long-path-safe syscalls."""

    value = os.fspath(path)
    if os.name != "nt" or value.startswith("\\\\?\\"):
        return value
    if value.startswith("\\\\"):
        return "\\\\?\\UNC\\" + value[2:]
    return "\\\\?\\" + value


def _path_key(path: Path) -> tuple[str, ...]:
    return tuple(part.rstrip(" .").casefold() for part in path.resolve().parts)


def _is_same_or_descendant(candidate: Path, parent: Path) -> bool:
    candidate_key = _path_key(candidate)
    parent_key = _path_key(parent)
    return (
        len(candidate_key) >= len(parent_key)
        and candidate_key[: len(parent_key)] == parent_key
    )


def _git_directory_from_marker(marker: Path) -> Path | None:
    if not marker.is_file():
        return None
    try:
        line = marker.read_text(encoding="utf-8").splitlines()[0]
    except (OSError, UnicodeError, IndexError):
        return None
    if not line.casefold().startswith("gitdir:"):
        return None
    raw = line[len("gitdir:") :].strip()
    if not raw:
        return None
    result = Path(raw)
    if not result.is_absolute():
        result = marker.parent / result
    return result.resolve()


def _control_roots(repo_root: Path) -> tuple[Path, ...]:
    root = repo_root.resolve()
    git_marker = root / ".git"
    roots = [(root / ".verification").resolve(), git_marker.resolve()]
    git_directory = _git_directory_from_marker(git_marker)
    if git_directory is not None:
        roots.append(git_directory)
        common_marker = git_directory / "commondir"
        if common_marker.is_file():
            try:
                raw = common_marker.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeError):
                raw = ""
            if raw:
                common = Path(raw)
                if not common.is_absolute():
                    common = git_directory / common
                roots.append(common.resolve())
    unique: dict[tuple[str, ...], Path] = {}
    for path in roots:
        unique.setdefault(_path_key(path), path)
    return tuple(unique.values())


def _is_control_path(path: Path, repo_root: Path) -> bool:
    return any(
        _is_same_or_descendant(path, control)
        for control in _control_roots(repo_root)
    )


def _git(repo_root: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise H6ValidationCandidateError(
            f"Git source capture failed: {exc}"
        ) from exc
    return completed.stdout


def _git_head(repo_root: Path) -> str:
    try:
        value = _git(repo_root, "rev-parse", "HEAD").decode(
            "ascii", errors="strict"
        ).strip()
    except UnicodeError as exc:
        raise H6ValidationCandidateError(
            "Git HEAD is not ASCII"
        ) from exc
    return _require_git_head(value)


def _dirty_content_digest(repo_root: Path, artifact_root: Path) -> str:
    root = repo_root.resolve()
    configured = artifact_root.resolve()
    if _is_control_path(configured, root):
        raise H6ValidationCandidateError(
            "artifact_root must not enter a repository control tree"
        )
    if _is_same_or_descendant(root, configured):
        raise H6ValidationCandidateError(
            "artifact_root must not contain the repository"
        )
    excluded_root = (
        configured if _is_same_or_descendant(configured, root) else None
    )
    try:
        tracked = {
            name
            for name in _git(
                root, "ls-files", "--cached", "-z"
            ).decode("utf-8", errors="strict").split("\0")
            if name
        }
        untracked = {
            name
            for name in _git(
                root,
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
            ).decode("utf-8", errors="strict").split("\0")
            if name
        }
    except UnicodeError as exc:
        raise H6ValidationCandidateError(
            "Git source path is not strict UTF-8"
        ) from exc

    digest = hashlib.sha256()
    for name in sorted(tracked | untracked):
        relative = PurePosixPath(name)
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in ("", ".", "..") for part in relative.parts)
            or relative.as_posix() != name
        ):
            raise H6ValidationCandidateError(
                f"Git source path is noncanonical or escaping: {name!r}"
            )
        normalized = relative.as_posix()
        if normalized == ".git" or normalized.startswith(".git/"):
            continue
        if normalized == ".verification" or normalized.startswith(
            ".verification/"
        ):
            continue
        absolute = (root / Path(*relative.parts)).resolve()
        if not _is_same_or_descendant(absolute, root):
            raise H6ValidationCandidateError(
                f"Git source path escapes repository: {name}"
            )
        if (
            name not in tracked
            and excluded_root is not None
            and _is_same_or_descendant(absolute, excluded_root)
        ):
            continue
        try:
            content = absolute.read_bytes()
        except FileNotFoundError:
            content = b"<deleted>"
        except OSError as exc:
            raise H6ValidationCandidateError(
                f"Git source content is unreadable: {name}: {exc}"
            ) from exc
        encoded_name = normalized.encode("utf-8", errors="strict")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FixtureBuildSourceIdentity:
    """Exact repository revision and content identity used by the producer."""

    git_head: str
    dirty_digest: str
    source_sha256: str

    def __post_init__(self) -> None:
        if type(self) is not FixtureBuildSourceIdentity:
            raise TypeError("source identity requires its exact record type")
        _require_git_head(self.git_head)
        _require_sha256(self.dirty_digest, "dirty_digest")
        _require_sha256(self.source_sha256, "source_sha256")
        expected = _sha256(
            _SOURCE_DOMAIN
            + bytes.fromhex(self.git_head)
            + bytes.fromhex(self.dirty_digest)
        )
        if self.source_sha256 != expected:
            raise H6ValidationCandidateError(
                "source_sha256 is stale for Git head and dirty digest"
            )

    @classmethod
    def create(
        cls, *, git_head: str, dirty_digest: str
    ) -> "FixtureBuildSourceIdentity":
        _require_git_head(git_head)
        _require_sha256(dirty_digest, "dirty_digest")
        return cls(
            git_head=git_head,
            dirty_digest=dirty_digest,
            source_sha256=_sha256(
                _SOURCE_DOMAIN
                + bytes.fromhex(git_head)
                + bytes.fromhex(dirty_digest)
            ),
        )


def capture_fixture_build_source_identity(
    repo_root: Path, artifact_root: Path
) -> FixtureBuildSourceIdentity:
    """Capture the exact current stdlib-only source identity."""

    if not isinstance(repo_root, Path) or not isinstance(artifact_root, Path):
        raise H6ValidationCandidateError(
            "repo_root and artifact_root must be pathlib.Path values"
        )
    root = repo_root.resolve()
    return FixtureBuildSourceIdentity.create(
        git_head=_git_head(root),
        dirty_digest=_dirty_content_digest(root, artifact_root),
    )


def _fixture_reference_payload(
    reference: ValidationSafetyFixtureReference,
) -> dict[str, object]:
    return {
        "access_policy_sha256": reference.access_policy_sha256,
        "binary_directory_manifest_sha256": (
            reference.binary_directory_manifest_sha256
        ),
        "data_identity_sha256": reference.data_identity_sha256,
        "fixture_raw_length": reference.fixture_raw_length,
        "fixture_raw_sha256": reference.fixture_raw_sha256,
        "local_payload_path": reference.local_payload_path.as_posix(),
        "logical_payload_name": reference.logical_payload_name,
        "reference_sha256": reference.reference_sha256,
        "row_count": reference.row_count,
        "schema_version": reference.schema_version,
        "validation_token_sha256": reference.validation_token_sha256,
    }


def _source_payload(
    source: FixtureBuildSourceIdentity,
) -> dict[str, str]:
    return {
        "dirty_digest": source.dirty_digest,
        "git_head": source.git_head,
        "source_sha256": source.source_sha256,
    }


def _resolved_config_payload(
    *,
    source: FixtureBuildSourceIdentity,
    reference: ValidationSafetyFixtureReference,
    artifact_root: Path,
) -> dict[str, object]:
    return {
        "artifact_root": artifact_root.as_posix(),
        "authorization_sha256": _AUTHORIZATION_SHA256,
        "expected_source": _source_payload(source),
        "fixture_reference": _fixture_reference_payload(reference),
        "full_count": _FULL_COUNT,
        "generator_version": _PERTURBATION_SCHEMA,
        "operation": _OPERATION,
        "perturbation_schema_version": _PERTURBATION_SCHEMA,
        "schema_version": _CONFIG_SCHEMA,
        "seed": _SEED,
        "vocabulary": {
            "size": _VOCABULARY_SIZE,
            "tokenizer_spec_sha256": _TOKENIZER_SPEC_SHA256,
            "vocabulary_id": _VOCABULARY_ID,
            "vocabulary_sha256": _VOCABULARY_SHA256,
        },
    }


@dataclass(frozen=True, slots=True)
class H6ValidationPerturbationBuildResolvedConfig:
    """Canonical, source-bound configuration for the full candidate build."""

    schema_version: Literal[
        "h6-validation-perturbation-build-config-v1"
    ]
    operation: Literal["H6-Validation-Perturbations"]
    source: FixtureBuildSourceIdentity
    fixture_reference: ValidationSafetyFixtureReference
    perturbation_schema_version: Literal[
        "h6-validation-perturbations-v1"
    ]
    generator_version: Literal["h6-validation-perturbations-v1"]
    seed: Literal[2026072197]
    full_count: Literal[4096]
    vocabulary_id: Literal["wikitext-2-byte-v1"]
    vocabulary_size: Literal[258]
    tokenizer_spec_sha256: str
    vocabulary_sha256: str
    authorization_sha256: str
    artifact_root: Path
    canonical_config_bytes: bytes
    config_sha256: str

    def __post_init__(self) -> None:
        if type(self) is not H6ValidationPerturbationBuildResolvedConfig:
            raise TypeError("resolved config requires its exact record type")
        if self.schema_version != _CONFIG_SCHEMA:
            raise H6ValidationCandidateError("build config schema is closed")
        if self.operation != _OPERATION:
            raise H6ValidationCandidateError("build operation is closed")
        if type(self.source) is not FixtureBuildSourceIdentity:
            raise H6ValidationCandidateError(
                "resolved config source identity type is invalid"
            )
        self.source.__post_init__()
        if type(self.fixture_reference) is not ValidationSafetyFixtureReference:
            raise H6ValidationCandidateError(
                "resolved config fixture reference type is invalid"
            )
        self.fixture_reference.__post_init__()
        exact_values = (
            self.perturbation_schema_version == _PERTURBATION_SCHEMA,
            self.generator_version == _PERTURBATION_SCHEMA,
            self.seed == _SEED and type(self.seed) is int,
            self.full_count == _FULL_COUNT and type(self.full_count) is int,
            self.vocabulary_id == _VOCABULARY_ID,
            self.vocabulary_size == _VOCABULARY_SIZE
            and type(self.vocabulary_size) is int,
            self.tokenizer_spec_sha256 == _TOKENIZER_SPEC_SHA256,
            self.vocabulary_sha256 == _VOCABULARY_SHA256,
            self.authorization_sha256 == _AUTHORIZATION_SHA256,
        )
        if not all(exact_values):
            raise H6ValidationCandidateError(
                "resolved build constants are not the frozen H6 values"
            )
        if (
            not isinstance(self.artifact_root, Path)
            or not self.artifact_root.is_absolute()
            or self.artifact_root.resolve(strict=False) != self.artifact_root
        ):
            raise H6ValidationCandidateError(
                "artifact_root must be normalized and absolute"
            )
        if type(self.canonical_config_bytes) is not bytes:
            raise H6ValidationCandidateError(
                "canonical config must be immutable bytes"
            )
        expected_bytes = _canonical_json_bytes(
            _resolved_config_payload(
                source=self.source,
                reference=self.fixture_reference,
                artifact_root=self.artifact_root,
            )
        )
        if self.canonical_config_bytes != expected_bytes:
            raise H6ValidationCandidateError(
                "canonical config bytes are stale"
            )
        _require_sha256(self.config_sha256, "config_sha256")
        if self.config_sha256 != _sha256(expected_bytes):
            raise H6ValidationCandidateError(
                "config_sha256 is stale for canonical config bytes"
            )

    @classmethod
    def create(
        cls,
        *,
        source: FixtureBuildSourceIdentity,
        fixture_reference: ValidationSafetyFixtureReference,
        artifact_root: Path,
    ) -> "H6ValidationPerturbationBuildResolvedConfig":
        if not isinstance(artifact_root, Path):
            raise H6ValidationCandidateError(
                "artifact_root must be a pathlib.Path"
            )
        normalized = artifact_root.resolve(strict=False)
        payload = _resolved_config_payload(
            source=source,
            reference=fixture_reference,
            artifact_root=normalized,
        )
        canonical = _canonical_json_bytes(payload)
        return cls(
            schema_version=_CONFIG_SCHEMA,
            operation=_OPERATION,
            source=source,
            fixture_reference=fixture_reference,
            perturbation_schema_version=_PERTURBATION_SCHEMA,
            generator_version=_PERTURBATION_SCHEMA,
            seed=_SEED,
            full_count=_FULL_COUNT,
            vocabulary_id=_VOCABULARY_ID,
            vocabulary_size=_VOCABULARY_SIZE,
            tokenizer_spec_sha256=_TOKENIZER_SPEC_SHA256,
            vocabulary_sha256=_VOCABULARY_SHA256,
            authorization_sha256=_AUTHORIZATION_SHA256,
            artifact_root=normalized,
            canonical_config_bytes=canonical,
            config_sha256=_sha256(canonical),
        )


@dataclass(frozen=True, slots=True)
class _ValidatedCandidate:
    canonical_bytes: bytes
    raw_sha256: str
    inner_manifest_sha256: str
    schema_version: str
    generator_version: str
    seed: int
    validation_token_sha256: str
    fixture_raw_sha256: str
    vocabulary_id: str
    vocabulary_size: int
    tokenizer_spec_sha256: str
    vocabulary_sha256: str
    full_count: int
    materialized_count: int


def _load_oracle_module() -> ModuleType:
    """Execute only the standalone stdlib oracle, bypassing its NumPy package."""

    root = Path(__file__).resolve().parent
    oracle_path = root / Path(
        *PurePosixPath(_STDLIB_ORACLE_RELATIVE_PATH).parts
    )
    source = _safe_read(
        root,
        _STDLIB_ORACLE_RELATIVE_PATH,
        maximum_length=1_000_000,
    )
    source_sha256 = _sha256(source)
    cached = sys.modules.get(_STDLIB_ORACLE_MODULE_NAME)
    if cached is not None:
        if (
            type(cached) is not ModuleType
            or getattr(cached, "__file__", None) != os.fspath(oracle_path)
            or getattr(cached, "_vfe4_source_sha256", None) != source_sha256
        ):
            raise H6ValidationCandidateError(
                "private stdlib oracle module cache is inconsistent"
            )
        return cached

    module = ModuleType(_STDLIB_ORACLE_MODULE_NAME)
    module.__file__ = os.fspath(oracle_path)
    module.__package__ = ""
    module._vfe4_source_sha256 = source_sha256
    sys.modules[_STDLIB_ORACLE_MODULE_NAME] = module
    try:
        code = compile(
            source,
            os.fspath(oracle_path),
            "exec",
            dont_inherit=True,
        )
        exec(code, module.__dict__)
    except Exception as exc:
        if sys.modules.get(_STDLIB_ORACLE_MODULE_NAME) is module:
            del sys.modules[_STDLIB_ORACLE_MODULE_NAME]
        raise H6ValidationCandidateError(
            f"standalone stdlib oracle import failed: {exc}"
        ) from exc
    return module


def _oracle_vocabulary(oracle_module: object) -> object:
    try:
        factory = oracle_module.IndependentVocabularyIdentity
        vocabulary = factory.create(
            vocabulary_id=_VOCABULARY_ID,
            size=_VOCABULARY_SIZE,
            tokenizer_spec_sha256=_TOKENIZER_SPEC_SHA256,
        )
        if (
            vocabulary.vocabulary_id != _VOCABULARY_ID
            or vocabulary.size != _VOCABULARY_SIZE
            or type(vocabulary.size) is not int
            or vocabulary.tokenizer_spec_sha256 != _TOKENIZER_SPEC_SHA256
            or vocabulary.vocabulary_sha256 != _VOCABULARY_SHA256
        ):
            raise ValueError(
                "independent vocabulary differs from the frozen identity"
            )
        return vocabulary
    except (AttributeError, TypeError, ValueError) as exc:
        raise H6ValidationCandidateError(
            f"independent oracle vocabulary identity failed: {exc}"
        ) from exc


def validate_h6_validation_candidate_bytes(
    candidate_bytes: bytes,
    config: H6ValidationPerturbationBuildResolvedConfig,
    *,
    _oracle_module: object | None = None,
) -> _ValidatedCandidate:
    """Require a complete, canonical, fully cross-bound oracle payload."""

    if type(config) is not H6ValidationPerturbationBuildResolvedConfig:
        raise H6ValidationCandidateError(
            "candidate validation requires an exact resolved config"
        )
    config.__post_init__()
    if type(candidate_bytes) is not bytes:
        raise H6ValidationCandidateError(
            "candidate payload must be exact immutable bytes"
        )
    oracle = (
        _oracle_module
        if _oracle_module is not None
        else _load_oracle_module()
    )
    raw_sha256 = _sha256(candidate_bytes)
    try:
        parsed = oracle.load_frozen_validation_perturbations(
            candidate_bytes,
            expected_raw_sha256=raw_sha256,
            expected_vocabulary_sha256=config.vocabulary_sha256,
            expected_validation_token_sha256=(
                config.fixture_reference.validation_token_sha256
            ),
            expected_validation_safety_fixture_sha256=(
                config.fixture_reference.fixture_raw_sha256
            ),
            require_complete=True,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise H6ValidationCandidateError(
            f"complete independent oracle reparse failed: {exc}"
        ) from exc
    try:
        vocabulary = parsed.vocabulary
        records = parsed.records
        indices = tuple(record.case_index for record in records)
        exact = (
            type(parsed.canonical_bytes) is bytes
            and parsed.canonical_bytes == candidate_bytes
            and parsed.raw_sha256 == raw_sha256
            and parsed.schema_version == config.perturbation_schema_version
            and parsed.generator_version == config.generator_version
            and parsed.seed == config.seed
            and type(parsed.seed) is int
            and parsed.validation_token_sha256
            == config.fixture_reference.validation_token_sha256
            and parsed.validation_safety_fixture_sha256
            == config.fixture_reference.fixture_raw_sha256
            and parsed.full_count == config.full_count
            and type(parsed.full_count) is int
            and parsed.materialized_count == config.full_count
            and type(parsed.materialized_count) is int
            and indices == tuple(range(config.full_count))
            and vocabulary.vocabulary_id == config.vocabulary_id
            and vocabulary.size == config.vocabulary_size
            and type(vocabulary.size) is int
            and vocabulary.tokenizer_spec_sha256
            == config.tokenizer_spec_sha256
            and vocabulary.vocabulary_sha256 == config.vocabulary_sha256
        )
        inner_manifest_sha256 = _require_sha256(
            parsed.manifest_sha256, "perturbation inner manifest SHA-256"
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise H6ValidationCandidateError(
            f"independent oracle payload shape is invalid: {exc}"
        ) from exc
    if not exact:
        raise H6ValidationCandidateError(
            "complete independent oracle payload is not exactly cross-bound"
        )
    return _ValidatedCandidate(
        canonical_bytes=candidate_bytes,
        raw_sha256=raw_sha256,
        inner_manifest_sha256=inner_manifest_sha256,
        schema_version=config.perturbation_schema_version,
        generator_version=config.generator_version,
        seed=config.seed,
        validation_token_sha256=(
            config.fixture_reference.validation_token_sha256
        ),
        fixture_raw_sha256=config.fixture_reference.fixture_raw_sha256,
        vocabulary_id=config.vocabulary_id,
        vocabulary_size=config.vocabulary_size,
        tokenizer_spec_sha256=config.tokenizer_spec_sha256,
        vocabulary_sha256=config.vocabulary_sha256,
        full_count=config.full_count,
        materialized_count=config.full_count,
    )


def _provenance_payload(
    config: H6ValidationPerturbationBuildResolvedConfig,
    candidate: _ValidatedCandidate,
) -> dict[str, object]:
    reference = config.fixture_reference
    return {
        "access_policy_sha256": reference.access_policy_sha256,
        "binary_directory_manifest_sha256": (
            reference.binary_directory_manifest_sha256
        ),
        "config_sha256": config.config_sha256,
        "data_identity_sha256": reference.data_identity_sha256,
        "fixture_raw_sha256": reference.fixture_raw_sha256,
        "full_count": candidate.full_count,
        "generator_version": candidate.generator_version,
        "materialized_count": candidate.materialized_count,
        "nonclaim": _NONCLAIM,
        "operation": config.operation,
        "perturbation_inner_manifest_sha256": (
            candidate.inner_manifest_sha256
        ),
        "perturbation_raw_sha256": candidate.raw_sha256,
        "perturbation_schema_version": candidate.schema_version,
        "producer_source": _source_payload(config.source),
        "seed": candidate.seed,
        "status": _STATUS,
        "validation_fixture_reference_sha256": reference.reference_sha256,
        "validation_token_sha256": reference.validation_token_sha256,
        "vocabulary": {
            "size": candidate.vocabulary_size,
            "tokenizer_spec_sha256": candidate.tokenizer_spec_sha256,
            "vocabulary_id": candidate.vocabulary_id,
            "vocabulary_sha256": candidate.vocabulary_sha256,
        },
    }


def _artifact_reference_identity_payload(
    *,
    config: H6ValidationPerturbationBuildResolvedConfig,
    candidate: _ValidatedCandidate,
    payload_sha256s: tuple[tuple[str, str], ...],
    directory_manifest_sha256: str,
) -> dict[str, object]:
    fixture = config.fixture_reference
    return {
        "access_policy_sha256": fixture.access_policy_sha256,
        "binary_directory_manifest_sha256": (
            fixture.binary_directory_manifest_sha256
        ),
        "config_sha256": config.config_sha256,
        "data_identity_sha256": fixture.data_identity_sha256,
        "directory_manifest_sha256": directory_manifest_sha256,
        "fixture_raw_sha256": fixture.fixture_raw_sha256,
        "full_count": candidate.full_count,
        "generator_version": candidate.generator_version,
        "materialized_count": candidate.materialized_count,
        "payload_sha256s": [
            {"path": path, "sha256": sha256}
            for path, sha256 in payload_sha256s
        ],
        "perturbation_inner_manifest_sha256": (
            candidate.inner_manifest_sha256
        ),
        "perturbation_raw_sha256": candidate.raw_sha256,
        "perturbation_schema_version": candidate.schema_version,
        "seed": candidate.seed,
        "source": _source_payload(config.source),
        "validation_fixture_reference_sha256": fixture.reference_sha256,
        "validation_token_sha256": fixture.validation_token_sha256,
        "vocabulary": {
            "size": candidate.vocabulary_size,
            "tokenizer_spec_sha256": candidate.tokenizer_spec_sha256,
            "vocabulary_id": candidate.vocabulary_id,
            "vocabulary_sha256": candidate.vocabulary_sha256,
        },
    }


def _artifact_reference_hash(
    *,
    source: FixtureBuildSourceIdentity,
    config_sha256: str,
    fixture: object,
    candidate: _ValidatedCandidate,
    payload_sha256s: tuple[tuple[str, str], ...],
    directory_manifest_sha256: str,
) -> str:
    shim = SimpleNamespace(
        source=source,
        config_sha256=config_sha256,
        fixture_reference=fixture,
    )
    identity = _artifact_reference_identity_payload(
        config=shim,
        candidate=candidate,
        payload_sha256s=payload_sha256s,
        directory_manifest_sha256=directory_manifest_sha256,
    )
    return _sha256(_REFERENCE_DOMAIN + _canonical_json_bytes(identity))


def _make_artifact_reference(
    *,
    local_artifact_path: Path,
    config: H6ValidationPerturbationBuildResolvedConfig,
    candidate: _ValidatedCandidate,
    payload_sha256s: tuple[tuple[str, str], ...],
    directory_manifest_sha256: str,
) -> H6ValidationPerturbationArtifactReference:
    fixture = config.fixture_reference
    reference_sha256 = _artifact_reference_hash(
        source=config.source,
        config_sha256=config.config_sha256,
        fixture=fixture,
        candidate=candidate,
        payload_sha256s=payload_sha256s,
        directory_manifest_sha256=directory_manifest_sha256,
    )
    return H6ValidationPerturbationArtifactReference(
        local_artifact_path=local_artifact_path.resolve(strict=False),
        git_head=config.source.git_head,
        dirty_digest=config.source.dirty_digest,
        source_sha256=config.source.source_sha256,
        config_sha256=config.config_sha256,
        validation_fixture_reference_sha256=fixture.reference_sha256,
        binary_directory_manifest_sha256=(
            fixture.binary_directory_manifest_sha256
        ),
        data_identity_sha256=fixture.data_identity_sha256,
        access_policy_sha256=fixture.access_policy_sha256,
        validation_token_sha256=fixture.validation_token_sha256,
        fixture_raw_sha256=fixture.fixture_raw_sha256,
        vocabulary_id=config.vocabulary_id,
        vocabulary_size=config.vocabulary_size,
        tokenizer_spec_sha256=config.tokenizer_spec_sha256,
        vocabulary_sha256=config.vocabulary_sha256,
        perturbation_schema_version=config.perturbation_schema_version,
        generator_version=config.generator_version,
        seed=config.seed,
        full_count=config.full_count,
        materialized_count=candidate.materialized_count,
        perturbation_inner_manifest_sha256=(
            candidate.inner_manifest_sha256
        ),
        perturbation_raw_sha256=candidate.raw_sha256,
        payload_sha256s=payload_sha256s,
        directory_manifest_sha256=directory_manifest_sha256,
        reference_sha256=reference_sha256,
    )


@dataclass(frozen=True, slots=True)
class _OwnedDirectoryIdentity:
    device: int
    inode: int
    marker_name: str
    marker_token: bytes
    marker_device: int
    marker_inode: int

    def matches(self, path: Path, *, require_marker: bool) -> bool:
        try:
            metadata = path.lstat()
        except OSError:
            return False
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or _is_redirect(path, metadata)
            or metadata.st_dev != self.device
            or metadata.st_ino != self.inode
        ):
            return False
        marker = path / self.marker_name
        try:
            marker_metadata = marker.lstat()
        except FileNotFoundError:
            return not require_marker
        except OSError:
            return False
        if not require_marker:
            return False
        if (
            not stat.S_ISREG(marker_metadata.st_mode)
            or _is_redirect(marker, marker_metadata)
            or marker_metadata.st_nlink != 1
            or marker_metadata.st_dev != self.marker_device
            or marker_metadata.st_ino != self.marker_inode
        ):
            return False
        try:
            return marker.read_bytes() == self.marker_token
        except OSError:
            return False


def _install_marker(staging: Path) -> _OwnedDirectoryIdentity:
    metadata = staging.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or _is_redirect(staging, metadata)
        or metadata.st_ino == 0
    ):
        raise H6ValidationCandidateError(
            "exclusive staging directory lacks a stable identity"
        )
    marker_name = f".owner-{uuid.uuid4().hex}"
    marker_token = uuid.uuid4().hex.encode("ascii")
    marker = staging / marker_name
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(marker, flags, 0o600)
    try:
        if os.write(descriptor, marker_token) != len(marker_token):
            raise H6ValidationCandidateError(
                "staging ownership marker write was incomplete"
            )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    marker_metadata = marker.lstat()
    identity = _OwnedDirectoryIdentity(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        marker_name=marker_name,
        marker_token=marker_token,
        marker_device=marker_metadata.st_dev,
        marker_inode=marker_metadata.st_ino,
    )
    if not identity.matches(staging, require_marker=True):
        raise H6ValidationCandidateError(
            "staging ownership changed during marker installation"
        )
    return identity


def _exclusive_write(path: Path, content: bytes) -> tuple[int, int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    try:
        written = 0
        while written < len(content):
            count = os.write(descriptor, content[written:])
            if count <= 0:
                raise H6ValidationCandidateError(
                    f"exclusive artifact write was incomplete: {path}"
                )
            written += count
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = path.lstat()
    if (
        not stat.S_ISREG(current.st_mode)
        or _is_redirect(path, current)
        or current.st_nlink != 1
        or (current.st_dev, current.st_ino, current.st_size)
        != (opened.st_dev, opened.st_ino, opened.st_size)
    ):
        raise H6ValidationCandidateError(
            f"exclusive artifact identity changed: {path}"
        )
    return (opened.st_dev, opened.st_ino, opened.st_size)


def _fsync_directory(path: Path) -> None:
    if os.name != "nt":
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(path, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return

    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    flush = kernel32.FlushFileBuffers
    flush.argtypes = (wintypes.HANDLE,)
    flush.restype = wintypes.BOOL
    close = kernel32.CloseHandle
    close.argtypes = (wintypes.HANDLE,)
    close.restype = wintypes.BOOL
    handle = create_file(
        os.fspath(path),
        0x80000000 | 0x40000000,
        0x00000001 | 0x00000002 | 0x00000004,
        None,
        3,
        0x02000000,
        None,
    )
    invalid = wintypes.HANDLE(-1).value
    if handle == invalid:
        error = ctypes.get_last_error()
        raise OSError(error, f"directory durability open failed: {path}")
    try:
        if not flush(handle):
            error = ctypes.get_last_error()
            raise OSError(error, f"directory durability flush failed: {path}")
    finally:
        close(handle)


def _destination_exists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    if os.name == "nt":
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        move = kernel32.MoveFileExW
        move.argtypes = (
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
            wintypes.DWORD,
        )
        move.restype = wintypes.BOOL
        if move(os.fspath(source), os.fspath(destination), 0):
            return
        error = ctypes.get_last_error()
        if _destination_exists(destination):
            raise H6ValidationCandidateError(
                f"candidate directory already exists: {destination}"
            )
        raise OSError(
            error, f"atomic no-replace publication failed: {destination}"
        )
    if os.name != "posix":
        raise H6ValidationCandidateError(
            f"atomic no-replace publication is unsupported on {os.name!r}"
        )
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform.startswith("linux"):
        try:
            rename = libc.renameat2
        except AttributeError as exc:
            raise H6ValidationCandidateError(
                "Linux libc lacks renameat2(RENAME_NOREPLACE)"
            ) from exc
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(-100, source_bytes, -100, destination_bytes, 1)
    elif sys.platform == "darwin":
        try:
            rename = libc.renamex_np
        except AttributeError as exc:
            raise H6ValidationCandidateError(
                "Darwin libc lacks renamex_np(RENAME_EXCL)"
            ) from exc
        rename.argtypes = (
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(source_bytes, destination_bytes, 0x00000004)
    else:
        raise H6ValidationCandidateError(
            "this POSIX platform lacks an atomic no-replace primitive"
        )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in (errno.EEXIST, errno.ENOTEMPTY) or _destination_exists(
        destination
    ):
        raise H6ValidationCandidateError(
            f"candidate directory already exists: {destination}"
        )
    raise OSError(
        error, f"atomic no-replace publication failed: {destination}"
    )


def _manifest_bytes(
    payloads: tuple[tuple[str, bytes], ...],
) -> tuple[bytes, tuple[tuple[str, str], ...]]:
    identities = tuple((name, _sha256(content)) for name, content in payloads)
    return (
        b"".join(
            f"{sha256}  {name}\n".encode("ascii")
            for name, sha256 in identities
        ),
        identities,
    )


def publish_h6_validation_perturbation_candidate(
    config: H6ValidationPerturbationBuildResolvedConfig,
    candidate: _ValidatedCandidate,
    *,
    _oracle_module: object | None = None,
) -> H6ValidationPerturbationArtifactReference:
    """Publish exactly one immutable candidate by atomic no-replace rename."""

    if type(config) is not H6ValidationPerturbationBuildResolvedConfig:
        raise H6ValidationCandidateError(
            "publication requires an exact resolved config"
        )
    config.__post_init__()
    if type(candidate) is not _ValidatedCandidate:
        raise H6ValidationCandidateError(
            "publication requires a validated candidate"
        )
    candidate = validate_h6_validation_candidate_bytes(
        candidate.canonical_bytes,
        config,
        _oracle_module=_oracle_module,
    )
    payloads = (
        ("config.json", config.canonical_config_bytes),
        (
            "provenance.json",
            _canonical_json_bytes(_provenance_payload(config, candidate)),
        ),
        (
            "validation/h6_validation_perturbations_v1.json",
            candidate.canonical_bytes,
        ),
    )
    manifest_bytes, payload_sha256s = _manifest_bytes(payloads)
    directory_manifest_sha256 = _sha256(manifest_bytes)
    root = config.artifact_root
    final = root / (
        _DIRECTORY_PREFIX + directory_manifest_sha256
    )
    staging = root / f".h6-candidate-staging-{uuid.uuid4().hex}"
    try:
        root.mkdir(parents=True, exist_ok=True)
        root_metadata = root.lstat()
        if not stat.S_ISDIR(root_metadata.st_mode) or _is_redirect(
            root, root_metadata
        ):
            raise H6ValidationCandidateError(
                "artifact_root must be a regular nonredirected directory"
            )
        staging.mkdir()
        identity = _install_marker(staging)
        file_identities: dict[str, tuple[int, int, int]] = {}
        for name, content in payloads:
            file_identities[name] = _exclusive_write(
                staging / Path(*PurePosixPath(name).parts), content
            )
        file_identities[_MANIFEST_NAME] = _exclusive_write(
            staging / _MANIFEST_NAME, manifest_bytes
        )
        validation_directory = staging / "validation"
        validation_identity = validation_directory.lstat()
        if (
            not stat.S_ISDIR(validation_identity.st_mode)
            or _is_redirect(validation_directory, validation_identity)
            or validation_identity.st_ino == 0
        ):
            raise H6ValidationCandidateError(
                "staged validation directory lacks a stable identity"
            )
        _fsync_directory(staging / "validation")
        _fsync_directory(staging)
        if not identity.matches(staging, require_marker=True):
            raise H6ValidationCandidateError(
                "staging ownership changed before publication"
            )
        for name, expected in file_identities.items():
            metadata = (staging / Path(*PurePosixPath(name).parts)).lstat()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or _is_redirect(
                    staging / Path(*PurePosixPath(name).parts), metadata
                )
                or metadata.st_nlink != 1
                or (metadata.st_dev, metadata.st_ino, metadata.st_size)
                != expected
            ):
                raise H6ValidationCandidateError(
                    f"staged payload identity changed: {name}"
                )
        validation_current = validation_directory.lstat()
        if (
            _is_redirect(validation_directory, validation_current)
            or (
                validation_current.st_dev,
                validation_current.st_ino,
            )
            != (
                validation_identity.st_dev,
                validation_identity.st_ino,
            )
        ):
            raise H6ValidationCandidateError(
                "staged validation directory identity changed"
            )
        (staging / identity.marker_name).unlink()
        if not identity.matches(staging, require_marker=False):
            raise H6ValidationCandidateError(
                "staging ownership changed before final commit"
            )
        _fsync_directory(staging)
        if not identity.matches(staging, require_marker=False):
            raise H6ValidationCandidateError(
                "staging ownership changed after marker-unlink durability"
            )
        _rename_directory_no_replace(staging, final)
        if not identity.matches(final, require_marker=False):
            raise H6ValidationCandidateError(
                "installed directory identity differs from staging"
            )
        try:
            _fsync_directory(root)
        except OSError as exc:
            raise H6ValidationCandidateError(
                "publication outcome is unknown after final rename; "
                f"parent-directory durability failed for {final}: {exc}"
            ) from exc
    except H6ValidationCandidateError:
        raise
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        raise H6ValidationCandidateError(
            f"candidate publication failed: {exc}"
        ) from exc
    return load_h6_validation_perturbation_artifact(
        final,
        _oracle_module=_oracle_module,
    )


def _safe_read(
    root: Path,
    relative_name: str,
    *,
    maximum_length: int,
) -> bytes:
    relative = PurePosixPath(relative_name)
    path = root / Path(*relative.parts)
    if path.resolve(strict=False) != path or not _is_same_or_descendant(
        path, root
    ):
        raise H6ValidationCandidateError(
            f"artifact path escapes or redirects: {relative_name}"
        )
    intermediate_identities: list[tuple[Path, int, int]] = []
    try:
        root_before = os.stat(_io_path(root), follow_symlinks=False)
        intermediate = root
        for component in relative.parts[:-1]:
            intermediate = intermediate / component
            metadata = os.stat(
                _io_path(intermediate), follow_symlinks=False
            )
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or _is_redirect(intermediate, metadata)
                or metadata.st_ino == 0
            ):
                raise H6ValidationCandidateError(
                    "artifact intermediate directory is not a stable "
                    f"regular nonredirected directory: {intermediate}"
                )
            intermediate_identities.append(
                (intermediate, metadata.st_dev, metadata.st_ino)
            )
        before = os.stat(_io_path(path), follow_symlinks=False)
    except H6ValidationCandidateError:
        raise
    except OSError as exc:
        raise H6ValidationCandidateError(
            f"artifact payload is unavailable: {relative_name}"
        ) from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or _is_redirect(path, before)
        or before.st_nlink != 1
        or before.st_size > maximum_length
    ):
        raise H6ValidationCandidateError(
            f"artifact payload is not a bounded regular nonlink: {relative_name}"
        )
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    try:
        descriptor = os.open(_io_path(path), flags)
    except OSError as exc:
        raise H6ValidationCandidateError(
            f"artifact payload cannot be opened safely: {relative_name}"
        ) from exc
    chunks: list[bytes] = []
    total = 0
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino, opened.st_size)
            != (before.st_dev, before.st_ino, before.st_size)
        ):
            raise H6ValidationCandidateError(
                f"artifact payload changed before opening: {relative_name}"
            )
        while True:
            chunk = os.read(
                descriptor, min(65_536, maximum_length + 1 - total)
            )
            if not chunk:
                break
            total += len(chunk)
            if total > maximum_length:
                raise H6ValidationCandidateError(
                    f"artifact payload exceeds bound: {relative_name}"
                )
            chunks.append(chunk)
        after_open = os.fstat(descriptor)
        try:
            after = os.stat(_io_path(path), follow_symlinks=False)
            intermediate_after = tuple(
                (
                    intermediate,
                    os.stat(
                        _io_path(intermediate), follow_symlinks=False
                    ),
                    expected_device,
                    expected_inode,
                )
                for (
                    intermediate,
                    expected_device,
                    expected_inode,
                ) in intermediate_identities
            )
            root_after = os.stat(
                _io_path(root), follow_symlinks=False
            )
        except OSError as exc:
            raise H6ValidationCandidateError(
                "artifact path identity became unavailable while reading: "
                f"{relative_name}"
            ) from exc
        if any(
            not stat.S_ISDIR(metadata.st_mode)
            or _is_redirect(intermediate, metadata)
            or (metadata.st_dev, metadata.st_ino)
            != (expected_device, expected_inode)
            for (
                intermediate,
                metadata,
                expected_device,
                expected_inode,
            ) in intermediate_after
        ):
            raise H6ValidationCandidateError(
                "artifact intermediate directory identity changed while "
                f"reading: {relative_name}"
            )
        if (
            _is_redirect(path, after)
            or after.st_nlink != 1
            or (after_open.st_dev, after_open.st_ino, after_open.st_size)
            != (opened.st_dev, opened.st_ino, opened.st_size)
            or (after.st_dev, after.st_ino, after.st_size)
            != (opened.st_dev, opened.st_ino, opened.st_size)
            or (root_after.st_dev, root_after.st_ino)
            != (root_before.st_dev, root_before.st_ino)
        ):
            raise H6ValidationCandidateError(
                f"artifact payload changed while reading: {relative_name}"
            )
    finally:
        os.close(descriptor)
    content = b"".join(chunks)
    if len(content) != before.st_size:
        raise H6ValidationCandidateError(
            f"artifact payload length changed: {relative_name}"
        )
    return content


def _exact_inventory(root: Path) -> None:
    try:
        root_names = {entry.name for entry in os.scandir(_io_path(root))}
        validation = root / "validation"
        validation_metadata = os.stat(
            _io_path(validation), follow_symlinks=False
        )
    except OSError as exc:
        raise H6ValidationCandidateError(
            f"candidate inventory is unavailable: {exc}"
        ) from exc
    if root_names != {
        "config.json",
        "provenance.json",
        "validation",
        "manifest.sha256",
    }:
        raise H6ValidationCandidateError(
            "candidate inventory is not the exact four-file layout"
        )
    if (
        not stat.S_ISDIR(validation_metadata.st_mode)
        or _is_redirect(validation, validation_metadata)
    ):
        raise H6ValidationCandidateError(
            "candidate validation inventory is not exact"
        )
    try:
        validation_names = {
            entry.name for entry in os.scandir(_io_path(validation))
        }
    except OSError as exc:
        raise H6ValidationCandidateError(
            f"candidate validation inventory is unavailable: {exc}"
        ) from exc
    if validation_names != {"h6_validation_perturbations_v1.json"}:
        raise H6ValidationCandidateError(
            "candidate validation inventory is not exact"
        )


def _reject_duplicate_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise H6ValidationCandidateError(
                f"duplicate JSON key is forbidden: {key}"
            )
        result[key] = value
    return result


def _load_canonical_object(data: bytes, label: str) -> dict[str, object]:
    try:
        text = data.decode("utf-8", errors="strict")
        value = json.loads(text, object_pairs_hook=_reject_duplicate_pairs)
    except H6ValidationCandidateError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise H6ValidationCandidateError(
            f"{label} JSON is invalid: {exc}"
        ) from exc
    if type(value) is not dict:
        raise H6ValidationCandidateError(f"{label} must be a JSON object")
    if _canonical_json_bytes(value) != data:
        raise H6ValidationCandidateError(f"{label} JSON is not canonical")
    return value


def _exact_keys(
    value: Mapping[str, object], expected: set[str], label: str
) -> None:
    if set(value) != expected:
        raise H6ValidationCandidateError(f"{label} keys are not exact")


def _fixture_reference_from_payload(
    value: object,
) -> ValidationSafetyFixtureReference:
    if type(value) is not dict:
        raise H6ValidationCandidateError(
            "fixture reference config must be an object"
        )
    expected = {
        "access_policy_sha256",
        "binary_directory_manifest_sha256",
        "data_identity_sha256",
        "fixture_raw_length",
        "fixture_raw_sha256",
        "local_payload_path",
        "logical_payload_name",
        "reference_sha256",
        "row_count",
        "schema_version",
        "validation_token_sha256",
    }
    _exact_keys(value, expected, "fixture reference")
    if type(value["local_payload_path"]) is not str:
        raise H6ValidationCandidateError(
            "fixture reference local path must be exact text"
        )
    try:
        reference = ValidationSafetyFixtureReference(
            schema_version=value["schema_version"],
            logical_payload_name=value["logical_payload_name"],
            local_payload_path=Path(value["local_payload_path"]),
            binary_directory_manifest_sha256=value[
                "binary_directory_manifest_sha256"
            ],
            data_identity_sha256=value["data_identity_sha256"],
            access_policy_sha256=value["access_policy_sha256"],
            validation_token_sha256=value["validation_token_sha256"],
            fixture_raw_sha256=value["fixture_raw_sha256"],
            fixture_raw_length=value["fixture_raw_length"],
            row_count=value["row_count"],
            reference_sha256=value["reference_sha256"],
        )
    except (TypeError, ValueError) as exc:
        raise H6ValidationCandidateError(
            f"fixture reference config is invalid: {exc}"
        ) from exc
    return reference


def _source_from_payload(value: object) -> FixtureBuildSourceIdentity:
    if type(value) is not dict:
        raise H6ValidationCandidateError("source config must be an object")
    _exact_keys(
        value,
        {"dirty_digest", "git_head", "source_sha256"},
        "source",
    )
    return FixtureBuildSourceIdentity(
        git_head=value["git_head"],
        dirty_digest=value["dirty_digest"],
        source_sha256=value["source_sha256"],
    )


def _config_from_payload(
    value: dict[str, object],
) -> H6ValidationPerturbationBuildResolvedConfig:
    expected = {
        "artifact_root",
        "authorization_sha256",
        "expected_source",
        "fixture_reference",
        "full_count",
        "generator_version",
        "operation",
        "perturbation_schema_version",
        "schema_version",
        "seed",
        "vocabulary",
    }
    _exact_keys(value, expected, "build config")
    vocabulary = value["vocabulary"]
    if type(vocabulary) is not dict:
        raise H6ValidationCandidateError(
            "build config vocabulary must be an object"
        )
    _exact_keys(
        vocabulary,
        {
            "size",
            "tokenizer_spec_sha256",
            "vocabulary_id",
            "vocabulary_sha256",
        },
        "build vocabulary",
    )
    source = _source_from_payload(value["expected_source"])
    fixture = _fixture_reference_from_payload(value["fixture_reference"])
    if type(value["artifact_root"]) is not str:
        raise H6ValidationCandidateError(
            "artifact_root config must be exact text"
        )
    try:
        artifact_root = Path(value["artifact_root"])
    except TypeError as exc:
        raise H6ValidationCandidateError(
            "artifact_root config must be text"
        ) from exc
    if (
        not artifact_root.is_absolute()
        or artifact_root.resolve(strict=False) != artifact_root
    ):
        raise H6ValidationCandidateError(
            "artifact_root config must be normalized and absolute"
        )
    config = H6ValidationPerturbationBuildResolvedConfig.create(
        source=source,
        fixture_reference=fixture,
        artifact_root=artifact_root,
    )
    if (
        value["authorization_sha256"] != config.authorization_sha256
        or value["full_count"] != config.full_count
        or type(value["full_count"]) is not int
        or value["generator_version"] != config.generator_version
        or value["operation"] != config.operation
        or value["perturbation_schema_version"]
        != config.perturbation_schema_version
        or value["schema_version"] != config.schema_version
        or value["seed"] != config.seed
        or type(value["seed"]) is not int
        or vocabulary["size"] != config.vocabulary_size
        or type(vocabulary["size"]) is not int
        or vocabulary["tokenizer_spec_sha256"]
        != config.tokenizer_spec_sha256
        or vocabulary["vocabulary_id"] != config.vocabulary_id
        or vocabulary["vocabulary_sha256"] != config.vocabulary_sha256
    ):
        raise H6ValidationCandidateError(
            "build config constants differ from the frozen values"
        )
    return config


def load_h6_validation_perturbation_artifact(
    artifact_path: Path,
    *,
    _oracle_module: object | None = None,
) -> H6ValidationPerturbationArtifactReference:
    """Strictly load one explicitly named immutable candidate directory."""

    if not isinstance(artifact_path, Path):
        raise H6ValidationCandidateError(
            "artifact_path must be a pathlib.Path"
        )
    root = artifact_path.resolve(strict=False)
    try:
        metadata = os.stat(_io_path(root), follow_symlinks=False)
    except OSError as exc:
        raise H6ValidationCandidateError(
            "candidate directory is unavailable"
        ) from exc
    if (
        root != artifact_path
        or not stat.S_ISDIR(metadata.st_mode)
        or _is_redirect(root, metadata)
    ):
        raise H6ValidationCandidateError(
            "candidate directory must be normalized, regular, and nonredirected"
        )
    suffix = root.name.removeprefix(_DIRECTORY_PREFIX)
    if (
        not root.name.startswith(_DIRECTORY_PREFIX)
        or len(suffix) != 64
        or any(character not in _LOWER_HEX for character in suffix)
    ):
        raise H6ValidationCandidateError(
            "candidate directory name is not exact"
        )
    _exact_inventory(root)
    config_bytes = _safe_read(root, "config.json", maximum_length=1_000_000)
    provenance_bytes = _safe_read(
        root, "provenance.json", maximum_length=1_000_000
    )
    candidate_bytes = _safe_read(
        root,
        "validation/h6_validation_perturbations_v1.json",
        maximum_length=128_000_000,
    )
    manifest_bytes = _safe_read(
        root, "manifest.sha256", maximum_length=1_000
    )
    payloads = (
        ("config.json", config_bytes),
        ("provenance.json", provenance_bytes),
        (
            "validation/h6_validation_perturbations_v1.json",
            candidate_bytes,
        ),
    )
    expected_manifest, payload_sha256s = _manifest_bytes(payloads)
    if manifest_bytes != expected_manifest:
        raise H6ValidationCandidateError(
            "candidate manifest order, bytes, or payload hashes are invalid"
        )
    directory_manifest_sha256 = _sha256(manifest_bytes)
    if (
        directory_manifest_sha256 != suffix
        or root.name != _DIRECTORY_PREFIX + directory_manifest_sha256
    ):
        raise H6ValidationCandidateError(
            "candidate directory manifest identity is stale"
        )
    config_payload = _load_canonical_object(config_bytes, "config")
    config = _config_from_payload(config_payload)
    if config.canonical_config_bytes != config_bytes:
        raise H6ValidationCandidateError(
            "loaded config bytes differ from the resolved config"
        )
    provenance = _load_canonical_object(provenance_bytes, "provenance")
    candidate = validate_h6_validation_candidate_bytes(
        candidate_bytes,
        config,
        _oracle_module=_oracle_module,
    )
    expected_provenance = _provenance_payload(config, candidate)
    if provenance != expected_provenance:
        raise H6ValidationCandidateError(
            "candidate provenance cross-binding is invalid"
        )
    return _make_artifact_reference(
        local_artifact_path=root,
        config=config,
        candidate=candidate,
        payload_sha256s=payload_sha256s,
        directory_manifest_sha256=directory_manifest_sha256,
    )


def build_h6_validation_perturbation_candidate(
    config: H6ValidationPerturbationBuildResolvedConfig,
    *,
    repo_root: Path,
) -> H6ValidationPerturbationArtifactReference:
    """Generate and publish the full independent 4,096-record candidate."""

    if type(config) is not H6ValidationPerturbationBuildResolvedConfig:
        raise H6ValidationCandidateError(
            "build requires an exact resolved config"
        )
    config.__post_init__()
    current = capture_fixture_build_source_identity(
        repo_root, config.artifact_root
    )
    if current != config.source:
        raise H6ValidationCandidateError(
            "current repository source identity differs from resolved config"
        )
    payload = read_validation_safety_fixture_payload(
        config.fixture_reference
    )
    oracle = _load_oracle_module()
    vocabulary = _oracle_vocabulary(oracle)
    try:
        generated = oracle.generate_frozen_validation_perturbations(
            payload.fixture_bytes,
            vocabulary=vocabulary,
            seed=config.seed,
            expected_validation_safety_fixture_sha256=(
                config.fixture_reference.fixture_raw_sha256
            ),
            expected_validation_token_sha256=(
                config.fixture_reference.validation_token_sha256
            ),
        )
        canonical_bytes = generated.canonical_bytes
    except (AttributeError, TypeError, ValueError) as exc:
        raise H6ValidationCandidateError(
            f"full independent oracle generation failed: {exc}"
        ) from exc
    candidate = validate_h6_validation_candidate_bytes(
        canonical_bytes,
        config,
        _oracle_module=oracle,
    )
    return publish_h6_validation_perturbation_candidate(
        config,
        candidate,
        _oracle_module=oracle,
    )


def run_h6_validation_perturbation_build(
    raw_config: object,
) -> dict[str, object]:
    """Resolve an explicitly supplied click config and run the full producer."""

    if type(raw_config) is not dict:
        raise H6ValidationCandidateError(
            "authorized build config must be an exact dictionary"
        )
    config = _config_from_payload(raw_config)
    repo_root = Path(__file__).resolve().parents[1]
    artifact = build_h6_validation_perturbation_candidate(
        config, repo_root=repo_root
    )
    return {
        "artifact_reference": artifact.to_payload(),
        "status": _STATUS,
    }


__all__ = [
    "FixtureBuildSourceIdentity",
    "H6ValidationCandidateError",
    "H6ValidationPerturbationArtifactReference",
    "H6ValidationPerturbationBuildResolvedConfig",
    "build_h6_validation_perturbation_candidate",
    "capture_fixture_build_source_identity",
    "load_h6_validation_perturbation_artifact",
    "publish_h6_validation_perturbation_candidate",
    "run_h6_validation_perturbation_build",
    "validate_h6_validation_candidate_bytes",
]
