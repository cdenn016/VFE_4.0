"""Authorized WikiText-103 source-lock and production orchestration.

This module is imported only after the click launcher has enforced the
operation-specific authorization boundary.  Source acquisition, live
distribution inspection, and the first tiktoken import are consequently
unreachable from launcher import and idle/synthetic modes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
from dataclasses import dataclass, fields, replace
from pathlib import Path, PurePosixPath
from typing import Literal, Protocol

import numpy as np
import torch

from vfe4.artifacts.durability import (
    ContentAddressedDurabilityBackend,
    DurableFileIdentity,
    DurabilityBackend,
    DurabilityCollisionError,
    PosixDurabilityBackend,
    WindowsDurabilityBackend,
    canonical_json_bytes_generic,
    validate_regular_nonlink_sha256,
)
from vfe4.artifacts.environment import (
    DistributionIdentity,
    LockInputManifest,
    parse_lock_input_manifest,
    render_dependency_lock,
)
from vfe4.artifacts.provenance import (
    production_token_cache_set_sha256,
)
from vfe4.config.schema import TrainingConfig
from vfe4.data.windows import (
    CausalBatch,
    CausalWindow,
    WT103_BATCH_SIZE,
    WT103_EOT_TOKEN_ID,
    WT103_IGNORE_TARGET_ID,
    WT103_SEQUENCE_LENGTH,
    WT103WindowRow,
    WindowSchedule,
    _schedule_sha256,
    enumerate_wt103_window_rows,
)
from vfe4.data.wikitext103 import (
    BoundedHttpClient,
    BoundedHttpObservation,
    HttpRedirectObservation,
    StagedAcquisitionRequest,
    StagedWikiText103AcquisitionRecord,
    WIKITEXT103_ARCHIVE_REQUEST_URL,
    WIKITEXT103_SOURCE_PAGE_REQUEST_URL,
    reopen_staged_wikitext103,
    stage_wikitext103_acquisition_record,
)
from vfe4.types.results import GateStatus
from vfe4.types.training import (
    A0ArchitectureProfile,
    A0FormulaRecord,
    ArchiveMemberIdentity,
    DataCursor,
    FinalizedWikiText103SourceRecord,
    PermutationManifest,
    ProductionTokenCacheIdentity,
    ProductionTokenizerSpec,
    RedirectHop,
    WT103_A0_HIDDEN_WIDTH_CANDIDATES,
    WindowManifest,
    owned_sha256,
    production_tokenizer_tables_sha256,
)

from .factories import (
    A0MatchRow,
    ArmMatchingReport,
)
from .formulas import (
    A0FlopLedger,
    A0FlopTerm,
    A0FlopWorkload,
    build_a0_architecture_profile,
    build_a0_formula_record,
    reconstruct_a0_flops,
    reconstruct_a0_parameters,
)
from .wt103_models import WT103A0Model
from .wt103_runtime import (
    WT103PrimaryParameterInventory,
    WT103PrimaryParameterRow,
    reconstruct_wt103_primary_parameters,
)
from .tokenizer_stream import (
    ExactPieceTokenizerAdapter,
    PublishedTokenPayload,
    TokenizerStreamLimits,
    publish_exact_bounded_token_payload,
    verify_piece_stream_golden_vectors,
)


_SOURCE_BUNDLE_RELATIVE_PATH = (
    "production-source-lock/finalized-source-bundle.json"
)
_LOCK_INPUT_NAME = "requirements-wt103.lock-input.json"
_LOCK_NAME = "requirements-wt103.lock"
_WINDOW_ROWS_DOMAIN = b"VFE4-WT103-WINDOW-ROWS-V1\x00"
_LOWER_HEX = frozenset("0123456789abcdef")
_CORPUS_VALIDATION_BLOCK_SIZE = 1_048_576


class ProductionOperationError(RuntimeError):
    """An authorized production operation failed closed."""


class _SourceLockExecutionLease:
    """Process-lifetime OS lease for one shared source-lock mutation path."""

    __slots__ = ("path", "_file_descriptor")

    def __init__(self, *, path: Path, file_descriptor: int) -> None:
        self.path = path
        self._file_descriptor: int | None = file_descriptor

    def release(self) -> None:
        file_descriptor = self._file_descriptor
        if file_descriptor is None:
            return
        self._file_descriptor = None
        try:
            os.lseek(file_descriptor, 0, os.SEEK_SET)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(file_descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.lockf(file_descriptor, fcntl.LOCK_UN, 1)
        finally:
            os.close(file_descriptor)

    def __del__(self) -> None:
        try:
            self.release()
        except OSError:
            pass


def _acquire_source_lock_execution_lease(
    source_record_path: Path,
) -> _SourceLockExecutionLease:
    """Acquire before inspecting the final marker or any live dependency."""

    if (
        not isinstance(source_record_path, Path)
        or not source_record_path.is_absolute()
    ):
        raise ProductionOperationError(
            "source-lock lease requires an absolute shared-mutation path"
        )
    lease_root = Path(tempfile.gettempdir()) / "vfe4-source-lock-leases-v1"
    try:
        lease_root.mkdir(parents=True, exist_ok=True)
        metadata = lease_root.lstat()
    except OSError as exc:
        raise ProductionOperationError(
            "source-lock lease root is unavailable"
        ) from exc
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    is_junction = getattr(lease_root, "is_junction", None)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or bool(getattr(metadata, "st_file_attributes", 0) & reparse)
        or bool(is_junction is not None and is_junction())
    ):
        raise ProductionOperationError(
            "source-lock lease root must be a regular local directory"
        )
    canonical_path = os.path.normcase(
        os.path.realpath(str(source_record_path))
    )
    key = hashlib.sha256(canonical_path.encode("utf-8")).hexdigest()
    path = lease_root / f"{key}.lock"
    flags = os.O_RDWR | os.O_CREAT
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    file_descriptor: int | None = None
    locked = False
    try:
        file_descriptor = os.open(path, flags, 0o600)
        os.set_inheritable(file_descriptor, False)
        opened = os.fstat(file_descriptor)
        observed = path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(observed.st_mode)
            or stat.S_ISLNK(observed.st_mode)
            or bool(getattr(observed, "st_file_attributes", 0) & reparse)
            or (
                opened.st_ino
                and observed.st_ino
                and (
                    opened.st_dev != observed.st_dev
                    or opened.st_ino != observed.st_ino
                )
            )
        ):
            raise ProductionOperationError(
                "source-lock lease must be a regular nonlink file"
            )
        if opened.st_size == 0:
            os.lseek(file_descriptor, 0, os.SEEK_SET)
            if os.write(file_descriptor, b"\0") != 1:
                raise ProductionOperationError(
                    "source-lock lease initialization was incomplete"
                )
            os.fsync(file_descriptor)
        elif opened.st_size != 1:
            raise ProductionOperationError(
                "source-lock lease file is malformed"
            )
        os.lseek(file_descriptor, 0, os.SEEK_SET)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(file_descriptor, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.lockf(
                    file_descriptor,
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                    1,
                )
            locked = True
        except OSError as exc:
            raise ProductionOperationError(
                "another source-lock transaction already owns this "
                "shared-mutation path"
            ) from exc
        os.lseek(file_descriptor, 0, os.SEEK_SET)
        if os.read(file_descriptor, 1) != b"\0":
            raise ProductionOperationError(
                "source-lock lease file is malformed"
            )
        return _SourceLockExecutionLease(
            path=path,
            file_descriptor=file_descriptor,
        )
    except BaseException:
        if file_descriptor is not None:
            if locked:
                try:
                    os.lseek(file_descriptor, 0, os.SEEK_SET)
                    if os.name == "nt":
                        import msvcrt

                        msvcrt.locking(
                            file_descriptor,
                            msvcrt.LK_UNLCK,
                            1,
                        )
                    else:
                        import fcntl

                        fcntl.lockf(file_descriptor, fcntl.LOCK_UN, 1)
                except OSError:
                    pass
            os.close(file_descriptor)
        raise


def _acquire_source_lock_execution_leases(
    paths: tuple[Path, ...],
) -> tuple[_SourceLockExecutionLease, ...]:
    """Acquire every shared mutation identity in canonical order."""

    if type(paths) is not tuple or not paths:
        raise ProductionOperationError(
            "source-lock transaction lease paths must be a nonempty tuple"
        )
    canonical_paths: dict[str, Path] = {}
    for path in paths:
        if not isinstance(path, Path) or not path.is_absolute():
            raise ProductionOperationError(
                "source-lock transaction lease paths must be absolute Paths"
            )
        canonical = os.path.normcase(os.path.realpath(str(path)))
        canonical_paths.setdefault(canonical, path)
    leases: list[_SourceLockExecutionLease] = []
    try:
        for canonical in sorted(canonical_paths):
            leases.append(
                _acquire_source_lock_execution_lease(
                    canonical_paths[canonical]
                )
            )
    except BaseException:
        for lease in reversed(leases):
            lease.release()
        raise
    return tuple(leases)


def _sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ProductionOperationError(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _semantic_payload(
    value: object,
    *,
    omit: tuple[str, ...],
) -> dict[str, object]:
    return {
        field.name: getattr(value, field.name)
        for field in fields(value)
        if field.name not in omit
    }


def _regular_nonlink_bytes(
    path: Path,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ProductionOperationError(
            f"required production artifact is unavailable: {path}: {exc}"
        ) from exc
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    is_junction = getattr(path, "is_junction", None)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or bool(getattr(metadata, "st_file_attributes", 0) & reparse)
        or bool(is_junction is not None and is_junction())
    ):
        raise ProductionOperationError(
            f"production artifact must be a regular nonlink file: {path}"
        )
    payload = path.read_bytes()
    if expected_size is not None and len(payload) != expected_size:
        raise ProductionOperationError(
            f"production artifact size changed: {path}"
        )
    if (
        expected_sha256 is not None
        and hashlib.sha256(payload).hexdigest() != expected_sha256
    ):
        raise ProductionOperationError(
            f"production artifact hash changed: {path}"
        )
    return payload


def _validate_corpus_file(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
) -> None:
    try:
        facts = validate_regular_nonlink_sha256(
            path,
            expected_size_bytes=expected_size,
            expected_sha256=expected_sha256,
            block_size=_CORPUS_VALIDATION_BLOCK_SIZE,
        )
    except Exception as exc:
        raise ProductionOperationError(
            f"production corpus validation failed: {path}: {exc}"
        ) from exc
    if (
        facts.size_bytes != expected_size
        or facts.sha256 != expected_sha256
    ):
        raise ProductionOperationError(
            "production corpus validator returned inconsistent facts"
        )


def _platform_backend() -> ContentAddressedDurabilityBackend:
    return (
        WindowsDurabilityBackend()
        if os.name == "nt"
        else PosixDurabilityBackend()
    )


def _content_type(observation: BoundedHttpObservation) -> str | None:
    values = tuple(
        value.split(";", 1)[0].strip().lower()
        for name, value in observation.headers
        if name == "content-type"
    )
    if not values:
        return None
    if len(values) != 1 or not values[0]:
        raise ProductionOperationError(
            "HTTP content-type observation is ambiguous"
        )
    return values[0]


class ProductionTokenizerAdapter(ExactPieceTokenizerAdapter, Protocol):
    distribution_name: str
    distribution_version: str
    encoding_name: str
    vocabulary_size: int
    eot_token_id: int
    distribution_record_sha256: str
    regex_pattern_sha256: str
    mergeable_ranks_sha256: str
    special_tokens_sha256: str
    golden_vectors_sha256: str
    tokenizer_tables_sha256: str
    regex_engine_distribution_name: str
    regex_engine_distribution_version: str
    regex_engine_distribution_record_sha256: str

    def encode_ordinary(self, text: str) -> list[int]: ...

    def decode(self, token_ids: list[int]) -> str: ...


@dataclass(frozen=True, slots=True)
class ProductionSourceLockDependencies:
    """Injected seam for hermetic source-lock tests and live defaults."""

    http_client: BoundedHttpClient
    tokenizer: ProductionTokenizerAdapter
    installed_distributions: tuple[DistributionIdentity, ...]
    pytorch_version: str
    sdpa_api_sha256: str
    flash_backend_sha256: str
    repository_root: Path
    durability_backend: ContentAddressedDurabilityBackend

    def __post_init__(self) -> None:
        if (
            not callable(getattr(self.http_client, "fetch", None))
            or not callable(
                getattr(
                    self.durability_backend,
                    "publish_content_addressed_stream",
                    None,
                )
            )
            or not callable(
                getattr(self.tokenizer, "encode_ordinary", None)
            )
            or not callable(getattr(self.tokenizer, "decode", None))
            or type(self.installed_distributions) is not tuple
            or tuple(item.name for item in self.installed_distributions)
            != tuple(
                sorted(item.name for item in self.installed_distributions)
            )
            or any(
                type(item) is not DistributionIdentity
                for item in self.installed_distributions
            )
            or type(self.pytorch_version) is not str
            or not self.pytorch_version
            or not isinstance(self.repository_root, Path)
            or not self.repository_root.is_absolute()
        ):
            raise ProductionOperationError(
                "source-lock dependency seam is malformed"
            )
        for item in self.installed_distributions:
            item.__post_init__()
        for name in (
            "sdpa_api_sha256",
            "flash_backend_sha256",
        ):
            _sha256(getattr(self, name), name)


class _RecordingHttpClient:
    def __init__(self, delegate: BoundedHttpClient) -> None:
        self._delegate = delegate
        self.observations: dict[str, BoundedHttpObservation] = {}

    def fetch(
        self,
        url: str,
        *,
        maximum_bytes: int,
    ) -> BoundedHttpObservation:
        if url in self.observations:
            raise ProductionOperationError(
                "source-lock attempted a duplicate candidate download"
            )
        observed = self._delegate.fetch(url, maximum_bytes=maximum_bytes)
        if type(observed) is not BoundedHttpObservation:
            raise ProductionOperationError(
                "HTTP client returned an untyped observation"
            )
        observed.__post_init__()
        self.observations[url] = observed
        return observed


class _LiveBoundedHttpClient:
    """Bounded HTTPS client with explicit manual redirect observations."""

    def fetch(
        self,
        url: str,
        *,
        maximum_bytes: int,
    ) -> BoundedHttpObservation:
        # Deliberately live and reachable only from run_source_lock's
        # authorized default-dependency construction.
        import urllib.error
        import urllib.request
        from urllib.parse import urljoin

        class _NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args: object, **kwargs: object):
                del args, kwargs
                return None

        opener = urllib.request.build_opener(_NoRedirect)
        current = url
        redirects: list[HttpRedirectObservation] = []
        for _ in range(6):
            request = urllib.request.Request(
                current,
                headers={
                    "Accept": "*/*",
                    "User-Agent": "VFE4-WT103-SourceLock/1",
                },
                method="GET",
            )
            try:
                response = opener.open(request, timeout=120)
            except urllib.error.HTTPError as exc:
                if exc.code not in (301, 302, 303, 307, 308):
                    raise ProductionOperationError(
                        f"source-lock HTTP request failed: {exc.code}"
                    ) from exc
                location = exc.headers.get("Location")
                if type(location) is not str or not location:
                    raise ProductionOperationError(
                        "HTTP redirect omitted Location"
                    ) from exc
                resolved = urljoin(current, location)
                redirects.append(
                    HttpRedirectObservation(
                        status_code=exc.code,
                        location=location,
                        resolved_url=resolved,
                    )
                )
                current = resolved
                continue
            with response:
                status_code = int(response.status)
                header_map: dict[str, list[str]] = {}
                for name, value in response.headers.items():
                    header_map.setdefault(name.lower(), []).append(
                        value.strip()
                    )
                headers = tuple(
                    (
                        name,
                        ",".join(values),
                    )
                    for name, values in sorted(header_map.items())
                )
                chunks: list[bytes] = []
                observed_size = 0
                while True:
                    chunk = response.read(
                        min(1024 * 1024, maximum_bytes + 1 - observed_size)
                    )
                    if not chunk:
                        break
                    observed_size += len(chunk)
                    if observed_size > maximum_bytes:
                        raise ProductionOperationError(
                            "HTTP response exceeded its byte bound"
                        )
                    chunks.append(chunk)
                return BoundedHttpObservation.create(
                    request_url=url,
                    final_url=current,
                    redirect_chain=tuple(redirects),
                    status_code=status_code,
                    headers=headers,
                    body=b"".join(chunks),
                )
        raise ProductionOperationError(
            "HTTP redirect chain exceeded five hops"
        )


class _LiveTokenizer:
    def __init__(
        self,
        *,
        encoding: object,
        distribution_record_sha256: str,
        regex_pattern_sha256: str,
        mergeable_ranks_sha256: str,
        special_tokens_sha256: str,
        golden_vectors_sha256: str,
        tokenizer_tables_sha256: str,
        regex_engine_distribution_name: str,
        regex_engine_distribution_version: str,
        regex_engine_distribution_record_sha256: str,
        regex_pattern: object,
    ) -> None:
        self.distribution_name = "tiktoken"
        self.distribution_version = "0.12.0"
        self.encoding_name = "gpt2"
        self.vocabulary_size = 50_257
        self.eot_token_id = 50_256
        self.distribution_record_sha256 = distribution_record_sha256
        self.regex_pattern_sha256 = regex_pattern_sha256
        self.mergeable_ranks_sha256 = mergeable_ranks_sha256
        self.special_tokens_sha256 = special_tokens_sha256
        self.golden_vectors_sha256 = golden_vectors_sha256
        self.tokenizer_tables_sha256 = tokenizer_tables_sha256
        self.regex_engine_distribution_name = (
            regex_engine_distribution_name
        )
        self.regex_engine_distribution_version = (
            regex_engine_distribution_version
        )
        self.regex_engine_distribution_record_sha256 = (
            regex_engine_distribution_record_sha256
        )
        self._encoding = encoding
        self._regex_pattern = regex_pattern

    def encode_ordinary(self, text: str) -> list[int]:
        result = self._encoding.encode_ordinary(text)
        if type(result) is not list or any(
            type(item) is not int for item in result
        ):
            raise ProductionOperationError(
                "live tokenizer returned noncanonical token IDs"
            )
        return result

    def decode(self, token_ids: list[int]) -> str:
        result = self._encoding.decode(token_ids)
        if type(result) is not str:
            raise ProductionOperationError(
                "live tokenizer returned nontext decode output"
            )
        return result

    def split_regex_pieces(self, text: str) -> tuple[str, ...]:
        pieces = tuple(
            match.group(0) for match in self._regex_pattern.finditer(text)
        )
        if any(type(piece) is not str for piece in pieces):
            raise ProductionOperationError(
                "live tokenizer regex returned nontext pieces"
            )
        return pieces

    def encode_single_piece(self, piece: str) -> list[int]:
        result = self._encoding._core_bpe.encode_single_piece(  # noqa: SLF001
            piece.encode("utf-8")
        )
        if type(result) is not list or any(
            type(item) is not int for item in result
        ):
            raise ProductionOperationError(
                "live tokenizer returned noncanonical single-piece token IDs"
            )
        return result

    def decode_token_bytes(self, token_ids: list[int]) -> bytes:
        result = self._encoding.decode_bytes(token_ids)
        if type(result) is not bytes:
            raise ProductionOperationError(
                "live tokenizer returned nonbytes token decode"
            )
        return result


def _installed_distribution_identity(
    name: str,
    *,
    metadata_module: object,
) -> DistributionIdentity:
    distribution = metadata_module.distribution(name)
    normalized_name = str(distribution.metadata["Name"]).lower()
    version = str(distribution.version)
    records: list[tuple[str, int, str]] = []
    for entry in sorted(
        distribution.files or (),
        key=lambda item: str(item).replace("\\", "/"),
    ):
        path = Path(distribution.locate_file(entry))
        if not path.is_file():
            continue
        payload = _regular_nonlink_bytes(path)
        records.append(
            (
                str(entry).replace("\\", "/"),
                len(payload),
                hashlib.sha256(payload).hexdigest(),
            )
        )
    if not records:
        raise ProductionOperationError(
            f"installed distribution {name!r} exposes no file inventory"
        )
    return DistributionIdentity(
        name=normalized_name,
        version=version,
        record_sha256=owned_sha256(
            "vfe4.wt103.installed-distribution-record.v1",
            tuple(records),
        ),
    )


def _live_source_lock_dependencies() -> ProductionSourceLockDependencies:
    """Perform all first-live imports for the authorized source-lock call."""

    # These imports must stay inside this function. The launcher calls it only
    # after exact SOURCE_LOCK_AUTHORIZATION validation.
    import importlib.metadata
    import inspect

    import regex as regex_module
    import tiktoken
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    installed = tuple(
        sorted(
            (
                _installed_distribution_identity(
                    name,
                    metadata_module=importlib.metadata,
                )
                for name in ("matplotlib", "tiktoken")
            ),
            key=lambda item: item.name,
        )
    )
    tiktoken_distribution = next(
        item for item in installed if item.name == "tiktoken"
    )
    regex_distribution = _installed_distribution_identity(
        "regex",
        metadata_module=importlib.metadata,
    )
    if (
        tiktoken_distribution.version != "0.12.0"
        or tiktoken.__version__ != "0.12.0"
    ):
        raise ProductionOperationError(
            "installed tiktoken version differs from 0.12.0"
        )
    encoding = tiktoken.get_encoding("gpt2")
    if (
        encoding.name != "gpt2"
        or encoding.n_vocab != 50_257
        or encoding.eot_token != 50_256
    ):
        raise ProductionOperationError(
            "live GPT-2 tokenizer literals changed"
        )
    regex = str(encoding._pat_str)  # noqa: SLF001
    compiled_regex = regex_module.compile(regex)
    ranks = tuple(
        sorted(
            (
                bytes(token).hex(),
                int(rank),
            )
            for token, rank in encoding._mergeable_ranks.items()  # noqa: SLF001
        )
    )
    special = tuple(
        sorted(
            (
                str(token),
                int(token_id),
            )
            for token, token_id in encoding._special_tokens.items()  # noqa: SLF001
        )
    )
    regex_sha = hashlib.sha256(regex.encode("utf-8")).hexdigest()
    ranks_sha = owned_sha256(
        "vfe4.wt103.gpt2-mergeable-ranks.v1",
        ranks,
    )
    special_sha = owned_sha256(
        "vfe4.wt103.gpt2-special-tokens.v1",
        special,
    )
    golden_texts = (
        ("ascii", "The quick brown fox."),
        ("unicode", "naïve café 世界"),
        ("newlines", "alpha\n\nbeta\n"),
    )
    golden_rows: list[tuple[str, str, tuple[int, ...]]] = []
    for label, text in golden_texts:
        token_ids = encoding.encode_ordinary(text)
        if encoding.decode(token_ids) != text:
            raise ProductionOperationError(
                f"live tokenizer golden round trip failed: {label}"
            )
        golden_rows.append((label, text, tuple(token_ids)))
    golden_sha = owned_sha256(
        "vfe4.wt103.gpt2-golden-vectors.v1",
        tuple(golden_rows),
    )
    tables_sha = production_tokenizer_tables_sha256(
        regex_pattern_sha256=regex_sha,
        regex_engine_distribution_name=regex_distribution.name,
        regex_engine_distribution_version=regex_distribution.version,
        regex_engine_distribution_record_sha256=(
            regex_distribution.record_sha256
        ),
        mergeable_ranks_sha256=ranks_sha,
        special_tokens_sha256=special_sha,
        golden_vectors_sha256=golden_sha,
    )
    live_tokenizer = _LiveTokenizer(
        encoding=encoding,
        distribution_record_sha256=(
            tiktoken_distribution.record_sha256
        ),
        regex_pattern_sha256=regex_sha,
        mergeable_ranks_sha256=ranks_sha,
        special_tokens_sha256=special_sha,
        golden_vectors_sha256=golden_sha,
        tokenizer_tables_sha256=tables_sha,
        regex_engine_distribution_name=regex_distribution.name,
        regex_engine_distribution_version=regex_distribution.version,
        regex_engine_distribution_record_sha256=(
            regex_distribution.record_sha256
        ),
        regex_pattern=compiled_regex,
    )
    verify_piece_stream_golden_vectors(
        adapter=live_tokenizer,
        golden_vectors=tuple(golden_rows),
        limits=TokenizerStreamLimits(
            input_chunk_size_bytes=7,
            retained_piece_size_bytes=4_096,
            output_chunk_size_bytes=64,
        ),
    )
    try:
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            pass
    except Exception as exc:
        raise ProductionOperationError(
            "exact Flash-only SDPA context is unavailable"
        ) from exc
    try:
        sdpa_source = inspect.getsource(sdpa_kernel).encode("utf-8")
    except (OSError, TypeError):
        sdpa_source = repr(sdpa_kernel).encode("utf-8")
    sdpa_sha = hashlib.sha256(sdpa_source).hexdigest()
    flash_sha = owned_sha256(
        "vfe4.wt103.flash-backend-identity.v1",
        {
            "backend": str(SDPBackend.FLASH_ATTENTION),
            "enum_module": SDPBackend.__module__,
            "sdpa_api_sha256": sdpa_sha,
        },
    )
    return ProductionSourceLockDependencies(
        http_client=_LiveBoundedHttpClient(),
        tokenizer=live_tokenizer,
        installed_distributions=installed,
        pytorch_version=str(torch.__version__),
        sdpa_api_sha256=sdpa_sha,
        flash_backend_sha256=flash_sha,
        repository_root=Path(__file__).resolve().parents[2],
        durability_backend=_platform_backend(),
    )


@dataclass(frozen=True, slots=True)
class ProductionTokenCacheRecord:
    schema_version: Literal["wt103-production-token-cache-record-v1"]
    split: Literal["train", "validation", "test"]
    raw_parent_sha256: str
    tokenizer: ProductionTokenizerSpec
    cache_identity: ProductionTokenCacheIdentity
    token_count: int
    minimum_token_id: int
    maximum_token_id: int
    payload_size_bytes: int
    payload_sha256: str
    cache_relative_path: str
    record_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != "wt103-production-token-cache-record-v1"
            or self.split not in ("train", "validation", "test")
            or type(self.tokenizer) is not ProductionTokenizerSpec
            or type(self.cache_identity)
            is not ProductionTokenCacheIdentity
            or self.cache_identity.tokenizer != self.tokenizer
            or self.cache_identity.split != self.split
            or self.cache_identity.payload_sha256 != self.payload_sha256
            or type(self.token_count) is not int
            or self.token_count < 2
            or type(self.minimum_token_id) is not int
            or type(self.maximum_token_id) is not int
            or not 0 <= self.minimum_token_id
            <= self.maximum_token_id
            < self.tokenizer.vocabulary_size
            or self.payload_size_bytes != 4 * self.token_count
            or self.cache_relative_path
            != (
                f"production-token-cache/{self.split}/"
                f"{self.payload_sha256}.int32le"
            )
        ):
            raise ProductionOperationError(
                "production token-cache record is inconsistent"
            )
        self.tokenizer.__post_init__()
        self.cache_identity.__post_init__()
        for name in ("raw_parent_sha256", "payload_sha256"):
            _sha256(getattr(self, name), name)
        expected = owned_sha256(
            "vfe4.wt103.production-token-cache-record.v1",
            _semantic_payload(self, omit=("record_sha256",)),
        )
        _sha256(self.record_sha256, "record_sha256")
        if self.record_sha256 != expected:
            raise ProductionOperationError(
                "production token-cache record hash does not match"
            )

    @classmethod
    def create(cls, **values: object) -> "ProductionTokenCacheRecord":
        payload = {
            "schema_version": (
                "wt103-production-token-cache-record-v1"
            ),
            **values,
        }
        return cls(
            **payload,
            record_sha256=owned_sha256(
                "vfe4.wt103.production-token-cache-record.v1",
                payload,
            ),
        )  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ProductionScheduleSet:
    schema_version: Literal["wt103-production-schedule-set-v1"]
    window_manifests: tuple[
        WindowManifest,
        WindowManifest,
        WindowManifest,
    ]
    window_row_relative_paths: tuple[str, str, str]
    permutation_manifests: tuple[
        PermutationManifest,
        PermutationManifest,
    ]
    permutation_relative_paths: tuple[str, str]
    schedule_sha256s: tuple[str, str, str, str]
    validation_boundary_batch_ordinals: tuple[int, ...]
    checkpoint_roles: tuple[
        Literal["resume_only"],
        Literal["terminal_scoring"],
    ]
    cadence_sha256: str
    schedule_set_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != "wt103-production-schedule-set-v1"
            or type(self.window_manifests) is not tuple
            or len(self.window_manifests) != 3
            or tuple(item.split for item in self.window_manifests)
            != ("train", "validation", "test")
            or any(
                type(item) is not WindowManifest
                for item in self.window_manifests
            )
            or type(self.permutation_manifests) is not tuple
            or len(self.permutation_manifests) != 2
            or tuple(
                item.pass_index for item in self.permutation_manifests
            )
            != (0, 1)
            or type(self.schedule_sha256s) is not tuple
            or len(self.schedule_sha256s) != 4
            or type(self.validation_boundary_batch_ordinals) is not tuple
            or not self.validation_boundary_batch_ordinals
            or tuple(sorted(set(self.validation_boundary_batch_ordinals)))
            != self.validation_boundary_batch_ordinals
            or self.checkpoint_roles
            != ("resume_only", "terminal_scoring")
        ):
            raise ProductionOperationError(
                "production schedule set is inconsistent"
            )
        for item in (
            *self.window_manifests,
            *self.permutation_manifests,
        ):
            item.__post_init__()
        for value in (
            *self.schedule_sha256s,
            self.cadence_sha256,
            self.schedule_set_sha256,
        ):
            _sha256(value, "schedule identity")
        expected = owned_sha256(
            "vfe4.wt103.production-schedule-set.v1",
            _semantic_payload(self, omit=("schedule_set_sha256",)),
        )
        if self.schedule_set_sha256 != expected:
            raise ProductionOperationError(
                "production schedule-set hash does not match"
            )

    @classmethod
    def create(cls, **values: object) -> "ProductionScheduleSet":
        payload = {
            "schema_version": "wt103-production-schedule-set-v1",
            **values,
        }
        return cls(
            **payload,
            schedule_set_sha256=owned_sha256(
                "vfe4.wt103.production-schedule-set.v1",
                payload,
            ),
        )  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class A5SemanticFlopOperator:
    """One actual PRIMARY phase/operator row in the source-lock ledger."""

    operator_id: str
    phase: Literal[
        "objective_forward",
        "recognition_backward",
        "model_backward",
        "gradient_control",
        "adamw",
        "scheduler",
    ]
    invocations_per_batch: int
    arithmetic_flops_per_batch: int | None
    unresolved_obligation: str | None

    def __post_init__(self) -> None:
        if (
            type(self.operator_id) is not str
            or not self.operator_id
            or self.phase
            not in (
                "objective_forward",
                "recognition_backward",
                "model_backward",
                "gradient_control",
                "adamw",
                "scheduler",
            )
            or type(self.invocations_per_batch) is not int
            or self.invocations_per_batch <= 0
        ):
            raise ProductionOperationError(
                "PRIMARY A5 FLOP operator row is malformed"
            )
        if self.arithmetic_flops_per_batch is None:
            if (
                type(self.unresolved_obligation) is not str
                or not self.unresolved_obligation
            ):
                raise ProductionOperationError(
                    "unresolved PRIMARY A5 operator requires an obligation"
                )
        elif (
            type(self.arithmetic_flops_per_batch) is not int
            or self.arithmetic_flops_per_batch < 0
            or self.unresolved_obligation is not None
        ):
            raise ProductionOperationError(
                "resolved PRIMARY A5 operator FLOPs are inconsistent"
            )


@dataclass(frozen=True, slots=True)
class A5SemanticTrainingFlopLedger:
    """Explicit actual-phase ledger that refuses an unfrozen scalar total."""

    schema_version: Literal[
        "wt103-primary-a5-semantic-train-flops-v1"
    ]
    primary_parameters: WT103PrimaryParameterInventory
    optimizer_steps: int
    objective_forward_evaluations_per_batch: Literal[4]
    recognition_backward_evaluations_per_batch: Literal[1]
    model_backward_evaluations_per_batch: Literal[1]
    adamw_scalar_flops_per_parameter: Literal[15]
    validation_scoring_included: Literal[False]
    operators: tuple[A5SemanticFlopOperator, ...]
    semantic_train_flops: None
    status: GateStatus
    obligations: tuple[str, ...]
    ledger_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != "wt103-primary-a5-semantic-train-flops-v1"
            or type(self.primary_parameters)
            is not WT103PrimaryParameterInventory
            or type(self.optimizer_steps) is not int
            or self.optimizer_steps <= 0
            or self.objective_forward_evaluations_per_batch != 4
            or self.recognition_backward_evaluations_per_batch != 1
            or self.model_backward_evaluations_per_batch != 1
            or self.adamw_scalar_flops_per_parameter != 15
            or self.validation_scoring_included is not False
            or type(self.operators) is not tuple
            or not self.operators
            or any(
                type(item) is not A5SemanticFlopOperator
                for item in self.operators
            )
            or len({item.operator_id for item in self.operators})
            != len(self.operators)
            or self.semantic_train_flops is not None
            or self.status is not GateStatus.INCONCLUSIVE
        ):
            raise ProductionOperationError(
                "PRIMARY A5 semantic FLOP ledger is inconsistent"
            )
        self.primary_parameters.__post_init__()
        for item in self.operators:
            item.__post_init__()
        expected_obligations = tuple(
            dict.fromkeys(
                item.unresolved_obligation
                for item in self.operators
                if item.unresolved_obligation is not None
            )
        )
        if (
            not expected_obligations
            or self.obligations != expected_obligations
        ):
            raise ProductionOperationError(
                "PRIMARY A5 FLOP obligations do not match its operator rows"
            )
        expected = owned_sha256(
            "vfe4.wt103.primary-a5-semantic-train-flops.v1",
            _semantic_payload(self, omit=("ledger_sha256",)),
        )
        _sha256(self.ledger_sha256, "PRIMARY A5 FLOP ledger SHA-256")
        if self.ledger_sha256 != expected:
            raise ProductionOperationError(
                "PRIMARY A5 FLOP ledger hash does not match"
            )

    @classmethod
    def create(
        cls,
        *,
        primary_parameters: WT103PrimaryParameterInventory,
        optimizer_steps: int,
        operators: tuple[A5SemanticFlopOperator, ...],
    ) -> "A5SemanticTrainingFlopLedger":
        obligations = tuple(
            dict.fromkeys(
                item.unresolved_obligation
                for item in operators
                if item.unresolved_obligation is not None
            )
        )
        payload = {
            "schema_version": (
                "wt103-primary-a5-semantic-train-flops-v1"
            ),
            "primary_parameters": primary_parameters,
            "optimizer_steps": optimizer_steps,
            "objective_forward_evaluations_per_batch": 4,
            "recognition_backward_evaluations_per_batch": 1,
            "model_backward_evaluations_per_batch": 1,
            "adamw_scalar_flops_per_parameter": 15,
            "validation_scoring_included": False,
            "operators": operators,
            "semantic_train_flops": None,
            "status": GateStatus.INCONCLUSIVE,
            "obligations": obligations,
        }
        return cls(
            **payload,
            ledger_sha256=owned_sha256(
                "vfe4.wt103.primary-a5-semantic-train-flops.v1",
                payload,
            ),
        )  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class A0ParameterMatchRow:
    """Corpus-free parameter half of one frozen A0 candidate row."""

    hidden_width: int
    parameter_count: int
    parameter_ratio: float
    parameter_relative_error: float
    parameter_eligible: bool

    def __post_init__(self) -> None:
        if self.hidden_width not in WT103_A0_HIDDEN_WIDTH_CANDIDATES:
            raise ProductionOperationError(
                "A0 parameter-match width is not preregistered"
            )
        expected_count = (
            2 * 50_257 * self.hidden_width
            + 128 * self.hidden_width
            + 12 * self.hidden_width**2
            + 15 * self.hidden_width
            + 50_257
        )
        if (
            type(self.parameter_count) is not int
            or self.parameter_count != expected_count
            or type(self.parameter_ratio) is not float
            or not math.isfinite(self.parameter_ratio)
            or type(self.parameter_relative_error) is not float
            or not math.isfinite(self.parameter_relative_error)
            or self.parameter_relative_error
            != abs(self.parameter_ratio - 1.0)
            or type(self.parameter_eligible) is not bool
            or self.parameter_eligible
            is not (self.parameter_relative_error <= 0.01)
        ):
            raise ProductionOperationError(
                "A0 parameter-match row is inconsistent"
            )


@dataclass(frozen=True, slots=True)
class A0SourceLockMatchingAssessment:
    """Fail-closed match record before an exact PRIMARY FLOP total exists."""

    schema_version: Literal[
        "wt103-a0-source-lock-matching-assessment-v1"
    ]
    primary_arm_spec_sha256: str
    endpoint_inventory_sha256: str
    primary_parameters: WT103PrimaryParameterInventory
    primary_flop_ledger: A5SemanticTrainingFlopLedger
    primary_semantic_train_flops: None
    parameter_relative_tolerance: Literal[0.01]
    flop_relative_tolerance: Literal[0.05]
    candidate_hidden_widths: tuple[int, ...]
    rows: tuple[A0ParameterMatchRow, ...]
    selected_hidden_width: None
    status: GateStatus
    obligations: tuple[str, ...]
    matching_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != "wt103-a0-source-lock-matching-assessment-v1"
            or type(self.primary_parameters)
            is not WT103PrimaryParameterInventory
            or type(self.primary_flop_ledger)
            is not A5SemanticTrainingFlopLedger
            or self.primary_semantic_train_flops is not None
            or self.parameter_relative_tolerance != 0.01
            or self.flop_relative_tolerance != 0.05
            or self.candidate_hidden_widths
            != WT103_A0_HIDDEN_WIDTH_CANDIDATES
            or type(self.rows) is not tuple
            or tuple(item.hidden_width for item in self.rows)
            != self.candidate_hidden_widths
            or any(
                type(item) is not A0ParameterMatchRow
                for item in self.rows
            )
            or self.selected_hidden_width is not None
            or self.status is not GateStatus.INCONCLUSIVE
        ):
            raise ProductionOperationError(
                "A0 source-lock matching assessment is inconsistent"
            )
        self.primary_parameters.__post_init__()
        self.primary_flop_ledger.__post_init__()
        for item in self.rows:
            item.__post_init__()
        _sha256(
            self.primary_arm_spec_sha256,
            "PRIMARY arm specification SHA-256",
        )
        _sha256(
            self.endpoint_inventory_sha256,
            "endpoint inventory SHA-256",
        )
        if (
            self.primary_arm_spec_sha256
            != self.primary_parameters.arm_spec_sha256
            or self.primary_flop_ledger.primary_parameters
            != self.primary_parameters
            or any(item.parameter_eligible for item in self.rows)
            or self.obligations
            != (
                "no_a0_candidate_satisfies_parameter_margin",
                *self.primary_flop_ledger.obligations,
            )
        ):
            raise ProductionOperationError(
                "A0 source-lock matching obligations are not exact"
            )
        expected = owned_sha256(
            "vfe4.wt103.a0-source-lock-matching-assessment.v1",
            _semantic_payload(self, omit=("matching_sha256",)),
        )
        _sha256(self.matching_sha256, "A0 matching SHA-256")
        if self.matching_sha256 != expected:
            raise ProductionOperationError(
                "A0 matching assessment hash does not match"
            )

    @classmethod
    def create(
        cls,
        *,
        endpoint_inventory_sha256: str,
        primary_parameters: WT103PrimaryParameterInventory,
        primary_flop_ledger: A5SemanticTrainingFlopLedger,
        rows: tuple[A0ParameterMatchRow, ...],
    ) -> "A0SourceLockMatchingAssessment":
        payload = {
            "schema_version": (
                "wt103-a0-source-lock-matching-assessment-v1"
            ),
            "primary_arm_spec_sha256": (
                primary_parameters.arm_spec_sha256
            ),
            "endpoint_inventory_sha256": endpoint_inventory_sha256,
            "primary_parameters": primary_parameters,
            "primary_flop_ledger": primary_flop_ledger,
            "primary_semantic_train_flops": None,
            "parameter_relative_tolerance": 0.01,
            "flop_relative_tolerance": 0.05,
            "candidate_hidden_widths": (
                WT103_A0_HIDDEN_WIDTH_CANDIDATES
            ),
            "rows": rows,
            "selected_hidden_width": None,
            "status": GateStatus.INCONCLUSIVE,
            "obligations": (
                "no_a0_candidate_satisfies_parameter_margin",
                *primary_flop_ledger.obligations,
            ),
        }
        return cls(
            **payload,
            matching_sha256=owned_sha256(
                "vfe4.wt103.a0-source-lock-matching-assessment.v1",
                payload,
            ),
        )  # type: ignore[arg-type]


def _tokenizer_authority_cross_links_match(
    *,
    finalized_spec_sha256: str,
    finalized_tables_sha256: str,
    tokenizer: ProductionTokenizerSpec,
) -> bool:
    return (
        type(tokenizer) is ProductionTokenizerSpec
        and finalized_spec_sha256 == tokenizer.spec_sha256
        and finalized_tables_sha256 == tokenizer.tokenizer_tables_sha256
    )


@dataclass(frozen=True, slots=True)
class ProductionSourceLock:
    schema_version: Literal["wt103-production-source-lock-v1"]
    finalized_source: FinalizedWikiText103SourceRecord
    tokenizer: ProductionTokenizerSpec
    token_caches: tuple[
        ProductionTokenCacheRecord,
        ProductionTokenCacheRecord,
        ProductionTokenCacheRecord,
    ]
    schedules: ProductionScheduleSet
    a0_architecture: A0ArchitectureProfile
    a0_formula: A0FormulaRecord
    a0_flop_ledger: A0FlopLedger
    a0_matching: ArmMatchingReport | A0SourceLockMatchingAssessment
    source_lock_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != "wt103-production-source-lock-v1"
            or type(self.finalized_source)
            is not FinalizedWikiText103SourceRecord
            or type(self.tokenizer) is not ProductionTokenizerSpec
            or type(self.token_caches) is not tuple
            or len(self.token_caches) != 3
            or tuple(item.split for item in self.token_caches)
            != ("train", "validation", "test")
            or any(
                type(item) is not ProductionTokenCacheRecord
                for item in self.token_caches
            )
            or type(self.schedules) is not ProductionScheduleSet
            or type(self.a0_architecture) is not A0ArchitectureProfile
            or type(self.a0_formula) is not A0FormulaRecord
            or type(self.a0_flop_ledger) is not A0FlopLedger
            or type(self.a0_matching)
            not in (
                ArmMatchingReport,
                A0SourceLockMatchingAssessment,
            )
            or self.a0_architecture.source_lock_scope
            != "production_source_lock_verified"
        ):
            raise ProductionOperationError(
                "production source-lock bundle is inconsistent"
            )
        self.finalized_source.__post_init__()
        self.tokenizer.__post_init__()
        self.schedules.__post_init__()
        self.a0_architecture.__post_init__()
        self.a0_formula.__post_init__()
        self.a0_flop_ledger.__post_init__()
        self.a0_matching.__post_init__()
        for item in self.token_caches:
            item.__post_init__()
        cache_set = production_token_cache_set_sha256(
            tuple(item.cache_identity for item in self.token_caches)
        )
        member_cache_cross_links_valid = all(
            cache.split == member.split
            and cache.raw_parent_sha256 == member.payload_sha256
            and cache.tokenizer == self.tokenizer
            and cache.cache_identity.tokenizer == self.tokenizer
            for member, cache in zip(
                self.finalized_source.members,
                self.token_caches,
                strict=True,
            )
        )
        if type(self.a0_matching) is ArmMatchingReport:
            matching_cross_links_valid = (
                self.a0_formula.hidden_width
                == self.a0_matching.selected_hidden_width
                and self.a0_matching.status is GateStatus.PASS
            )
        else:
            matching_cross_links_valid = (
                self.a0_matching.status is GateStatus.INCONCLUSIVE
                and self.a0_matching.selected_hidden_width is None
                and any(
                    row.hidden_width == self.a0_formula.hidden_width
                    and row.parameter_count
                    == self.a0_formula.parameter_count
                    for row in self.a0_matching.rows
                )
            )
        if (
            not _tokenizer_authority_cross_links_match(
                finalized_spec_sha256=(
                    self.finalized_source.production_tokenizer_spec_sha256
                ),
                finalized_tables_sha256=(
                    self.finalized_source.tokenizer_tables_sha256
                ),
                tokenizer=self.tokenizer,
            )
            or not member_cache_cross_links_valid
            or self.finalized_source.production_token_cache_set_sha256
            != cache_set
            or self.finalized_source.schedule_set_sha256
            != self.schedules.schedule_set_sha256
            or self.a0_architecture.formula_sha256
            != self.a0_formula.formula_sha256
            or self.a0_formula.semantic_train_flops
            != self.a0_flop_ledger.semantic_train_flops
            or self.a0_formula.parameter_count
            != self.a0_flop_ledger.parameter_count
            or not matching_cross_links_valid
        ):
            raise ProductionOperationError(
                "source-lock bundle cross-links disagree"
            )
        expected = owned_sha256(
            "vfe4.wt103.production-source-lock.v1",
            _semantic_payload(self, omit=("source_lock_sha256",)),
        )
        _sha256(self.source_lock_sha256, "source_lock_sha256")
        if self.source_lock_sha256 != expected:
            raise ProductionOperationError(
                "source-lock bundle hash does not match"
            )

    @classmethod
    def create(cls, **values: object) -> "ProductionSourceLock":
        payload = {
            "schema_version": "wt103-production-source-lock-v1",
            **values,
        }
        return cls(
            **payload,
            source_lock_sha256=owned_sha256(
                "vfe4.wt103.production-source-lock.v1",
                payload,
            ),
        )  # type: ignore[arg-type]


def _owned_path(
    root: Path,
    relative: str,
    *,
    prepare_parents: bool = False,
) -> Path:
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or "\\" in relative
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise ProductionOperationError(
            "production artifact relative path is noncanonical"
        )
    current = root
    try:
        root_metadata = current.lstat()
    except FileNotFoundError:
        if not prepare_parents:
            return root.joinpath(*pure.parts)
        current.mkdir(parents=True, exist_ok=False)
        root_metadata = current.lstat()
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    root_is_junction = getattr(current, "is_junction", None)
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or stat.S_ISLNK(root_metadata.st_mode)
        or bool(
            getattr(root_metadata, "st_file_attributes", 0) & reparse
        )
        or bool(
            root_is_junction is not None and root_is_junction()
        )
    ):
        raise ProductionOperationError(
            "production artifact root is not a regular directory"
        )
    for part in pure.parts[:-1]:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if not prepare_parents:
                return root.joinpath(*pure.parts)
            current.mkdir(exist_ok=False)
            metadata = current.lstat()
        is_junction = getattr(current, "is_junction", None)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or bool(
                getattr(metadata, "st_file_attributes", 0) & reparse
            )
            or bool(is_junction is not None and is_junction())
        ):
            raise ProductionOperationError(
                "production artifact parent is not a regular directory"
            )
    return root.joinpath(*pure.parts)


def _publish(
    backend: DurabilityBackend,
    path: Path,
    payload: bytes,
) -> None:
    try:
        path.lstat()
    except FileNotFoundError:
        try:
            backend.create_exclusive(path, payload)
        except DurabilityCollisionError as exc:
            observed = _regular_nonlink_bytes(path)
            if observed != payload:
                raise ProductionOperationError(
                    "immutable production artifact differs after a "
                    f"competing exclusive create: {path}"
                ) from exc
    except OSError as exc:
        raise ProductionOperationError(
            f"production publication target is unavailable: {path}: {exc}"
        ) from exc
    else:
        observed = _regular_nonlink_bytes(path)
        if observed != payload:
            raise ProductionOperationError(
                f"immutable production artifact differs: {path}"
            )
        return
    _regular_nonlink_bytes(
        path,
        expected_size=len(payload),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    )


def _replace_exact_reviewed_predecessor(
    backend: DurabilityBackend,
    path: Path,
    *,
    predecessor: bytes,
    replacement: bytes,
) -> None:
    """Perform the sole reviewed Task13 candidate-to-resolution transition.

    ``DurabilityBackend`` has no atomic compare-and-swap operation.  This
    exceptional one-way transition therefore performs a second exact
    predecessor observation immediately before its explicit durable replace.
    Any mutation visible before that call fails closed.  An uncooperative
    writer in the final observation-to-replace instruction gap is outside the
    backend contract; ordinary finalized artifacts never enter this seam.
    """

    observed = _regular_nonlink_bytes(path)
    if observed == replacement:
        return
    if observed != predecessor:
        raise ProductionOperationError(
            f"reviewed dependency-lock predecessor differs: {path}"
        )
    if _regular_nonlink_bytes(path) != predecessor:
        raise ProductionOperationError(
            "reviewed dependency-lock predecessor changed before the "
            f"authorized transition: {path}"
        )
    backend.replace_durable(path, replacement)
    _regular_nonlink_bytes(
        path,
        expected_size=len(replacement),
        expected_sha256=hashlib.sha256(replacement).hexdigest(),
    )


def _validate_lock_writer_source(manifest: LockInputManifest) -> str:
    source_path = Path(render_dependency_lock.__code__.co_filename)
    observed_sha256 = hashlib.sha256(
        _regular_nonlink_bytes(source_path)
    ).hexdigest()
    if observed_sha256 != manifest.writer_code_sha256:
        raise ProductionOperationError(
            "lock-writer source identity changed: "
            f"{source_path}"
        )
    return observed_sha256


def _resolve_dependency_lock(
    dependencies: ProductionSourceLockDependencies,
) -> tuple[bytes, str]:
    manifest_path = dependencies.repository_root / _LOCK_INPUT_NAME
    lock_path = dependencies.repository_root / _LOCK_NAME
    manifest_payload = _regular_nonlink_bytes(manifest_path)
    manifest = parse_lock_input_manifest(manifest_payload)
    _validate_lock_writer_source(manifest)
    lock_payload = _regular_nonlink_bytes(lock_path)
    unresolved_flags = tuple(
        requirement.expected_installed_record_sha256 is None
        for requirement in manifest.requirements
    )
    if all(unresolved_flags):
        candidate_manifest = manifest
    elif not any(unresolved_flags):
        candidate_manifest = LockInputManifest.create(
            writer_code_sha256=manifest.writer_code_sha256,
            target_python_version=manifest.target_python_version,
            requirements=tuple(
                replace(
                    requirement,
                    expected_installed_record_sha256=None,
                    task13_obligation=(
                        "task13_capture_exact_installed_record_sha256:"
                        f"{requirement.name}"
                    ),
                )
                for requirement in manifest.requirements
            ),
        )
    else:
        raise ProductionOperationError(
            "dependency-lock manifest is partially resolved"
        )
    candidate_manifest_payload = (
        canonical_json_bytes_generic(candidate_manifest) + b"\n"
    )
    candidate_lock_payload = render_dependency_lock(candidate_manifest)
    installed = {
        item.name: item for item in dependencies.installed_distributions
    }
    resolved_requirements = []
    for requirement in candidate_manifest.requirements:
        observed = installed.get(requirement.name)
        if observed is None or observed.version != requirement.version:
            raise ProductionOperationError(
                "installed dependency differs from the reviewed lock input: "
                f"{requirement.name}"
            )
        resolved_requirements.append(
            replace(
                requirement,
                expected_installed_record_sha256=(
                    observed.record_sha256
                ),
                task13_obligation=None,
            )
        )
    resolved_manifest = LockInputManifest.create(
        writer_code_sha256=candidate_manifest.writer_code_sha256,
        target_python_version=candidate_manifest.target_python_version,
        requirements=tuple(resolved_requirements),
    )
    resolved_manifest_payload = (
        canonical_json_bytes_generic(resolved_manifest) + b"\n"
    )
    resolved_lock_payload = render_dependency_lock(resolved_manifest)
    if manifest_payload not in (
        candidate_manifest_payload,
        resolved_manifest_payload,
    ):
        raise ProductionOperationError(
            "resolved dependency lock differs from its reviewed transition"
        )
    if lock_payload not in (
        candidate_lock_payload,
        resolved_lock_payload,
    ):
        message = (
            "reviewed unresolved dependency-lock pair differs"
            if manifest_payload == candidate_manifest_payload
            else "resolved dependency lock differs"
        )
        raise ProductionOperationError(message)
    if (
        manifest_payload == resolved_manifest_payload
        and lock_payload == resolved_lock_payload
    ):
        return (
            resolved_lock_payload,
            hashlib.sha256(resolved_lock_payload).hexdigest(),
        )
    if manifest_payload == resolved_manifest_payload:
        # Recover the old writer's bundle-before-lock ordering only when the
        # remaining predecessor is the exact reviewed candidate lock.
        _replace_exact_reviewed_predecessor(
            dependencies.durability_backend,
            lock_path,
            predecessor=candidate_lock_payload,
            replacement=resolved_lock_payload,
        )
        return (
            resolved_lock_payload,
            hashlib.sha256(resolved_lock_payload).hexdigest(),
        )
    # Publish the resolved lock first.  A crash here leaves the exact
    # candidate manifest plus exact resolved lock, which the next call can
    # recognize and close without recomputing or replacing different bytes.
    _replace_exact_reviewed_predecessor(
        dependencies.durability_backend,
        lock_path,
        predecessor=candidate_lock_payload,
        replacement=resolved_lock_payload,
    )
    _replace_exact_reviewed_predecessor(
        dependencies.durability_backend,
        manifest_path,
        predecessor=candidate_manifest_payload,
        replacement=resolved_manifest_payload,
    )
    return (
        resolved_lock_payload,
        hashlib.sha256(resolved_lock_payload).hexdigest(),
    )


def _production_tokenizer_spec(
    adapter: ProductionTokenizerAdapter,
) -> ProductionTokenizerSpec:
    if (
        adapter.distribution_name != "tiktoken"
        or adapter.distribution_version != "0.12.0"
        or adapter.encoding_name != "gpt2"
        or adapter.vocabulary_size != 50_257
        or adapter.eot_token_id != 50_256
    ):
        raise ProductionOperationError(
            "live tokenizer differs from the candidate contract"
        )
    try:
        return ProductionTokenizerSpec.create_verified(
            distribution_record_sha256=(
                adapter.distribution_record_sha256
            ),
            regex_pattern_sha256=adapter.regex_pattern_sha256,
            regex_engine_distribution_name=(
                adapter.regex_engine_distribution_name
            ),
            regex_engine_distribution_version=(
                adapter.regex_engine_distribution_version
            ),
            regex_engine_distribution_record_sha256=(
                adapter.regex_engine_distribution_record_sha256
            ),
            mergeable_ranks_sha256=adapter.mergeable_ranks_sha256,
            special_tokens_sha256=adapter.special_tokens_sha256,
            golden_vectors_sha256=adapter.golden_vectors_sha256,
            tokenizer_tables_sha256=adapter.tokenizer_tables_sha256,
        )
    except (AttributeError, ValueError) as exc:
        raise ProductionOperationError(
            "live tokenizer table authority failed validation"
        ) from exc


def _build_cache_record(
    *,
    split: Literal["train", "validation", "test"],
    staged: StagedWikiText103AcquisitionRecord,
    staging_root: Path,
    cache_root: Path,
    tokenizer: ProductionTokenizerSpec,
    adapter: ProductionTokenizerAdapter,
    backend: ContentAddressedDurabilityBackend,
) -> ProductionTokenCacheRecord:
    sealed = next(item for item in staged.sealed_splits if item.split == split)
    raw_path = staging_root.joinpath(
        *PurePosixPath(sealed.cache_relative_path).parts
    )
    destination_probe = _owned_path(
        cache_root,
        f"production-token-cache/{split}/payload.int32le",
        prepare_parents=True,
    )
    try:
        publication = publish_exact_bounded_token_payload(
            raw_path=raw_path,
            expected_raw_size_bytes=sealed.payload_size_bytes,
            expected_raw_sha256=sealed.payload_sha256,
            adapter=adapter,
            backend=backend,
            destination_directory=destination_probe.parent,
            suffix=".int32le",
            limits=TokenizerStreamLimits(),
        )
    except Exception as exc:
        raise ProductionOperationError(
            f"{split} bounded token publication failed: {exc}"
        ) from exc
    if type(publication) is not PublishedTokenPayload:
        raise ProductionOperationError(
            f"{split} token publication returned an untyped result"
        )
    facts = publication.facts
    identity = publication.durable_identity
    relative = (
        f"production-token-cache/{split}/"
        f"{facts.payload_sha256}.int32le"
    )
    expected_path = _owned_path(cache_root, relative)
    if (
        type(identity) is not DurableFileIdentity
        or identity.operation != "content_addressed"
        or identity.reopen_verified is not True
        or identity.size_bytes != facts.payload_size_bytes
        or identity.sha256 != facts.payload_sha256
        or facts.payload_size_bytes != facts.token_count * 4
        or facts.token_count <= 0
        or not (
            0
            <= facts.minimum_token_id
            <= facts.maximum_token_id
            < tokenizer.vocabulary_size
        )
        or publication.path != expected_path
        or publication.path.parent != destination_probe.parent
    ):
        raise ProductionOperationError(
            f"{split} token stream facts and durable identity disagree"
        )
    cache_identity = ProductionTokenCacheIdentity.create(
        tokenizer=tokenizer,
        split=split,
        payload_sha256=facts.payload_sha256,
    )
    return ProductionTokenCacheRecord.create(
        split=split,
        raw_parent_sha256=sealed.payload_sha256,
        tokenizer=tokenizer,
        cache_identity=cache_identity,
        token_count=facts.token_count,
        minimum_token_id=facts.minimum_token_id,
        maximum_token_id=facts.maximum_token_id,
        payload_size_bytes=facts.payload_size_bytes,
        payload_sha256=facts.payload_sha256,
        cache_relative_path=relative,
    )


def _validation_boundaries(batch_count: int) -> tuple[int, ...]:
    if type(batch_count) is not int or batch_count <= 0:
        raise ProductionOperationError(
            "training pass must contain a positive batch count"
        )
    return tuple(
        sorted(
            {
                min(
                    batch_count,
                    math.ceil(index * batch_count / 20),
                )
                for index in range(1, 21)
            }
        )
    )


def _build_schedule_set(
    *,
    caches: tuple[
        ProductionTokenCacheRecord,
        ProductionTokenCacheRecord,
        ProductionTokenCacheRecord,
    ],
    cache_root: Path,
    backend: DurabilityBackend,
    training: TrainingConfig,
) -> ProductionScheduleSet:
    manifests: list[WindowManifest] = []
    row_paths: list[str] = []
    for cache in caches:
        rows = enumerate_wt103_window_rows(cache.token_count)
        row_payload = _WINDOW_ROWS_DOMAIN + b"".join(
            row.canonical_bytes() for row in rows
        )
        row_sha = hashlib.sha256(row_payload).hexdigest()
        relative = (
            f"production-window-manifests/{cache.split}/"
            f"{row_sha}.rows"
        )
        _publish(
            backend,
            _owned_path(
                cache_root,
                relative,
                prepare_parents=True,
            ),
            row_payload,
        )
        manifests.append(
            WindowManifest.create(
                split=cache.split,
                token_payload_sha256=cache.payload_sha256,
                window_count=len(rows),
                counted_targets=cache.token_count - 1,
                payload_sha256=row_sha,
            )
        )
        row_paths.append(relative)
    train_manifest, validation_manifest, test_manifest = manifests
    permutation_manifests: list[PermutationManifest] = []
    permutation_paths: list[str] = []
    train_schedule_hashes: list[str] = []
    for pass_index in (0, 1):
        generator = np.random.Generator(
            np.random.PCG64(
                np.random.SeedSequence((2026072199, pass_index))
            )
        )
        permutation = generator.permutation(
            train_manifest.window_count
        ).astype("<u8", copy=False)
        payload = permutation.tobytes(order="C")
        payload_sha = hashlib.sha256(payload).hexdigest()
        manifest = PermutationManifest.create(
            pass_index=pass_index,  # type: ignore[arg-type]
            numpy_version=np.__version__,
            window_manifest_sha256=(
                train_manifest.manifest_sha256
            ),
            payload_sha256=payload_sha,
        )
        relative = (
            f"production-permutations/pass-{pass_index}/"
            f"{payload_sha}.u64le"
        )
        _publish(
            backend,
            _owned_path(
                cache_root,
                relative,
                prepare_parents=True,
            ),
            payload,
        )
        ids = tuple(int(item) for item in permutation.tolist())
        train_schedule_hashes.append(
            _schedule_sha256(
                split="train",
                pass_index=pass_index,
                window_manifest_sha256=(
                    train_manifest.manifest_sha256
                ),
                permutation_manifest=manifest,
                window_ids=ids,
                batch_size=WT103_BATCH_SIZE,
            )
        )
        permutation_manifests.append(manifest)
        permutation_paths.append(relative)
    evaluation_hashes = tuple(
        _schedule_sha256(
            split=manifest.split,
            pass_index=0,
            window_manifest_sha256=manifest.manifest_sha256,
            permutation_manifest=None,
            window_ids=tuple(range(manifest.window_count)),
            batch_size=WT103_BATCH_SIZE,
        )
        for manifest in (validation_manifest, test_manifest)
    )
    pass_batch_count = math.ceil(
        train_manifest.window_count / WT103_BATCH_SIZE
    )
    boundaries = _validation_boundaries(pass_batch_count)
    cadence_sha = owned_sha256(
        "vfe4.wt103.production-cadence-checkpoints.v1",
        {
            "cadence": training.profile.cadence,
            "checkpoints": training.profile.checkpoints,
            "batch_size": WT103_BATCH_SIZE,
            "train_window_count": train_manifest.window_count,
            "pass_batch_count": pass_batch_count,
            "validation_boundary_batch_ordinals": boundaries,
            "checkpoint_roles": (
                "resume_only",
                "terminal_scoring",
            ),
        },
    )
    return ProductionScheduleSet.create(
        window_manifests=tuple(manifests),
        window_row_relative_paths=tuple(row_paths),
        permutation_manifests=tuple(permutation_manifests),
        permutation_relative_paths=tuple(permutation_paths),
        schedule_sha256s=tuple(
            (*train_schedule_hashes, *evaluation_hashes)
        ),
        validation_boundary_batch_ordinals=boundaries,
        checkpoint_roles=("resume_only", "terminal_scoring"),
        cadence_sha256=cadence_sha,
    )


def _primary_a5_flop_ledger(
    *,
    primary_parameters: WT103PrimaryParameterInventory,
    optimizer_steps: int,
) -> A5SemanticTrainingFlopLedger:
    """Freeze the actual phase/operator inventory without inventing FLOPs."""

    unresolved = {
        "recognition": (
            "a5_forward_primitive_ledger_unresolved:"
            "recognition_affine_softplus_tanh"
        ),
        "backsolve": (
            "a5_forward_primitive_ledger_unresolved:"
            "block_bidiagonal_backsolve"
        ),
        "decoder": (
            "a5_forward_primitive_ledger_unresolved:"
            "chunked_decoder_selected_log_softmax"
        ),
        "source": (
            "a5_forward_primitive_ledger_unresolved:"
            "parent_specific_source_banks"
        ),
        "source_kl": (
            "a5_forward_primitive_ledger_unresolved:"
            "source_posterior_kl"
        ),
        "transport": (
            "a5_forward_primitive_ledger_unresolved:"
            "matrix_exp_inverse_transport"
        ),
        "transition": (
            "a5_forward_primitive_ledger_unresolved:"
            "state_model_transition_gaussians"
        ),
        "entropy": (
            "a5_forward_primitive_ledger_unresolved:"
            "joint_recognition_entropy"
        ),
        "reduction": (
            "a5_forward_primitive_ledger_unresolved:"
            "complete_elbo_reductions"
        ),
        "recognition_backward": (
            "a5_backward_primitive_ledger_unresolved:"
            "recognition_active_leaves"
        ),
        "model_backward": (
            "a5_backward_primitive_ledger_unresolved:"
            "model_active_leaves"
        ),
        "scheduler": "a5_scheduler_scalar_flop_policy_unresolved",
    }

    def open_row(
        operator_id: str,
        phase: Literal[
            "objective_forward",
            "recognition_backward",
            "model_backward",
            "scheduler",
        ],
        invocations: int,
        obligation: str,
    ) -> A5SemanticFlopOperator:
        return A5SemanticFlopOperator(
            operator_id=operator_id,
            phase=phase,
            invocations_per_batch=invocations,
            arithmetic_flops_per_batch=None,
            unresolved_obligation=obligation,
        )

    operators = (
        open_row(
            "recognition_embeddings_affine_and_band_factors",
            "objective_forward",
            4,
            unresolved["recognition"],
        ),
        open_row(
            "block_bidiagonal_reparameterization_backsolve",
            "objective_forward",
            4,
            unresolved["backsolve"],
        ),
        open_row(
            "chunked_decoder_affine_and_selected_log_softmax",
            "objective_forward",
            4,
            unresolved["decoder"],
        ),
        open_row(
            "parent_specific_state_and_model_source_banks",
            "objective_forward",
            4,
            unresolved["source"],
        ),
        open_row(
            "state_and_model_source_posterior_kl",
            "objective_forward",
            4,
            unresolved["source_kl"],
        ),
        open_row(
            "frame_matrix_exp_inverse_and_transport",
            "objective_forward",
            4,
            unresolved["transport"],
        ),
        open_row(
            "state_and_model_transition_gaussian_terms",
            "objective_forward",
            4,
            unresolved["transition"],
        ),
        open_row(
            "joint_block_recognition_entropy",
            "objective_forward",
            4,
            unresolved["entropy"],
        ),
        open_row(
            "complete_elbo_q_weighted_reductions",
            "objective_forward",
            4,
            unresolved["reduction"],
        ),
        open_row(
            "recognition_active_leaf_backward",
            "recognition_backward",
            1,
            unresolved["recognition_backward"],
        ),
        open_row(
            "model_active_leaf_backward",
            "model_backward",
            1,
            unresolved["model_backward"],
        ),
        A5SemanticFlopOperator(
            operator_id="recognition_global_l2_clip_and_scale",
            phase="gradient_control",
            invocations_per_batch=1,
            arithmetic_flops_per_batch=(
                3 * primary_parameters.recognition_parameter_count + 3
            ),
            unresolved_obligation=None,
        ),
        A5SemanticFlopOperator(
            operator_id="model_global_l2_clip_and_scale",
            phase="gradient_control",
            invocations_per_batch=1,
            arithmetic_flops_per_batch=(
                3 * primary_parameters.model_parameter_count + 3
            ),
            unresolved_obligation=None,
        ),
        A5SemanticFlopOperator(
            operator_id="recognition_adamw",
            phase="adamw",
            invocations_per_batch=1,
            arithmetic_flops_per_batch=(
                15 * primary_parameters.recognition_parameter_count
            ),
            unresolved_obligation=None,
        ),
        A5SemanticFlopOperator(
            operator_id="model_adamw",
            phase="adamw",
            invocations_per_batch=1,
            arithmetic_flops_per_batch=(
                15 * primary_parameters.model_parameter_count
            ),
            unresolved_obligation=None,
        ),
        open_row(
            "recognition_and_model_warmup_cosine_scheduler",
            "scheduler",
            2,
            unresolved["scheduler"],
        ),
    )
    return A5SemanticTrainingFlopLedger.create(
        primary_parameters=primary_parameters,
        optimizer_steps=optimizer_steps,
        operators=operators,
    )


def _resolved_a0_records(
    training: TrainingConfig,
    schedules: ProductionScheduleSet,
    dependencies: ProductionSourceLockDependencies,
) -> tuple[
    A0ArchitectureProfile,
    A0FormulaRecord,
    A0FlopLedger,
    A0SourceLockMatchingAssessment,
]:
    profile = training.profile
    train_windows = schedules.window_manifests[0].window_count
    optimizer_steps = (
        profile.cadence.confirmatory_passes
        * math.ceil(train_windows / profile.batch_size)
    )
    primary_parameters = reconstruct_wt103_primary_parameters(
        training,
        device=torch.device("meta"),
        dtype=torch.float32,
    )
    primary_flop_ledger = _primary_a5_flop_ledger(
        primary_parameters=primary_parameters,
        optimizer_steps=optimizer_steps,
    )
    meta_model = WT103A0Model(
        vocabulary_size=profile.vocabulary_size,
        positional_capacity=profile.sequence_length,
        hidden_width=training.a0_architecture.hidden_width,
        attention_heads=training.a0_architecture.attention_heads,
        layer_norm_epsilon=1.0e-5,
        device=torch.device("meta"),
        dtype=torch.float32,
    )
    inventory = reconstruct_a0_parameters(
        meta_model,
        vocabulary_size=profile.vocabulary_size,
        positional_capacity=profile.sequence_length,
        hidden_width=training.a0_architecture.hidden_width,
    )
    workload = A0FlopWorkload(
        batch_size=profile.batch_size,
        sequence_length=profile.sequence_length,
        vocabulary_size=profile.vocabulary_size,
        hidden_width=training.a0_architecture.hidden_width,
        parameter_count=inventory.parameter_count,
        decoder_chunk_size=profile.decoder_train_token_chunk,
        optimizer_steps=optimizer_steps,
        # Capacity matching is train-only. Validation/SMC/checkpoint work is
        # separately reported and cannot alter the frozen capacity decision.
        validation_batches=0,
    )
    ledger = reconstruct_a0_flops(workload)
    formula = build_a0_formula_record(
        inventory=inventory,
        ledger=ledger,
    )
    rows = tuple(
        A0ParameterMatchRow(
            hidden_width=hidden_width,
            parameter_count=(
                2 * profile.vocabulary_size * hidden_width
                + profile.sequence_length * hidden_width
                + 12 * hidden_width**2
                + 15 * hidden_width
                + profile.vocabulary_size
            ),
            parameter_ratio=(
                (
                    2 * profile.vocabulary_size * hidden_width
                    + profile.sequence_length * hidden_width
                    + 12 * hidden_width**2
                    + 15 * hidden_width
                    + profile.vocabulary_size
                )
                / primary_parameters.parameter_count
            ),
            parameter_relative_error=abs(
                (
                    (
                        2 * profile.vocabulary_size * hidden_width
                        + profile.sequence_length * hidden_width
                        + 12 * hidden_width**2
                        + 15 * hidden_width
                        + profile.vocabulary_size
                    )
                    / primary_parameters.parameter_count
                )
                - 1.0
            ),
            parameter_eligible=(
                abs(
                    (
                        (
                            2 * profile.vocabulary_size * hidden_width
                            + profile.sequence_length * hidden_width
                            + 12 * hidden_width**2
                            + 15 * hidden_width
                            + profile.vocabulary_size
                        )
                        / primary_parameters.parameter_count
                    )
                    - 1.0
                )
                <= 0.01
            ),
        )
        for hidden_width in WT103_A0_HIDDEN_WIDTH_CANDIDATES
    )
    if any(row.parameter_eligible for row in rows):
        raise ProductionOperationError(
            "A0 parameter-gate result changed; a complete PRIMARY FLOP "
            "ledger and closed finite match are required"
        )
    matching = A0SourceLockMatchingAssessment.create(
        endpoint_inventory_sha256=(
            training.endpoint_inventory.endpoint_inventory_sha256
        ),
        primary_parameters=primary_parameters,
        primary_flop_ledger=primary_flop_ledger,
        rows=rows,
    )
    architecture = build_a0_architecture_profile(
        hidden_width=training.a0_architecture.hidden_width,
        formula=formula,
        source_lock_scope="production_source_lock_verified",
        pytorch_version=dependencies.pytorch_version,
        sdpa_api_sha256=dependencies.sdpa_api_sha256,
        flash_backend_sha256=dependencies.flash_backend_sha256,
    )
    dynamic = {
        "schema_version",
        "source_lock_scope",
        "pytorch_version",
        "sdpa_api_sha256",
        "flash_backend_sha256",
        "formula_sha256",
        "architecture_sha256",
    }
    for descriptor in fields(A0ArchitectureProfile):
        if (
            descriptor.name not in dynamic
            and getattr(architecture, descriptor.name)
            != getattr(training.a0_architecture, descriptor.name)
        ):
            raise ProductionOperationError(
                "resolved A0 architecture changed a preregistered field: "
                f"{descriptor.name}"
            )
    return architecture, formula, ledger, matching


def _redirect_hops(
    *,
    request_url: str,
    redirects: tuple[HttpRedirectObservation, ...],
) -> tuple[RedirectHop, ...]:
    current = request_url
    result: list[RedirectHop] = []
    for item in redirects:
        result.append(
            RedirectHop(
                request_url=current,
                response_url=item.resolved_url,
                status_code=item.status_code,
            )
        )
        current = item.resolved_url
    return tuple(result)


def _finalized_source_record(
    *,
    staged: StagedWikiText103AcquisitionRecord,
    recorder: _RecordingHttpClient,
    tokenizer: ProductionTokenizerSpec,
    adapter: ProductionTokenizerAdapter,
    caches: tuple[
        ProductionTokenCacheRecord,
        ProductionTokenCacheRecord,
        ProductionTokenCacheRecord,
    ],
    schedules: ProductionScheduleSet,
    dependency_lock_sha256: str,
) -> FinalizedWikiText103SourceRecord:
    archive_http = recorder.observations.get(
        WIKITEXT103_ARCHIVE_REQUEST_URL
    )
    source_http = recorder.observations.get(
        WIKITEXT103_SOURCE_PAGE_REQUEST_URL
    )
    if archive_http is None or source_http is None:
        raise ProductionOperationError(
            "source-lock HTTP observations are incomplete"
        )
    source_content_type = _content_type(source_http)
    if source_content_type is None:
        raise ProductionOperationError(
            "source page lacks an exact content type"
        )
    validator_sha = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    return FinalizedWikiText103SourceRecord.create(
        acquisition_observation_sha256=(
            staged.observation.observation_sha256
        ),
        archive_request_url=staged.archive_request_url,
        archive_final_url=staged.archive_final_url,
        archive_redirect_chain=_redirect_hops(
            request_url=staged.archive_request_url,
            redirects=staged.archive_redirect_chain,
        ),
        source_page_request_url=staged.source_page_request_url,
        source_page_final_url=staged.source_page_final_url,
        source_page_redirect_chain=_redirect_hops(
            request_url=staged.source_page_request_url,
            redirects=staged.source_page_redirect_chain,
        ),
        archive_size_bytes=staged.archive_size_bytes,
        archive_sha256=staged.archive_sha256,
        archive_content_type=_content_type(archive_http),
        central_directory_sha256=(
            staged.observation.central_directory_sha256
        ),
        members=tuple(item.identity for item in staged.members),
        source_page_size_bytes=staged.source_page_size_bytes,
        source_page_content_type=source_content_type,
        source_page_sha256=staged.source_page_sha256,
        license_paragraph_start_byte=(
            staged.license.paragraph_start_offset
        ),
        license_paragraph_end_byte=(
            staged.license.paragraph_end_offset
        ),
        license_raw_slice_sha256=staged.license.raw_slice_sha256,
        license_declaration=staged.license.visible_text,
        license_hrefs=staged.license.hrefs,
        installed_distribution_sha256=(
            adapter.distribution_record_sha256
        ),
        tokenizer_tables_sha256=tokenizer.tokenizer_tables_sha256,
        production_tokenizer_spec_sha256=tokenizer.spec_sha256,
        production_token_cache_set_sha256=(
            production_token_cache_set_sha256(
                tuple(item.cache_identity for item in caches)
            )
        ),
        schedule_set_sha256=schedules.schedule_set_sha256,
        dependency_lock_sha256=dependency_lock_sha256,
        validator_sha256=validator_sha,
    )


def run_source_lock(
    *,
    training: TrainingConfig,
    paths: object,
    dependencies: ProductionSourceLockDependencies | None = None,
) -> ProductionSourceLock:
    """Serialize one source-lock transaction before any marker inspection."""

    if type(training) is not TrainingConfig or training.operation != "source_lock":
        raise ProductionOperationError(
            "source-lock requires an exact source_lock TrainingConfig"
        )
    cache_root = getattr(paths, "cache_root", None)
    source_record_path = getattr(paths, "source_record_path", None)
    if (
        not isinstance(cache_root, Path)
        or not cache_root.is_absolute()
        or not isinstance(source_record_path, Path)
        or not source_record_path.is_absolute()
    ):
        raise ProductionOperationError(
            "source-lock paths are not exact absolute Paths"
        )
    if dependencies is None:
        repository_root = Path(__file__).resolve().parents[2]
    elif type(dependencies) is ProductionSourceLockDependencies:
        repository_root = dependencies.repository_root
    else:
        raise ProductionOperationError(
            "source-lock dependencies must be exact"
        )
    if (
        not isinstance(repository_root, Path)
        or not repository_root.is_absolute()
    ):
        raise ProductionOperationError(
            "source-lock repository root must be an exact absolute Path"
        )
    leases = _acquire_source_lock_execution_leases(
        (repository_root, cache_root, source_record_path)
    )
    try:
        return _run_source_lock_under_lease(
            training=training,
            paths=paths,
            dependencies=dependencies,
        )
    finally:
        for lease in reversed(leases):
            lease.release()


def _run_source_lock_under_lease(
    *,
    training: TrainingConfig,
    paths: object,
    dependencies: ProductionSourceLockDependencies | None = None,
) -> ProductionSourceLock:
    """Acquire, derive, close, and publish the production source lock."""

    if type(training) is not TrainingConfig or training.operation != "source_lock":
        raise ProductionOperationError(
            "source-lock requires an exact source_lock TrainingConfig"
        )
    cache_root = getattr(paths, "cache_root", None)
    source_record_path = getattr(paths, "source_record_path", None)
    if (
        not isinstance(cache_root, Path)
        or not cache_root.is_absolute()
        or not isinstance(source_record_path, Path)
        or not source_record_path.is_absolute()
    ):
        raise ProductionOperationError(
            "source-lock paths are not exact absolute Paths"
        )
    try:
        source_record_path.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise ProductionOperationError(
            "finalized source marker metadata is unavailable: "
            f"{source_record_path}: {exc}"
        ) from exc
    else:
        if dependencies is None:
            repository_root = Path(__file__).resolve().parents[2]
        elif type(dependencies) is ProductionSourceLockDependencies:
            repository_root = dependencies.repository_root
        else:
            raise ProductionOperationError(
                "source-lock dependencies must be exact"
            )
        return _reopen_source_lock(
            training=training,
            paths=paths,
            repository_root=repository_root,
        )
    exact_dependencies = (
        _live_source_lock_dependencies()
        if dependencies is None
        else dependencies
    )
    if type(exact_dependencies) is not ProductionSourceLockDependencies:
        raise ProductionOperationError(
            "source-lock dependencies must be exact"
        )
    exact_dependencies.__post_init__()
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_probe = exact_dependencies.durability_backend.probe(cache_root)
    repo_probe = exact_dependencies.durability_backend.probe(
        exact_dependencies.repository_root
    )
    if cache_probe.status != "pass" or repo_probe.status != "pass":
        raise ProductionOperationError(
            "source-lock durability probes did not pass"
        )
    staging_root = cache_root / "production-source-lock" / "staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    recorder = _RecordingHttpClient(exact_dependencies.http_client)
    staged = stage_wikitext103_acquisition_record(
        StagedAcquisitionRequest(
            archive_request_url=WIKITEXT103_ARCHIVE_REQUEST_URL,
            source_page_request_url=WIKITEXT103_SOURCE_PAGE_REQUEST_URL,
            staging_root=staging_root,
            allow_network=True,
        ),
        http_client=recorder,
        durability_backend=exact_dependencies.durability_backend,
    )
    reopen_staged_wikitext103(
        observation=staged,
        staging_root=staging_root,
    )
    lock_bytes, lock_sha = _resolve_dependency_lock(
        exact_dependencies
    )
    if hashlib.sha256(lock_bytes).hexdigest() != lock_sha:
        raise ProductionOperationError(
            "resolved dependency lock failed immediate reopen"
        )
    tokenizer = _production_tokenizer_spec(
        exact_dependencies.tokenizer
    )
    caches = tuple(
        _build_cache_record(
            split=split,
            staged=staged,
            staging_root=staging_root,
            cache_root=cache_root,
            tokenizer=tokenizer,
            adapter=exact_dependencies.tokenizer,
            backend=exact_dependencies.durability_backend,
        )
        for split in ("train", "validation", "test")
    )
    schedules = _build_schedule_set(
        caches=caches,
        cache_root=cache_root,
        backend=exact_dependencies.durability_backend,
        training=training,
    )
    (
        a0_architecture,
        a0_formula,
        a0_flop_ledger,
        a0_matching,
    ) = _resolved_a0_records(
        training,
        schedules,
        exact_dependencies,
    )
    finalized = _finalized_source_record(
        staged=staged,
        recorder=recorder,
        tokenizer=tokenizer,
        adapter=exact_dependencies.tokenizer,
        caches=caches,
        schedules=schedules,
        dependency_lock_sha256=lock_sha,
    )
    source_lock = ProductionSourceLock.create(
        finalized_source=finalized,
        tokenizer=tokenizer,
        token_caches=caches,
        schedules=schedules,
        a0_architecture=a0_architecture,
        a0_formula=a0_formula,
        a0_flop_ledger=a0_flop_ledger,
        a0_matching=a0_matching,
    )
    bundle_payload = canonical_json_bytes_generic(source_lock) + b"\n"
    bundle_path = _owned_path(
        cache_root,
        _SOURCE_BUNDLE_RELATIVE_PATH,
        prepare_parents=True,
    )
    _publish(
        exact_dependencies.durability_backend,
        bundle_path,
        bundle_payload,
    )
    source_record_path.parent.mkdir(parents=True, exist_ok=True)
    source_payload = canonical_json_bytes_generic(finalized) + b"\n"
    # The finalized tracked record is the transaction's final publication.
    _publish(
        exact_dependencies.durability_backend,
        source_record_path,
        source_payload,
    )
    reopened = _reopen_source_lock(
        training=training,
        paths=paths,
        repository_root=exact_dependencies.repository_root,
    )
    if reopened != source_lock:
        raise ProductionOperationError(
            "published source lock differs after durable reopen"
        )
    return source_lock


def _json_mapping(value: object, name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ProductionOperationError(f"{name} must be a JSON object")
    return value


def _json_tuple(value: object, name: str) -> tuple[object, ...]:
    if type(value) is not list:
        raise ProductionOperationError(f"{name} must be a JSON array")
    return tuple(value)


def _parse_tokenizer(value: object) -> ProductionTokenizerSpec:
    return ProductionTokenizerSpec(
        **_json_mapping(value, "production tokenizer")
    )


def _parse_cache_record(value: object) -> ProductionTokenCacheRecord:
    raw = dict(_json_mapping(value, "production token-cache record"))
    tokenizer = _parse_tokenizer(raw["tokenizer"])
    identity_raw = dict(
        _json_mapping(raw["cache_identity"], "token-cache identity")
    )
    identity_raw["tokenizer"] = _parse_tokenizer(
        identity_raw["tokenizer"]
    )
    raw["tokenizer"] = tokenizer
    raw["cache_identity"] = ProductionTokenCacheIdentity(**identity_raw)
    return ProductionTokenCacheRecord(**raw)


def _parse_window_manifest(value: object) -> WindowManifest:
    return WindowManifest(
        **_json_mapping(value, "window manifest")
    )


def _parse_permutation_manifest(value: object) -> PermutationManifest:
    return PermutationManifest(
        **_json_mapping(value, "permutation manifest")
    )


def _parse_schedules(value: object) -> ProductionScheduleSet:
    raw = dict(_json_mapping(value, "production schedules"))
    raw["window_manifests"] = tuple(
        _parse_window_manifest(item)
        for item in _json_tuple(
            raw["window_manifests"],
            "window manifests",
        )
    )
    raw["window_row_relative_paths"] = _json_tuple(
        raw["window_row_relative_paths"],
        "window row paths",
    )
    raw["permutation_manifests"] = tuple(
        _parse_permutation_manifest(item)
        for item in _json_tuple(
            raw["permutation_manifests"],
            "permutation manifests",
        )
    )
    raw["permutation_relative_paths"] = _json_tuple(
        raw["permutation_relative_paths"],
        "permutation paths",
    )
    raw["schedule_sha256s"] = _json_tuple(
        raw["schedule_sha256s"],
        "schedule hashes",
    )
    raw["validation_boundary_batch_ordinals"] = _json_tuple(
        raw["validation_boundary_batch_ordinals"],
        "validation boundaries",
    )
    raw["checkpoint_roles"] = _json_tuple(
        raw["checkpoint_roles"],
        "checkpoint roles",
    )
    return ProductionScheduleSet(**raw)


def _parse_source_record(value: object) -> FinalizedWikiText103SourceRecord:
    raw = dict(_json_mapping(value, "finalized source record"))
    for name in (
        "archive_redirect_chain",
        "source_page_redirect_chain",
    ):
        raw[name] = tuple(
            RedirectHop(**_json_mapping(item, name))
            for item in _json_tuple(raw[name], name)
        )
    raw["members"] = tuple(
        ArchiveMemberIdentity(**_json_mapping(item, "archive member"))
        for item in _json_tuple(raw["members"], "archive members")
    )
    raw["license_hrefs"] = _json_tuple(
        raw["license_hrefs"],
        "license hrefs",
    )
    return FinalizedWikiText103SourceRecord(**raw)


def _parse_a0(value: object) -> A0ArchitectureProfile:
    raw = dict(_json_mapping(value, "A0 architecture"))
    raw["enabled_backends"] = _json_tuple(
        raw["enabled_backends"],
        "A0 enabled backends",
    )
    raw["candidate_hidden_widths"] = _json_tuple(
        raw["candidate_hidden_widths"],
        "A0 hidden-width candidates",
    )
    return A0ArchitectureProfile(**raw)


def _parse_a0_formula(value: object) -> A0FormulaRecord:
    return A0FormulaRecord(
        **_json_mapping(value, "A0 formula")
    )


def _parse_a0_flop_ledger(value: object) -> A0FlopLedger:
    raw = dict(_json_mapping(value, "A0 FLOP ledger"))
    raw["terms"] = tuple(
        A0FlopTerm(**_json_mapping(item, "A0 FLOP term"))
        for item in _json_tuple(raw["terms"], "A0 FLOP terms")
    )
    return A0FlopLedger(**raw)


def _parse_primary_parameters(
    value: object,
) -> WT103PrimaryParameterInventory:
    raw = dict(_json_mapping(value, "PRIMARY parameter inventory"))
    raw["rows"] = tuple(
        WT103PrimaryParameterRow(
            **{
                **_json_mapping(item, "PRIMARY parameter row"),
                "shape": _json_tuple(
                    _json_mapping(item, "PRIMARY parameter row")["shape"],
                    "PRIMARY parameter shape",
                ),
            }
        )
        for item in _json_tuple(raw["rows"], "PRIMARY parameter rows")
    )
    return WT103PrimaryParameterInventory(**raw)


def _parse_primary_flop_ledger(
    value: object,
) -> A5SemanticTrainingFlopLedger:
    raw = dict(_json_mapping(value, "PRIMARY A5 FLOP ledger"))
    raw["primary_parameters"] = _parse_primary_parameters(
        raw["primary_parameters"]
    )
    raw["operators"] = tuple(
        A5SemanticFlopOperator(
            **_json_mapping(item, "PRIMARY A5 FLOP operator")
        )
        for item in _json_tuple(
            raw["operators"],
            "PRIMARY A5 FLOP operators",
        )
    )
    raw["status"] = GateStatus(raw["status"])
    raw["obligations"] = _json_tuple(
        raw["obligations"],
        "PRIMARY A5 FLOP obligations",
    )
    return A5SemanticTrainingFlopLedger(**raw)


def _parse_a0_matching(
    value: object,
) -> ArmMatchingReport | A0SourceLockMatchingAssessment:
    raw = dict(_json_mapping(value, "A0 matching"))
    if (
        raw.get("schema_version")
        == "wt103-a0-source-lock-matching-assessment-v1"
    ):
        raw["primary_parameters"] = _parse_primary_parameters(
            raw["primary_parameters"]
        )
        raw["primary_flop_ledger"] = _parse_primary_flop_ledger(
            raw["primary_flop_ledger"]
        )
        raw["candidate_hidden_widths"] = _json_tuple(
            raw["candidate_hidden_widths"],
            "A0 matching widths",
        )
        raw["rows"] = tuple(
            A0ParameterMatchRow(
                **_json_mapping(item, "A0 parameter-match row")
            )
            for item in _json_tuple(raw["rows"], "A0 matching rows")
        )
        raw["status"] = GateStatus(raw["status"])
        raw["obligations"] = _json_tuple(
            raw["obligations"],
            "A0 matching obligations",
        )
        return A0SourceLockMatchingAssessment(**raw)
    rows: list[A0MatchRow] = []
    for item in _json_tuple(raw["rows"], "A0 matching rows"):
        row = dict(_json_mapping(item, "A0 matching row"))
        row["selection_key"] = _json_tuple(
            row["selection_key"],
            "A0 matching selection key",
        )
        rows.append(A0MatchRow(**row))
    raw["candidate_hidden_widths"] = _json_tuple(
        raw["candidate_hidden_widths"],
        "A0 matching widths",
    )
    raw["rows"] = tuple(rows)
    raw["status"] = GateStatus(raw["status"])
    raw["obligations"] = _json_tuple(
        raw["obligations"],
        "A0 matching obligations",
    )
    return ArmMatchingReport(**raw)


def _parse_source_lock(payload: bytes) -> ProductionSourceLock:
    try:
        document = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionOperationError(
            "production source-lock bundle JSON is invalid"
        ) from exc
    raw = dict(_json_mapping(document, "production source-lock bundle"))
    raw["finalized_source"] = _parse_source_record(
        raw["finalized_source"]
    )
    raw["tokenizer"] = _parse_tokenizer(raw["tokenizer"])
    raw["token_caches"] = tuple(
        _parse_cache_record(item)
        for item in _json_tuple(raw["token_caches"], "token caches")
    )
    raw["schedules"] = _parse_schedules(raw["schedules"])
    raw["a0_architecture"] = _parse_a0(raw["a0_architecture"])
    raw["a0_formula"] = _parse_a0_formula(raw["a0_formula"])
    raw["a0_flop_ledger"] = _parse_a0_flop_ledger(
        raw["a0_flop_ledger"]
    )
    raw["a0_matching"] = _parse_a0_matching(raw["a0_matching"])
    result = ProductionSourceLock(**raw)
    if payload != canonical_json_bytes_generic(result) + b"\n":
        raise ProductionOperationError(
            "production source-lock bundle is not canonical JSONL"
        )
    return result


def _revalidate_schedule_payloads(
    source_lock: ProductionSourceLock,
    *,
    cache_root: Path,
) -> None:
    schedules = source_lock.schedules
    for manifest, relative in zip(
        schedules.window_manifests,
        schedules.window_row_relative_paths,
        strict=True,
    ):
        payload = _regular_nonlink_bytes(
            _owned_path(cache_root, relative),
            expected_sha256=manifest.payload_sha256,
        )
        if not payload.startswith(_WINDOW_ROWS_DOMAIN):
            raise ProductionOperationError(
                "window-row payload domain changed"
            )
    observed_schedule_hashes: list[str] = []
    train_manifest = schedules.window_manifests[0]
    for manifest, relative in zip(
        schedules.permutation_manifests,
        schedules.permutation_relative_paths,
        strict=True,
    ):
        payload = _regular_nonlink_bytes(
            _owned_path(cache_root, relative),
            expected_size=8 * train_manifest.window_count,
            expected_sha256=manifest.payload_sha256,
        )
        permutation = np.frombuffer(payload, dtype=np.dtype("<u8"))
        ids = tuple(int(item) for item in permutation.tolist())
        if sorted(ids) != list(range(train_manifest.window_count)):
            raise ProductionOperationError(
                "production permutation is not complete and unique"
            )
        observed_schedule_hashes.append(
            _schedule_sha256(
                split="train",
                pass_index=manifest.pass_index,
                window_manifest_sha256=(
                    train_manifest.manifest_sha256
                ),
                permutation_manifest=manifest,
                window_ids=ids,
                batch_size=WT103_BATCH_SIZE,
            )
        )
    for manifest in schedules.window_manifests[1:]:
        observed_schedule_hashes.append(
            _schedule_sha256(
                split=manifest.split,
                pass_index=0,
                window_manifest_sha256=manifest.manifest_sha256,
                permutation_manifest=None,
                window_ids=tuple(range(manifest.window_count)),
                batch_size=WT103_BATCH_SIZE,
            )
        )
    if tuple(observed_schedule_hashes) != schedules.schedule_sha256s:
        raise ProductionOperationError(
            "production schedule identities changed on reopen"
        )


def _reopen_source_lock(
    *,
    training: TrainingConfig,
    paths: object,
    repository_root: Path,
) -> ProductionSourceLock:
    if type(training) is not TrainingConfig:
        raise ProductionOperationError(
            "source-lock reopen requires exact TrainingConfig"
        )
    cache_root = getattr(paths, "cache_root", None)
    source_record_path = getattr(paths, "source_record_path", None)
    if not isinstance(cache_root, Path) or not isinstance(
        source_record_path,
        Path,
    ):
        raise ProductionOperationError("source-lock paths are malformed")
    bundle_path = _owned_path(
        cache_root,
        _SOURCE_BUNDLE_RELATIVE_PATH,
    )
    source_lock = _parse_source_lock(
        _regular_nonlink_bytes(bundle_path)
    )
    source_payload = _regular_nonlink_bytes(source_record_path)
    if (
        source_payload
        != canonical_json_bytes_generic(source_lock.finalized_source)
        + b"\n"
    ):
        raise ProductionOperationError(
            "tracked finalized source differs from its frozen bundle"
        )
    if (
        source_lock.finalized_source.validator_sha256
        != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    ):
        raise ProductionOperationError(
            "source-lock validator revision changed"
        )
    if (
        not isinstance(repository_root, Path)
        or not repository_root.is_absolute()
    ):
        raise ProductionOperationError(
            "source-lock repository root is malformed"
        )
    lock_path = repository_root / _LOCK_NAME
    lock_bytes = _regular_nonlink_bytes(
        lock_path,
        expected_sha256=(
            source_lock.finalized_source.dependency_lock_sha256
        ),
    )
    if not lock_bytes:
        raise ProductionOperationError("dependency lock is empty")
    staging_root = cache_root / "production-source-lock" / "staging"
    for member in source_lock.finalized_source.members:
        _validate_corpus_file(
            staging_root
            / "staged"
            / "splits"
            / member.split
            / f"{member.payload_sha256}.raw",
            expected_size=member.uncompressed_size_bytes,
            expected_sha256=member.payload_sha256,
        )
    for cache in source_lock.token_caches:
        _validate_corpus_file(
            _owned_path(cache_root, cache.cache_relative_path),
            expected_size=cache.payload_size_bytes,
            expected_sha256=cache.payload_sha256,
        )
    _revalidate_schedule_payloads(
        source_lock,
        cache_root=cache_root,
    )
    return source_lock


def reopen_source_lock(
    *,
    training: TrainingConfig,
    paths: object,
) -> ProductionSourceLock:
    """Durably reopen every frozen artifact without mutating its roots."""

    return _reopen_source_lock(
        training=training,
        paths=paths,
        repository_root=Path(__file__).resolve().parents[2],
    )


@dataclass(frozen=True, slots=True)
class ProductionWindowSet:
    """Read-only, target-complete view of one finalized production split."""

    split: Literal["train", "validation"]
    cache_record: ProductionTokenCacheRecord
    manifest: WindowManifest
    rows: tuple[WT103WindowRow, ...]
    token_payload_path: Path
    tokens: np.memmap

    def __post_init__(self) -> None:
        if (
            self.split not in ("train", "validation")
            or type(self.cache_record) is not ProductionTokenCacheRecord
            or type(self.manifest) is not WindowManifest
            or type(self.rows) is not tuple
            or not self.rows
            or any(type(item) is not WT103WindowRow for item in self.rows)
            or tuple(item.window_id for item in self.rows)
            != tuple(range(len(self.rows)))
            or type(self.tokens) is not np.memmap
            or self.tokens.dtype != np.dtype("<i4")
            or tuple(self.tokens.shape)
            != (self.cache_record.token_count,)
        ):
            raise ProductionOperationError(
                "production window set is malformed"
            )
        self.cache_record.__post_init__()
        self.manifest.__post_init__()
        for row in self.rows:
            row.__post_init__()
        if (
            self.cache_record.split != self.split
            or self.manifest.split != self.split
            or self.manifest.token_payload_sha256
            != self.cache_record.payload_sha256
            or self.manifest.window_count != len(self.rows)
            or self.manifest.counted_targets
            != sum(row.counted_targets for row in self.rows)
            or self.manifest.counted_targets
            != self.cache_record.token_count - 1
        ):
            raise ProductionOperationError(
                "production windows do not bind the source lock"
            )

    def window(self, window_id: int) -> CausalWindow:
        self.__post_init__()
        if type(window_id) is not int or not 0 <= window_id < len(self.rows):
            raise IndexError("production window ID is outside the split")
        row = self.rows[window_id]
        start = row.start_transition
        count = row.counted_targets
        inputs = torch.full(
            (WT103_SEQUENCE_LENGTH,),
            WT103_EOT_TOKEN_ID,
            dtype=torch.int64,
            device="cpu",
        )
        targets = torch.full(
            (WT103_SEQUENCE_LENGTH,),
            WT103_IGNORE_TARGET_ID,
            dtype=torch.int64,
            device="cpu",
        )
        attention_mask = torch.zeros(
            (WT103_SEQUENCE_LENGTH,),
            dtype=torch.bool,
            device="cpu",
        )
        input_values = np.asarray(
            self.tokens[start : start + count],
            dtype=np.int64,
        ).copy()
        target_values = np.asarray(
            self.tokens[start + 1 : start + count + 1],
            dtype=np.int64,
        ).copy()
        inputs[:count] = torch.from_numpy(input_values)
        targets[:count] = torch.from_numpy(target_values)
        attention_mask[:count] = True
        return CausalWindow(
            window_id=window_id,
            start_transition=start,
            inputs=inputs,
            targets=targets,
            attention_mask=attention_mask,
            counted_targets=count,
        )


def _parse_window_rows(payload: bytes) -> tuple[WT103WindowRow, ...]:
    if (
        not payload.startswith(_WINDOW_ROWS_DOMAIN)
        or (len(payload) - len(_WINDOW_ROWS_DOMAIN)) % 20
    ):
        raise ProductionOperationError(
            "production window-row payload is malformed"
        )
    body = payload[len(_WINDOW_ROWS_DOMAIN) :]
    rows = tuple(
        WT103WindowRow(
            window_id=int.from_bytes(body[offset : offset + 8], "little"),
            start_transition=int.from_bytes(
                body[offset + 8 : offset + 16],
                "little",
            ),
            counted_targets=int.from_bytes(
                body[offset + 16 : offset + 20],
                "little",
            ),
        )
        for offset in range(0, len(body), 20)
    )
    if not rows:
        raise ProductionOperationError(
            "production window-row payload is empty"
        )
    return rows


def open_production_training_split(
    *,
    source_lock: ProductionSourceLock,
    cache_root: Path,
    split: Literal["train", "validation"],
    pass_index: Literal[0, 1] = 0,
) -> tuple[ProductionWindowSet, WindowSchedule]:
    """Open train/validation only; the held-out test cache is unreachable."""

    if type(source_lock) is not ProductionSourceLock:
        raise ProductionOperationError(
            "production split requires an exact source lock"
        )
    source_lock.__post_init__()
    if (
        split not in ("train", "validation")
        or type(pass_index) is not int
        or pass_index not in (0, 1)
        or (split == "validation" and pass_index != 0)
    ):
        raise ProductionOperationError(
            "production training split request is invalid"
        )
    cache = next(
        item for item in source_lock.token_caches if item.split == split
    )
    manifest_index = 0 if split == "train" else 1
    manifest = source_lock.schedules.window_manifests[manifest_index]
    row_relative = source_lock.schedules.window_row_relative_paths[
        manifest_index
    ]
    row_payload = _regular_nonlink_bytes(
        _owned_path(cache_root, row_relative),
        expected_sha256=manifest.payload_sha256,
    )
    rows = _parse_window_rows(row_payload)
    token_path = _owned_path(cache_root, cache.cache_relative_path)
    _validate_corpus_file(
        token_path,
        expected_size=cache.payload_size_bytes,
        expected_sha256=cache.payload_sha256,
    )
    tokens = np.memmap(
        token_path,
        mode="r",
        dtype=np.dtype("<i4"),
        shape=(cache.token_count,),
    )
    windows = ProductionWindowSet(
        split=split,
        cache_record=cache,
        manifest=manifest,
        rows=rows,
        token_payload_path=token_path,
        tokens=tokens,
    )
    permutation: PermutationManifest | None
    if split == "train":
        permutation = source_lock.schedules.permutation_manifests[
            pass_index
        ]
        relative = source_lock.schedules.permutation_relative_paths[
            pass_index
        ]
        payload = _regular_nonlink_bytes(
            _owned_path(cache_root, relative),
            expected_size=8 * manifest.window_count,
            expected_sha256=permutation.payload_sha256,
        )
        values = np.frombuffer(payload, dtype=np.dtype("<u8"))
        window_ids = tuple(int(item) for item in values.tolist())
        expected_schedule_sha = source_lock.schedules.schedule_sha256s[
            pass_index
        ]
    else:
        permutation = None
        window_ids = tuple(range(manifest.window_count))
        expected_schedule_sha = source_lock.schedules.schedule_sha256s[2]
    schedule = WindowSchedule(
        schema_version="wt103-window-schedule-v1",
        split=split,
        pass_index=pass_index,
        window_manifest_sha256=manifest.manifest_sha256,
        permutation_manifest=permutation,
        window_ids=window_ids,
        batch_size=WT103_BATCH_SIZE,
        schedule_sha256=expected_schedule_sha,
    )
    schedule.__post_init__()
    return windows, schedule


def production_cursor_after_batches(
    *,
    windows: ProductionWindowSet,
    schedule: WindowSchedule,
    completed_batch_count: int,
) -> DataCursor:
    windows.__post_init__()
    schedule.__post_init__()
    batches = tuple(
        schedule.window_ids[offset : offset + schedule.batch_size]
        for offset in range(
            0,
            len(schedule.window_ids),
            schedule.batch_size,
        )
    )
    if (
        type(completed_batch_count) is not int
        or not 0 <= completed_batch_count <= len(batches)
    ):
        raise ProductionOperationError(
            "completed production batch count is outside the schedule"
        )
    consumed = tuple(
        window_id
        for batch in batches[:completed_batch_count]
        for window_id in batch
    )
    binding = (
        schedule.schedule_sha256
        if schedule.permutation_manifest is None
        else schedule.permutation_manifest.manifest_sha256
    )
    return DataCursor.create(
        split=schedule.split,
        pass_index=schedule.pass_index,
        permutation_sha256=binding,
        next_batch_ordinal=completed_batch_count,
        next_window_ids=(
            batches[completed_batch_count]
            if completed_batch_count < len(batches)
            else ()
        ),
        counted_targets=sum(
            windows.rows[item].counted_targets for item in consumed
        ),
    )


def iter_production_batches(
    *,
    windows: ProductionWindowSet,
    schedule: WindowSchedule,
    cursor: DataCursor | None = None,
):
    """Yield exact CPU CausalBatch values from a frozen production schedule."""

    windows.__post_init__()
    schedule.__post_init__()
    batches = tuple(
        schedule.window_ids[offset : offset + schedule.batch_size]
        for offset in range(
            0,
            len(schedule.window_ids),
            schedule.batch_size,
        )
    )
    start = 0
    if cursor is not None:
        cursor.__post_init__()
        start = cursor.next_batch_ordinal
        if cursor != production_cursor_after_batches(
            windows=windows,
            schedule=schedule,
            completed_batch_count=start,
        ):
            raise ProductionOperationError(
                "production resume cursor differs from the frozen schedule"
            )
    for window_ids in batches[start:]:
        rows = tuple(windows.window(item) for item in window_ids)
        yield CausalBatch(
            window_ids=window_ids,
            inputs=torch.stack(
                tuple(row.inputs for row in rows),
                dim=0,
            ),
            targets=torch.stack(
                tuple(row.targets for row in rows),
                dim=0,
            ),
            attention_mask=torch.stack(
                tuple(row.attention_mask for row in rows),
                dim=0,
            ),
            counted_targets=sum(row.counted_targets for row in rows),
        )


@dataclass(frozen=True, slots=True)
class ProductionReadinessResult:
    schema_version: Literal["wt103-production-readiness-result-v1"]
    source_lock_sha256: str
    status: GateStatus
    obligations: tuple[str, ...]
    readiness_bundle: object | None
    readiness: object | None
    readiness_token: object | None
    result_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != "wt103-production-readiness-result-v1"
            or type(self.status) is not GateStatus
            or type(self.obligations) is not tuple
            or any(type(item) is not str or not item for item in self.obligations)
            or (
                self.status is GateStatus.PASS
                and (
                    self.obligations
                    or self.readiness_bundle is None
                    or self.readiness is None
                    or self.readiness_token is None
                )
            )
            or (
                self.status is not GateStatus.PASS
                and (
                    not self.obligations
                    or self.readiness_bundle is not None
                    or self.readiness_token is not None
                )
            )
        ):
            raise ProductionOperationError(
                "production readiness result is inconsistent"
            )
        _sha256(self.source_lock_sha256, "source_lock_sha256")
        expected = owned_sha256(
            "vfe4.wt103.production-readiness-result.v1",
            {
                "schema_version": self.schema_version,
                "source_lock_sha256": self.source_lock_sha256,
                "status": self.status,
                "obligations": self.obligations,
                "readiness_bundle_sha256": getattr(
                    self.readiness_bundle,
                    "bundle_sha256",
                    None,
                ),
                "readiness_assessment_sha256": getattr(
                    self.readiness,
                    "assessment_sha256",
                    None,
                ),
                "readiness_token_sha256": getattr(
                    self.readiness_token,
                    "token_sha256",
                    None,
                ),
            },
        )
        _sha256(self.result_sha256, "result_sha256")
        if self.result_sha256 != expected:
            raise ProductionOperationError(
                "production readiness result hash does not match"
            )

    @classmethod
    def create(
        cls,
        *,
        source_lock_sha256: str,
        status: GateStatus,
        obligations: tuple[str, ...],
        readiness_bundle: object | None,
        readiness: object | None,
        readiness_token: object | None,
    ) -> "ProductionReadinessResult":
        payload = {
            "schema_version": "wt103-production-readiness-result-v1",
            "source_lock_sha256": source_lock_sha256,
            "status": status,
            "obligations": obligations,
            "readiness_bundle_sha256": getattr(
                readiness_bundle,
                "bundle_sha256",
                None,
            ),
            "readiness_assessment_sha256": getattr(
                readiness,
                "assessment_sha256",
                None,
            ),
            "readiness_token_sha256": getattr(
                readiness_token,
                "token_sha256",
                None,
            ),
        }
        return cls(
            schema_version="wt103-production-readiness-result-v1",
            source_lock_sha256=source_lock_sha256,
            status=status,
            obligations=obligations,
            readiness_bundle=readiness_bundle,
            readiness=readiness,
            readiness_token=readiness_token,
            result_sha256=owned_sha256(
                "vfe4.wt103.production-readiness-result.v1",
                payload,
            ),
        )


def run_readiness(
    *,
    training: TrainingConfig,
    paths: object,
    source_lock: object,
) -> ProductionReadinessResult:
    """Reopen Task 14 evidence and issue no optimizer update."""

    if type(source_lock) is not ProductionSourceLock:
        raise ProductionOperationError(
            "readiness requires an exact ProductionSourceLock"
        )
    source_lock.__post_init__()
    if (
        type(source_lock.a0_matching) is not ArmMatchingReport
        or source_lock.a0_matching.status is not GateStatus.PASS
    ):
        return ProductionReadinessResult.create(
            source_lock_sha256=source_lock.source_lock_sha256,
            status=GateStatus.INCONCLUSIVE,
            obligations=tuple(
                f"capacity_matching:{item}"
                for item in source_lock.a0_matching.obligations
            ),
            readiness_bundle=None,
            readiness=None,
            readiness_token=None,
        )
    evidence_path = (
        getattr(paths, "run_root")
        / "readiness"
        / "task14-live-readiness-bundle.json"
    )
    if not evidence_path.is_file():
        return ProductionReadinessResult.create(
            source_lock_sha256=source_lock.source_lock_sha256,
            status=GateStatus.INCONCLUSIVE,
            obligations=(
                "task14_live_readiness_bundle_missing",
            ),
            readiness_bundle=None,
            readiness=None,
            readiness_token=None,
        )
    from vfe4.artifacts.live_readiness import (
        read_task14_readiness_bundle,
        reopen_and_issue_task14_readiness,
    )

    readiness_bundle = read_task14_readiness_bundle(evidence_path)
    readiness, token = reopen_and_issue_task14_readiness(
        path=evidence_path,
        training=training,
        source_lock=source_lock,
    )
    return ProductionReadinessResult.create(
        source_lock_sha256=source_lock.source_lock_sha256,
        status=GateStatus.PASS,
        obligations=(),
        readiness_bundle=readiness_bundle,
        readiness=readiness,
        readiness_token=token,
    )


def run_training(
    *,
    training: TrainingConfig,
    paths: object,
    source_lock: object,
    readiness: object,
    mode: Literal["train", "resume"],
) -> object:
    """Enter the concrete production attempt orchestrator after PASS only."""

    if (
        type(source_lock) is not ProductionSourceLock
        or type(readiness) is not ProductionReadinessResult
        or readiness.status is not GateStatus.PASS
        or readiness.readiness_token is None
    ):
        raise ProductionOperationError(
            "production training requires exact unchanged PASS readiness"
        )
    from vfe4.training.production_attempt import run_production_attempts

    return run_production_attempts(
        training=training,
        paths=paths,
        source_lock=source_lock,
        readiness=readiness,
        mode=mode,
    )


__all__ = [
    "A0ParameterMatchRow",
    "A0SourceLockMatchingAssessment",
    "A5SemanticFlopOperator",
    "A5SemanticTrainingFlopLedger",
    "ProductionOperationError",
    "ProductionReadinessResult",
    "ProductionScheduleSet",
    "ProductionSourceLock",
    "ProductionSourceLockDependencies",
    "ProductionTokenCacheRecord",
    "ProductionTokenizerAdapter",
    "ProductionWindowSet",
    "iter_production_batches",
    "open_production_training_split",
    "production_cursor_after_batches",
    "reopen_source_lock",
    "run_readiness",
    "run_source_lock",
    "run_training",
]
