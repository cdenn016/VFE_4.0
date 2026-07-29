from __future__ import annotations

import hashlib
import io
import os
import zipfile
from pathlib import Path

import pytest


ARCHIVE_URL = (
    "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip"
)
SOURCE_URL = (
    "https://blog.salesforceairesearch.com/"
    "the-wikitext-long-term-dependency-language-modeling-dataset/"
)
MEMBERS = {
    "wikitext-103-raw/wiki.train.raw": b"train text\n",
    "wikitext-103-raw/wiki.valid.raw": b"validation text\n",
    "wikitext-103-raw/wiki.test.raw": b"test text\n",
}


def _archive(
    *,
    members: dict[str, bytes] | None = None,
    compression: int = zipfile.ZIP_DEFLATED,
    include_directory: bool = True,
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", allowZip64=False) as handle:
        if include_directory:
            directory = zipfile.ZipInfo("wikitext-103-raw/")
            directory.external_attr = (0o40755 << 16) | 0x10
            directory.compress_type = zipfile.ZIP_STORED
            handle.writestr(directory, b"")
        for name, payload in (members or MEMBERS).items():
            info = zipfile.ZipInfo(name)
            info.external_attr = 0o100600 << 16
            info.compress_type = compression
            handle.writestr(info, payload)
    return buffer.getvalue()


def _source_page(paragraph: bytes | None = None) -> bytes:
    license_paragraph = paragraph or (
        b'<p class="license">Released under a Creative Commons '
        b'<a href="https://creativecommons.org/licenses/by-sa/4.0/">'
        b"Attribution-ShareAlike license</a>.</p>"
    )
    return b"<html><body><h1>WikiText</h1>" + license_paragraph + b"</body></html>"


class _Client:
    def __init__(
        self,
        *,
        archive: bytes | None = None,
        source_page: bytes | None = None,
        source_content_type: str = "text/html; charset=utf-8",
        archive_content_type: str | None = "application/zip",
    ) -> None:
        self.archive = archive if archive is not None else _archive()
        self.source_page = source_page if source_page is not None else _source_page()
        self.source_content_type = source_content_type
        self.archive_content_type = archive_content_type
        self.calls: list[tuple[str, int]] = []

    def fetch(self, url: str, *, maximum_bytes: int):
        from vfe4.data.wikitext103 import (
            BoundedHttpObservation,
            HttpRedirectObservation,
        )

        self.calls.append((url, maximum_bytes))
        if url == ARCHIVE_URL:
            headers = (
                ()
                if self.archive_content_type is None
                else (("content-type", self.archive_content_type),)
            )
            return BoundedHttpObservation.create(
                request_url=url,
                final_url=url,
                redirect_chain=(),
                status_code=200,
                headers=headers,
                body=self.archive,
            )
        if url == SOURCE_URL:
            return BoundedHttpObservation.create(
                request_url=url,
                final_url="https://www.salesforce.com/research/wikitext/",
                redirect_chain=(
                    HttpRedirectObservation(
                        status_code=301,
                        location="https://www.salesforce.com/research/wikitext/",
                        resolved_url="https://www.salesforce.com/research/wikitext/",
                    ),
                ),
                status_code=200,
                headers=(("content-type", self.source_content_type),),
                body=self.source_page,
            )
        raise AssertionError(f"unexpected URL {url}")


class _Backend:
    def __init__(self, *, reject_corpus_bytes: bool = False) -> None:
        self.writes: list[tuple[Path, bytes]] = []
        self.stream_writes: list[tuple[Path, tuple[int, ...]]] = []
        self.reject_corpus_bytes = reject_corpus_bytes

    def publish_bytes(self, path: Path, payload: bytes) -> None:
        if self.reject_corpus_bytes and path.suffix in (".raw", ".int32le"):
            raise AssertionError("corpus payload reached publish_bytes")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        self.writes.append((path, payload))

    def publish_content_addressed_stream(
        self,
        directory: Path,
        chunks: object,
        *,
        suffix: str,
        chunk_size_limit: int,
        reopen_block_size: int = 1_048_576,
    ) -> object:
        from vfe4.artifacts.durability import DurableFileIdentity

        del reopen_block_size
        parts = tuple(chunks)
        assert all(type(part) is bytes for part in parts)
        assert all(len(part) <= chunk_size_limit for part in parts)
        payload = b"".join(parts)
        digest = hashlib.sha256(payload).hexdigest()
        path = directory / f"{digest}{suffix}"
        path.write_bytes(payload)
        self.stream_writes.append(
            (path, tuple(len(part) for part in parts))
        )
        return DurableFileIdentity.create_verified(
            operation="content_addressed",
            size_bytes=len(payload),
            sha256=digest,
            volume_identity="wt103-source-test-volume",
        )


def _request(tmp_path: Path, *, allow_network: bool = True):
    from vfe4.data.wikitext103 import StagedAcquisitionRequest

    return StagedAcquisitionRequest(
        archive_request_url=ARCHIVE_URL,
        source_page_request_url=SOURCE_URL,
        staging_root=tmp_path,
        allow_network=allow_network,
    )


def test_stages_exact_archive_license_and_three_opaque_split_refs(
    tmp_path: Path,
) -> None:
    from vfe4.data.wikitext103 import (
        StagedWikiText103AcquisitionRecord,
        reopen_staged_wikitext103,
        stage_wikitext103_acquisition_record,
    )
    from vfe4.types.training import StagedWikiText103AcquisitionObservation

    client = _Client()
    backend = _Backend()
    observation = stage_wikitext103_acquisition_record(
        _request(tmp_path),
        http_client=client,
        durability_backend=backend,
    )

    assert type(observation) is StagedWikiText103AcquisitionRecord
    assert type(observation.observation) is StagedWikiText103AcquisitionObservation
    assert observation.schema_version == "wt103-staged-acquisition-v1"
    assert observation.authority == "nonproduction_staged_observation"
    assert observation.archive_request_url == ARCHIVE_URL
    assert observation.archive_final_url == ARCHIVE_URL
    assert observation.source_page_final_url.startswith("https://")
    assert tuple(row.member_path for row in observation.members) == tuple(MEMBERS)
    assert tuple(row.split for row in observation.sealed_splits) == (
        "train",
        "validation",
        "test",
    )
    assert observation.license.visible_text.startswith(
        "Released under a Creative Commons"
    )
    assert observation.license.hrefs == (
        "https://creativecommons.org/licenses/by-sa/4.0/",
    )
    assert (
        observation.license.raw_slice_sha256
        == hashlib.sha256(
            client.source_page[
                observation.license.paragraph_start_offset : observation.license.paragraph_end_offset
            ]
        ).hexdigest()
    )
    assert len(backend.writes) == 2  # archive and source page stay small-bytes
    assert len(backend.stream_writes) == 3
    reopened = reopen_staged_wikitext103(
        observation=observation,
        staging_root=tmp_path,
    )
    assert reopened.record_sha256 == observation.record_sha256
    assert (
        reopened.observation.observation_sha256
        == observation.observation.observation_sha256
    )


def test_zip_members_are_decompressed_and_published_in_bounded_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vfe4.data.wikitext103 as source

    member_chunk_bound = 4
    train_payload = b"larger-than-one-chunk"
    client = _Client(
        archive=_archive(
            members={
                **MEMBERS,
                "wikitext-103-raw/wiki.train.raw": train_payload,
            }
        )
    )
    backend = _Backend(reject_corpus_bytes=True)
    observed_reads: list[int] = []
    real_read = zipfile.ZipExtFile.read

    def guarded_read(
        handle: zipfile.ZipExtFile,
        size: int = -1,
    ) -> bytes:
        observed_reads.append(size)
        if size < 0 or size > member_chunk_bound:
            raise AssertionError("ZIP member was read outside the chunk bound")
        return real_read(handle, size)

    monkeypatch.setattr(
        source,
        "ARCHIVE_MEMBER_CHUNK_SIZE_BYTES",
        member_chunk_bound,
        raising=False,
    )
    monkeypatch.setattr(zipfile.ZipExtFile, "read", guarded_read)

    observation = source.stage_wikitext103_acquisition_record(
        _request(tmp_path),
        http_client=client,
        durability_backend=backend,
    )

    assert observation.members[0].uncompressed_size_bytes == len(train_payload)
    assert len(backend.writes) == 2
    assert len(backend.stream_writes) == 3
    assert all(
        size <= member_chunk_bound
        for _, sizes in backend.stream_writes
        for size in sizes
    )
    assert observed_reads
    assert all(0 <= size <= member_chunk_bound for size in observed_reads)


def test_staged_raw_reopen_never_uses_whole_file_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vfe4.data.wikitext103 as source

    observation = source.stage_wikitext103_acquisition_record(
        _request(tmp_path),
        http_client=_Client(),
        durability_backend=_Backend(),
    )
    real_regular_bytes = source._regular_bytes

    def reject_corpus_read(
        path: Path,
        *,
        size: int,
        sha256: str,
    ) -> bytes:
        if path.suffix == ".raw":
            raise AssertionError("staged raw reached whole-file helper")
        return real_regular_bytes(path, size=size, sha256=sha256)

    monkeypatch.setattr(source, "_regular_bytes", reject_corpus_read)

    assert (
        source.reopen_staged_wikitext103(
            observation=observation,
            staging_root=tmp_path,
        )
        == observation
    )


def test_public_acquisition_api_returns_exact_canonical_observation(
    tmp_path: Path,
) -> None:
    from vfe4.data.wikitext103 import stage_wikitext103_acquisition
    from vfe4.types.training import StagedWikiText103AcquisitionObservation

    observation = stage_wikitext103_acquisition(
        _request(tmp_path),
        http_client=_Client(),
        durability_backend=_Backend(),
    )

    assert type(observation) is StagedWikiText103AcquisitionObservation


def test_fresh_root_provisions_parents_for_real_durability_backend(
    tmp_path: Path,
) -> None:
    from vfe4.artifacts.durability import (
        PosixDurabilityBackend,
        WindowsDurabilityBackend,
    )
    from vfe4.data.wikitext103 import (
        reopen_staged_wikitext103,
        stage_wikitext103_acquisition_record,
    )

    backend = (
        WindowsDurabilityBackend() if os.name == "nt" else PosixDurabilityBackend()
    )
    observation = stage_wikitext103_acquisition_record(
        _request(tmp_path),
        http_client=_Client(),
        durability_backend=backend,
    )

    assert (
        len(tuple(path for path in (tmp_path / "staged").rglob("*") if path.is_file()))
        == 5
    )
    assert (
        reopen_staged_wikitext103(
            observation=observation,
            staging_root=tmp_path,
        )
        == observation
    )


def test_network_requires_explicit_source_lock_permission_before_client_call(
    tmp_path: Path,
) -> None:
    from vfe4.data.wikitext103 import (
        SourceAcquisitionError,
        stage_wikitext103_acquisition,
    )

    client = _Client()
    with pytest.raises(SourceAcquisitionError, match="allow_network"):
        stage_wikitext103_acquisition(
            _request(tmp_path, allow_network=False),
            http_client=client,
            durability_backend=_Backend(),
        )
    assert client.calls == []


@pytest.mark.parametrize(
    "source_page",
    (
        b"<html><body><p>No license declaration.</p></body></html>",
        (b"<p>Creative Commons one</p><p>Creative Commons two</p>"),
        (b"<p>Creative Commons</p><p>All rights reserved.</p>"),
        b"<p>Creative Commons",
        b"<p>" + b"x" * 4097 + b"Creative Commons</p>",
        b"<picture>Creative Commons</p>",
        b"<pre>Creative Commons</p>",
    ),
    ids=(
        "missing",
        "duplicate",
        "contradictory",
        "unclosed",
        "oversized-paragraph",
        "picture-is-not-a-paragraph",
        "pre-is-not-a-paragraph",
    ),
)
def test_license_extraction_fails_closed_on_ambiguity(
    tmp_path: Path, source_page: bytes
) -> None:
    from vfe4.data.wikitext103 import (
        SourceAcquisitionError,
        stage_wikitext103_acquisition,
    )

    with pytest.raises(
        SourceAcquisitionError, match="license|Creative Commons|paragraph"
    ):
        stage_wikitext103_acquisition(
            _request(tmp_path),
            http_client=_Client(source_page=source_page),
            durability_backend=_Backend(),
        )


@pytest.mark.parametrize(
    ("source_page", "content_type"),
    (
        (b"x" * (4_194_304 + 1), "text/html"),
        (_source_page(), "application/json"),
        (_source_page() + b"\xff", "text/html"),
    ),
    ids=("oversized", "wrong-content-type", "invalid-utf8"),
)
def test_source_page_bounds_content_type_and_utf8_are_enforced(
    tmp_path: Path, source_page: bytes, content_type: str
) -> None:
    from vfe4.data.wikitext103 import (
        SourceAcquisitionError,
        stage_wikitext103_acquisition,
    )

    with pytest.raises(
        SourceAcquisitionError, match="source page|content type|UTF-8|size"
    ):
        stage_wikitext103_acquisition(
            _request(tmp_path),
            http_client=_Client(
                source_page=source_page, source_content_type=content_type
            ),
            durability_backend=_Backend(),
        )


@pytest.mark.parametrize(
    "members",
    (
        {**MEMBERS, "wikitext-103-raw/extra.raw": b"x"},
        {
            "wikitext-103-raw/wiki.train.raw": b"x",
            "wikitext-103-raw/wiki.valid.raw": b"y",
            "../wiki.test.raw": b"z",
        },
        {
            "wikitext-103-raw/wiki.train.raw": b"x",
            "wikitext-103-raw/WIKI.TRAIN.RAW": b"y",
            "wikitext-103-raw/wiki.valid.raw": b"z",
            "wikitext-103-raw/wiki.test.raw": b"q",
        },
    ),
)
def test_archive_inventory_extra_traversal_and_case_collisions_fail(
    tmp_path: Path, members: dict[str, bytes]
) -> None:
    from vfe4.data.wikitext103 import (
        SourceAcquisitionError,
        stage_wikitext103_acquisition,
    )

    with pytest.raises(
        SourceAcquisitionError, match="archive|member|inventory|path|collision"
    ):
        stage_wikitext103_acquisition(
            _request(tmp_path),
            http_client=_Client(archive=_archive(members=members)),
            durability_backend=_Backend(),
        )


def test_archive_rejects_unsupported_compression_and_zip_bomb_ratio(
    tmp_path: Path,
) -> None:
    from vfe4.data.wikitext103 import (
        SourceAcquisitionError,
        stage_wikitext103_acquisition,
    )

    with pytest.raises(SourceAcquisitionError, match="compression"):
        stage_wikitext103_acquisition(
            _request(tmp_path),
            http_client=_Client(archive=_archive(compression=zipfile.ZIP_BZIP2)),
            durability_backend=_Backend(),
        )
    bomb_members = {
        **MEMBERS,
        "wikitext-103-raw/wiki.train.raw": b"0" * 100_000,
    }
    with pytest.raises(SourceAcquisitionError, match="ratio|compression"):
        stage_wikitext103_acquisition(
            _request(tmp_path),
            http_client=_Client(archive=_archive(members=bomb_members)),
            durability_backend=_Backend(),
        )


@pytest.mark.parametrize(
    "status_code",
    (300, 304, 305, 306, 309, 399),
)
def test_redirect_status_is_one_of_the_canonical_http_redirect_codes(
    status_code: int,
) -> None:
    from vfe4.data.wikitext103 import (
        HttpRedirectObservation,
        SourceAcquisitionError,
    )

    with pytest.raises(SourceAcquisitionError, match="redirect status"):
        HttpRedirectObservation(
            status_code=status_code,
            location="https://www.salesforce.com/research/wikitext/",
            resolved_url="https://www.salesforce.com/research/wikitext/",
        )


@pytest.mark.parametrize(
    ("final_url", "redirect_chain"),
    (
        (
            "https://www.salesforce.com/research/wikitext/",
            (),
        ),
        (
            "https://www.salesforce.com/other/",
            (
                (
                    301,
                    "https://www.salesforce.com/research/wikitext/",
                    "https://www.salesforce.com/other/",
                ),
            ),
        ),
        (
            SOURCE_URL,
            (
                (
                    301,
                    SOURCE_URL,
                    SOURCE_URL,
                ),
            ),
        ),
        (
            "https://example.test/wikitext/",
            (
                (
                    301,
                    "https://example.test/wikitext/",
                    "https://example.test/wikitext/",
                ),
            ),
        ),
    ),
    ids=(
        "empty-chain-final-mismatch",
        "location-resolution-mismatch",
        "redirect-loop",
        "unapproved-origin",
    ),
)
def test_redirect_chain_enforces_location_continuity_loop_and_origin(
    final_url: str,
    redirect_chain: tuple[tuple[int, str, str], ...],
) -> None:
    from vfe4.data.wikitext103 import (
        BoundedHttpObservation,
        HttpRedirectObservation,
        SourceAcquisitionError,
    )

    chain = tuple(
        HttpRedirectObservation(
            status_code=status_code,
            location=location,
            resolved_url=resolved_url,
        )
        for status_code, location, resolved_url in redirect_chain
    )
    with pytest.raises(
        SourceAcquisitionError,
        match="redirect|origin|final|location|loop",
    ):
        BoundedHttpObservation.create(
            request_url=SOURCE_URL,
            final_url=final_url,
            redirect_chain=chain,
            status_code=200,
            headers=(("content-type", "text/html"),),
            body=b"x",
        )


def test_archive_content_type_rule_and_https_final_origin_are_enforced(
    tmp_path: Path,
) -> None:
    from vfe4.data.wikitext103 import (
        BoundedHttpObservation,
        SourceAcquisitionError,
        stage_wikitext103_acquisition,
    )

    with pytest.raises(SourceAcquisitionError, match="content type"):
        stage_wikitext103_acquisition(
            _request(tmp_path),
            http_client=_Client(archive_content_type="text/plain"),
            durability_backend=_Backend(),
        )

    class Downgrade(_Client):
        def fetch(self, url: str, *, maximum_bytes: int):
            observed = super().fetch(url, maximum_bytes=maximum_bytes)
            if url == ARCHIVE_URL:
                return BoundedHttpObservation.create(
                    request_url=url,
                    final_url="http://example.test/archive.zip",
                    redirect_chain=observed.redirect_chain,
                    status_code=200,
                    headers=observed.headers,
                    body=observed.body,
                )
            return observed

    with pytest.raises(SourceAcquisitionError, match="HTTPS"):
        stage_wikitext103_acquisition(
            _request(tmp_path),
            http_client=Downgrade(),
            durability_backend=_Backend(),
        )


def test_offline_reopen_rehashes_every_payload_and_quarantines_v3(
    tmp_path: Path,
) -> None:
    from vfe4.data.wikitext103 import (
        SourceAcquisitionError,
        reopen_staged_wikitext103,
        stage_wikitext103_acquisition_record,
    )

    observation = stage_wikitext103_acquisition_record(
        _request(tmp_path),
        http_client=_Client(),
        durability_backend=_Backend(),
    )
    split = observation.sealed_splits[0]
    path = tmp_path / split.cache_relative_path
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(SourceAcquisitionError, match="size|hash"):
        reopen_staged_wikitext103(
            observation=observation,
            staging_root=tmp_path,
        )

    v3_root = tmp_path / "V3_Transformer"
    with pytest.raises(SourceAcquisitionError, match="V3"):
        reopen_staged_wikitext103(
            observation=observation,
            staging_root=v3_root,
        )


def test_target_rejects_original_component_junction_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vfe4.data.wikitext103 import SourceAcquisitionError, _target

    staged = tmp_path / "staged"
    staged.mkdir()
    original = getattr(Path, "is_junction", None)

    def is_junction(path: Path) -> bool:
        if path == staged:
            return True
        return bool(original is not None and original(path))

    monkeypatch.setattr(Path, "is_junction", is_junction, raising=False)
    with pytest.raises(SourceAcquisitionError, match="redirect|reparse|junction"):
        _target(tmp_path.resolve(), "staged/archive/payload.zip")


def test_target_rejects_resolved_identity_different_from_declared_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vfe4.data.wikitext103 import SourceAcquisitionError, _target

    root = tmp_path.resolve()
    (root / "staged" / "archive").mkdir(parents=True)
    redirected = root / "V3_Transformer"
    redirected.mkdir()
    declared = root / "staged" / "archive" / "payload.zip"
    actual = redirected / "payload.zip"
    original = Path.resolve

    def resolve(path: Path, *args: object, **kwargs: object) -> Path:
        resolved = original(path, *args, **kwargs)
        return actual if resolved == declared else resolved

    monkeypatch.setattr(Path, "resolve", resolve)
    with pytest.raises(SourceAcquisitionError, match="identity|declared|V3"):
        _target(root, "staged/archive/payload.zip")


def test_task3_exposes_no_finalized_production_record_or_test_unsealer() -> None:
    import vfe4.data.wikitext103 as module

    assert not hasattr(module, "FinalizedWikiText103SourceRecord")
    assert not hasattr(module, "ProductionTokenizerSpec")
    assert not hasattr(module, "open_test")
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "V3_Transformer" not in source.replace('"V3_Transformer"', "")
