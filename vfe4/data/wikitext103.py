"""Bounded, injected WikiText-103 staging with no production promotion.

The live network adapter belongs to the separately authorized source-lock
operation.  This module accepts an injected bounded response seam so Tasks
1--12 can exercise archive, source-page, license, and offline-integrity logic
with generated fixtures only.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import stat
import unicodedata
import zipfile
import zlib
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal, Protocol
from urllib.parse import urljoin, urlparse

from vfe4.types.training import (
    ArchiveMemberIdentity,
    RedirectHop,
    StagedWikiText103AcquisitionObservation,
)


WIKITEXT103_ARCHIVE_REQUEST_URL = (
    "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip"
)
WIKITEXT103_SOURCE_PAGE_REQUEST_URL = (
    "https://blog.salesforceairesearch.com/"
    "the-wikitext-long-term-dependency-language-modeling-dataset/"
)
WIKITEXT103_DIRECTORY = "wikitext-103-raw/"
WIKITEXT103_MEMBER_PATHS = (
    "wikitext-103-raw/wiki.train.raw",
    "wikitext-103-raw/wiki.valid.raw",
    "wikitext-103-raw/wiki.test.raw",
)
WIKITEXT103_SPLITS: tuple[Literal["train", "validation", "test"], ...] = (
    "train",
    "validation",
    "test",
)
_CANONICAL_REDIRECT_STATUS_CODES = frozenset((301, 302, 303, 307, 308))
_ALLOWED_REDIRECT_ORIGINS = {
    WIKITEXT103_ARCHIVE_REQUEST_URL: frozenset((("https", "s3.amazonaws.com", 443),)),
    WIKITEXT103_SOURCE_PAGE_REQUEST_URL: frozenset(
        (
            ("https", "blog.salesforceairesearch.com", 443),
            ("https", "salesforce.com", 443),
            ("https", "www.salesforce.com", 443),
        )
    ),
}

MAXIMUM_ARCHIVE_BYTES = 268_435_456
MAXIMUM_SOURCE_PAGE_BYTES = 4_194_304
MAXIMUM_MEMBER_BYTES = 671_088_640
MAXIMUM_TOTAL_MEMBER_BYTES = 805_306_368
MAXIMUM_COMPRESSION_RATIO = 100.0
MAXIMUM_LICENSE_PARAGRAPH_BYTES = 4_096

_HTTP_DOMAIN = b"VFE4-WT103-BOUNDED-HTTP-OBSERVATION-V1\x00"
_LICENSE_DOMAIN = b"VFE4-WT103-LICENSE-OBSERVATION-V1\x00"
_MEMBER_DOMAIN = b"VFE4-WT103-ARCHIVE-MEMBER-V1\x00"
_CENTRAL_DIRECTORY_DOMAIN = b"VFE4-WT103-CENTRAL-DIRECTORY-V1\x00"
_SEALED_SPLIT_DOMAIN = b"VFE4-WT103-SEALED-SPLIT-V1\x00"
_STAGED_DOMAIN = b"VFE4-WT103-STAGED-ACQUISITION-V1\x00"

_ACCEPTED_ZIP_CONTENT_TYPES = frozenset(
    (
        "application/zip",
        "application/x-zip-compressed",
        "application/octet-stream",
    )
)
_ACCEPTED_HTML_CONTENT_TYPES = frozenset(("text/html", "application/xhtml+xml"))


class SourceAcquisitionError(ValueError):
    """The candidate source could not be staged without ambiguity."""


def _canonical_bytes(value: object) -> bytes:
    def convert(item: object) -> object:
        if dataclasses.is_dataclass(item) and not isinstance(item, type):
            return {
                field.name: convert(getattr(item, field.name))
                for field in dataclasses.fields(item)
            }
        if isinstance(item, Path):
            return item.as_posix()
        if isinstance(item, tuple):
            return [convert(child) for child in item]
        if item is None or type(item) in (str, bool, int):
            return item
        if type(item) is float:
            if not (item >= 0.0 and item < float("inf")):
                raise SourceAcquisitionError("canonical floats must be finite")
            return item
        raise SourceAcquisitionError(
            f"unsupported canonical value: {type(item).__name__}"
        )

    return json.dumps(
        convert(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_bytes(value)).hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SourceAcquisitionError(f"{field} must be a lowercase SHA-256")
    return value


def _require_https(url: object, *, field: str) -> str:
    if type(url) is not str or not url:
        raise SourceAcquisitionError(f"{field} must be a nonempty HTTPS URL")
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise SourceAcquisitionError(f"{field} must retain an unambiguous HTTPS origin")
    return url


def _https_origin(url: str, *, field: str) -> tuple[str, str, int]:
    _require_https(url, field=field)
    parsed = urlparse(url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise SourceAcquisitionError(f"{field} has an invalid HTTPS origin") from exc
    if parsed.hostname is None or port not in (None, 443):
        raise SourceAcquisitionError(f"{field} must retain an approved HTTPS origin")
    return (parsed.scheme, parsed.hostname.casefold(), port or 443)


def _headers(value: object) -> tuple[tuple[str, str], ...]:
    if type(value) is not tuple:
        raise SourceAcquisitionError("HTTP headers must be an immutable tuple")
    result: list[tuple[str, str]] = []
    names: set[str] = set()
    for row in value:
        if (
            type(row) is not tuple
            or len(row) != 2
            or type(row[0]) is not str
            or type(row[1]) is not str
            or not row[0]
            or row[0] != row[0].lower()
        ):
            raise SourceAcquisitionError("HTTP headers must be lowercase string pairs")
        if row[0] in names:
            raise SourceAcquisitionError("duplicate HTTP response header")
        names.add(row[0])
        result.append(row)
    return tuple(result)


@dataclasses.dataclass(frozen=True, slots=True)
class HttpRedirectObservation:
    status_code: int
    location: str
    resolved_url: str

    def __post_init__(self) -> None:
        if (
            type(self.status_code) is not int
            or self.status_code not in _CANONICAL_REDIRECT_STATUS_CODES
        ):
            raise SourceAcquisitionError(
                "redirect status is not a canonical HTTP redirect code"
            )
        if type(self.location) is not str or not self.location:
            raise SourceAcquisitionError("redirect location must be nonempty")
        _require_https(self.resolved_url, field="redirect resolved_url")


@dataclasses.dataclass(frozen=True, slots=True)
class BoundedHttpObservation:
    schema_version: str
    request_url: str
    final_url: str
    redirect_chain: tuple[HttpRedirectObservation, ...]
    status_code: int
    headers: tuple[tuple[str, str], ...]
    body: bytes = dataclasses.field(repr=False)
    body_size_bytes: int = 0
    body_sha256: str = ""
    observation_sha256: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-bounded-http-observation-v1":
            raise SourceAcquisitionError("unsupported HTTP observation schema")
        _require_https(self.request_url, field="request_url")
        _require_https(self.final_url, field="final_url")
        if type(self.redirect_chain) is not tuple or any(
            type(row) is not HttpRedirectObservation for row in self.redirect_chain
        ):
            raise SourceAcquisitionError(
                "redirect_chain must contain exact RedirectHop rows"
            )
        for row in self.redirect_chain:
            row.__post_init__()
        allowed_origins = _ALLOWED_REDIRECT_ORIGINS.get(self.request_url)
        if allowed_origins is None:
            raise SourceAcquisitionError(
                "request URL has no preregistered redirect-origin policy"
            )
        current_url = self.request_url
        observed_urls = {current_url}
        for index, row in enumerate(self.redirect_chain):
            expected_url = urljoin(current_url, row.location)
            _require_https(
                expected_url,
                field=f"redirect_chain[{index}] Location",
            )
            if expected_url != row.resolved_url:
                raise SourceAcquisitionError(
                    "redirect Location does not resolve to the recorded URL"
                )
            if (
                _https_origin(
                    row.resolved_url,
                    field=f"redirect_chain[{index}] resolved_url",
                )
                not in allowed_origins
            ):
                raise SourceAcquisitionError(
                    "redirect chain left its approved HTTPS origins"
                )
            if row.resolved_url in observed_urls:
                raise SourceAcquisitionError("redirect chain contains a loop")
            try:
                RedirectHop(
                    request_url=current_url,
                    response_url=row.resolved_url,
                    status_code=row.status_code,
                )
            except ValueError as exc:
                raise SourceAcquisitionError("redirect hop is not canonical") from exc
            observed_urls.add(row.resolved_url)
            current_url = row.resolved_url
        if current_url != self.final_url:
            raise SourceAcquisitionError(
                "redirect chain does not terminate exactly at final_url"
            )
        if _https_origin(self.final_url, field="final_url") not in allowed_origins:
            raise SourceAcquisitionError("final URL has an unapproved HTTPS origin")
        if type(self.status_code) is not int:
            raise SourceAcquisitionError("status_code must be an exact integer")
        headers = _headers(self.headers)
        if type(self.body) is not bytes:
            raise SourceAcquisitionError("HTTP body must be exact bytes")
        if self.body_size_bytes != len(self.body):
            raise SourceAcquisitionError("HTTP body size is inconsistent")
        body_sha256 = hashlib.sha256(self.body).hexdigest()
        if self.body_sha256 != body_sha256:
            raise SourceAcquisitionError("HTTP body hash is inconsistent")
        expected = _digest(
            _HTTP_DOMAIN,
            (
                self.schema_version,
                self.request_url,
                self.final_url,
                self.redirect_chain,
                self.status_code,
                headers,
                self.body_size_bytes,
                self.body_sha256,
            ),
        )
        if self.observation_sha256 != expected:
            raise SourceAcquisitionError("HTTP observation hash is inconsistent")

    @classmethod
    def create(
        cls,
        *,
        request_url: str,
        final_url: str,
        redirect_chain: tuple[HttpRedirectObservation, ...],
        status_code: int,
        headers: tuple[tuple[str, str], ...],
        body: bytes,
    ) -> "BoundedHttpObservation":
        values = (
            "wt103-bounded-http-observation-v1",
            request_url,
            final_url,
            redirect_chain,
            status_code,
            headers,
            len(body),
            hashlib.sha256(body).hexdigest(),
        )
        return cls(
            schema_version=values[0],
            request_url=values[1],
            final_url=values[2],
            redirect_chain=values[3],
            status_code=values[4],
            headers=values[5],
            body=body,
            body_size_bytes=values[6],
            body_sha256=values[7],
            observation_sha256=_digest(_HTTP_DOMAIN, values),
        )


class BoundedHttpClient(Protocol):
    def fetch(self, url: str, *, maximum_bytes: int) -> BoundedHttpObservation:
        """Return one already-bounded response observation."""


class SourceDurabilityBackend(Protocol):
    def publish_bytes(self, path: Path, payload: bytes) -> object:
        """Publish bytes durably and reopen-validate them."""


@dataclasses.dataclass(frozen=True, slots=True)
class StagedAcquisitionRequest:
    archive_request_url: str
    source_page_request_url: str
    staging_root: Path
    allow_network: bool

    def __post_init__(self) -> None:
        if self.archive_request_url != WIKITEXT103_ARCHIVE_REQUEST_URL:
            raise SourceAcquisitionError(
                "archive_request_url is not the preregistered candidate"
            )
        if self.source_page_request_url != WIKITEXT103_SOURCE_PAGE_REQUEST_URL:
            raise SourceAcquisitionError(
                "source_page_request_url is not the preregistered candidate"
            )
        if not isinstance(self.staging_root, Path):
            raise SourceAcquisitionError("staging_root must be a Path")
        if type(self.allow_network) is not bool:
            raise SourceAcquisitionError("allow_network must be a plain bool")


@dataclasses.dataclass(frozen=True, slots=True)
class LicenseObservation:
    schema_version: str
    paragraph_start_offset: int
    paragraph_end_offset: int
    raw_slice_sha256: str
    visible_text: str
    hrefs: tuple[str, ...]
    observation_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-license-observation-v1":
            raise SourceAcquisitionError("unsupported license observation schema")
        if (
            type(self.paragraph_start_offset) is not int
            or type(self.paragraph_end_offset) is not int
            or self.paragraph_start_offset < 0
            or self.paragraph_end_offset <= self.paragraph_start_offset
            or self.paragraph_end_offset - self.paragraph_start_offset
            > MAXIMUM_LICENSE_PARAGRAPH_BYTES
        ):
            raise SourceAcquisitionError("license paragraph offsets are invalid")
        _require_sha256(self.raw_slice_sha256, field="license raw_slice_sha256")
        if type(self.visible_text) is not str or not self.visible_text.strip():
            raise SourceAcquisitionError("license visible_text is empty")
        if (
            type(self.hrefs) is not tuple
            or any(type(href) is not str or not href for href in self.hrefs)
            or len(set(self.hrefs)) != len(self.hrefs)
        ):
            raise SourceAcquisitionError("license hrefs are invalid")
        expected = _digest(
            _LICENSE_DOMAIN,
            (
                self.schema_version,
                self.paragraph_start_offset,
                self.paragraph_end_offset,
                self.raw_slice_sha256,
                self.visible_text,
                self.hrefs,
            ),
        )
        if self.observation_sha256 != expected:
            raise SourceAcquisitionError("license observation hash is invalid")


class _LicenseParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text: list[str] = []
        self.hrefs: list[str] = []
        self.start_tags: list[str] = []
        self.end_tags: list[str] = []

    def handle_data(self, data: str) -> None:
        self.text.append(data)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized_tag = tag.casefold()
        self.start_tags.append(normalized_tag)
        if normalized_tag != "a":
            return
        for key, value in attrs:
            if key.casefold() == "href" and value:
                self.hrefs.append(value)

    def handle_endtag(self, tag: str) -> None:
        self.end_tags.append(tag.casefold())


def _extract_license(body: bytes) -> LicenseObservation:
    lowered = body.lower()
    needle = b"creative commons"
    occurrences: list[int] = []
    offset = 0
    while True:
        found = lowered.find(needle, offset)
        if found < 0:
            break
        occurrences.append(found)
        offset = found + len(needle)
    if len(occurrences) != 1:
        raise SourceAcquisitionError(
            "source page must contain exactly one Creative Commons occurrence"
        )
    occurrence = occurrences[0]
    paragraph_start = -1
    search_end = occurrence + 1
    while search_end > 0:
        candidate = lowered.rfind(b"<p", 0, search_end)
        if candidate < 0:
            break
        boundary = candidate + 2
        if boundary < len(lowered) and lowered[boundary] in b" \t\r\n\f>":
            paragraph_start = candidate
            break
        search_end = candidate
    if paragraph_start < 0:
        raise SourceAcquisitionError(
            "Creative Commons declaration is not inside a paragraph"
        )
    opening_end = lowered.find(b">", paragraph_start, occurrence + 1)
    if opening_end < 0 or opening_end >= occurrence:
        raise SourceAcquisitionError("license paragraph opening tag is malformed")
    closing_start = lowered.find(b"</p>", occurrence + len(needle))
    if closing_start < 0:
        raise SourceAcquisitionError("license paragraph is not syntactically closed")
    paragraph_end = closing_start + len(b"</p>")
    if paragraph_end - paragraph_start > MAXIMUM_LICENSE_PARAGRAPH_BYTES:
        raise SourceAcquisitionError("license paragraph exceeds 4,096 bytes")
    prior_close = lowered.rfind(b"</p>", paragraph_start, occurrence)
    nested_open = lowered.find(b"<p", opening_end + 1, paragraph_end)
    if prior_close >= paragraph_start or nested_open >= 0:
        raise SourceAcquisitionError("license paragraph containment is ambiguous")
    outside = lowered[:paragraph_start] + lowered[paragraph_end:]
    if b"all rights reserved" in outside:
        raise SourceAcquisitionError(
            "source page contains a contradictory license declaration"
        )
    raw_slice = body[paragraph_start:paragraph_end]
    try:
        decoded = raw_slice.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise SourceAcquisitionError("license paragraph is not strict UTF-8") from exc
    parser = _LicenseParser()
    try:
        parser.feed(decoded)
        parser.close()
    except (ValueError, UnicodeError) as exc:
        raise SourceAcquisitionError("license paragraph HTML is malformed") from exc
    if (
        parser.start_tags.count("p") != 1
        or parser.end_tags.count("p") != 1
        or not parser.start_tags
        or parser.start_tags[0] != "p"
        or not parser.end_tags
        or parser.end_tags[-1] != "p"
    ):
        raise SourceAcquisitionError(
            "license declaration must have one exact paragraph-tag structure"
        )
    visible_text = " ".join("".join(parser.text).split())
    hrefs = tuple(parser.hrefs)
    plausible = tuple(
        href for href in hrefs if "creativecommons.org" in href.casefold()
    )
    if len(plausible) > 1:
        raise SourceAcquisitionError(
            "license paragraph contains multiple plausible license links"
        )
    for href in hrefs:
        _require_https(href, field="license href")
    values = (
        "wt103-license-observation-v1",
        paragraph_start,
        paragraph_end,
        hashlib.sha256(raw_slice).hexdigest(),
        visible_text,
        hrefs,
    )
    return LicenseObservation(*values, _digest(_LICENSE_DOMAIN, values))


def _canonical_member_path(name: object) -> str:
    if type(name) is not str or not name or "\\" in name or "\x00" in name:
        raise SourceAcquisitionError("archive member path is invalid")
    if unicodedata.normalize("NFC", name) != name:
        raise SourceAcquisitionError("archive member path is not canonical Unicode")
    is_directory = name.endswith("/")
    normalized_name = name[:-1] if is_directory else name
    if is_directory and name != WIKITEXT103_DIRECTORY:
        raise SourceAcquisitionError("archive contains an unexpected directory")
    posix = PurePosixPath(normalized_name)
    windows = PureWindowsPath(normalized_name)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or bool(windows.root)
        or posix.as_posix() != normalized_name
        or any(part in ("", ".", "..") for part in posix.parts)
    ):
        raise SourceAcquisitionError("archive member path is noncanonical or escapes")
    return name


@dataclasses.dataclass(frozen=True, slots=True)
class ArchiveMemberObservation:
    schema_version: str
    split: Literal["train", "validation", "test"]
    member_path: str
    uncompressed_size_bytes: int
    compressed_size_bytes: int
    compression_method: int
    flag_bits: int
    central_crc32: int
    recomputed_crc32: int
    sha256: str
    observation_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-archive-member-v1":
            raise SourceAcquisitionError("unsupported archive member schema")
        if self.split not in WIKITEXT103_SPLITS:
            raise SourceAcquisitionError("archive member split is invalid")
        expected_path = WIKITEXT103_MEMBER_PATHS[WIKITEXT103_SPLITS.index(self.split)]
        if self.member_path != expected_path:
            raise SourceAcquisitionError("archive member path/split mismatch")
        _canonical_member_path(self.member_path)
        if (
            type(self.uncompressed_size_bytes) is not int
            or type(self.compressed_size_bytes) is not int
            or not 0 < self.uncompressed_size_bytes <= MAXIMUM_MEMBER_BYTES
            or self.compressed_size_bytes <= 0
            or self.uncompressed_size_bytes / self.compressed_size_bytes
            > MAXIMUM_COMPRESSION_RATIO
        ):
            raise SourceAcquisitionError(
                "archive member size/compression ratio is invalid"
            )
        if self.compression_method not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            raise SourceAcquisitionError("archive member compression is unsupported")
        if type(self.flag_bits) is not int or self.flag_bits & 0x9:
            raise SourceAcquisitionError(
                "archive member encryption/data-descriptor flags are forbidden"
            )
        for value, field in (
            (self.central_crc32, "central_crc32"),
            (self.recomputed_crc32, "recomputed_crc32"),
        ):
            if type(value) is not int or not 0 <= value <= 0xFFFFFFFF:
                raise SourceAcquisitionError(f"{field} is invalid")
        if self.central_crc32 != self.recomputed_crc32:
            raise SourceAcquisitionError("archive member CRC32 does not match")
        _require_sha256(self.sha256, field="archive member sha256")
        expected = _digest(
            _MEMBER_DOMAIN,
            tuple(
                getattr(self, field.name)
                for field in dataclasses.fields(self)
                if field.name != "observation_sha256"
            ),
        )
        if self.observation_sha256 != expected:
            raise SourceAcquisitionError("archive member observation hash is invalid")

    @property
    def identity(self) -> ArchiveMemberIdentity:
        """Project the detailed observation to the canonical Task 1 identity."""

        return ArchiveMemberIdentity(
            split=self.split,
            member_name=self.member_path,
            compression_method=self.compression_method,
            compressed_size_bytes=self.compressed_size_bytes,
            uncompressed_size_bytes=self.uncompressed_size_bytes,
            crc32=self.central_crc32,
            payload_sha256=self.sha256,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class SealedSplitRef:
    schema_version: str
    authority: str
    split: Literal["train", "validation", "test"]
    member_observation_sha256: str
    payload_size_bytes: int
    payload_sha256: str
    cache_relative_path: str
    sealed_ref_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-sealed-split-ref-v1":
            raise SourceAcquisitionError("unsupported sealed split schema")
        if self.authority != "nonproduction_staged_observation":
            raise SourceAcquisitionError("sealed staged split cannot be promoted")
        if self.split not in WIKITEXT103_SPLITS:
            raise SourceAcquisitionError("sealed split name is invalid")
        _require_sha256(
            self.member_observation_sha256, field="member_observation_sha256"
        )
        if type(self.payload_size_bytes) is not int or self.payload_size_bytes <= 0:
            raise SourceAcquisitionError("sealed split payload size is invalid")
        _require_sha256(self.payload_sha256, field="sealed split payload_sha256")
        expected_path = f"staged/splits/{self.split}/{self.payload_sha256}.raw"
        if self.cache_relative_path != expected_path:
            raise SourceAcquisitionError(
                "sealed split cache path is not content addressed"
            )
        expected = _digest(
            _SEALED_SPLIT_DOMAIN,
            tuple(
                getattr(self, field.name)
                for field in dataclasses.fields(self)
                if field.name != "sealed_ref_sha256"
            ),
        )
        if self.sealed_ref_sha256 != expected:
            raise SourceAcquisitionError("sealed split reference hash is invalid")


@dataclasses.dataclass(frozen=True, slots=True)
class StagedWikiText103AcquisitionRecord:
    schema_version: str
    authority: str
    observation: StagedWikiText103AcquisitionObservation
    archive_request_url: str
    archive_final_url: str
    archive_redirect_chain: tuple[HttpRedirectObservation, ...]
    archive_size_bytes: int
    archive_sha256: str
    archive_relative_path: str
    source_page_request_url: str
    source_page_final_url: str
    source_page_redirect_chain: tuple[HttpRedirectObservation, ...]
    source_page_size_bytes: int
    source_page_sha256: str
    source_page_relative_path: str
    license: LicenseObservation
    members: tuple[ArchiveMemberObservation, ...]
    sealed_splits: tuple[SealedSplitRef, ...]
    record_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-staged-acquisition-v1":
            raise SourceAcquisitionError("unsupported staged acquisition schema")
        if self.authority != "nonproduction_staged_observation":
            raise SourceAcquisitionError("staged observation cannot be promoted")
        if type(self.observation) is not StagedWikiText103AcquisitionObservation:
            raise SourceAcquisitionError(
                "observation must be the exact canonical staged observation"
            )
        self.observation.__post_init__()
        if self.archive_request_url != WIKITEXT103_ARCHIVE_REQUEST_URL:
            raise SourceAcquisitionError("archive request URL is not frozen")
        if self.source_page_request_url != WIKITEXT103_SOURCE_PAGE_REQUEST_URL:
            raise SourceAcquisitionError("source-page request URL is not frozen")
        _require_https(self.archive_final_url, field="archive_final_url")
        _require_https(self.source_page_final_url, field="source_page_final_url")
        if (
            type(self.archive_redirect_chain) is not tuple
            or type(self.source_page_redirect_chain) is not tuple
            or any(
                type(row) is not HttpRedirectObservation
                for row in self.archive_redirect_chain + self.source_page_redirect_chain
            )
        ):
            raise SourceAcquisitionError("staged redirect chains are invalid")
        if (
            type(self.archive_size_bytes) is not int
            or not 0 < self.archive_size_bytes <= MAXIMUM_ARCHIVE_BYTES
            or type(self.source_page_size_bytes) is not int
            or not 0 < self.source_page_size_bytes <= MAXIMUM_SOURCE_PAGE_BYTES
        ):
            raise SourceAcquisitionError("staged source sizes are invalid")
        _require_sha256(self.archive_sha256, field="archive_sha256")
        _require_sha256(self.source_page_sha256, field="source_page_sha256")
        if self.archive_relative_path != (f"staged/archive/{self.archive_sha256}.zip"):
            raise SourceAcquisitionError("archive path is not content addressed")
        if self.source_page_relative_path != (
            f"staged/source/{self.source_page_sha256}.html"
        ):
            raise SourceAcquisitionError("source-page path is not content addressed")
        if type(self.license) is not LicenseObservation:
            raise SourceAcquisitionError("license must be an exact observation")
        self.license.__post_init__()
        if (
            type(self.members) is not tuple
            or tuple(row.split for row in self.members) != WIKITEXT103_SPLITS
            or any(type(row) is not ArchiveMemberObservation for row in self.members)
        ):
            raise SourceAcquisitionError("archive member inventory is not exact")
        if (
            type(self.sealed_splits) is not tuple
            or tuple(row.split for row in self.sealed_splits) != WIKITEXT103_SPLITS
            or any(type(row) is not SealedSplitRef for row in self.sealed_splits)
        ):
            raise SourceAcquisitionError("sealed split inventory is not exact")
        for member, sealed in zip(self.members, self.sealed_splits, strict=True):
            member.__post_init__()
            sealed.__post_init__()
            if (
                sealed.member_observation_sha256 != member.observation_sha256
                or sealed.payload_size_bytes != member.uncompressed_size_bytes
                or sealed.payload_sha256 != member.sha256
            ):
                raise SourceAcquisitionError(
                    "sealed split does not match archive member observation"
                )
        central_directory_sha256 = _digest(
            _CENTRAL_DIRECTORY_DOMAIN,
            tuple(member.identity for member in self.members),
        )
        if (
            self.observation.archive_sha256 != self.archive_sha256
            or self.observation.central_directory_sha256 != central_directory_sha256
            or self.observation.source_page_sha256 != self.source_page_sha256
            or self.observation.license_raw_slice_sha256
            != self.license.raw_slice_sha256
        ):
            raise SourceAcquisitionError(
                "canonical staged observation does not bind the detailed record"
            )
        expected = _digest(
            _STAGED_DOMAIN,
            tuple(
                getattr(self, field.name)
                for field in dataclasses.fields(self)
                if field.name != "record_sha256"
            ),
        )
        if self.record_sha256 != expected:
            raise SourceAcquisitionError("staged record hash is invalid")


def _content_type(observation: BoundedHttpObservation) -> str | None:
    header = dict(observation.headers).get("content-type")
    if header is None:
        return None
    return header.split(";", 1)[0].strip().lower()


def _validate_http(
    observation: object,
    *,
    expected_request_url: str,
    maximum_bytes: int,
    kind: Literal["archive", "source page"],
) -> BoundedHttpObservation:
    if type(observation) is not BoundedHttpObservation:
        raise SourceAcquisitionError(
            f"{kind} client returned an untyped response observation"
        )
    observation.__post_init__()
    if observation.request_url != expected_request_url:
        raise SourceAcquisitionError(f"{kind} request URL changed")
    _require_https(observation.final_url, field=f"{kind} final URL")
    if observation.status_code != 200:
        raise SourceAcquisitionError(f"{kind} status must be exactly 200")
    if observation.body_size_bytes <= 0 or observation.body_size_bytes > maximum_bytes:
        raise SourceAcquisitionError(f"{kind} response size exceeds its bound")
    return observation


def _entry_is_redirect(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0xFFFF
    file_type = stat.S_IFMT(mode)
    return file_type in (
        stat.S_IFLNK,
        stat.S_IFCHR,
        stat.S_IFBLK,
        stat.S_IFIFO,
        stat.S_IFSOCK,
    )


def _inspect_archive(
    archive: bytes,
) -> tuple[tuple[ArchiveMemberObservation, bytes], ...]:
    if not archive.startswith((b"PK\x03\x04", b"PK\x05\x06")):
        raise SourceAcquisitionError("archive lacks a valid ZIP signature")
    try:
        handle = zipfile.ZipFile(io.BytesIO(archive), "r")
    except (zipfile.BadZipFile, OSError, ValueError) as exc:
        raise SourceAcquisitionError("archive is not a valid bounded ZIP") from exc
    with handle:
        infos = handle.infolist()
        names = tuple(_canonical_member_path(info.filename) for info in infos)
        expected_names = (WIKITEXT103_DIRECTORY,) + WIKITEXT103_MEMBER_PATHS
        if names != expected_names:
            raise SourceAcquisitionError(
                "archive member inventory/order does not match the frozen four entries"
            )
        if len({name.casefold() for name in names}) != len(names):
            raise SourceAcquisitionError("archive member names have a case collision")
        directory = infos[0]
        if not directory.is_dir() or directory.file_size != 0:
            raise SourceAcquisitionError(
                "archive directory entry is not an empty directory"
            )
        if _entry_is_redirect(directory):
            raise SourceAcquisitionError("archive directory entry is a redirect/device")
        rows: list[tuple[ArchiveMemberObservation, bytes]] = []
        total_size = 0
        for split, expected_name, info in zip(
            WIKITEXT103_SPLITS,
            WIKITEXT103_MEMBER_PATHS,
            infos[1:],
            strict=True,
        ):
            if info.filename != expected_name or info.is_dir():
                raise SourceAcquisitionError("archive member path/type is invalid")
            if _entry_is_redirect(info):
                raise SourceAcquisitionError(
                    "archive member must be a regular nonlink file"
                )
            if info.compress_type not in (
                zipfile.ZIP_STORED,
                zipfile.ZIP_DEFLATED,
            ):
                raise SourceAcquisitionError(
                    "archive member compression is unsupported"
                )
            if info.flag_bits & 0x9:
                raise SourceAcquisitionError(
                    "archive encryption/data-descriptor flags are forbidden"
                )
            if (
                info.file_size <= 0
                or info.file_size > MAXIMUM_MEMBER_BYTES
                or info.compress_size <= 0
                or info.file_size / info.compress_size > MAXIMUM_COMPRESSION_RATIO
            ):
                raise SourceAcquisitionError(
                    "archive member size or compression ratio is invalid"
                )
            total_size += info.file_size
            if total_size > MAXIMUM_TOTAL_MEMBER_BYTES:
                raise SourceAcquisitionError(
                    "archive total uncompressed size exceeds the bound"
                )
            try:
                payload = handle.read(info)
            except (zipfile.BadZipFile, OSError, RuntimeError, ValueError) as exc:
                raise SourceAcquisitionError(
                    "archive member extraction/CRC validation failed"
                ) from exc
            if len(payload) != info.file_size:
                raise SourceAcquisitionError(
                    "archive member extracted size does not match central directory"
                )
            recomputed_crc = zlib.crc32(payload) & 0xFFFFFFFF
            values = (
                "wt103-archive-member-v1",
                split,
                info.filename,
                info.file_size,
                info.compress_size,
                info.compress_type,
                info.flag_bits,
                info.CRC,
                recomputed_crc,
                hashlib.sha256(payload).hexdigest(),
            )
            rows.append(
                (
                    ArchiveMemberObservation(*values, _digest(_MEMBER_DOMAIN, values)),
                    payload,
                )
            )
    return tuple(rows)


def _root(path: Path) -> Path:
    if not isinstance(path, Path):
        raise SourceAcquisitionError("staging root must be a Path")
    declared = path.absolute()
    _require_regular_directory(
        declared,
        field="staging root",
    )
    try:
        resolved = declared.resolve(strict=True)
    except OSError as exc:
        raise SourceAcquisitionError(
            f"staging root identity cannot be resolved: {exc}"
        ) from exc
    if resolved != declared:
        raise SourceAcquisitionError(
            "staging root resolved identity differs from its declared path"
        )
    if any("v3_transformer" in part.casefold() for part in resolved.parts):
        raise SourceAcquisitionError("V3 cache roots are quarantined")
    return resolved


def _is_redirect_or_reparse(path: Path, status: object) -> bool:
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if stat.S_ISLNK(getattr(status, "st_mode", 0)) or bool(
        getattr(status, "st_file_attributes", 0) & reparse
    ):
        return True
    is_junction = getattr(path, "is_junction", None)
    try:
        return bool(is_junction is not None and is_junction())
    except OSError as exc:
        raise SourceAcquisitionError(
            f"redirect/reparse/junction metadata cannot be read: {path}: {exc}"
        ) from exc


def _require_regular_directory(path: Path, *, field: str) -> None:
    try:
        status = path.lstat()
    except OSError as exc:
        raise SourceAcquisitionError(f"{field} is unavailable: {exc}") from exc
    if not stat.S_ISDIR(status.st_mode) or _is_redirect_or_reparse(path, status):
        raise SourceAcquisitionError(
            f"{field} must be a regular directory, not a redirect/reparse/junction"
        )


def _require_declared_identity(path: Path, *, field: str) -> None:
    try:
        resolved = path.resolve(strict=False)
    except OSError as exc:
        raise SourceAcquisitionError(
            f"{field} identity cannot be resolved: {exc}"
        ) from exc
    if resolved != path:
        raise SourceAcquisitionError(
            f"{field} resolved identity differs from its declared path"
        )
    if any("v3_transformer" in part.casefold() for part in resolved.parts):
        raise SourceAcquisitionError("V3 cache paths are quarantined")


def _provision_directory(root: Path, relative_path: str) -> None:
    relative = PurePosixPath(relative_path)
    if (
        relative.is_absolute()
        or any(part in ("", ".", "..") for part in relative.parts)
        or "\\" in relative_path
    ):
        raise SourceAcquisitionError("staging directory path is noncanonical")
    current = root
    for part in relative.parts:
        current = current / part
        try:
            current.mkdir()
        except FileExistsError:
            pass
        except OSError as exc:
            raise SourceAcquisitionError(
                f"staging directory cannot be provisioned: {current}: {exc}"
            ) from exc
        _require_regular_directory(
            current,
            field="staging directory component",
        )
        _require_declared_identity(
            current,
            field="staging directory component",
        )


def _provision_staging_layout(root: Path) -> None:
    for relative_path in (
        "staged/archive",
        "staged/source",
        "staged/splits/train",
        "staged/splits/validation",
        "staged/splits/test",
    ):
        _provision_directory(root, relative_path)


def _target(root: Path, relative_path: str) -> Path:
    posix = PurePosixPath(relative_path)
    windows = PureWindowsPath(relative_path)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or bool(windows.root)
        or any(part in ("", ".", "..") for part in posix.parts)
        or "\\" in relative_path
    ):
        raise SourceAcquisitionError("staged payload path is noncanonical")
    target = root / Path(*posix.parts)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise SourceAcquisitionError("staged payload path escapes its root") from exc
    current = root
    for part in posix.parts[:-1]:
        current = current / part
        try:
            status = current.lstat()
        except FileNotFoundError:
            break
        except OSError as exc:
            raise SourceAcquisitionError(
                f"staged path component metadata is unavailable: {exc}"
            ) from exc
        if not stat.S_ISDIR(status.st_mode) or _is_redirect_or_reparse(current, status):
            raise SourceAcquisitionError(
                "staged path component is a redirect/reparse/junction"
            )
        _require_declared_identity(
            current,
            field="staged path component",
        )
    try:
        status = target.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise SourceAcquisitionError(
            f"staged target metadata is unavailable: {exc}"
        ) from exc
    else:
        if not stat.S_ISREG(status.st_mode) or _is_redirect_or_reparse(target, status):
            raise SourceAcquisitionError("staged target must be a regular nonlink file")
    _require_declared_identity(target, field="staged target")
    return target


def _regular_bytes(path: Path, *, size: int, sha256: str) -> bytes:
    try:
        status = path.lstat()
    except OSError as exc:
        raise SourceAcquisitionError(f"staged payload is missing: {exc}") from exc
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if (
        not stat.S_ISREG(status.st_mode)
        or stat.S_ISLNK(status.st_mode)
        or bool(getattr(status, "st_file_attributes", 0) & reparse)
    ):
        raise SourceAcquisitionError("staged payload must be a regular nonlink file")
    if status.st_size != size:
        raise SourceAcquisitionError("staged payload size does not match")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise SourceAcquisitionError(
            f"staged payload cannot be reopened: {exc}"
        ) from exc
    if hashlib.sha256(payload).hexdigest() != sha256:
        raise SourceAcquisitionError("staged payload hash does not match")
    return payload


def _publish(backend: SourceDurabilityBackend, path: Path, payload: bytes) -> None:
    if not callable(getattr(backend, "publish_bytes", None)):
        raise SourceAcquisitionError("durability backend must expose publish_bytes")
    try:
        backend.publish_bytes(path, payload)
    except SourceAcquisitionError:
        raise
    except Exception as exc:
        raise SourceAcquisitionError(
            f"durability-backed publication failed: {exc}"
        ) from exc
    _regular_bytes(
        path,
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def stage_wikitext103_acquisition_record(
    request: StagedAcquisitionRequest,
    *,
    http_client: BoundedHttpClient,
    durability_backend: SourceDurabilityBackend,
) -> StagedWikiText103AcquisitionRecord:
    """Stage and return the richer nonproduction operational record."""

    if type(request) is not StagedAcquisitionRequest:
        raise SourceAcquisitionError(
            "request must be an exact StagedAcquisitionRequest"
        )
    request.__post_init__()
    if request.allow_network is not True:
        raise SourceAcquisitionError(
            "allow_network must be explicitly true for source-lock staging"
        )
    if not callable(getattr(http_client, "fetch", None)):
        raise SourceAcquisitionError("http_client must expose bounded fetch")
    root = _root(request.staging_root)
    _provision_staging_layout(root)
    archive_response = _validate_http(
        http_client.fetch(
            request.archive_request_url,
            maximum_bytes=MAXIMUM_ARCHIVE_BYTES,
        ),
        expected_request_url=request.archive_request_url,
        maximum_bytes=MAXIMUM_ARCHIVE_BYTES,
        kind="archive",
    )
    archive_content_type = _content_type(archive_response)
    if archive_content_type is not None and (
        archive_content_type not in _ACCEPTED_ZIP_CONTENT_TYPES
    ):
        raise SourceAcquisitionError("archive content type is not accepted")
    if not archive_response.body.startswith((b"PK\x03\x04", b"PK\x05\x06")):
        raise SourceAcquisitionError(
            "archive content type/signature rule is not satisfied"
        )
    member_payloads = _inspect_archive(archive_response.body)

    source_response = _validate_http(
        http_client.fetch(
            request.source_page_request_url,
            maximum_bytes=MAXIMUM_SOURCE_PAGE_BYTES,
        ),
        expected_request_url=request.source_page_request_url,
        maximum_bytes=MAXIMUM_SOURCE_PAGE_BYTES,
        kind="source page",
    )
    source_content_type = _content_type(source_response)
    if source_content_type not in _ACCEPTED_HTML_CONTENT_TYPES:
        raise SourceAcquisitionError("source page content type is not accepted")
    try:
        source_response.body.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise SourceAcquisitionError("source page is not strict UTF-8") from exc
    license_observation = _extract_license(source_response.body)

    archive_relative = f"staged/archive/{archive_response.body_sha256}.zip"
    source_relative = f"staged/source/{source_response.body_sha256}.html"
    _publish(
        durability_backend,
        _target(root, archive_relative),
        archive_response.body,
    )
    _publish(
        durability_backend,
        _target(root, source_relative),
        source_response.body,
    )
    members: list[ArchiveMemberObservation] = []
    sealed_splits: list[SealedSplitRef] = []
    for member, payload in member_payloads:
        relative = f"staged/splits/{member.split}/{member.sha256}.raw"
        _publish(durability_backend, _target(root, relative), payload)
        values = (
            "wt103-sealed-split-ref-v1",
            "nonproduction_staged_observation",
            member.split,
            member.observation_sha256,
            member.uncompressed_size_bytes,
            member.sha256,
            relative,
        )
        members.append(member)
        sealed_splits.append(
            SealedSplitRef(*values, _digest(_SEALED_SPLIT_DOMAIN, values))
        )
    canonical_observation = StagedWikiText103AcquisitionObservation.create(
        archive_sha256=archive_response.body_sha256,
        central_directory_sha256=_digest(
            _CENTRAL_DIRECTORY_DOMAIN,
            tuple(member.identity for member in members),
        ),
        source_page_sha256=source_response.body_sha256,
        license_raw_slice_sha256=license_observation.raw_slice_sha256,
    )
    values = (
        "wt103-staged-acquisition-v1",
        "nonproduction_staged_observation",
        canonical_observation,
        archive_response.request_url,
        archive_response.final_url,
        archive_response.redirect_chain,
        archive_response.body_size_bytes,
        archive_response.body_sha256,
        archive_relative,
        source_response.request_url,
        source_response.final_url,
        source_response.redirect_chain,
        source_response.body_size_bytes,
        source_response.body_sha256,
        source_relative,
        license_observation,
        tuple(members),
        tuple(sealed_splits),
    )
    return StagedWikiText103AcquisitionRecord(*values, _digest(_STAGED_DOMAIN, values))


def stage_wikitext103_acquisition(
    request: StagedAcquisitionRequest,
    *,
    http_client: BoundedHttpClient,
    durability_backend: SourceDurabilityBackend,
) -> StagedWikiText103AcquisitionObservation:
    """Stage sources and return only the canonical Task 1 observation."""

    return stage_wikitext103_acquisition_record(
        request,
        http_client=http_client,
        durability_backend=durability_backend,
    ).observation


def reopen_staged_wikitext103(
    *,
    observation: StagedWikiText103AcquisitionRecord,
    staging_root: Path,
) -> StagedWikiText103AcquisitionRecord:
    """Revalidate every staged byte without issuing a data capability."""

    if type(observation) is not StagedWikiText103AcquisitionRecord:
        raise SourceAcquisitionError(
            "observation must be an exact staged acquisition record"
        )
    observation.__post_init__()
    root = _root(staging_root)
    archive = _regular_bytes(
        _target(root, observation.archive_relative_path),
        size=observation.archive_size_bytes,
        sha256=observation.archive_sha256,
    )
    source_page = _regular_bytes(
        _target(root, observation.source_page_relative_path),
        size=observation.source_page_size_bytes,
        sha256=observation.source_page_sha256,
    )
    if _extract_license(source_page) != observation.license:
        raise SourceAcquisitionError(
            "source-page license observation changed during offline reuse"
        )
    archive_rows = _inspect_archive(archive)
    if tuple(row for row, _ in archive_rows) != observation.members:
        raise SourceAcquisitionError(
            "archive member observations changed during offline reuse"
        )
    for sealed, (_, archive_payload) in zip(
        observation.sealed_splits, archive_rows, strict=True
    ):
        staged_payload = _regular_bytes(
            _target(root, sealed.cache_relative_path),
            size=sealed.payload_size_bytes,
            sha256=sealed.payload_sha256,
        )
        if staged_payload != archive_payload:
            raise SourceAcquisitionError(
                "sealed split bytes differ from the staged archive member"
            )
    return observation


__all__ = [
    "ArchiveMemberIdentity",
    "ArchiveMemberObservation",
    "BoundedHttpClient",
    "BoundedHttpObservation",
    "HttpRedirectObservation",
    "LicenseObservation",
    "MAXIMUM_ARCHIVE_BYTES",
    "MAXIMUM_COMPRESSION_RATIO",
    "MAXIMUM_LICENSE_PARAGRAPH_BYTES",
    "MAXIMUM_MEMBER_BYTES",
    "MAXIMUM_SOURCE_PAGE_BYTES",
    "MAXIMUM_TOTAL_MEMBER_BYTES",
    "RedirectHop",
    "SealedSplitRef",
    "SourceAcquisitionError",
    "SourceDurabilityBackend",
    "StagedAcquisitionRequest",
    "StagedWikiText103AcquisitionObservation",
    "StagedWikiText103AcquisitionRecord",
    "WIKITEXT103_ARCHIVE_REQUEST_URL",
    "WIKITEXT103_DIRECTORY",
    "WIKITEXT103_MEMBER_PATHS",
    "WIKITEXT103_SOURCE_PAGE_REQUEST_URL",
    "WIKITEXT103_SPLITS",
    "reopen_staged_wikitext103",
    "stage_wikitext103_acquisition",
    "stage_wikitext103_acquisition_record",
]
