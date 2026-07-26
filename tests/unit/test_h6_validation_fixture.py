from __future__ import annotations

import hashlib
import importlib
import os
import struct
from pathlib import Path

import pytest


_FIXTURE_DOMAIN = b"VFE4-H6-VALIDATION-SAFETY-FIXTURE-V1\x00"
_ROW_COUNT = 4096
_ROW = struct.Struct("<QH33H")
_TOKEN_SHA256 = hashlib.sha256(b"synthetic validation tokens").hexdigest()
_DIRECTORY_MANIFEST_SHA256 = hashlib.sha256(
    b"synthetic blinded binary directory"
).hexdigest()


def _fixture_bytes() -> bytes:
    raw = bytearray(
        _FIXTURE_DOMAIN + bytes.fromhex(_TOKEN_SHA256) + struct.pack("<I", _ROW_COUNT)
    )
    for index in range(_ROW_COUNT):
        raw += _ROW.pack(
            index * 32,
            index % 32 + 1,
            *((index + offset) % 258 for offset in range(33)),
        )
    result = bytes(raw)
    assert len(result) == 311_369
    return result


def _module():
    return importlib.import_module("vfe4.h6_validation_fixture")


def _write_reference(root: Path, raw: bytes):
    module = _module()
    root.mkdir()
    payload_path = root / "validation_safety_fixture.bin"
    payload_path.write_bytes(raw)
    (root / "manifest.sha256").write_bytes(
        (_DIRECTORY_MANIFEST_SHA256 + "\n").encode("ascii")
    )
    reference = module.ValidationSafetyFixtureReference.create(
        local_payload_path=payload_path,
        binary_directory_manifest_sha256=_DIRECTORY_MANIFEST_SHA256,
        data_identity_sha256="1" * 64,
        access_policy_sha256="2" * 64,
        validation_token_sha256=_TOKEN_SHA256,
        fixture_raw_sha256=hashlib.sha256(raw).hexdigest(),
        fixture_raw_length=311_369,
        row_count=4096,
    )
    return module, reference


def test_fixture_payload_reader_opens_only_manifest_and_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _fixture_bytes()
    module, reference = _write_reference(tmp_path / "fixture-root", raw)
    opened: list[Path] = []
    original_open = os.open

    def tracking_open(path: os.PathLike[str] | str, flags: int, *args: int) -> int:
        opened.append(Path(path).resolve())
        return original_open(path, flags, *args)

    monkeypatch.setattr(
        Path,
        "rglob",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("fixture reader must not enumerate its directory")
        ),
    )
    monkeypatch.setattr(module.os, "open", tracking_open)

    payload = module.read_validation_safety_fixture_payload(reference)

    assert opened == [
        (tmp_path / "fixture-root" / "manifest.sha256").resolve(),
        (
            tmp_path
            / "fixture-root"
            / "validation_safety_fixture.bin"
        ).resolve(),
    ]
    assert type(payload).__name__ == "ValidationSafetyFixturePayload"
    assert payload.reference is reference
    assert payload.fixture_bytes == raw
    assert payload.validation_token_sha256 == _TOKEN_SHA256
    assert payload.starts == tuple(index * 32 for index in range(_ROW_COUNT))
    assert payload.real_target_counts == tuple(
        index % 32 + 1 for index in range(_ROW_COUNT)
    )
    assert type(payload.fixture_bytes) is bytes
    assert type(payload.starts) is tuple
    assert type(payload.real_target_counts) is tuple
    assert not hasattr(payload, "__dict__")
    assert not (tmp_path / "fixture-root" / "data_identity.json").exists()


def test_fixture_payload_reader_rejects_redirect_hash_header_or_row_mutation(
    tmp_path: Path,
) -> None:
    valid = _fixture_bytes()
    header_length = len(_FIXTURE_DOMAIN) + 32 + 4
    first_row = header_length
    second_row = header_length + _ROW.size

    header_mutation = bytearray(valid)
    header_mutation[0] ^= 1
    token_mutation = bytearray(valid)
    token_mutation[len(_FIXTURE_DOMAIN)] ^= 1
    row_count_mutation = bytearray(valid)
    row_count_mutation[len(_FIXTURE_DOMAIN) + 32 : header_length] = struct.pack(
        "<I", _ROW_COUNT - 1
    )
    duplicate_start_mutation = bytearray(valid)
    duplicate_start_mutation[second_row : second_row + 8] = valid[
        first_row : first_row + 8
    ]
    target_count_mutation = bytearray(valid)
    target_count_mutation[first_row + 8 : first_row + 10] = b"\x00\x00"
    token_id_mutation = bytearray(valid)
    token_id_mutation[first_row + 10 : first_row + 12] = struct.pack("<H", 258)

    cases: list[tuple[str, bytes, str]] = [
        ("manifest-hash", valid, "manifest"),
        ("fixture-hash", valid, "fixture"),
        ("header", bytes(header_mutation), "header"),
        ("token-sha", bytes(token_mutation), "token"),
        ("row-count", bytes(row_count_mutation), "row"),
        ("duplicate-start", bytes(duplicate_start_mutation), "start"),
        ("target-count", bytes(target_count_mutation), "target"),
        ("token-id", bytes(token_id_mutation), "token"),
        ("truncated", valid, "length"),
        ("oversize", valid, "length"),
    ]
    for name, raw, message in cases:
        root = tmp_path / name
        module, reference = _write_reference(root, raw)
        if name == "manifest-hash":
            (root / "manifest.sha256").write_bytes((("0" * 64) + "\n").encode())
        elif name == "fixture-hash":
            mutated = bytearray(raw)
            mutated[-1] ^= 1
            (root / "validation_safety_fixture.bin").write_bytes(mutated)
        elif name == "truncated":
            (root / "validation_safety_fixture.bin").write_bytes(raw[:-1])
        elif name == "oversize":
            (root / "validation_safety_fixture.bin").write_bytes(raw + b"\x00")
        with pytest.raises(ValueError, match=message):
            module.read_validation_safety_fixture_payload(reference)

    redirect_root = tmp_path / "redirect"
    module, redirect_reference = _write_reference(redirect_root, valid)
    fixture_path = redirect_root / "validation_safety_fixture.bin"
    fixture_path.unlink()
    source = redirect_root / "fixture-source.bin"
    source.write_bytes(valid)
    os.link(source, fixture_path)
    with pytest.raises(ValueError, match="link|redirect"):
        module.read_validation_safety_fixture_payload(redirect_reference)

    direct_root = tmp_path / "direct-construction"
    module, direct_reference = _write_reference(direct_root, valid)
    payload = module.read_validation_safety_fixture_payload(direct_reference)
    direct_mutations = (
        ("starts", (False, *payload.starts[1:])),
        ("starts", (0.0, *payload.starts[1:])),
        ("starts", payload.starts[:-1]),
        (
            "real_target_counts",
            (True, *payload.real_target_counts[1:]),
        ),
        (
            "real_target_counts",
            (1.0, *payload.real_target_counts[1:]),
        ),
        ("real_target_counts", payload.real_target_counts[:-1]),
    )
    for field_name, values in direct_mutations:
        arguments = {
            "reference": payload.reference,
            "fixture_bytes": payload.fixture_bytes,
            "validation_token_sha256": payload.validation_token_sha256,
            "starts": payload.starts,
            "real_target_counts": payload.real_target_counts,
        }
        arguments[field_name] = values
        with pytest.raises(ValueError, match="parsed (starts|real-target counts)"):
            module.ValidationSafetyFixturePayload(**arguments)
