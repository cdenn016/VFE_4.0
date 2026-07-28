"""Lexical, nonreparse path preparation for owned artifact trees."""

from __future__ import annotations

import os
import stat
from pathlib import Path, PurePosixPath, PureWindowsPath


class OwnedArtifactPathError(ValueError):
    """An owned artifact path was ambiguous, redirected, or unsafe."""


def _is_redirect_or_reparse(path: Path, status: os.stat_result) -> bool:
    if stat.S_ISLNK(status.st_mode):
        return True
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if bool(getattr(status, "st_file_attributes", 0) & reparse):
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(os.path.normpath(os.fspath(left))) == os.path.normcase(
        os.path.normpath(os.fspath(right))
    )


def _canonical_relative_path(value: object) -> PurePosixPath:
    if type(value) is not str or not value or "\\" in value:
        raise OwnedArtifactPathError(
            "relative_path must be a nonempty canonical POSIX path"
        )
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or bool(windows.root)
        or posix.as_posix() != value
        or any(part in ("", ".", "..") for part in posix.parts)
    ):
        raise OwnedArtifactPathError(
            "relative_path is noncanonical or escapes its owned root"
        )
    return posix


def _regular_nonlink_directory(path: Path) -> None:
    try:
        status = path.lstat()
    except OSError as exc:
        raise OwnedArtifactPathError(
            f"owned directory is unavailable: {path}: {exc}"
        ) from exc
    if not stat.S_ISDIR(status.st_mode) or _is_redirect_or_reparse(path, status):
        raise OwnedArtifactPathError(
            f"owned directory must be a regular nonlink directory: {path}"
        )


def _prepare_root(root: Path) -> None:
    try:
        root.lstat()
    except FileNotFoundError:
        _regular_nonlink_directory(root.parent)
        try:
            root.mkdir()
        except FileExistsError:
            pass
        except OSError as exc:
            raise OwnedArtifactPathError(
                f"owned root creation failed: {root}: {exc}"
            ) from exc
    except OSError as exc:
        raise OwnedArtifactPathError(
            f"owned root metadata failed: {root}: {exc}"
        ) from exc
    _regular_nonlink_directory(root)


def _walk_parent(
    root: Path,
    relative_parent: PurePosixPath,
    *,
    prepare: bool,
) -> None:
    current = root
    for part in relative_parent.parts:
        current = current / part
        try:
            status = current.lstat()
        except FileNotFoundError:
            if not prepare:
                raise OwnedArtifactPathError(
                    f"owned directory is unavailable: {current}"
                ) from None
            try:
                current.mkdir()
            except FileExistsError:
                pass
            except OSError as exc:
                raise OwnedArtifactPathError(
                    f"owned directory creation failed: {current}: {exc}"
                ) from exc
            _regular_nonlink_directory(current)
            continue
        except OSError as exc:
            raise OwnedArtifactPathError(
                f"owned directory metadata failed: {current}: {exc}"
            ) from exc
        if not stat.S_ISDIR(status.st_mode) or _is_redirect_or_reparse(
            current,
            status,
        ):
            raise OwnedArtifactPathError(
                f"owned directory must be a regular nonlink directory: {current}"
            )


def owned_payload_path(
    *,
    root: Path,
    relative_path: str,
    prepare_parents: bool,
    forbidden_component_substrings: tuple[str, ...] = (),
) -> Path:
    """Return a lexical target after checking every owned path component.

    ``prepare_parents`` may create only the explicit root and descendants under
    it. The root's parent must already exist as a regular nonlink directory.
    """

    if not isinstance(root, Path) or not root.is_absolute():
        raise OwnedArtifactPathError("owned root must be an absolute Path")
    relative = _canonical_relative_path(relative_path)
    lexical_root = Path(os.path.abspath(root))
    if prepare_parents:
        _prepare_root(lexical_root)
    else:
        _regular_nonlink_directory(lexical_root)
    resolved_root = lexical_root.resolve(strict=True)
    if not _same_path(lexical_root, resolved_root):
        raise OwnedArtifactPathError(
            "owned root contains a symlink, junction, or reparse component"
        )
    _walk_parent(
        lexical_root,
        relative.parent,
        prepare=prepare_parents,
    )
    target = lexical_root / Path(*relative.parts)
    try:
        status = target.lstat()
    except FileNotFoundError:
        status = None
    except OSError as exc:
        raise OwnedArtifactPathError(
            f"owned payload metadata failed: {target}: {exc}"
        ) from exc
    if status is not None and _is_redirect_or_reparse(target, status):
        raise OwnedArtifactPathError(
            "owned payload cannot be a symlink, junction, or reparse point"
        )
    resolved_target = target.resolve(strict=False)
    if not _same_path(target, resolved_target):
        raise OwnedArtifactPathError(
            "owned payload does not preserve its declared path identity"
        )
    try:
        resolved_target.relative_to(resolved_root)
    except ValueError as exc:
        raise OwnedArtifactPathError(
            "owned payload escapes its declared root"
        ) from exc
    forbidden = tuple(item.casefold() for item in forbidden_component_substrings)
    if any(
        substring in part.casefold()
        for part in (*target.parts, *resolved_target.parts)
        for substring in forbidden
    ):
        raise OwnedArtifactPathError(
            "owned payload path contains a forbidden provenance component"
        )
    return target


def regular_nonlink_payload(
    path: Path,
    *,
    expected_size: int | None = None,
) -> os.stat_result:
    """Validate an existing payload without following a final redirect."""

    try:
        status = path.lstat()
    except OSError as exc:
        raise OwnedArtifactPathError(
            f"owned payload is unavailable: {path}: {exc}"
        ) from exc
    if not stat.S_ISREG(status.st_mode) or _is_redirect_or_reparse(path, status):
        raise OwnedArtifactPathError(
            f"owned payload must be a regular nonlink file: {path}"
        )
    if expected_size is not None and status.st_size != expected_size:
        raise OwnedArtifactPathError("owned payload size does not match")
    return status


__all__ = [
    "OwnedArtifactPathError",
    "owned_payload_path",
    "regular_nonlink_payload",
]
