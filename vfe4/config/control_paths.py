"""Canonical repository control-tree path checks."""

from __future__ import annotations

from pathlib import Path


def _path_key(path: Path) -> tuple[str, ...]:
    return tuple(part.casefold() for part in path.resolve().parts)


def _is_same_or_descendant(candidate: Path, parent: Path) -> bool:
    candidate_parts = _path_key(candidate)
    parent_parts = _path_key(parent)
    return (
        len(candidate_parts) >= len(parent_parts)
        and candidate_parts[: len(parent_parts)] == parent_parts
    )


def _git_directory_from_marker(marker: Path) -> Path | None:
    if not marker.is_file():
        return None
    try:
        first_line = marker.read_text(encoding="utf-8").splitlines()[0]
    except (OSError, UnicodeError, IndexError):
        return None
    prefix = "gitdir:"
    if not first_line.casefold().startswith(prefix):
        return None
    raw = first_line[len(prefix) :].strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = marker.parent / path
    return path.resolve()


def repository_control_roots(repo_root: Path) -> tuple[Path, ...]:
    """Return lexical and resolved Git/verification control roots."""
    repository = repo_root.resolve()
    git_marker = repository / ".git"
    roots = [(repository / ".verification").resolve(), git_marker.resolve()]
    git_directory = _git_directory_from_marker(git_marker)
    if git_directory is not None:
        roots.append(git_directory)
        common_marker = git_directory / "commondir"
        if common_marker.is_file():
            try:
                raw_common = common_marker.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeError):
                raw_common = ""
            if raw_common:
                common = Path(raw_common)
                if not common.is_absolute():
                    common = git_directory / common
                roots.append(common.resolve())
    unique: dict[tuple[str, ...], Path] = {}
    for root in roots:
        unique.setdefault(_path_key(root), root)
    return tuple(unique.values())


def is_repository_control_path(path: Path, repo_root: Path) -> bool:
    """Return whether path is equal to or inside any repository control tree."""
    candidate = path.resolve()
    return any(
        _is_same_or_descendant(candidate, control)
        for control in repository_control_roots(repo_root)
    )


__all__ = ["is_repository_control_path", "repository_control_roots"]
