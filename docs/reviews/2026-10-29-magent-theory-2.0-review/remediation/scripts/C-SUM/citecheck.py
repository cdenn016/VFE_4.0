"""Check that every citation key of each document resolves in its bibliography files.

Usage: python citecheck.py <manuscript root>
Prints, per document, the keys that are cited but missing, and duplicate keys
across the bibliography files a document loads.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

CITE = re.compile(r"\\cite[a-zA-Z*]*\s*(?:\[[^\]]*\]\s*){0,2}\{([^}]*)\}")
INPUT = re.compile(r"\\(?:input|include)\{([^}]*)\}")
BIBLIO = re.compile(r"\\bibliography\{([^}]*)\}")
ENTRY = re.compile(r"^\s*@(\w+)\s*\{\s*([^,\s]+)\s*,", re.M)


def strip_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", text)


def collect(root: Path, rel: str, seen: set[Path]) -> list[tuple[Path, str]]:
    path = root / rel
    if path.suffix != ".tex":
        path = path.with_suffix(".tex")
    if not path.exists() or path in seen:
        return []
    seen.add(path)
    text = strip_comments(path.read_text(encoding="utf-8", errors="replace"))
    out = [(path, text)]
    for m in INPUT.finditer(text):
        out += collect(root, m.group(1), seen)
    return out


def bib_keys(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [k for t, k in ENTRY.findall(text) if t.lower() not in {"comment", "string", "preamble"}]


def main(root: Path) -> None:
    docs = [
        "MAgent_Theory_2.0.tex",
        "MAgent_Theory_2.0_supplement.tex",
        "MAgent_Theory_2.0_companion.tex",
        "Item_1_communication_links.tex",
        "Item_2_link_dynamics.tex",
        "Item_3_curvature_hierarchy.tex",
        "Item_7_formation_feedback.tex",
    ]
    for doc in docs:
        files = collect(root, doc, set())
        main_text = files[0][1]
        bibs = [b.strip() for m in BIBLIO.finditer(main_text) for b in m.group(1).split(",")]
        keys: dict[str, list[str]] = {}
        for b in bibs:
            for k in bib_keys(root / f"{b}.bib"):
                keys.setdefault(k, []).append(b)
        cited: dict[str, set[str]] = {}
        for path, text in files:
            for m in CITE.finditer(text):
                for k in m.group(1).split(","):
                    k = k.strip()
                    if k:
                        cited.setdefault(k, set()).add(path.relative_to(root).as_posix())
        missing = sorted(k for k in cited if k not in keys)
        dups = sorted(k for k, v in keys.items() if len(v) > 1)
        print(f"== {doc}: {len(files)} files, bibs {bibs}, {len(cited)} cited keys")
        for k in missing:
            print(f"   MISSING {k}: {sorted(cited[k])}")
        for k in dups:
            print(f"   DUPLICATE {k}: {keys[k]}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
