r"""Inventory every citation command in the MAgent Theory 2.0 sources.

Prints, per file, each citation command with its line number and a window of
context, and then lists cited keys with the bib file that defines them.
Read-only: it only reads the manuscript directory.

Usage: python cite_inventory.py <file relative to the manuscript root> ...
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(r"C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0")
CITE_RE = re.compile(r"\\(cite[a-zA-Z]*)\*?\s*(?:\[[^\]]*\]\s*){0,2}\{([^}]*)\}")
ENTRY_RE = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", re.I)
BIBS = ["references.bib", "item1_refs.bib", "item2_refs.bib", "item7_refs.bib"]


def bib_keys(path: Path) -> dict[str, str]:
    """Return a map from bib key to entry type for one bib file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return {m.group(2): m.group(1).lower() for m in ENTRY_RE.finditer(text)}


def cites(path: Path, width: int = 220) -> list[tuple[int, str, list[str], str]]:
    """Return (line, command, keys, context) for every citation command."""
    text = path.read_text(encoding="utf-8", errors="replace")
    out = []
    for m in CITE_RE.finditer(text):
        line = text.count("\n", 0, m.start()) + 1
        keys = [k.strip() for k in m.group(2).split(",") if k.strip()]
        a = max(0, m.start() - width)
        b = min(len(text), m.end() + width // 2)
        ctx = " ".join(text[a:b].split())
        out.append((line, m.group(0)[:80], keys, ctx))
    return out


def main(argv: list[str]) -> None:
    files = argv[1:]
    bibs = {name: bib_keys(ROOT / name) for name in BIBS if (ROOT / name).exists()}
    allkeys: dict[str, list[str]] = {}
    for f in files:
        p = ROOT / f
        res = cites(p)
        print(f"=== {f}: {len(res)} citation commands")
        for line, cmd, keys, ctx in res:
            print(f"  L{line}: {cmd}")
            print(f"      ... {ctx} ...")
            for k in keys:
                allkeys.setdefault(k, []).append(f"{f}:{line}")
    print("\n=== cited keys and defining bib files")
    for k in sorted(allkeys):
        where = [b for b, ks in bibs.items() if k in ks]
        print(f"  {k}: defined in {where or 'UNDEFINED'}; cited at {', '.join(allkeys[k])}")
    print(f"\n{len(allkeys)} distinct keys")
    dup = {}
    for b, ks in bibs.items():
        for k in ks:
            dup.setdefault(k, []).append(b)
    multi = {k: v for k, v in dup.items() if len(v) > 1}
    print(f"keys defined in more than one bib file: {multi}")


if __name__ == "__main__":
    main(sys.argv)
