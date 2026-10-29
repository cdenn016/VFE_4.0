"""List result headings (macro, title, label) in the MAgent Theory 2.0 chapters and appendices.

Usage: python extract_headings.py [kind ...]
where kind is one of theorem, proposition, corollary, lemma, definition,
openproblem, construction, remark, hypothesis, example.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(r"C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0")
BS = chr(92)
PAT = re.compile(
    re.escape(BS)
    + r"(theorem|proposition|corollary|lemma|definition|openproblem|construction|remark|hypothesis|example)heading"
    + r"\{([^}]*)\}\{([^}]*)\}"
)


def main() -> None:
    kinds = set(sys.argv[1:]) if len(sys.argv) > 1 else None
    for sub in ["chapters", "appendices"]:
        for f in sorted((ROOT / sub).glob("*.tex")):
            for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                for m in PAT.finditer(line):
                    if kinds is None or m.group(1) in kinds:
                        print(f"{sub}/{f.name}:{n} | {m.group(1)} | {m.group(2)} | {m.group(3)}")


if __name__ == "__main__":
    main()
