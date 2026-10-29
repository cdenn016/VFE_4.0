"""Sentence-level diff of two LaTeX sources (read-only). Prints removed (-) and added (+) sentences.

Usage: python diff_sentences.py OLD NEW
"""
import difflib
import re
import sys

BS = chr(92)  # backslash
SPLIT = re.compile(r"(?<=[.?!])\s+(?=[A-Z" + re.escape(BS) + r"])")


def sentences(path: str) -> list[str]:
    text = open(path, encoding="utf-8").read()
    text = re.sub(r"\s+", " ", text)
    parts = SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def main() -> None:
    a, b = sys.argv[1], sys.argv[2]
    sa, sb = sentences(a), sentences(b)
    sm = difflib.SequenceMatcher(a=sa, b=sb, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        print(f"=== {tag} old[{i1}:{i2}] new[{j1}:{j2}]")
        for s in sa[i1:i2]:
            print("- " + s)
        for s in sb[j1:j2]:
            print("+ " + s)


if __name__ == "__main__":
    main()
