"""Find and optionally repair single-backslash row endings and backslash-digit corruption."""
import glob
import re
import sys

BS = chr(92)
root = sys.argv[1]
fix = "--fix" in sys.argv
files = []
for d in ("chapters", "companion", "appendices", "supplements"):
    files += glob.glob(f"{root}/{d}/*.tex") + glob.glob(f"{root}/{d}/figures/*.tex")
files += [f"{root}/frontmatter.tex"]
for p in files:
    text = open(p, encoding="utf-8").read()
    lines = text.split("\n")
    changed = False
    for n, line in enumerate(lines):
        stripped = line.rstrip()
        if stripped.endswith(BS) and not stripped.endswith(BS * 2):
            print(p, n + 1, "single-backslash row end")
            if fix:
                lines[n] = stripped + BS
                changed = True
        for m in re.finditer(r"(?<![\\])\\[0-9]", line):
            print(p, n + 1, "backslash-digit:", line[max(0, m.start() - 15):m.end() + 10])
    if changed:
        open(p, "w", encoding="utf-8").write("\n".join(lines))
