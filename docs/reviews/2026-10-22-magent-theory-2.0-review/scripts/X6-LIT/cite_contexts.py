"""Print every citation command in the MAgent 2.0 sources with surrounding context."""
import re
import pathlib

root = pathlib.Path(r"C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0")
BS = chr(92)
pat = re.compile(re.escape(BS) + r"cite[a-z]*(?:\[[^\]]*\])*\{[^}]*\}")
for f in sorted(root.rglob("*.tex")):
    if "build" in f.parts:
        continue
    for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
        for m in pat.finditer(line):
            s = max(0, m.start() - 450)
            e = min(len(line), m.end() + 150)
            print(f"=== {f.relative_to(root)}:{n}  {m.group(0)}")
            print(line[s:e].strip())
            print()
