"""Apply exact string replacements from a JSON file [[old, new], ...] to a target file; each old must occur once."""
import json
import sys

target, repfile = sys.argv[1], sys.argv[2]
s = open(target, encoding="utf-8").read()
reps = json.load(open(repfile, encoding="utf-8"))
for old, new in reps:
    n = s.count(old)
    if n != 1:
        sys.exit(f"count={n} for: {old[:90]!r}")
for old, new in reps:
    s = s.replace(old, new)
open(target, "w", encoding="utf-8").write(s)
print(f"applied {len(reps)} replacements to {target}")
