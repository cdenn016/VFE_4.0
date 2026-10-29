"""Extract finding blocks by ID from the 2026-10-22 lens reports (read-only)."""
import re, sys, pathlib
base = pathlib.Path(r"C:/Users/chris and christine/Desktop/VFE_4.0/docs/reviews/2026-10-22-magent-theory-2.0-review/lenses")
ids = sys.argv[1:]
for f in sorted(base.glob("*.md")):
    text = f.read_text(encoding="utf-8")
    blocks = re.split(r"(?m)^### ", text)
    for b in blocks:
        head = b.split("\n",1)[0]
        for i in ids:
            if head.startswith(i + ":") or head.startswith(i + " "):
                print("=== " + f.name + " :: ### " + b[:6000])
                print()
