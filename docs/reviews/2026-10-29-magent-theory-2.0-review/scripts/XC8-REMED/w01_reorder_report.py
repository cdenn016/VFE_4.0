r"""Writer helper for lenses/XC8-REMED.md: put the high-severity finding first and renumber.

Swaps the blocks of findings 01 and 02, exchanges every reference XC8-REMED-01 <-> XC8-REMED-02,
and applies three wording corrections (the tilt conjugacy is a sufficient condition, not an iff).
"""
import re

PATH = r"C:/Users/chris and christine/Desktop/VFE_4.0/docs/reviews/2026-10-29-magent-theory-2.0-review/lenses/XC8-REMED.md"

text = open(PATH, encoding="utf-8").read()
start1 = text.index("### XC8-REMED-01:")
start2 = text.index("### XC8-REMED-02:")
start3 = text.index("### XC8-REMED-03:")
block1 = text[start1:start2]
block2 = text[start2:start3]
text = text[:start1] + block2 + block1 + text[start3:]
text = text.replace("XC8-REMED-01", "@@TMP@@").replace("XC8-REMED-02", "XC8-REMED-01").replace("@@TMP@@", "XC8-REMED-02")

repl = [
    ("which characterizes exactly the selectors that fail (F5)",
     "which gives a sharp sufficient condition for failing (F5) and shows how tilt-sensitive selectors pass it"),
    ("and gives the iff the remark is reaching for: a selector produces",
     "and gives the sharp form of the remark: a selector produces"),
    ("The exact tilt conjugacy sketched in XC8-REMED-13 can replace the remark and states precisely which selectors fail.",
     "The exact tilt conjugacy sketched in XC8-REMED-13 can replace the remark; it gives a sharp sufficient condition for failing (F5)."),
    ("also the definitions geo:det-canonical and def:public-controlled-projection",
     "also the canonical-scale statement geo:det-canonical, the definition def:public-controlled-projection"),
]
for old, new in repl:
    if old not in text:
        raise SystemExit("missing: " + old)
    text = text.replace(old, new)
open(PATH, "w", encoding="utf-8").write(text)
heads = re.findall(r"^### (XC8-REMED-\d\d): (.*)$", text, flags=re.M)
for h in heads:
    print(h[0], h[1][:90])
