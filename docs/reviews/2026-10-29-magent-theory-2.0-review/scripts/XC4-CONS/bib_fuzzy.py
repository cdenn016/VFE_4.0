"""Fuzzy duplicate-work search across the four .bib files and plainnat case-folding risk.  Read-only."""
import difflib, re, itertools, pathlib
from texscan import parse_bib
BIBS = ["references.bib", "item1_refs.bib", "item2_refs.bib", "item7_refs.bib"]
ents = []
for b in BIBS:
    for k, v in parse_bib(b).items():
        ents.append((k, b, v))
def norm(t):
    t = re.sub(r"\[a-zA-Z]+|[{}]", "", t.lower())
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", t).split())
out = []
for (k1, b1, v1), (k2, b2, v2) in itertools.combinations(ents, 2):
    r = difflib.SequenceMatcher(None, norm(v1.get("title", "")), norm(v2.get("title", ""))).ratio()
    a1 = norm(v1.get("author", "").split(" and ")[0])[:12]
    a2 = norm(v2.get("author", "").split(" and ")[0])[:12]
    if r > 0.75 or (a1 and a1 == a2):
        out.append(f"{r:.2f} {k1} [{b1}] {v1.get('title','')[:60]!r} ({v1.get('year')}) <> {k2} [{b2}] {v2.get('title','')[:60]!r} ({v2.get('year')})")
out.append("\n== titles whose capitalized words are unbraced and will be lowercased by plainnat (article/inproceedings/incollection/misc)")
for k, b, v in ents:
    if v["type"] not in ("article", "inproceedings", "incollection", "misc", "techreport", "phdthesis", "unpublished"):
        continue
    t = v.get("title", "")
    # remove braced groups
    depth = 0; s = []
    for ch in t:
        if ch == "{": depth += 1; s.append(" "); continue
        if ch == "}": depth -= 1; s.append(" "); continue
        s.append(ch if depth == 0 else " ")
    bare = "".join(s)
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", bare)
    caps = [w for w in words[1:] if w[0].isupper()]
    # after a colon the first word may be capitalized legitimately (still lowercased by plainnat)
    if caps:
        out.append(f"  {b}:{v['_line']} {k} [{v['type']}]: unbraced capitals {caps[:8]} in {t[:90]!r}")
pathlib.Path(__file__).with_name("bib_fuzzy.out.txt").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
