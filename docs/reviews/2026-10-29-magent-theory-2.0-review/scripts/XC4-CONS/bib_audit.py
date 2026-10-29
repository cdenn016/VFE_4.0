"""Bibliography audit: duplicate keys across the four .bib files, one work under two keys,
missing standard fields, cited keys absent from each driver's bibliographies, unused entries,
and author self-citations.  Read-only."""
from __future__ import annotations

import collections
import pathlib
import re
import unicodedata

from texscan import DOCS, doc_files, parse_bib, scan

OUT = pathlib.Path(__file__).with_name("bib_audit.out.txt")
out: list[str] = []
emit = out.append

BIBS = ["references.bib", "item1_refs.bib", "item2_refs.bib", "item7_refs.bib"]
DRIVER_BIBS = {
    "main": ["references.bib", "item1_refs.bib", "item2_refs.bib"],
    "supplement": ["references.bib"],
    "companion": ["references.bib", "item1_refs.bib", "item2_refs.bib"],
    "item1": ["references.bib", "item1_refs.bib"],
    "item2": ["references.bib", "item1_refs.bib", "item2_refs.bib"],
    "item3": ["references.bib", "item1_refs.bib", "item2_refs.bib"],
    "item7": ["references.bib", "item1_refs.bib", "item2_refs.bib", "item7_refs.bib"],
}

bib = {b: parse_bib(b) for b in BIBS}
for b, e in bib.items():
    emit(f"{b}: {len(e)} entries")
    for k, v in e.items():
        if v.get("_duplicate_in_same_file"):
            emit(f"  DUPLICATE KEY IN SAME FILE: {k}")

# duplicate keys across files
emit("\n== keys defined in more than one .bib file")
where = collections.defaultdict(list)
for b, e in bib.items():
    for k in e:
        where[k].append(b)
for k, bs in sorted(where.items()):
    if len(bs) > 1:
        same = len({(bib[b][k].get("title", ""), bib[b][k].get("year", "")) for b in bs}) == 1
        emit(f"  {k}: {bs}  identical title/year: {same}")
        for b in bs:
            emit(f"     {b}:{bib[b][k]['_line']} {bib[b][k].get('title','')[:90]} ({bib[b][k].get('year','')})")


def norm_title(t: str) -> str:
    t = unicodedata.normalize("NFKD", t)
    t = re.sub(r"\\[a-zA-Z]+", "", t)
    t = re.sub(r"[^a-z0-9 ]", "", t.lower())
    return " ".join(t.split())


emit("\n== possible same work under two keys (normalized title, DOI, or eprint match)")
seen_t = collections.defaultdict(list)
seen_d = collections.defaultdict(list)
for b, e in bib.items():
    for k, v in e.items():
        nt = norm_title(v.get("title", ""))
        if nt:
            seen_t[nt[:60]].append((k, b))
        for fld in ("doi", "eprint"):
            if v.get(fld):
                seen_d[v[fld].lower()].append((k, b))
for nt, ks in seen_t.items():
    if len({k for k, _ in ks}) > 1:
        emit(f"  TITLE {nt}: {ks}")
for d, ks in seen_d.items():
    if len({k for k, _ in ks}) > 1:
        emit(f"  DOI/EPRINT {d}: {ks}")
# same first author surname + year, different keys
ay = collections.defaultdict(list)
for b, e in bib.items():
    for k, v in e.items():
        a = v.get("author", "").split(" and ")[0]
        sur = a.split(",")[0].strip() if "," in a else (a.split()[-1] if a.split() else "")
        ay[(norm_title(sur), v.get("year", ""))].append((k, b, v.get("title", "")[:70]))
emit("\n== same first-author surname and year under different keys (inspect by hand)")
for (s, y), ks in sorted(ay.items()):
    if len({k for k, _, _ in ks}) > 1:
        emit(f"  {s} {y}: " + " | ".join(f"{k} [{b}] {t}" for k, b, t in ks))

REQ = {
    "article": ["author", "title", "journal", "year", "volume", "pages"],
    "book": ["author|editor", "title", "publisher", "year"],
    "inproceedings": ["author", "title", "booktitle", "year", "pages"],
    "incollection": ["author", "title", "booktitle", "publisher", "year", "pages"],
    "phdthesis": ["author", "title", "school", "year"],
    "techreport": ["author", "title", "institution", "year"],
    "misc": ["author", "title", "year"],
    "unpublished": ["author", "title", "note", "year"],
}
emit("\n== entries missing standard fields (doi absent listed separately)")
for b, e in bib.items():
    for k, v in sorted(e.items()):
        req = REQ.get(v["type"], ["author", "title", "year"])
        miss = [r for r in req if not any(v.get(x) for x in r.split("|"))]
        nodoi = not v.get("doi") and v["type"] in ("article", "inproceedings", "incollection")
        if miss or nodoi:
            emit(f"  {b}:{v['_line']} {k} [{v['type']}] missing={miss} {'no-doi' if nodoi else ''}")

emit("\n== arXiv-only or preprint entries (check for published versions)")
for b, e in bib.items():
    for k, v in sorted(e.items()):
        blob = " ".join(str(x) for x in v.values()).lower()
        if "arxiv" in blob and v["type"] in ("misc", "article", "unpublished", "techreport"):
            emit(f"  {b}:{v['_line']} {k} [{v['type']}] journal={v.get('journal','')!r} note={v.get('note','')[:80]!r} eprint={v.get('eprint','')}")

emit("\n== author self-citations (Dennis) and links to unpublished work")
for b, e in bib.items():
    for k, v in e.items():
        if "dennis" in v.get("author", "").lower():
            emit(f"  SELF {b}:{v['_line']} {k}: {v.get('title','')}")
        for fld in ("url", "howpublished", "note"):
            if "github" in v.get(fld, "").lower() or "zenodo" in v.get(fld, "").lower():
                emit(f"  LINK {b}:{v['_line']} {k} {fld}={v[fld]}")

emit("\n== per-document citation resolution")
cited_any = collections.Counter()
for doc in DOCS:
    files = doc_files(doc)
    _, _, _, _, cites = scan(files)
    keys = collections.Counter(k for c in cites for k in c.keys)
    for k in keys:
        cited_any[k] += keys[k]
    avail = {}
    for b in DRIVER_BIBS[doc]:
        for k in bib[b]:
            avail.setdefault(k, b)
    missing = sorted(k for k in keys if k not in avail)
    from_other = sorted(k for k in keys if k in avail and avail[k] != "references.bib")
    emit(f"  {doc}: {len(cites)} cite commands, {len(keys)} distinct keys; missing from driver bibs: {missing}")
    emit(f"     keys resolved from item bibs: {from_other}")
    # keys that resolve to different entries depending on bib order
    for k in keys:
        holders = [b for b in DRIVER_BIBS[doc] if k in bib[b]]
        if len(holders) > 1:
            emit(f"     KEY IN SEVERAL DRIVER BIBS: {k} -> {holders} (first wins in bibtex? bibtex warns 'repeated entry')")

emit("\n== entries never cited by any of the seven documents")
for b, e in bib.items():
    unused = sorted(k for k in e if cited_any[k] == 0)
    emit(f"  {b}: {len(unused)} unused: {unused}")

OUT.write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
