r"""Query Crossref for bibliographic metadata of DOIs and compare with the bib record.

Usage: python crossref_check.py <batch name>

For each (key, doi) in the chosen batch, fetch https://api.crossref.org/works/<doi>
and print authors, title, container, volume, issue, pages, year.  The bib fields
are parsed from the manuscript bib files (read-only).  Single-threaded, sequential,
with a short per-request timeout so that the whole batch stays under 30 s.
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(r"C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0")
BIBS = ["references.bib", "item1_refs.bib", "item2_refs.bib", "item7_refs.bib"]
UA = "XC5-LIT-review/1.0 (mailto:noreply@example.org)"


def parse_bib() -> dict[str, dict[str, str]]:
    """Parse the bib files into {key: {field: value}} with a brace-balanced scan."""
    out: dict[str, dict[str, str]] = {}
    for name in BIBS:
        p = ROOT / name
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
            key = m.group(2)
            i = m.end()
            depth = 1
            j = i
            while j < len(text) and depth > 0:
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                j += 1
            body = text[i:j - 1]
            fields: dict[str, str] = {"_type": m.group(1).lower(), "_file": name}
            for fm in re.finditer(r"(\w+)\s*=\s*", body):
                k = fm.group(1).lower()
                s = fm.end()
                if s < len(body) and body[s] == "{":
                    d = 1
                    t = s + 1
                    while t < len(body) and d > 0:
                        if body[t] == "{":
                            d += 1
                        elif body[t] == "}":
                            d -= 1
                        t += 1
                    fields[k] = " ".join(body[s + 1:t - 1].split())
                else:
                    vm = re.match(r"([^,\n]+)", body[s:])
                    if vm:
                        fields[k] = vm.group(1).strip()
            out[key] = fields
    return out


def crossref(doi: str) -> dict:
    """Return the Crossref message for one DOI, or an error dict."""
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="/")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=6) as r:
            return json.loads(r.read().decode("utf-8"))["message"]
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}


def summarize(msg: dict) -> str:
    """Format the Crossref fields that matter for a bibliography check."""
    if "error" in msg:
        return "  CROSSREF ERROR " + msg["error"]
    auth = "; ".join(f"{a.get('family','?')}, {a.get('given','')}" for a in msg.get("author", []))
    title = " ".join(msg.get("title", [""])[0].split())
    cont = (msg.get("container-title") or [""])[0]
    year = None
    for k in ("published-print", "issued", "published-online"):
        if k in msg and msg[k].get("date-parts"):
            year = msg[k]["date-parts"][0]
            break
    return (f"  CR authors: {auth}\n  CR title:   {title}\n  CR venue:   {cont}\n"
            f"  CR vol/iss/pages: {msg.get('volume')}/{msg.get('issue')}/{msg.get('page')}"
            f"  article-number: {msg.get('article-number')}\n  CR date: {year}")


BATCHES: dict[str, list[str]] = {
    "flagged": ["ONeill1966", "Procesi1976", "LeBruynProcesi1990", "Reinhart1959", "Zamir1998",
                "FelsOlver1998", "CarreiraPerpinan2006", "Abbas2009", "Lojasiewicz1963"],
}


def main(argv: list[str]) -> None:
    bib = parse_bib()
    if len(argv) == 4 and argv[1] == "doi":
        doikeys = sorted(k for k, f in bib.items() if "doi" in f)
        print(f"# {len(doikeys)} bib entries carry a DOI; checking indices {argv[2]}..{argv[3]}")
        keys = doikeys[int(argv[2]):int(argv[3])]
    else:
        keys = BATCHES.get(argv[1], argv[1:]) if len(argv) > 1 else []
    t0 = time.time()
    for key in keys:
        f = bib.get(key)
        if f is None:
            print(f"== {key}: NOT IN BIB")
            continue
        print(f"== {key} [{f['_file']}, @{f['_type']}]")
        for fld in ("author", "title", "journal", "booktitle", "volume", "number", "pages", "year", "doi"):
            if fld in f:
                print(f"  bib {fld}: {f[fld]}")
        doi = f.get("doi")
        if doi and time.time() - t0 < 24:
            print(summarize(crossref(doi)))
        elif doi:
            print("  (skipped: time budget)")
        else:
            print("  (no DOI in bib)")


if __name__ == "__main__":
    main(sys.argv)
