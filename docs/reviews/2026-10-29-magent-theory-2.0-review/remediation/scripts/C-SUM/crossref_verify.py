"""Fetch the Crossref record of every DOI in a .bib file and print it beside the entry.

Usage: python crossref_verify.py staged.bib
Sequential requests, one at a time.
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request

ENTRY = re.compile(r"@(\w+)\{([^,]+),(.*?)\n\}", re.S)
FIELD = re.compile(r"(\w+)\s*=\s*\{(.*)\},?\s*$", re.M)


def fetch(doi: str) -> dict | None:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="/()")
    req = urllib.request.Request(url, headers={"User-Agent": "bibcheck/1.0 (mailto:none@example.org)"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.load(r)["message"]
    except Exception as exc:  # noqa: BLE001
        print(f"   FETCH FAILED: {exc}")
        return None


def main(path: str) -> None:
    text = open(path, encoding="utf-8").read()
    for typ, key, body in ENTRY.findall(text):
        fields = {k.lower(): v for k, v in FIELD.findall(body)}
        print(f"== {key} ({typ})")
        print(f"   bib: {fields.get('author','')} | {fields.get('title','')} | {fields.get('journal', fields.get('series',''))} "
              f"| v{fields.get('volume','')} n{fields.get('number','')} p{fields.get('pages','')} | {fields.get('year','')}")
        doi = fields.get("doi")
        if not doi:
            print("   no DOI")
            continue
        m = fetch(doi)
        time.sleep(0.3)
        if not m:
            continue
        authors = "; ".join(f"{a.get('family','')}, {a.get('given','')}" for a in m.get("author", []))
        year = (m.get("published-print") or m.get("published-online") or m.get("issued") or {}).get("date-parts", [[None]])[0][0]
        print(f"   xref: {authors} | {' / '.join(m.get('title', []))} | {' / '.join(m.get('container-title', []))} "
              f"| v{m.get('volume','')} n{m.get('issue','')} p{m.get('page','')} | {year} | {m.get('publisher','')} | type {m.get('type','')}")
        if m.get("ISBN"):
            print(f"   isbn: {m['ISBN']}")


if __name__ == "__main__":
    main(sys.argv[1])
