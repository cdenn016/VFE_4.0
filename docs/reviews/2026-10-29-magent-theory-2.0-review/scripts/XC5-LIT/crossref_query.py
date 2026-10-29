r"""Find Crossref records for bib entries that carry no DOI and print them beside the bib fields.

Usage: python crossref_query.py <key> [<key> ...]

For each key, query https://api.crossref.org/works?query.bibliographic=<author title year>&rows=2
and print the top two candidates (authors, title, venue, volume, issue, pages, year, DOI).
A candidate is evidence only after its title and authors are compared by hand.
Single-threaded, sequential, short timeouts; the batch stops at about 25 s.
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request

from crossref_check import parse_bib, summarize, UA


def query(text: str, rows: int = 2) -> list[dict]:
    """Return up to `rows` Crossref items for a bibliographic query string."""
    url = ("https://api.crossref.org/works?rows=%d&query.bibliographic=" % rows
           + urllib.parse.quote(text))
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    time.sleep(1.5)  # polite pacing after a 429 from Crossref
    try:
        with urllib.request.urlopen(req, timeout=7) as r:
            return json.loads(r.read().decode("utf-8"))["message"]["items"]
    except Exception as exc:  # noqa: BLE001
        return [{"error": repr(exc)}]


def clean(s: str) -> str:
    s = re.sub(r"\\[a-zA-Z]+\s*", " ", s)
    return " ".join(s.replace("{", "").replace("}", "").replace("$", "").split())


def main(argv: list[str]) -> None:
    bib = parse_bib()
    t0 = time.time()
    for key in argv[1:]:
        f = bib.get(key)
        if f is None:
            print(f"== {key}: NOT IN BIB")
            continue
        print(f"== {key} [{f['_file']}, @{f['_type']}]")
        for fld in ("author", "title", "journal", "booktitle", "volume", "number", "pages", "year", "doi", "eprint", "note", "url", "publisher"):
            if fld in f:
                print(f"  bib {fld}: {f[fld]}")
        if time.time() - t0 > 22:
            print("  (skipped: time budget)")
            continue
        first_author = clean(f.get("author", "").split(" and ")[0].split(",")[0])
        q = f"{first_author} {clean(f.get('title', ''))} {f.get('year', '')}"
        for k, item in enumerate(query(q)):
            print(f"  -- candidate {k + 1}: DOI {item.get('DOI')}")
            print(summarize(item))


if __name__ == "__main__":
    main(sys.argv)
