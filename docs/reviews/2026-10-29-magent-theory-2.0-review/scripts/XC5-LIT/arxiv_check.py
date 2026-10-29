r"""Fetch arXiv metadata (title, authors, journal reference, DOI, versions) for arXiv identifiers.

Usage: python arxiv_check.py <arXiv id> [<arXiv id> ...]

Uses the public export.arxiv.org Atom API with one request for all identifiers.
Prints the fields needed to check a bib record: title, authors, first and latest
version dates, journal_ref and DOI when the authors registered them.
"""
from __future__ import annotations

import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def main(argv: list[str]) -> None:
    ids = argv[1:]
    url = ("http://export.arxiv.org/api/query?max_results=%d&id_list=" % len(ids)
           + urllib.parse.quote(",".join(ids), safe=",/"))
    req = urllib.request.Request(url, headers={"User-Agent": "XC5-LIT-review/1.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        root = ET.fromstring(r.read())
    for e in root.findall("a:entry", NS):
        aid = e.findtext("a:id", default="", namespaces=NS)
        title = " ".join(e.findtext("a:title", default="", namespaces=NS).split())
        authors = "; ".join(a.findtext("a:name", default="", namespaces=NS)
                            for a in e.findall("a:author", NS))
        pub = e.findtext("a:published", default="", namespaces=NS)
        upd = e.findtext("a:updated", default="", namespaces=NS)
        jref = e.findtext("arxiv:journal_ref", default="", namespaces=NS)
        doi = e.findtext("arxiv:doi", default="", namespaces=NS)
        comment = " ".join(e.findtext("arxiv:comment", default="", namespaces=NS).split())
        print(f"== {aid}\n  title: {title}\n  authors: {authors}\n  first: {pub}  latest: {upd}"
              f"\n  journal_ref: {jref or '-'}\n  doi: {doi or '-'}\n  comment: {comment or '-'}")


if __name__ == "__main__":
    main(sys.argv)
