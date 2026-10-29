r"""Compare bib fields with the Crossref fields saved by crossref_check.py.

Reads crossref_out_*.txt in this directory and flags, per entry, mismatches in
volume, issue, first and last page, year, author family names, and a crude title
similarity.  Prints only entries with at least one flag, plus a summary.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent


def norm(s: str) -> str:
    """Lowercase, strip TeX accents and braces, and fold Unicode to ASCII."""
    s = re.sub(r"\\[`'^\"~=.uvHtcdbk]\s*\{?(\w)\}?", r"\1", s)
    s = s.replace("\\L", "L").replace("\\l", "l").replace("\\o", "o").replace("\\i", "i")
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    s = s.replace("{", "").replace("}", "").replace("$", "")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split().__str__()


def words(s: str) -> list[str]:
    s = re.sub(r"\\[`'^\"~=.uvHtcdbk]\s*\{?(\w)\}?", r"\1", s)
    s = s.replace("\\L", "L").replace("\\l", "l")
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    s = s.replace("{", "").replace("}", "").replace("$", "")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split()


def parse(path: Path) -> list[dict[str, str]]:
    out = []
    cur: dict[str, str] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"== (\S+) \[", line)
        if m:
            cur = {"key": m.group(1)}
            out.append(cur)
            continue
        if cur is None:
            continue
        m = re.match(r"\s+bib (\w+): (.*)", line)
        if m:
            cur["bib_" + m.group(1)] = m.group(2)
            continue
        m = re.match(r"\s+CR ([\w/]+):\s*(.*)", line)
        if m:
            cur["cr_" + m.group(1)] = m.group(2)
    return out


def pages(s: str | None) -> tuple[str, str]:
    if not s:
        return ("", "")
    parts = re.split(r"-+|\u2013", s.replace(" ", ""))
    return (parts[0], parts[-1])


def main() -> None:
    entries = []
    for p in sorted(HERE.glob("crossref_out_*.txt")):
        entries.extend(parse(p))
    nflag = 0
    for e in entries:
        flags = []
        vip = e.get("cr_vol/iss/pages", "")
        m = re.match(r"(\S*)/(\S*)/(\S*)\s+article-number: (\S*)", vip)
        crv, cri, crp, cra = (m.groups() if m else ("", "", "", ""))
        bv = e.get("bib_volume", "")
        if bv and crv not in ("None", "") and bv != crv:
            flags.append(f"volume bib={bv} cr={crv}")
        bi = e.get("bib_number", "").replace("--", "-")
        if bi and cri not in ("None", "") and bi != cri:
            flags.append(f"issue bib={bi} cr={cri}")
        if not bi and cri not in ("None", ""):
            flags.append(f"issue missing in bib (cr={cri})")
        bp = pages(e.get("bib_pages"))
        cp = pages(crp if crp != "None" else (cra if cra != "None" else ""))
        if bp != ("", "") and cp != ("", "") and bp != cp:
            flags.append(f"pages bib={e.get('bib_pages')} cr={crp} art={cra}")
        if bp == ("", "") and cp != ("", ""):
            flags.append(f"pages missing in bib (cr={crp}, art={cra})")
        by = e.get("bib_year", "")
        cy = re.findall(r"\d{4}", e.get("cr_date", ""))
        if by and cy and by != cy[0]:
            flags.append(f"year bib={by} cr={cy[0]}")
        bt, ct = words(e.get("bib_title", "")), words(e.get("cr_title", ""))
        ratio = difflib.SequenceMatcher(None, bt, ct).ratio()
        if ratio < 0.85:
            flags.append(f"title similarity {ratio:.2f}: bib='{e.get('bib_title')}' cr='{e.get('cr_title')}'")
        bfam = [words(a.split(",")[0]) for a in e.get("bib_author", "").split(" and ")]
        cfam = [words(a.split(",")[0]) for a in e.get("cr_authors", "").split("; ")]
        bfam = [" ".join(x) for x in bfam if x]
        cfam = [" ".join(x) for x in cfam if x]
        if cfam and bfam != cfam:
            flags.append(f"authors bib={bfam} cr={cfam}")
        bj = words(e.get("bib_journal", e.get("bib_booktitle", "")))
        cj = words(e.get("cr_venue", ""))
        if cj and difflib.SequenceMatcher(None, bj, cj).ratio() < 0.6:
            flags.append(f"venue bib='{e.get('bib_journal', e.get('bib_booktitle'))}' cr='{e.get('cr_venue')}'")
        if flags:
            nflag += 1
            print(f"{e['key']}:")
            for f in flags:
                print("   - " + f)
    print(f"\n{len(entries)} entries compared; {nflag} with at least one flag")


if __name__ == "__main__":
    main()
