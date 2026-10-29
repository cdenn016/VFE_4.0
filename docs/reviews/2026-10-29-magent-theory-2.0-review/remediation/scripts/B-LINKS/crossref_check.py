"""Fetch Crossref records for the DOIs of new BibTeX entries (bibliographic verification only)."""
import json, sys, urllib.request
DOIS = sys.argv[1:]
for d in DOIS:
    try:
        with urllib.request.urlopen(f"https://api.crossref.org/works/{d}", timeout=20) as r:
            m = json.load(r)["message"]
        au = "; ".join(f"{a.get('family','')}, {a.get('given','')}" for a in m.get("author", m.get("editor", [])))
        yr = (m.get("published-print") or m.get("published-online") or m.get("issued"))["date-parts"][0][0]
        print(f"{d}\n  {au}\n  {m.get('title',[''])[0]}\n  {m.get('container-title',[''])[0] if m.get('container-title') else ''} vol={m.get('volume')} no={m.get('issue')} pp={m.get('page')} yr={yr} type={m.get('type')} publisher={m.get('publisher')}")
    except Exception as e:
        print(d, "ERROR", e)
