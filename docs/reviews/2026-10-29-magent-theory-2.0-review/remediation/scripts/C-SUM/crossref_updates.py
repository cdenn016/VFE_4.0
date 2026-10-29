"""Crossref and dblp checks for records updated in references.bib."""
import json, time, urllib.parse, urllib.request
H = {"User-Agent": "bibcheck/1.0 (mailto:none@example.org)"}
def get(url):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=25))
for doi in ["10.1016/j.jmva.2017.10.005", "10.1109/TMBMC.2016.2630054", "10.1023/A:1010388907793",
            "10.1007/s00220-004-1191-7", "10.1007/s10851-025-01270-w", "10.2307/2373037"]:
    m = get("https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="/():"))["message"]
    au = "; ".join(f"{a.get('family')}, {a.get('given')}" for a in m.get("author", []))
    y = (m.get("published-print") or m.get("published-online") or m.get("issued"))["date-parts"][0]
    print("==", doi, "|", au, "|", m.get("title"), "|", m.get("container-title"), "| v", m.get("volume"), "n", m.get("issue"), "p", m.get("page"), "art", m.get("article-number"), "|", y)
    time.sleep(0.3)
r = get("https://api.crossref.org/works?" + urllib.parse.urlencode({"query.bibliographic": "Gauge invariant composite fields out of connections, with examples Fournel Francois Lazzarini Masson", "rows": 2}))
for m in r["message"]["items"]:
    print("== search Fournel |", m.get("DOI"), m.get("title"), m.get("container-title"), "v", m.get("volume"), "n", m.get("issue"), "p", m.get("page"), m.get("issued"))
r = get("https://api.crossref.org/works?" + urllib.parse.urlencode({"query.bibliographic": "Kallenberg Foundations of Modern Probability third edition 2021", "rows": 2}))
for m in r["message"]["items"]:
    print("== search Kallenberg |", m.get("DOI"), m.get("title"), m.get("container-title"), "v", m.get("volume"), m.get("issued"))
for q in ["Independence clustering (without a matrix) Ryabko", "information bottleneck method Tishby Pereira Bialek"]:
    d = get("https://dblp.org/search/publ/api?" + urllib.parse.urlencode({"q": q, "format": "json", "h": 5}))
    for h in d["result"]["hits"].get("hit", []):
        i = h["info"]
        print("== dblp |", i.get("title"), "|", i.get("venue"), "|", i.get("volume"), "|", i.get("pages"), "|", i.get("year"), "|", i.get("type"), "|", i.get("ee"))
    time.sleep(1)
