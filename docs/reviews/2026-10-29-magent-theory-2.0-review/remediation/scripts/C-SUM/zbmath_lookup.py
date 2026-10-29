"""Look up bibliographic records on zbMATH Open, one query at a time."""
import json, sys, time, urllib.parse, urllib.request
QUERIES = [
 ("NarasimhanRamanan1963", "ti:universal connections II au:Narasimhan"),
 ("Palais1961", "ti:existence of slices au:Palais py:1961"),
 ("Kummer1981", "ti:reduced phase space au:Kummer py:1981"),
 ("GleissLeydoldStadler2003", "ti:circuit bases au:Gleiss"),
 ("Hartman2002", "ti:ordinary differential equations au:Hartman py:2002"),
 ("Lee2018", "ti:introduction to Riemannian manifolds au:Lee py:2018"),
 ("Taylor2011", "ti:partial differential equations III au:Taylor py:2011"),
 ("GolubitskySchaeffer1985", "ti:singularities and groups in bifurcation theory au:Golubitsky py:1985"),
 ("Topsoe1979", "ti:information-theoretical optimization techniques au:Topsoe"),
 ("AmariNagaoka2000", "ti:methods of information geometry au:Amari py:2000"),
 ("vanderVaart1998", "ti:asymptotic statistics au:Vaart py:1998"),
 ("MarsdenMontgomeryRatiu1990", "ti:reduction symmetry phases mechanics au:Marsden"),
]
for key, q in QUERIES:
    url = "https://api.zbmath.org/v1/document/_search?" + urllib.parse.urlencode({"search_string": q, "results_per_page": 3})
    try:
        d = json.load(urllib.request.urlopen(url, timeout=25))
    except Exception as e:
        print(key, "ERR", e); continue
    print("==", key)
    for r in (d.get("result") or [])[:3]:
        authors = [a.get("name") for a in r["contributors"]["authors"]]
        src = r.get("source", {})
        series = [(s.get("title"), s.get("volume"), s.get("year")) for s in src.get("series", [])]
        print("  ", r["title"]["title"], "|", authors, "|", src.get("source"), "|", src.get("pages"), "|", series)
    time.sleep(0.5)
