r"""Free-text Crossref bibliographic search: python crossref_free.py "<query>" ["<query>" ...]; prints top 3 hits."""
import sys
from crossref_query import query
from crossref_check import summarize
for q in sys.argv[1:]:
    print("== QUERY:", q)
    for k, item in enumerate(query(q, rows=3)):
        print(f"  -- hit {k+1}: DOI {item.get('DOI')}")
        print(summarize(item))
