r"""Print Crossref metadata for DOIs given on the command line (not taken from the bib)."""
import sys
from crossref_check import crossref, summarize
for d in sys.argv[1:]:
    print("== DOI", d)
    print(summarize(crossref(d)))
