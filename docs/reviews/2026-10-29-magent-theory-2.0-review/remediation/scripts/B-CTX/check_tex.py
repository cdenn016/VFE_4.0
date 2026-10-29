r"""Balance check: environments, braces, $ parity per paragraph, banned macros."""
import re
import sys
from collections import Counter

for p in sys.argv[1:]:
    s = open(p, encoding="utf-8").read()
    s_nc = re.sub(r"(?<!\\)%.*", "", s)
    b = re.findall(r"\\(begin|end)\{(\w+\*?)\}", s_nc)
    c = Counter(b)
    bad = {e: (c[("begin", e)], c[("end", e)]) for _, e in b if c[("begin", e)] != c[("end", e)]}
    braces = s_nc.count("{") - s_nc.count(r"\{") - (s_nc.count("}") - s_nc.count(r"\}"))
    dollars = []
    for i, par in enumerate(s_nc.split("\n\n")):
        n = len(re.findall(r"(?<!\\)\$", par))
        if n % 2:
            dollars.append(par[:80].replace("\n", " "))
    banned = [m for m in [r"\,", r"\;", r"\!", "---", r"\eqref", "ELBO"] if m in s_nc]
    print(p, "env-mismatch:", bad, "brace-diff:", braces, "odd-$ paragraphs:", dollars, "banned:", banned)
