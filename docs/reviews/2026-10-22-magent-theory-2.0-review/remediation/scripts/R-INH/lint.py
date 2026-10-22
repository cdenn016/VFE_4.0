import re, sys
root = "C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/"
files = ["chapters/05_inheritance.tex", "supplements/s04_inheritance.tex", "companion/c05_inheritance.tex"]
banned = r"key insight|crucial|critically|notably|importantly|fundamental|essentially|leverag|underscore|delve|robust|comprehensive|seamless|pivotal|paramount|nuanced|landscape|realm"
for f in files:
    s = open(root + f, encoding="utf-8").read()
    envs = set(re.findall(r"\\begin\{(\w+\*?)\}", s))
    for e in envs:
        b = s.count("\\begin{" + e + "}"); n = s.count("\\end{" + e + "}")
        if b != n:
            print(f, "UNBALANCED", e, b, n)
    print(f, "brace balance", s.count("{") - s.count("}"), "dollar parity", s.count("$") % 2)
    for i, line in enumerate(s.splitlines(), 1):
        if re.search(banned, line, re.I) or "---" in line or re.search(r"\\[;,!]", line):
            print(f, i, line.strip()[:120])
