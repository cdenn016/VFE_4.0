import re, sys
root = "C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/"
fs = sys.argv[1:]
bad = [chr(92) + ",", chr(92) + ";", chr(92) + "!", "\u2014"]
words = re.compile(r"\b(crucially|notably|importantly|essentially|fundamentally|leverages?|robust|delve|pivotal)\b", re.I)
for f in fs:
    s = open(root + f, encoding="utf-8").read()
    for i, l in enumerate(s.split("\n"), 1):
        if l.lstrip().startswith("%"):
            continue
        for b in bad:
            if b in l:
                print(f, i, repr(b), l[max(0, l.index(b) - 30): l.index(b) + 20])
        if words.search(l):
            print(f, i, "word", words.search(l).group(0))
    for env in ["equation", "proof", "aligned", "enumerate"]:
        a, b = s.count("\\begin{%s}" % env), s.count("\\end{%s}" % env)
        if a != b:
            print(f, "unbalanced", env, a, b)
    print(f, "brace balance", s.count("{") - s.count("}"), "dollar parity", s.count("$") % 2)
