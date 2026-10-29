import re, sys
root = "C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/"
keys = set()
for b in ["references.bib", "item1_refs.bib", "item2_refs.bib"]:
    keys |= set(re.findall(r"@\w+\{([^,]+),", open(root + b, encoding="utf-8").read()))
labels = set()
import glob
for f in glob.glob(root + "**/*.tex", recursive=True):
    t = open(f, encoding="utf-8").read()
    labels |= set(re.findall(r"\\label\{([^}]+)\}", t))
    labels |= set(re.findall(r"heading\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\{([^}]+)\}", t))
for f in sys.argv[1:]:
    s = open(root + f, encoding="utf-8").read()
    cited = set()
    for m in re.findall(r"\\cite[pt]?\*?(?:\[[^\]]*\])*\{([^}]*)\}", s):
        cited |= {k.strip() for k in m.split(",")}
    refs = set()
    for m in re.findall(r"\\[cC]ref\{([^}]*)\}", s):
        refs |= {k.strip() for k in m.split(",")}
    print(f, "missing keys:", sorted(cited - keys), "missing labels:", sorted(refs - labels))
