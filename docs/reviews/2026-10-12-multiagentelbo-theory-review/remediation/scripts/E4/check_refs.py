"""Check braces, cross-reference targets, and label uniqueness for E4's files."""
import glob
import re
import sys
from collections import Counter

root = sys.argv[1]
files = sys.argv[2:]
alltex = glob.glob(root + "/*.tex")
label_re = re.compile(r"\\label\{([^}]+)\}")
head_re = re.compile(r"heading\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}\{([^}]+)\}")
counts: Counter = Counter()
for g in alltex:
    s = open(g, encoding="utf-8").read()
    counts.update(label_re.findall(s))
    counts.update(head_re.findall(s))
labels = set(counts)
print("duplicate labels:", sorted(k for k, v in counts.items() if v > 1))
for f in files:
    s = open(root + "/" + f, encoding="utf-8").read()
    t = re.sub(r"\\[{}]", "", s)
    refs = set()
    for m in re.finditer(r"\\(?:[cC]ref|eqref)\{([^}]+)\}", s):
        refs.update(r.strip() for r in m.group(1).split(","))
    print(f, "braces", t.count("{") - t.count("}"), "missing refs:", sorted(r for r in refs if r not in labels))
