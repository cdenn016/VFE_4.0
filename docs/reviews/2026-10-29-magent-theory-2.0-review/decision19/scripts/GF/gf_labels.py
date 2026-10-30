r"""Check cross-references and gf: label uniqueness across the MAgent 2.0 documents."""
import collections
import glob
import os
import re

ROOT = r"C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0"
src_path = os.path.join(ROOT, "chapters", "05_global_formation.tex")
src = open(src_path, encoding="utf8").read()
HEAD = re.compile(r"heading\{(?:[^{}]|\{[^{}]*\})*\}\{([^}]*)\}")
LAB = re.compile(r"\\label\{([^}]*)\}")
refs = set()
for g in re.findall(r"\\[cC]ref\{([^}]*)\}", src):
    refs.update(x.strip() for x in g.split(","))
defs, other_gf = set(), []
for f in glob.glob(os.path.join(ROOT, "**", "*.tex"), recursive=True):
    t = open(f, encoding="utf8", errors="ignore").read()
    labs = set(LAB.findall(t)) | set(HEAD.findall(t))
    defs |= labs
    if os.path.abspath(f) != os.path.abspath(src_path):
        other_gf += [l for l in labs if l.startswith("gf:") or l == "ch:global-formation"]
mine = LAB.findall(src) + HEAD.findall(src)
print("missing refs:", sorted(r for r in refs if r not in defs))
print("duplicate labels in chapter:", [k for k, v in collections.Counter(mine).items() if v > 1])
print("gf labels defined elsewhere:", other_gf)
print("non-gf labels in chapter:", [l for l in mine if not l.startswith("gf:") and l != "ch:global-formation"])
print(len(mine), "labels:", sorted(mine))
for line_no, line in enumerate(src.splitlines(), 1):
    if "\u2014" in line or "\u2013" in line:
        print("dash at", line_no)
