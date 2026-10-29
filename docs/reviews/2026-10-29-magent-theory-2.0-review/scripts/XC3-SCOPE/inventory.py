"""Inventory of result headings (theorem, proposition, corollary, lemma, open problem,
hypothesis, definition) in the XC3-SCOPE scope files of MAgent Theory 2.0."""
import re, sys, pathlib
ROOT = pathlib.Path(r"C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0")
FILES = ["chapters/03_dynamics.tex","chapters/04_prediction.tex","chapters/05_inheritance.tex",
 "chapters/06_fusion.tex","chapters/06_outlook.tex","chapters/07_smooth_closure.tex",
 "chapters/08_smooth_erasure.tex","chapters/09_public_compression.tex","chapters/10_boundary_closure.tex",
 "chapters/11_conditional_formation.tex","chapters/12_open_formation.tex",
 "chapters/13_per_agent_inheritance.tex","chapters/14_collective_local_parent.tex",
 "chapters/15_contextual_coupling.tex","chapters/16_link_base_transport.tex",
 "chapters/17_communication_links.tex","chapters/18_link_dynamics.tex",
 "chapters/19_curvature_hierarchy.tex","appendices/conditional_memory.tex",
 "appendices/interacting_memory.tex"]
kinds = sys.argv[1].split(",") if len(sys.argv) > 1 else ["theorem","proposition","corollary","lemma","openproblem"]
BS = chr(92)
pat = re.compile(BS + BS + "(" + "|".join(kinds) + r")heading\{((?:[^{}]|\{[^{}]*\})*)\}\{([^}]*)\}")
tot = {}
for f in FILES:
    txt = (ROOT / f).read_text(encoding="utf-8")
    lines = txt.splitlines()
    print("==", f)
    for n, line in enumerate(lines, 1):
        for m in pat.finditer(line):
            k = m.group(1)
            tot[k] = tot.get(k, 0) + 1
            print(f"  {n:5d} {k[:5]:5s} {m.group(3):45s} {m.group(2)[:70]}")
print(tot)
