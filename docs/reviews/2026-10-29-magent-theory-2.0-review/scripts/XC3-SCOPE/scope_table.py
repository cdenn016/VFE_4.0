"""Generate the XC3-SCOPE classification table for every theorem, proposition, corollary and lemma
in chapters 03-19 and the two memory appendices.  Data: scope_table_data.py (reviewer reading)."""
import re, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from scope_table_data import C
ROOT = pathlib.Path(r"C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0")
FILES = ["chapters/03_dynamics.tex","chapters/04_prediction.tex","chapters/05_inheritance.tex",
 "chapters/06_fusion.tex","chapters/07_smooth_closure.tex","chapters/08_smooth_erasure.tex",
 "chapters/09_public_compression.tex","chapters/10_boundary_closure.tex","chapters/11_conditional_formation.tex",
 "chapters/12_open_formation.tex","chapters/13_per_agent_inheritance.tex","chapters/14_collective_local_parent.tex",
 "chapters/15_contextual_coupling.tex","chapters/16_link_base_transport.tex","chapters/17_communication_links.tex",
 "chapters/18_link_dynamics.tex","chapters/19_curvature_hierarchy.tex","appendices/conditional_memory.tex",
 "appendices/interacting_memory.tex"]
BS = chr(92); BT = chr(96); NL = chr(10)
pat = re.compile(BS + BS + "(theorem|proposition|corollary|lemma)heading" + BS + "{((?:[^{}]|" + BS + "{[^{}]*" + BS + "})*)" + BS + "}" + BS + "{([^}]*)" + BS + "}")
rows = []; missing = []
for f in FILES:
    for n, line in enumerate((ROOT / f).read_text(encoding="utf-8").splitlines(), 1):
        for mt in pat.finditer(line):
            lab = mt.group(3)
            if lab not in C:
                missing.append(lab); continue
            o, fam, reg = C[lab]
            short = f.split("/")[-1].replace(".tex", "")
            rows.append("| %s%s%s | %s:%d | %s | %s | %s |" % (BT, lab, BT, short, n, o, fam, reg))
out = ["| Label | File:line | Objective | Law family | Regularity |", "|---|---|---|---|---|"] + rows
sys.stdout.write(NL.join(out) + NL)
sys.stderr.write("rows=%d unclassified=%s" % (len(rows), missing) + NL)
