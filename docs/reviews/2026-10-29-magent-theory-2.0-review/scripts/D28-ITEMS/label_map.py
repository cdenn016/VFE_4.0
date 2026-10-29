"""List result headings (type, title, label) in each working document and its chapter; read-only."""
import re
import sys

M = sys.argv[1]
PAIRS = [("Item_1_communication_links.tex", "chapters/17_communication_links.tex"),
         ("Item_2_link_dynamics.tex", "chapters/18_link_dynamics.tex"),
         ("Item_3_curvature_hierarchy.tex", "chapters/19_curvature_hierarchy.tex")]
PAT = re.compile(r"\\([a-z]+)heading\{([^}]*)\}\{([^}]*)\}")


def heads(path):
    txt = open(path, encoding="utf-8").read()
    return [(m.group(3), m.group(1), m.group(2)) for m in PAT.finditer(txt)]


for a, b in PAIRS:
    ha, hb = heads(f"{M}/{a}"), heads(f"{M}/{b}")
    da, db = {x[0]: x for x in ha}, {x[0]: x for x in hb}
    print(f"=== {a}: {len(ha)} results; {b}: {len(hb)} results")
    for lab, typ, title in ha:
        if lab in db:
            t2 = db[lab]
            flag = "" if (t2[1] == typ and t2[2] == title) else f"   [chapter: {t2[1]} '{t2[2]}']"
            print(f"  {lab:28s} {typ:12s} {title}{flag}")
        else:
            print(f"  {lab:28s} {typ:12s} {title}   [NOT IN CHAPTER]")
    for lab, typ, title in hb:
        if lab not in da:
            print(f"  {lab:28s} {typ:12s} {title}   [CHAPTER ONLY]")
