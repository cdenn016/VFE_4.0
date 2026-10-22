r"""X2-FRAME: which numbered results of the general chapters (01-05, appendices)
are cited by the construction chapters (06_fusion ... 12_open_formation, outlook)?
Counts \cref/\Cref/\ref/\eqref targets that are result labels (heading macros).
"""
import re, pathlib
BS = chr(92)
ROOT = pathlib.Path(r"C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0")
general = ["chapters/01_foundations", "chapters/02_interactions", "chapters/02_geometry",
           "chapters/03_dynamics", "chapters/04_prediction", "chapters/05_inheritance",
           "appendices/record_process", "appendices/base_connection", "appendices/determinant_example"]
constr = ["chapters/06_fusion", "chapters/07_smooth_closure", "chapters/08_smooth_erasure",
          "chapters/09_public_compression", "chapters/10_boundary_closure",
          "chapters/11_conditional_formation", "chapters/12_open_formation", "chapters/06_outlook"]
HEAD = re.compile(re.escape(BS) + r"(theorem|proposition|lemma|corollary|definition|construction|remark|openproblem|hypothesis)heading\{((?:[^{}]|\{[^{}]*\})*)\}\{([^}]*)\}")
REF = re.compile(re.escape(BS) + r"(?:cref|Cref|ref|eqref)\{([^}]*)\}")
results = []
for f in general:
    t = (ROOT / (f + ".tex")).read_text(encoding="utf-8")
    for m in HEAD.finditer(t):
        results.append((m.group(3), m.group(1), m.group(2), f))
used = {}
for f in constr:
    t = (ROOT / (f + ".tex")).read_text(encoding="utf-8")
    for m in REF.finditer(t):
        for k in m.group(1).split(","):
            used.setdefault(k.strip(), set()).add(f.split("/")[-1])
n_used = 0
for lab, kind, title, f in results:
    u = used.get(lab)
    n_used += bool(u)
    print(f"{'USED  ' if u else 'unused'} {kind:12s} {lab:32s} {sorted(u) if u else ''}")
print(f"\n{n_used} of {len(results)} general-chapter results are cited by construction chapters or the outlook.")
