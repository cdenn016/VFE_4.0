r"""Cross-reference audit for MAgent Theory 2.0 (X2-FRAME lens).

For each of the three documents, list labels defined (via \label or the
semantic heading macros), references used (\cref, \Cref, \ref, \eqref),
undefined references, and forward references (reference position precedes
the definition position in input order). Read-only.
"""
import re, sys, pathlib

ROOT = pathlib.Path(r"C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0")
DOCS = {
    "main": ["frontmatter", "chapters/01_foundations", "chapters/02_interactions",
             "chapters/02_geometry", "chapters/03_dynamics", "chapters/04_prediction",
             "chapters/05_inheritance", "chapters/06_fusion", "chapters/07_smooth_closure",
             "chapters/08_smooth_erasure", "chapters/09_public_compression",
             "chapters/10_boundary_closure", "chapters/11_conditional_formation",
             "chapters/12_open_formation", "chapters/06_outlook",
             "appendices/record_process", "appendices/base_connection",
             "appendices/determinant_example", "appendices/notation"],
    "supplement": ["supplements/s01_probability", "supplements/s02_geometry",
                   "supplements/s03_variation", "supplements/s04_inheritance"],
    "companion": ["companion/c01_objects", "companion/c02_links", "companion/c06_geometry",
                  "companion/c03_dynamics", "companion/c04_communication",
                  "companion/c05_inheritance", "companion/c07_fusion", "companion/c08_smooth",
                  "companion/c09_public_compression", "companion/c10_boundary",
                  "companion/c11_formation", "companion/c12_open_formation",
                  "companion/glossary"],
}
BS = chr(92)
HEAD = re.compile(re.escape(BS) + r"(?:theorem|proposition|lemma|corollary|definition|construction|remark|openproblem|hypothesis)heading\{(?:[^{}]|\{[^{}]*\})*\}\{([^}]*)\}")
LAB = re.compile(re.escape(BS) + r"label\{([^}]*)\}")
REF = re.compile(re.escape(BS) + r"(?:cref|Cref|ref|eqref)\{([^}]*)\}")

for doc, files in DOCS.items():
    defs, refs = {}, []
    pos = 0
    for f in files:
        text = (ROOT / (f + ".tex")).read_text(encoding="utf-8")
        for ln, line in enumerate(text.splitlines(), 1):
            pos += 1
            for m in HEAD.finditer(line):
                defs.setdefault(m.group(1), (f, ln, pos))
            for m in LAB.finditer(line):
                defs.setdefault(m.group(1), (f, ln, pos))
            for m in REF.finditer(line):
                for key in m.group(1).split(","):
                    refs.append((key.strip(), f, ln, pos))
    undefined = [(k, f, ln) for k, f, ln, p in refs if k not in defs]
    forward = [(k, f, ln, defs[k][0], defs[k][1]) for k, f, ln, p in refs
               if k in defs and defs[k][2] > p and not k.startswith("eq") and defs[k][0] != f]
    print(f"== {doc}: {len(defs)} labels, {len(refs)} refs")
    print("  undefined:", undefined if undefined else "none")
    print("  cross-file forward refs (non-equation):")
    for row in forward:
        print("   ", row)
