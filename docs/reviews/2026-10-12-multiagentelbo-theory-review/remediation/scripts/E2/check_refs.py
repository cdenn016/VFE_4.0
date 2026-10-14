r"""Check cref targets, label uniqueness, and brace balance for the E2 files."""
import glob
import re
import os

os.chdir(r"C:\Users\chris and christine\Desktop\Research\manuscripts\MultiAgentELBO_Theory")
labels: dict[str, list[str]] = {}
head = re.compile(r"\\(?:definition|hypothesis|lemma|proposition|theorem|corollary|conjecture|openproblem|remark)heading\{(?:[^{}]|\{[^{}]*\})*\}\{([^}]*)\}")
for f in glob.glob("*.tex"):
    s = open(f, encoding="utf-8").read()
    for m in re.finditer(r"\\label\{([^}]*)\}", s):
        labels.setdefault(m.group(1), []).append(f)
    for m in head.finditer(s):
        labels.setdefault(m.group(1), []).append(f)
new = ["rem:link-reciprocal-limit", "eq:link-reciprocal-limit", "rem:link-chart-injectivity",
       "rem:link-normalizer-role", "eq:link-normalizer-integral"]
for k in new:
    print(k, labels.get(k))
for f in ["c08b_lie_links.tex", "ca_proofs_c08b.tex", "g11_latent_links.tex"]:
    s = open(f, encoding="utf-8").read()
    refs = set()
    for m in re.finditer(r"\\(?:c|C)ref\{([^}]*)\}|\\eqref\{([^}]*)\}", s):
        for r in (m.group(1) or m.group(2)).split(","):
            refs.add(r.strip())
    print(f, "missing:", sorted(r for r in refs if r not in labels),
          "brace diff:", s.count("{") - s.count("}"))
