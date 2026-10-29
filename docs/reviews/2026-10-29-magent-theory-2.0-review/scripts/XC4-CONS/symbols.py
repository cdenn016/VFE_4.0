r"""Symbol-usage census for overloaded letters across all seven documents (read-only).

For each symbol pattern, prints per-file counts and up to N short contexts, so that the
meaning in each chapter can be read off by hand.  Patterns are applied to the TeX source.
Usage: python symbols.py <pattern-name> [N]
"""
from __future__ import annotations

import pathlib
import re
import sys

from texscan import DOCS, doc_files, read_lines

PATTERNS = {
    # structure group versus evaluator versus other G's
    "G": r"(?<![A-Za-z\\])G(?![A-Za-z])(_[A-Za-z0-9{]|\^|\(|\[)?",
    # bare K used as a dimension, a kernel K_m, K[...], K_{...}
    "K": r"(?<![A-Za-z\\{])K(?![A-Za-z])(_\{?[A-Za-z0-9\\]+\}?|\^|\[|\(|=|\))?",
    # T as temperature, code, sending map, horizon
    "T": r"(?<![A-Za-z\\{])T(?![A-Za-z])(_\{?[A-Za-z0-9\\]+\}?|\^|\]|=|>|\\downarrow|\)|\\geq|<)?",
    "barT": r"\\bar T|\\bar\{T\}",
    "beta": r"\\beta(?![a-z])",
    "gamma": r"\\gamma(?![a-z])",
    "Gamma": r"\\Gamma(?![a-z])",
    "rho": r"\\rho(?![a-z])",
    "varrho": r"\\varrho(?![a-z])",
    "kappa": r"\\kappa(?![a-z])",
    "H": r"(?<![A-Za-z\\{])H(?![A-Za-z])(_\{?[A-Za-z0-9\\]+\}?|\^|\(|\[)?",
    "calE": r"\\mathcal E(?![a-z])(_\{?[A-Za-z0-9\\]+\}?)?",
    "calF": r"\\mathcal F(?![a-z])(_\{?[A-Za-z0-9\\]+\}?)?",
    "Omega": r"\\Omega(?![a-z])(_\{?[A-Za-z0-9\\]+\}?)?",
    "omega": r"\\omega(?![a-z])(_\{?[A-Za-z0-9\\]+\}?)?",
    "Theta": r"\\Theta(?![a-z])(_\{?[A-Za-z0-9\\]+\}?|\^)?",
    "theta": r"\\theta(?![a-z])(_\{?[A-Za-z0-9\\]+\}?)?",
    "bartheta": r"\\bar\\theta",
    "nu": r"\\nu(?![a-z])(_\{?[A-Za-z0-9\\]+\}?|\^)?",
    "mu": r"\\mu(?![a-z])(_\{?[A-Za-z0-9\\]+\}?|\^)?",
    "zeta": r"\\zeta(?![a-z])(_\{?[A-Za-z0-9\\]+\}?)?",
    "tau": r"\\tau(?![a-z])",
    "Xi": r"\\Xi(?![a-z])",
    "Pi": r"\\Pi(?![a-z])",
    "Lambda": r"\\Lambda(?![a-z])",
    "D": r"(?<![A-Za-z\\{])D(?![A-Za-z])(_\{?[A-Za-z0-9\\]+\}?|\^)?",
    "J": r"(?<![A-Za-z\\{])J(?![A-Za-z])(_\{?[A-Za-z0-9\\]+\}?|\^)?",
    "P": r"(?<![A-Za-z\\{])P(?![A-Za-z])(_\{?[A-Za-z0-9\\]+\}?|\^|\()?",
    "B": r"(?<![A-Za-z\\{])B(?![A-Za-z])(_\{?[A-Za-z0-9\\]+\}?|\^)?",
    "E": r"(?<![A-Za-z\\{])E(?![A-Za-z])(_\{?[A-Za-z0-9\\]+\}?|\^)?",
    "M": r"(?<![A-Za-z\\{])M(?![A-Za-z])(_\{?[A-Za-z0-9\\]+\}?|\^)?",
    "S": r"(?<![A-Za-z\\{])S(?![A-Za-z])(_\{?[A-Za-z0-9\\]+\}?|\^)?",
    "W": r"(?<![A-Za-z\\{])W(?![A-Za-z])(_\{?[A-Za-z0-9\\]+\}?|\^|\()?",
    "N": r"(?<![A-Za-z\\{])N(?![A-Za-z])(_\{?[A-Za-z0-9\\]+\}?|\^)?",
    "sfK": r"\\mathsf K(?![a-z])",
    "calK": r"\\mathcal K(?![a-z])",
    "calG": r"\\mathcal G(?![a-z])",
    "calL": r"\\mathcal L(?![a-z])",
    "ell": r"\\ell(?![a-z])(_\{?[A-Za-z0-9\\]+\}?)?",
    "lambda": r"\\lambda(?![a-z])(_\{?[A-Za-z0-9\\]+\}?)?",
    "sigma": r"\\sigma(?![a-z])(_\{?[A-Za-z0-9\\]+\}?)?",
    "Sigma": r"\\Sigma(?![a-z])(_\{?[A-Za-z0-9\\]+\}?)?",
    "phi": r"\\phi(?![a-z])(_\{?[A-Za-z0-9\\]+\}?)?",
    "varphi": r"\\varphi(?![a-z])",
    "psi": r"\\psi(?![a-z])",
    "chi": r"\\chi(?![a-z])",
    "pi": r"\\pi(?![a-z])(_\{?[A-Za-z0-9\\]+\}?|\^)?",
    "eta": r"\\eta(?![a-z])",
    "alpha": r"\\alpha(?![a-z])",
    "delta": r"\\delta(?![a-z])",
    "epsilon": r"\\(var)?epsilon(?![a-z])",
}

name = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 3
pat = re.compile(PATTERNS[name])
files = []
for d in DOCS:
    for f in doc_files(d):
        if f not in files and not f.startswith("MAgent_Theory_2.0"):
            files.append(f)
for d in ("item1", "item2", "item3", "item7"):
    pass
out = []
for f in files:
    lines = read_lines(f)
    hits = []
    for n, line in enumerate(lines, 1):
        # only inside math: crude test, keep everything but skip \label/\cref arguments
        clean = re.sub(r"\\(label|cref|Cref|eqref|ref|cite[a-z]*)(\[[^\]]*\])?\{[^}]*\}", "", line)
        for m in pat.finditer(clean):
            s = max(0, m.start() - 70)
            hits.append((n, clean[s:m.end() + 50].replace("\n", " ")))
    if hits:
        forms = {}
        for n, ctx in hits:
            pass
        out.append(f"== {f}: {len(hits)} hits")
        for n, ctx in hits[:N]:
            out.append(f"   {n}: ...{ctx}...")
txt = "\n".join(out)
pathlib.Path(__file__).with_name(f"symbols_{name}.out.txt").write_text(txt, encoding="utf-8")
print(txt)
