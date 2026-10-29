"""Structural lint of the C-SUM LaTeX files: braces, dollars, environments, banned macros."""
import re
import sys

root = sys.argv[1]
FILES = ["frontmatter.tex", "chapters/06_outlook.tex", "appendices/notation.tex",
         "companion/glossary.tex", "chapters/01_foundations.tex"]
BANNED = ["\\,", "\\;", "\\!", "---", "\u2014", "\\eqref"]
for rel in FILES:
    t = open(f"{root}/{rel}", encoding="utf-8").read()
    t = re.sub(r"(?<!\\)%.*", "", t)
    s = t.replace("\\{", "").replace("\\}", "").replace("\\$", "")
    depth = 0
    neg = False
    for c in s:
        depth += (c == "{") - (c == "}")
        neg |= depth < 0
    dollars = len(re.findall(r"\$", s)) % 2
    envs: dict[str, int] = {}
    for m in re.finditer(r"\\(begin|end)\{([A-Za-z*]+)\}", t):
        envs[m.group(2)] = envs.get(m.group(2), 0) + (1 if m.group(1) == "begin" else -1)
    bad = [b for b in BANNED if b in t]
    print(rel, "brace-depth", depth, "neg", neg, "odd$", dollars, "unbalanced-env",
          {k: v for k, v in envs.items() if v}, "banned", bad)
