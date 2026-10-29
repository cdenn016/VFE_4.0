r"""List sentences in which one chapter quotes a parameter, value or range of another chapter
('used in \cref{...}', 'as in \cref{...}' with an inline formula nearby), for hand comparison.  Read-only."""
import re

from texscan import doc_files, read_lines

out = []
for f in doc_files("main"):
    for n, l in enumerate(read_lines(f), 1):
        for m in re.finditer(r"(used in|of|from|as in)\s+\\[cC]ref\{(ch:[^}]*|app:[^}]*)\}", l):
            s = l[max(0, m.start() - 160):m.end() + 20]
            if "$" in s and re.search(r"\$[^$]*[0-9][^$]*\$", s):
                out.append(f"{f}:{n}: ...{s}...")
txt = "\n".join(out)
print(txt)
open(__file__.replace(".py", ".out.txt"), "w", encoding="utf-8").write(txt + "\n")
