r"""Recount LaTeX spacing macros \, \; \! and control spaces inside \operatorname (AUTHORING
prohibits \, \; \!).  Written with an editor rather than a shell heredoc so that backslashes in
the regular expressions are preserved.  Read-only."""
import pathlib
import re

from texscan import DOCS, doc_files, read_lines

files = []
for d in DOCS:
    for f in doc_files(d):
        if f not in files:
            files.append(f)
out = []
tot = 0
per_file = {}
for f in files:
    for n, line in enumerate(read_lines(f), 1):
        for m in re.finditer(r"\\[,;!]", line):
            # a preceding backslash would make this '\\' followed by punctuation (a line break)
            if m.start() > 0 and line[m.start() - 1] == "\\":
                continue
            tot += 1
            per_file[f] = per_file.get(f, 0) + 1
            out.append(f"{f}:{n}: ...{line[max(0, m.start() - 50):m.end() + 40]}...")
out.insert(0, f"total spacing macros: {tot}; per file: {per_file}")
pathlib.Path(__file__).with_name("spacing_recount.out.txt").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
