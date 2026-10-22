"""Report new control-space occurrences (backslash followed by space), which can hide a collapsed line break."""
import re
import subprocess
import sys

root = sys.argv[1]
files = subprocess.run(["git", "-C", root, "diff", "--name-only", "--", "."], capture_output=True, text=True).stdout.split()
pat = re.compile(r"(?<!\\)\\ ")
for rel in files:
    if not rel.endswith(".tex"):
        continue
    path = rel.split("MAgent_Theory_2.0/", 1)[-1]
    new = open(f"{root}/{path}", encoding="utf-8").read()
    old = subprocess.run(["git", "-C", root, "show", f"HEAD:./{path}"], capture_output=True, text=True, encoding="utf-8").stdout
    if len(pat.findall(new)) > len(pat.findall(old)):
        for n, line in enumerate(new.split("\n"), 1):
            for m in pat.finditer(line):
                print(path, n, repr(line[max(0, m.start() - 30):m.end() + 20]))
