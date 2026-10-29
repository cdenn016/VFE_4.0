"""Lowercase \\Cref at given line numbers (mid-sentence line starts) and normalize to CRLF.

Usage: python fix_linestart_cref.py FILE N1 N2 ...
"""
import sys

f = sys.argv[1]
nums = {int(a) for a in sys.argv[2:]}
data = open(f, "rb").read().replace(b"\r\n", b"\n").decode("utf-8")
lines = data.split("\n")
for k in nums:
    line = lines[k - 1]
    assert line.startswith("\\Cref{"), (k, line[:40])
    lines[k - 1] = "\\cref{" + line[len("\\Cref{"):]
out = "\n".join(lines).replace("\n", "\r\n")
open(f, "wb").write(out.encode("utf-8"))
print(f, len(nums))
