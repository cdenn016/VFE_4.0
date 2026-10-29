"""Convert hand-made proof blocks to the proof environment (text edit helper)."""
import sys

OPEN = r"\noindent\emph{Proof.}" + "\n"
SQ = r"$\square$"
for f in sys.argv[1:]:
    s = open(f, encoding="utf-8").read()
    n = 0
    while True:
        i = s.find(OPEN)
        if i < 0:
            break
        j = s.find(SQ, i)
        assert j > 0
        seg = s[i:j]
        assert r"\end{proof}" not in seg and "heading{" not in seg, (f, seg[:200])
        body = s[i + len(OPEN):j].rstrip()
        s = s[:i] + r"\begin{proof}" + "\n" + body + "\n" + r"\end{proof}" + s[j + len(SQ):]
        n += 1
    open(f, "w", encoding="utf-8").write(s)
    print(f, n, s.count(SQ))
