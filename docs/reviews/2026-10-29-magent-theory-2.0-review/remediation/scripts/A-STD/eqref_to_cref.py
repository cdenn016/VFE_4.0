"""Replace \\eqref by \\cref (\\Cref after 'Equation~' or at a sentence start)."""
import re
import sys

for f in sys.argv[1:]:
    s = open(f, encoding="utf-8").read()
    s = s.replace("Equation~\\eqref{", "\\Cref{")
    s = re.sub(r"\\eqref\{([^}]*)\} and \\eqref\{([^}]*)\}", r"\\cref{\1,\2}", s)
    s = re.sub(r"(^|[.!?]\s+)\\eqref\{", lambda m: m.group(1) + "\\Cref{", s, flags=re.M)
    n = s.count("\\eqref{")
    s = s.replace("\\eqref{", "\\cref{")
    open(f, "w", encoding="utf-8").write(s)
    print(f, n)
