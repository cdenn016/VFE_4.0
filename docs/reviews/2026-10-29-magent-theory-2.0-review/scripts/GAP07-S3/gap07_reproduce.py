r"""GAP07-S3: verbatim reproduction of checks (6)-(9), (16)-(20), the first part of (21), and (25) of checks.py.

The code of these blocks is executed exactly as written (lines 232-281, 379-410, 527-599, 608 and 654-670), after the
definition blocks of lines 1-231 and 413-525, with the generator calls of checks (11), (12), (14) replayed so that the
draws of (18)-(20) align; the printed lines are compared with checks_output.txt. Lines 600-605 (the enumeration) are skipped. Check (25) needs du_ of line
608 (check (22)) and roots of check (9). Checks (10), (11)-(14), (21)-(24), (26) are rerun elsewhere at reduced size.
"""
import os, sys, io, time, contextlib
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
SRC = r"C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/item7_checks/checks.py"
OUTF = r"C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/item7_checks/checks_output.txt"
T0 = time.time()
L = open(SRC, encoding="utf-8").readlines()
stored = open(OUTF, encoding="utf-8").read().splitlines()
assert L[233].startswith("# (6)") and L[282].startswith("# (10)") and L[378].startswith("# (16)")
assert L[388].startswith("# (17)") and L[607].startswith("du_ = ") and L[653].startswith("# (25)")
ns = {"__name__": "item7_repro"}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile("".join(L[:231]), "defs1", "exec"), ns)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile("".join(L[231:281]), "checks_6_9", "exec"), ns)          # lines 232-281: short, (6), (7), (8), (9)
    exec(compile("".join(L[378:410]), "checks_16_17", "exec"), ns)        # lines 379-410: (16), (17)
rng = ns["rng"]                                                       # replay rng use of checks (11), (12), (14)
for _ in range(4): rng.random(12)
for _ in range(12): rng.random(12)
for _ in range(3): rng.random(12)
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile("".join(L[412:525]), "defs2", "exec"), ns)               # derived-selector definitions (fresh rngd)
assert L[526].startswith("# (18)") and L[585].startswith("# (21)") and L[599].startswith("RG = all_rgs(12)")
with contextlib.redirect_stdout(buf):
    exec(compile("".join(L[526:585]), "checks_18_20", "exec"), ns)        # lines 527-585: (18), (19), (20)
    exec(compile("".join(L[585:599]), "check_21_part", "exec"), ns)       # lines 586-599: (21) without the enumeration
    exec(compile(L[607], "line608", "exec"), ns)
    exec(compile("".join(L[653:670]), "check_25", "exec"), ns)            # lines 654-670: (25)
got = buf.getvalue().splitlines()
nsame = 0
for line in got:
    hit = line in stored
    nsame += hit
    print(("SAME | " if hit else "DIFF | ") + line)
print(f"{nsame} of {len(got)} printed lines identical to checks_output.txt; elapsed {time.time() - T0:.1f} s")
