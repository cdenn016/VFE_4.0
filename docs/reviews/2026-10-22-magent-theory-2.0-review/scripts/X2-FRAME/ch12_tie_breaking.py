r"""X2-FRAME: interaction-necessary selection in the open-law routing limit field.
Core of four agents at (-6, -2+e, 2-e, 6) plus a distant pair at (R-2, R+2), R=40.
For e>0 the middle gap is the smallest initial gap. Compares the membership
history of the full field with the null field (peer forces deleted), same
selector (eps=1/10, delta=1/20). Exploratory; RK4 dt=2e-3, single thread.
"""
import numpy as np, pathlib
src = pathlib.Path(__file__).with_name("ch12_null_model_events.py").read_text(encoding="utf-8")
ns = {}; exec(src.split("for A, R, T in")[0], ns)
run, field_full, field_null = ns["run"], ns["field_full"], ns["field_null"]
for e in [0.0, 0.1, 0.2, 0.4, 0.8]:
    u0 = np.array([-6.0, -2.0 + e, 2.0 - e, 6.0, 38.0, 42.0])
    out = []
    for name, f in [("full", field_full), ("null", field_null)]:
        ev, bl, w0 = run(f, u0, 4.0, dt=2e-3)
        out.append((name, [x for x in ev if max(x[1]) < 4]))
    print(f"e={e}: max init offdiag w={w0:.2e}")
    for name, ev in out:
        print(f"   {name}: core events {ev}")
