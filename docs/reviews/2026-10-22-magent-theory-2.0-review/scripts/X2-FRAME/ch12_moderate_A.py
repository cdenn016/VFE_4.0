r"""X2-FRAME: the manuscript's six-agent configuration (-A-2,-A+2,A-2,A+2,R-2,R+2)
at moderate core separation A, full limit field vs null field (peer forces
deleted), same selector. Exhibits whether peer forces become necessary for the
second-stage (core-parent) join when the scale hierarchy is weakened.
"""
import numpy as np, pathlib
src = pathlib.Path(__file__).with_name("ch12_null_model_events.py").read_text(encoding="utf-8")
ns = {}; exec(src.split("for A, R, T in")[0], ns)
run, field_full, field_null = ns["run"], ns["field_full"], ns["field_null"]
for A in [2.5, 3.0, 4.0, 5.0]:
    R = 40.0
    u0 = np.array([-A - 2, -A + 2, A - 2, A + 2, R - 2, R + 2])
    for name, f in [("full", field_full), ("null", field_null)]:
        ev, bl, w0 = run(f, u0, 4.0, dt=2e-3)
        print(f"A={A} {name}: max init offdiag w={w0:.2e}; events={ev}")
