r"""X2-FRAME: contrast amplification by peer forces in the open-law routing limit
field (12_open_formation.tex, eq. open:limit). For the core configuration
u=(-a-b,-a+b,a-b,a+b), report the within-pair/between-pair contrast b/a at the
times the selector of open:selector fires, for the full field and for the
null field with peer forces deleted (where b/a is constant = 2/A).
Reuses ch12_null_model_events.py. RK4, dt=1e-3, single thread.
"""
import numpy as np
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location("ev", pathlib.Path(__file__).with_name("ch12_null_model_events.py"))
src = pathlib.Path(__file__).with_name("ch12_null_model_events.py").read_text(encoding="utf-8")
ns = {}
exec(src.split("for A, R, T in")[0], ns)
rows, field_full, field_null = ns["rows"], ns["field_full"], ns["field_null"]

def contrast_at(field, A, R, t_eval, dt=1e-3):
    u = np.array([-A - 2, -A + 2, A - 2, A + 2, R - 2, R + 2]); t = 0.0; out = {}
    for te in sorted(t_eval):
        while t < te - 1e-12:
            k1 = field(u); k2 = field(u + dt / 2 * k1); k3 = field(u + dt / 2 * k2); k4 = field(u + dt * k3)
            u = u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4); t += dt
        a = (u[2] + u[3]) / 2.0; b = (u[3] - u[2]) / 2.0
        out[te] = (a, b, b / a)
    return out

for A, R, tp_full, tc_full, tp_null, tc_null in [(8.0, 200.0, 0.589, 1.974, 0.647, 1.965),
                                                (64.0, 2000.0, 0.589, 4.055, 0.647, 4.111)]:
    f = contrast_at(field_full, A, R, [tp_full, tc_full])
    n = contrast_at(field_null, A, R, [tp_null, tc_null])
    print(f"A={A:g}: initial b/a={2/A:.3e}")
    print(f"  full: at pair event b/a={f[tp_full][2]:.3e}; at core-parent event a={f[tc_full][0]:.3f}, b={f[tc_full][1]:.3e}, b/a={f[tc_full][2]:.3e}")
    print(f"  null: at pair event b/a={n[tp_null][2]:.3e}; at core-parent event a={n[tc_null][0]:.3f}, b={n[tc_null][1]:.3e}, b/a={n[tc_null][2]:.3e}")
