r"""X2-FRAME exploratory search: does the open-law routing architecture ever
select a DIFFERENT membership history with peer forces than without them?
Random initial data for the limiting field (open:limit), N=5 agents on a line,
same selector (eps=1/10, delta=1/20), horizon T=5. Reports trials where the
final partitions (as sets) differ between full and null fields.
Exploratory only; dt=2e-3 RK4, single thread, fixed seed.
"""
import numpy as np
src = open(__file__.replace("ch12_random_search.py", "ch12_null_model_events.py"), encoding="utf-8").read()
ns = {}; exec(src.split("for A, R, T in")[0], ns)
run, field_full, field_null, rows = ns["run"], ns["field_full"], ns["field_null"], ns["rows"]
rng = np.random.default_rng(7)
diff = 0; trials = 0; examples = []
for k in range(400):
    if trials >= 6:
        break
    u0 = np.sort(rng.uniform(-25, 25, 5))
    w0 = rows(u0); off = (w0 - np.diag(np.diag(w0))).max()
    if off >= 0.1:
        continue
    trials += 1
    ef, bf, _ = run(field_full, u0, 5.0, dt=2e-3)
    en, bn, _ = run(field_null, u0, 5.0, dt=2e-3)
    sf = sorted(sorted(b) for b in bf); sn = sorted(sorted(b) for b in bn)
    first_f = [e[1] for e in ef[:3]]; first_n = [e[1] for e in en[:3]]
    if sf != sn or first_f != first_n:
        diff += 1
        if len(examples) < 4:
            examples.append((np.round(u0, 2).tolist(), ef, en))
print(f"admissible trials={trials}, membership histories differing={diff}")
for ex in examples:
    print("u0=", ex[0]); print("  full:", ex[1]); print("  null:", ex[2])
