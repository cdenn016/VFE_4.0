"""Ch.06 fus:two-stages and eq:fus:open-gap-condition by exact event-driven integration in tau,
plus a demonstration that explicit Euler on the subgradient inclusion chatters at sticking."""
import numpy as np

def sticky(z0, lam=1.0):
    """Exact event-driven solution of z' in -lam sum_j n_j Sign(z_i - z_j) (masses one)."""
    blocks = [[z, 1] for z in sorted(z0)]; tau = 0.0; events = []
    while len(blocks) > 1:
        M = sum(b[1] for b in blocks); left = 0; vel = []
        for b in blocks:
            right = M - left - b[1]; vel.append(lam * (right - left)); left += b[1]
        gaps = [(blocks[k + 1][0] - blocks[k][0]) / (vel[k] - vel[k + 1]) for k in range(len(blocks) - 1)]
        k = int(np.argmin(gaps)); dt = gaps[k]
        for b, v in zip(blocks, vel): b[0] += v * dt
        tau += dt
        merged = [blocks[k][0], blocks[k][1] + blocks[k + 1][1]]
        events.append((tau, blocks[k][1], blocks[k + 1][1]))
        blocks = blocks[:k] + [merged] + blocks[k + 2:]
    return events
d, lam, kap = 1.0, 0.7, 0.4
ev = sticky([-3 * d, -2 * d, 2 * d, 3 * d], lam)
t = [np.log(1 + kap * e[0]) / kap for e in ev]
print('events (tau, masses):', [(round(e[0], 12), e[1], e[2]) for e in ev])
print('t1, t2 numeric:', t[0], t[2], ' claimed:', np.log(1 + kap * d / (2 * lam)) / kap, np.log(1 + 5 * kap * d / (4 * lam)) / kap)
rng = np.random.default_rng(5); bad = 0; checked = 0
for _ in range(20000):
    g = rng.uniform(0.1, 3, 3); H = max(g[0], g[2]); D = g[1] + (g[0] + g[2]) / 2 - 2 * H
    z = np.cumsum(np.r_[0, g]); ev = sticky(list(z), lam)
    two_pairs = ev[0][1:] == (1, 1) and ev[1][1:] == (1, 1)
    if D > 0:
        checked += 1
        tf = (2 * g[1] + g[0] + g[2]) / (8 * lam)
        if not (two_pairs and abs(ev[2][0] - tf) < 1e-9): bad += 1
print(f'D>0 cases checked: {checked}, violations of two-pair-then-merge or tau_final: {bad}')
# converse (not claimed): how often does the two-pair structure occur with D<=0?
conv = sum(1 for _ in range(20000) for g in [rng.uniform(0.1, 3, 3)]
           if g[1] + (g[0] + g[2]) / 2 - 2 * max(g[0], g[2]) <= 0
           and (lambda e: e[0][1:] == (1, 1) and e[1][1:] == (1, 1))(sticky(list(np.cumsum(np.r_[0, g])), lam)))
print('two-pair structure with D<=0 (condition is sufficient, not necessary):', conv)
# explicit Euler chatter on two unit particles z' = -lam*sign(z1-z2), z2 symmetric
for dt in (1e-2, 1e-3):
    z = 1.0; traj = []
    for _ in range(int(3.0 / dt)):
        z = z - dt * 2 * lam * np.sign(z); traj.append(z)
    tail = np.array(traj[-200:])
    print(f'explicit Euler dt={dt}: post-collision gap oscillation amplitude ~ {np.abs(tail).max():.2e} (2*lam*dt={2 * lam * dt:.1e}); exact gap is 0')
