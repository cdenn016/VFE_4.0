r"""D12-SMOOTH check 2: the sufficient condition stated inside (H3) of smooth:hypotheses.

(H3) says channels "with kernel densities whose logarithms are bounded on bounded subsets and
which depend on the laws through a continuously differentiable map into L^infty with locally
Lipschitz derivative" satisfy (H3) (first variations Lipschitz into L^infty).

Test family (the 2026-10-22 G2-01 channel): Z = (0,1], rho = Lebesgue, agents i (receiver)
and j (source) share (Z, rho);
    b[mu_j](z1, z) = 1 + c(y_j) phi(z1) phi(z),  phi = cos(2 pi z),  y_j = int g dmu_j,
    g(z) = log(1/z) in L^1 minus L^infty,  c(y) = 0.5 sin(y - Y0) (any smooth |c| <= 1/2 works).
b lies in [1/2, 3/2], is normalized, and u_j -> y_j(u_j) = E_rho[g e^u]/E_rho[e^u] is C^infty
on the chart L^infty_0(rho), because v -> E_rho[g v] is a bounded linear functional on L^infty
(norm ||g||_1 = 1). So the family meets the stated sufficient condition. Its first variation is
    dD/dmu_j (z2) = -int (dmu_i/dnu)(z) b(z2, z) dz  -  c'(y_j) x_j J g(z2),
with x_j = int phi dmu_j and J = int (dmu_i/dnu) phi dz: unbounded in z2 when c' x_j J != 0.
On n-cell grids (exact cell averages of g, midpoint phi) we report sup|h - E h| (grows like
log n), the L^1(mu_j) norm of h - E h (bounded), and the chart operator norms of dy and db
(bounded uniformly in n). Grids illustrate; the continuum statement rests on the formula.
"""
import numpy as np

H_CS = 1e-30
Y0 = 1.0 + 0.8 * 0.2257  # approx y_j at the test state, so c'(y_j) is near 1/2


def setup(n):
    edges = np.linspace(0.0, 1.0, n + 1)
    zmid = 0.5 * (edges[1:] + edges[:-1])
    lo = edges[:-1]; hi = edges[1:]
    Fhi = hi - hi * np.log(hi)
    Flo = np.where(lo > 0, lo - lo * np.log(np.where(lo > 0, lo, 1.0)), 0.0)
    gbar = (Fhi - Flo) * n            # exact cell averages of -log z; gbar[0] = 1 + log n
    phi = np.cos(2 * np.pi * zmid)
    mui = 1.0 + 0.8 * np.cos(2 * np.pi * zmid) + 0.3 * np.sin(2 * np.pi * zmid); mui /= mui.sum()
    muj = 1.0 + 0.8 * np.cos(2 * np.pi * zmid); muj /= muj.sum()
    return gbar, phi, mui, muj


def D_of(mj, mi, gbar, phi):
    """D_ij = KL(mu_i || mu_j B[mu_j]); nu_z = (sum m + c (m.phi) phi_z) / n  (cell masses)."""
    n = phi.size
    y = np.sum(gbar * mj)
    c = 0.5 * np.sin(y - Y0)
    nu = (np.sum(mj) + c * np.sum(mj * phi) * phi) / n
    return np.sum(mi * np.log(mi / nu))


print(f"{'n':>6} {'sup|h-Eh|':>10} {'1+log n':>8} {'L1(mu_j)':>9} {'|dy|op':>8} {'|db|op':>8} {'c1*x*J':>9} {'formula':>9}")
rows = []
for n in [16, 64, 256, 1024, 4096]:
    gbar, phi, mui, muj = setup(n)
    h = np.zeros(n)
    base = muj.astype(complex)
    for k in range(n):
        base[k] += 1j * H_CS
        h[k] = D_of(base, mui, gbar, phi).imag / H_CS
        base[k] -= 1j * H_CS
    hc = h - np.sum(muj * h)
    y = np.sum(gbar * muj); c = 0.5 * np.sin(y - Y0); cp = 0.5 * np.cos(y - Y0)
    nu = (1.0 + c * np.sum(muj * phi) * phi) / n
    ratio = mui / nu                          # density ratio dmu_i/dnu on cells
    x_j = np.sum(phi * muj)
    J = np.sum(ratio * phi) / n               # int (dmu_i/dnu) phi dz
    src = -(np.sum(ratio) / n + c * phi * J)  # -int ratio(z) b(z2, z) dz
    chan = -cp * x_j * J * gbar               # -c'(y) x_j J g(z2)
    ferr = np.max(np.abs((src + chan) - h))
    dy_op = np.sum(np.abs(gbar - y) * muj)    # operator norm of du -> dy on the chart
    db_op = abs(cp) * dy_op                   # sup |phi(z1) phi(z)| = 1
    rows.append((n, np.max(np.abs(hc)), cp * x_j * J))
    print(f"{n:6d} {np.max(np.abs(hc)):10.4f} {1+np.log(n):8.4f} {np.sum(muj*np.abs(hc)):9.4f} {dy_op:8.4f} {db_op:8.4f} {cp*x_j*J:9.5f} {ferr:9.1e}")
slope = (rows[-1][1] - rows[0][1]) / (np.log(rows[-1][0]) - np.log(rows[0][0]))
print(f"fitted slope of sup|h-Eh| in log n: {slope:.5f}; predicted |c1 x_j J| = {abs(rows[-1][2]):.5f}")
print("sup|h - E h| grows like |c1 x_j J| (1 + log n); the L^1 norm and every quantity in the")
print("stated sufficient condition stay bounded. The stated condition does not imply (H3).")
