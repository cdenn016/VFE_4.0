r"""G1 s4: arithmetic of the quantifier chain with the manuscript's and with sharpened early bounds.

Early phase (a >= 4, 0 <= b <= 2), from open:core:
  manuscript:  -2a <= adot <= -a,  -4b <= bdot <= -b/2
  sharpened:   -(1 + 4e^-32 + 4e^-8) a <= adot <= -a,  bdot <= -(1 - 68 e^-8) b
Late phase: invariant-region certificate eps_b = 1 (s1), crossing within 1.3429 of t_A.
Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 20 python s4_sharpened_constants.py
"""
import math

ka = 1 + 4 * math.exp(-32) + 4 * math.exp(-8)  # max decay rate of a
kb = 1 - 68 * math.exp(-8)  # min decay rate of b
S = 1.426696
a_star = math.sqrt(math.log(2))
cross = 1.342873
print(f"ka = {ka:.6f}, kb = {kb:.6f}")

# manuscript chain at A = 4096
A = 4096.0
print("manuscript constants, A = 4096:")
print(f"  tau <= 2 log 4 = {2*math.log(4):.4f}; a(tau) >= A/256 = {A/256:.1f}")
print(f"  t_A >= 0.5 log(A/4) = {0.5*math.log(A/4):.4f}; b(t_A) <= 2 (A/4)^-1/4 = {2*(A/4)**-0.25:.4f}")

# sharpened chain: smallest A with a(tau) >= 8 and b(t_A) <= eps_b = 1
tau = math.log(4) / kb
A_min_early = 8 * math.exp(ka * tau)
A_min_gap = 4 * 2 ** (ka / kb)
print("sharpened constants:")
print(f"  tau <= log 4 / kb = {tau:.4f}; a(tau) >= A e^(-ka tau) = {math.exp(-ka*tau):.4f} A")
print(f"  a(tau) >= 8 needs A >= {A_min_early:.2f}; b(t_A) <= 2 (A/4)^(-kb/ka) <= 1 needs A >= {A_min_gap:.2f}")
for A in (34.0, 64.0, 4096.0):
    tA_lo, tA_hi = math.log(A / 4) / ka, math.log(A / 4)
    bt = 2 * (A / 4) ** (-kb / ka)
    L = tA_hi + S
    print(f"  A={A:6.0f}: t_A in [{tA_lo:.4f}, {tA_hi:.4f}], b(t_A) <= {bt:.4f}, "
          f"parent crossing <= t_A + {cross:.4f} < L = t_A + S <= {L:.4f}")
    for D in (8.0, 12.0):
        R = (D + A + 6) * math.exp(L)
        print(f"      D_R = {D:4.1f}: R = {R:.4e}; admissibility h < 1/(R+3) = {1/(R+3):.3e};"
              f" far-pair tube (radius 1/2, common mode) h < {math.sqrt(0.5/(0.1283*R**3)):.3e}")
