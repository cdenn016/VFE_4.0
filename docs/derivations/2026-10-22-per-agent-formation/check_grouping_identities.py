"""Small exact checks supporting the persistent-record grouping derivation."""

from pathlib import Path
import json
import sympy as sp


def main() -> None:
    """Check displayed finite identities; do not claim a universal flow proof."""
    root = Path(__file__).resolve().parent
    contract = json.loads((root / "problem-contract.json").read_text(encoding="utf-8"))
    eps = sp.symbols("epsilon", real=True)
    l = sp.symbols("l0:3", real=True)
    a, b = sp.Rational(1, 5), sp.Rational(2, 5)

    def divergence(x: sp.Expr, y: sp.Expr) -> sp.Expr:
        """Return KL of Bernoulli laws with log odds x and y."""
        return (x - y) * sp.tanh(x / 2) / 2 - sp.log(sp.cosh(x / 2)) + sp.log(sp.cosh(y / 2))

    fields = []
    for i in range(3):
        weights = [a * sp.exp(-divergence(l[i], 0)) if j == i else b * sp.exp(-divergence(l[i], l[j])) for j in range(3)]
        denominator = sum(weights)
        fields.append(-l[i] + sum(weights[j] * l[j] / denominator for j in range(3) if j != i))
    at_zero = {x: 0 for x in l}
    actual = sp.Matrix(fields).jacobian(l).subs(at_zero)
    expected = -sp.Rational(7, 5) * sp.eye(3) + sp.Rational(2, 5) * sp.ones(3)
    forcing = sp.Matrix([1, 1, -1])
    equilibrium = sp.Matrix([sp.Rational(15, 7), sp.Rational(15, 7), sp.Rational(5, 7)])
    checks = {
        "record_normalization_at_k_plus": sp.simplify(((sp.exp(eps / 2) + sp.exp(-eps / 2)) / (2 * sp.cosh(eps / 2)) - 1).rewrite(sp.exp)) == 0,
        "record_normalization_at_k_minus": sp.simplify(((sp.exp(-eps / 2) + sp.exp(eps / 2)) / (2 * sp.cosh(eps / 2)) - 1).rewrite(sp.exp)) == 0,
        "all_KL_first_derivatives_zero_at_uniform": all(sp.diff(divergence(l[i], l[j]), x).subs(at_zero) == 0 for i in range(3) for j in range(3) for x in l),
        "full_field_Jacobian": actual == expected,
        "equilibrium_equation": expected * equilibrium + forcing == sp.zeros(3, 1),
        "linear_eigenvalues": expected.eigenvals() == {sp.Rational(-1, 5): 1, sp.Rational(-7, 5): 2},
        "initial_and_final_selector_margins": sp.Rational(1, 8) < sp.Rational(1, 4) < sp.Rational(1, 3),
    }
    result = {key: contract[key] for key in ("schema_version", "contract_id", "target_digest")}
    result.update({"scope": "Exact finite algebra checks only; no simulation, basin computation, or full formation test.", "checks": {key: bool(value) for key, value in checks.items()}, "all_passed": all(checks.values())})
    (root / "grouping-identity-checks.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
