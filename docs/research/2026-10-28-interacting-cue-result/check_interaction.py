"""Exact finite-law checks for posterior-predictive communication.

All probabilities use Fraction arithmetic. No sampling, training, or device
libraries are used. The state order is (X0, X1, A0, A1, Y0, Y1, D0, D1).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from fractions import Fraction as F
from itertools import product
from pathlib import Path
from time import perf_counter

State = tuple[int, ...]
Law = dict[State, F]
PrivateLaw = dict[tuple[int, int], F]


def binary_channel(value: int, target: int, reliability: F = F(3, 4)) -> F:
    """Return the mass of a binary channel with the stated reliability."""
    return reliability if value == target else 1 - reliability


def prior() -> Law:
    """Build the normalized joint preparation, including private records."""
    result: Law = {}
    for state in product((0, 1), repeat=8):
        x0, x1, a0, a1, y0, y1, d0, d1 = state
        result[state] = (
            F(1, 2) * binary_channel(x1, x0)
            * binary_channel(a0, x0) * binary_channel(a1, x1)
            * binary_channel(y0, 1 - x0) * binary_channel(y1, 1 - x1)
            * binary_channel(d0, x0) * binary_channel(d1, x1)
        )
    assert sum(result.values()) == 1
    return result


def condition(law: Law, predicate: Callable[[State], bool]) -> Law:
    """Condition on a positive-probability event."""
    mass = sum(prob for state, prob in law.items() if predicate(state))
    assert mass > 0
    return {state: prob / mass for state, prob in law.items() if predicate(state)}


def probability(law: Law, predicate: Callable[[State], bool]) -> F:
    """Return the exact probability of an event."""
    return sum((prob for state, prob in law.items() if predicate(state)), F(0))


def local(law: Law, agent: int, private_record: int) -> Law:
    """Condition common information on the agent's actual private record."""
    return condition(law, lambda state: state[6 + agent] == private_record)


def emission(law: Law, speaker: int, message: int) -> dict[int, F]:
    """Compute kappa(message | private record, common posterior)."""
    return {
        d: probability(local(law, speaker, d), lambda state: state[4 + speaker] == message)
        for d in (0, 1)
    }


def update(law: Law, speaker: int, message: int) -> tuple[Law, F]:
    """Apply Bayes' rule to a fresh posterior-predictive message."""
    kernel = emission(law, speaker, message)
    unnormalized = {state: prob * kernel[state[6 + speaker]] for state, prob in law.items()}
    evidence = sum(unnormalized.values())
    assert evidence > 0
    posterior = {state: prob / evidence for state, prob in unnormalized.items()}
    for d in (0, 1):
        assert local(posterior, speaker, d) == local(law, speaker, d)
    return posterior, evidence


def moments(law: Law, agent: int) -> dict[str, F]:
    """Return exact binary marginal moments and nonzero local dependence."""
    px = probability(law, lambda state: state[agent] == 1)
    pa = probability(law, lambda state: state[2 + agent] == 1)
    py = probability(law, lambda state: state[4 + agent] == 1)
    pax = probability(law, lambda state: state[2 + agent] == state[agent] == 1)
    pxy = probability(law, lambda state: state[4 + agent] == state[agent] == 1)
    assert pa == F(1, 4) + px / 2
    assert py == F(3, 4) - px / 2
    assert pax - pa * px == px * (1 - px) / 2 > 0
    assert pxy - px * py == -px * (1 - px) / 2 < 0
    return {"P_X1": px, "P_A1": pa, "P_Y1": py, "Cov_A_X": pax - pa * px}


def private_marginal(law: Law) -> PrivateLaw:
    """Retain the joint private-record law, without factorizing its marginals."""
    return {
        d: probability(law, lambda state: state[6:] == d)
        for d in product((0, 1), repeat=2)
    }


def fixed_reconstruct(initial: Law, retained: PrivateLaw) -> Law:
    """Reconstruct Gamma from the fixed W(Z | D) and changing lambda(D)."""
    initial_private = private_marginal(initial)
    return {state: p / initial_private[state[6:]] * retained[state[6:]] for state, p in initial.items()}


def retained_update(initial: Law, retained: PrivateLaw, speaker: int, message: int) -> PrivateLaw:
    """Update four weights using a fixed conditional-result table."""
    fixed_result = {
        d: probability(condition(initial, lambda s: s[6:] == d), lambda s: s[4 + speaker] == message)
        for d in retained
    }
    kernel = {}
    for ds in (0, 1):
        marginal = sum(p for d, p in retained.items() if d[speaker] == ds)
        kernel[ds] = sum(p * fixed_result[d] for d, p in retained.items() if d[speaker] == ds) / marginal
    weights = {d: p * kernel[d[speaker]] for d, p in retained.items()}
    evidence = sum(weights.values())
    return {d: p / evidence for d, p in weights.items()}


def stringify(value: object) -> object:
    """Serialize exact probabilities without converting them to decimals."""
    if isinstance(value, F):
        return str(value)
    if isinstance(value, dict):
        return {str(k): stringify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [stringify(v) for v in value]
    return value


def main() -> None:
    """Check all two-round branches and the explicit record-reuse contrast."""
    start = perf_counter()
    gamma0 = prior()
    assert emission(gamma0, 1, 1) == {0: F(5, 8), 1: F(3, 8)}
    gamma1, z1 = update(gamma0, 1, 1)
    assert emission(gamma1, 0, 1) == {0: F(7, 11), 1: F(12, 31)}
    gamma2, z2 = update(gamma1, 0, 1)
    rows = {
        "agent0_D0_1_before": moments(local(gamma0, 0, 1), 0),
        "agent0_D0_1_after_peer": moments(local(gamma1, 0, 1), 0),
        "agent0_D0_1_after_own_reply": moments(local(gamma2, 0, 1), 0),
        "agent1_D1_0_before": moments(local(gamma0, 1, 0), 1),
        "agent1_D1_0_after_own_message": moments(local(gamma1, 1, 0), 1),
        "agent1_D1_0_after_reply": moments(local(gamma2, 1, 0), 1),
    }
    assert rows["agent0_D0_1_after_peer"]["P_X1"] == F(45, 62)
    assert rows["agent0_D0_1_after_peer"]["P_A1"] == F(19, 31)
    assert rows["agent0_D0_1_after_peer"]["P_Y1"] == F(12, 31)
    assert rows["agent0_D0_1_after_peer"] == rows["agent0_D0_1_after_own_reply"]
    assert rows["agent1_D1_0_before"] == rows["agent1_D1_0_after_own_message"]

    # The known two-round history likelihood is independently tabulated.
    k1 = {0: F(5, 8), 1: F(3, 8)}
    k2 = {0: F(7, 11), 1: F(12, 31)}
    direct = {s: p * k1[s[7]] * k2[s[6]] for s, p in gamma0.items()}
    norm = sum(direct.values())
    assert norm == z1 * z2
    assert {s: p / norm for s, p in direct.items()} == gamma2

    histories: dict[str, F] = {}
    for h1, h2 in product((0, 1), repeat=2):
        first, p1 = update(gamma0, 1, h1)
        second, p2 = update(first, 0, h2)
        lambda1 = retained_update(gamma0, private_marginal(gamma0), 1, h1)
        lambda2 = retained_update(gamma0, lambda1, 0, h2)
        assert fixed_reconstruct(gamma0, lambda1) == first
        assert fixed_reconstruct(gamma0, lambda2) == second
        assert sum(second.values()) == 1
        histories[f"{h1}{h2}"] = p1 * p2
        for agent, d in product((0, 1), repeat=2):
            moments(local(second, agent, d), agent)
    assert sum(histories.values()) == 1

    # A direct observation of the actual Y1 is a different model.
    physical = condition(local(gamma0, 0, 1), lambda s: s[5] == 1)
    wrong_sensor = probability(physical, lambda s: s[0] == 1)
    assert wrong_sensor == F(9, 14)

    # Two FRESH samples from the same peer share the private record D1.
    repeated, _ = update(gamma1, 1, 1)
    repeated_true = probability(local(repeated, 0, 1), lambda s: s[0] == 1)
    assert repeated_true == F(45, 64)
    one_step_effective = {0: F(9, 16), 1: F(7, 16)}
    naive = {s: p * one_step_effective[s[1]] ** 2 for s, p in local(gamma0, 0, 1).items()}
    naive_p = sum(p for s, p in naive.items() if s[0] == 1) / sum(naive.values())
    assert naive_p == F(171, 244) and naive_p != repeated_true
    # A literal echo H2=H1 has conditional likelihood one on the realized history.
    echo = {s: p * F(1) for s, p in gamma1.items()}
    assert echo == gamma1

    result = {
        "state_count": len(gamma0), "exact_arithmetic": "fractions.Fraction",
        "first_message_kernel_H1_1": k1, "reply_kernel_H2_1_given_H1_1": k2,
        "public_history_probabilities": histories, "actual_private_records": {"D0": 1, "D1": 0},
        "local_posterior_table": rows,
        "correct_first_receiver_P_X0_1": F(45, 62), "wrong_direct_sensor_P_X0_1": wrong_sensor,
        "two_fresh_peer_samples_P_X0_1": repeated_true,
        "wrong_independent_given_X1_P_X0_1": naive_p,
        "full_joint_matches_direct_product": True,
        "all_sender_self_message_invariance": True,
        "all_history_normalizers_and_local_correlations": True,
        "retained_joint_private_weights_initial": private_marginal(gamma0),
        "retained_joint_private_weights_after_H1_1": private_marginal(gamma1),
        "retained_joint_private_weights_after_H1_H2_11": private_marginal(gamma2),
        "four_weight_retention_exact_on_all_two_message_histories": True,
        "deterministic_echo_no_change": True,
        "wall_seconds": round(perf_counter() - start, 4),
        "scope": "Exact enumeration of one 256-state two-agent model; no sampling, trajectory scan, or universal numerical proof.",
    }
    output = Path(__file__).with_name("interaction-checks.json")
    output.write_text(json.dumps(stringify(result), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stringify(result), indent=2))


if __name__ == "__main__":
    main()
