"""Pure law-derived H7 ELBO component construction.

The only scientific input is one exact :class:`H7CompleteLawSnapshot`.
Every raw expected-log factor, recognition-entropy child, grouped emission,
and positive KL is recomputed from that snapshot.  The module has no route
for generic signed values and does not import the independent MP oracle.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Callable, Literal

import torch
from torch import Tensor

from vfe4.types.h7 import (
    H7AffineComponentSnapshot,
    H7CompleteLawSnapshot,
    H7GaussianComponentSnapshot,
    H7GenerativeSnapshot,
    H7RecognitionSnapshot,
    H7SourceScorerRowSnapshot,
    h7_owned_sha256,
)
from vfe4.types.h7_law import (
    H7ChronologicalEntropyOwnership,
    H7EntropyChild,
    H7EntropySlotOwnership,
    H7LawComponents,
    H7LawGroupedTerm,
    H7LawRawTerm,
    H7SourceAssemblyProfile,
    H7SourceAssemblyRow,
    H7_LAW_QUADRATURE_ORDER,
    H7_LAW_SEMANTICS,
    H7_SOURCE_ASSEMBLY_SEMANTICS,
    h7_law_equality_tolerance,
)


_FLOAT64_EPSILON = math.ulp(1.0)
_LOG_2_PI = math.log(2.0 * math.pi)


@dataclass(frozen=True)
class _SourcePath:
    path_id: str
    state_sources: tuple[int, int]
    model_sources: tuple[int, int]
    q_probability: float
    p_probability: float
    operand_sha256: str


@dataclass(frozen=True)
class _JointMoments:
    mean: Tensor
    covariance: Tensor
    operand_sha256: str


@dataclass(frozen=True)
class _FactorValues:
    raw_log_p: float
    entropy: float
    positive_kl: float
    raw_operands: tuple[str, ...]
    entropy_operands: tuple[str, ...]
    kl_operands: tuple[str, ...]


@dataclass(frozen=True)
class _InitialValues:
    raw_log_p: float
    entropy: float
    positive_kl: float
    raw_operands: tuple[str, ...]
    entropy_operands: tuple[str, ...]
    kl_operands: tuple[str, ...]


def derive_h7_source_assembly_profile(
    law: H7CompleteLawSnapshot,
) -> H7SourceAssemblyProfile:
    """Project exact fixed generative source rows from one complete law."""

    _require_exact_law(law)
    rows: list[H7SourceAssemblyRow] = []
    if law.fixture_id == "h1-v1":
        source_law = law.generative.scalar_source_law
        if source_law is None:
            raise ValueError("H1 complete law lacks its scalar source law")
        for partition, receiver_t in (
            ("model_source", 1),
            ("state_source", 1),
            ("model_source", 2),
            ("state_source", 2),
        ):
            bank_rows = (
                source_law.model_source_priors
                if partition == "model_source"
                else source_law.state_source_priors
            )
            snapshot = bank_rows[receiver_t - 1]
            probabilities = tuple(
                float(value) for value in snapshot.value().tolist()
            )
            rows.append(
                H7SourceAssemblyRow(
                    partition=partition,
                    receiver_t=receiver_t,
                    support=tuple(range(len(probabilities))),
                    probabilities=probabilities,
                    complete_law_operand_sha256s=_unique_hashes(
                        law.snapshot_sha256,
                        source_law.source_law_sha256,
                        snapshot.snapshot_sha256,
                    ),
                )
            )
    else:
        context = law.generative.source_context
        if context is None:
            raise ValueError("H7 complete law lacks its source context")
        for partition, receiver_t in (
            ("model_source", 1),
            ("state_source", 1),
            ("model_source", 2),
            ("state_source", 2),
        ):
            bank = "model" if partition == "model_source" else "state"
            row = _find_source_row(context.scorer_rows, bank, receiver_t)
            rows.append(
                H7SourceAssemblyRow(
                    partition=partition,
                    receiver_t=receiver_t,
                    support=row.support,
                    probabilities=tuple(
                        float(value) for value in row.probabilities.value().tolist()
                    ),
                    complete_law_operand_sha256s=_unique_hashes(
                        law.snapshot_sha256,
                        context.context_sha256,
                        row.row_sha256,
                        row.probabilities.snapshot_sha256,
                    ),
                )
            )
    return H7SourceAssemblyProfile(
        semantics=H7_SOURCE_ASSEMBLY_SEMANTICS,
        fixture_id=law.fixture_id,
        complete_law_snapshot_sha256=law.snapshot_sha256,
        rows=tuple(rows),
    )


def build_h7_law_components(
    law: H7CompleteLawSnapshot,
) -> H7LawComponents:
    """Derive the exact raw, grouped, and monolithic views of ``law``."""

    _require_exact_law(law)
    paths = _source_paths(law)
    q_moments = {
        path.path_id: _joint_moments(
            law,
            path,
            role="q",
        )
        for path in paths
    }
    p_moments = {
        path.path_id: _joint_moments(
            law,
            path,
            role="p",
        )
        for path in paths
    }

    initial = _initial_values(law)
    source_values = {
        (partition, receiver_t): _source_factor_values(
            law,
            partition=partition,
            receiver_t=receiver_t,
        )
        for receiver_t in (1, 2)
        for partition in ("model_source", "state_source")
    }
    transition_values = {
        (partition, receiver_t): _transition_factor_values(
            law,
            paths=paths,
            q_moments=q_moments,
            partition=partition,
            receiver_t=receiver_t,
        )
        for receiver_t in (1, 2)
        for partition in ("model_transition", "state_transition")
    }
    per_path_emissions = {
        (path.path_id, receiver_t): _expected_log_emission(
            law,
            q_moments[path.path_id],
            receiver_t=receiver_t,
        )
        for path in paths
        if path.q_probability > 0.0
        for receiver_t in (1, 2)
    }
    emission_values = {
        receiver_t: float(
            math.fsum(
                path.q_probability
                * per_path_emissions[(path.path_id, receiver_t)]
                for path in paths
                if path.q_probability > 0.0
            )
        )
        for receiver_t in (1, 2)
    }
    emission_operands = {
        receiver_t: _unique_hashes(
            law.snapshot_sha256,
            _decoder(law, receiver_t).decoder_sha256,
            *(
                operand
                for path in paths
                if path.q_probability > 0.0
                for operand in (
                    path.operand_sha256,
                    q_moments[path.path_id].operand_sha256,
                )
            ),
        )
        for receiver_t in (1, 2)
    }

    entropy_ownership = _entropy_ownership(
        law,
        initial=initial,
        source_values=source_values,
        transition_values=transition_values,
    )
    entropy_by_receiver = {
        slot.raw_slot[1]: slot for slot in entropy_ownership.slots
    }
    raw_value_by_slot = {
        ("initial", 0): initial.raw_log_p,
        ("model_source", 1): source_values[
            ("model_source", 1)
        ].raw_log_p,
        ("model_transition", 1): transition_values[
            ("model_transition", 1)
        ].raw_log_p,
        ("state_source", 1): source_values[
            ("state_source", 1)
        ].raw_log_p,
        ("state_transition", 1): transition_values[
            ("state_transition", 1)
        ].raw_log_p,
        ("emission", 1): emission_values[1],
        ("entropy", 1): entropy_by_receiver[1].value,
        ("model_source", 2): source_values[
            ("model_source", 2)
        ].raw_log_p,
        ("model_transition", 2): transition_values[
            ("model_transition", 2)
        ].raw_log_p,
        ("state_source", 2): source_values[
            ("state_source", 2)
        ].raw_log_p,
        ("state_transition", 2): transition_values[
            ("state_transition", 2)
        ].raw_log_p,
        ("emission", 2): emission_values[2],
        ("entropy", 2): entropy_by_receiver[2].value,
    }
    raw_operands_by_slot = {
        ("initial", 0): initial.raw_operands,
        ("model_source", 1): source_values[
            ("model_source", 1)
        ].raw_operands,
        ("model_transition", 1): transition_values[
            ("model_transition", 1)
        ].raw_operands,
        ("state_source", 1): source_values[
            ("state_source", 1)
        ].raw_operands,
        ("state_transition", 1): transition_values[
            ("state_transition", 1)
        ].raw_operands,
        ("emission", 1): emission_operands[1],
        ("entropy", 1): _unique_hashes(
            law.snapshot_sha256,
            *(
                operand
                for child in entropy_by_receiver[1].children
                for operand in child.complete_law_operand_sha256s
            ),
        ),
        ("model_source", 2): source_values[
            ("model_source", 2)
        ].raw_operands,
        ("model_transition", 2): transition_values[
            ("model_transition", 2)
        ].raw_operands,
        ("state_source", 2): source_values[
            ("state_source", 2)
        ].raw_operands,
        ("state_transition", 2): transition_values[
            ("state_transition", 2)
        ].raw_operands,
        ("emission", 2): emission_operands[2],
        ("entropy", 2): _unique_hashes(
            law.snapshot_sha256,
            *(
                operand
                for child in entropy_by_receiver[2].children
                for operand in child.complete_law_operand_sha256s
            ),
        ),
    }
    raw_terms = tuple(
        H7LawRawTerm(
            slot=slot,
            semantics=(
                "expected_log_emission"
                if slot[0] == "emission"
                else "recognition_entropy"
                if slot[0] == "entropy"
                else "expected_log_generative_factor"
            ),
            value=float(raw_value_by_slot[slot]),
            complete_law_operand_sha256s=raw_operands_by_slot[slot],
        )
        for slot in (
            ("initial", 0),
            ("model_source", 1),
            ("model_transition", 1),
            ("state_source", 1),
            ("state_transition", 1),
            ("emission", 1),
            ("entropy", 1),
            ("model_source", 2),
            ("model_transition", 2),
            ("state_source", 2),
            ("state_transition", 2),
            ("emission", 2),
            ("entropy", 2),
        )
    )

    emission_terms = tuple(
        H7LawGroupedTerm(
            term_id=f"expected_log_emission[{receiver_t}]",
            semantics="expected_log_emission",
            elbo_sign=1,
            value=emission_values[receiver_t],
            complete_law_operand_sha256s=emission_operands[receiver_t],
        )
        for receiver_t in (1, 2)
    )
    positive_kl_terms = (
        H7LawGroupedTerm(
            term_id="K0_joint_z0_m0",
            semantics="positive_kl_q_to_p",
            elbo_sign=-1,
            value=initial.positive_kl,
            complete_law_operand_sha256s=initial.kl_operands,
        ),
        *(
            H7LawGroupedTerm(
                term_id=f"{partition}_kl[{receiver_t}]",
                semantics="positive_kl_q_to_p",
                elbo_sign=-1,
                value=(
                    source_values[(partition, receiver_t)].positive_kl
                    if partition in ("model_source", "state_source")
                    else transition_values[
                        (partition, receiver_t)
                    ].positive_kl
                ),
                complete_law_operand_sha256s=(
                    source_values[(partition, receiver_t)].kl_operands
                    if partition in ("model_source", "state_source")
                    else transition_values[(partition, receiver_t)].kl_operands
                ),
            )
            for receiver_t in (1, 2)
            for partition in (
                "model_source",
                "state_source",
                "model_transition",
                "state_transition",
            )
        ),
    )

    raw_total = float(math.fsum(term.value for term in raw_terms))
    grouped_signed_values = tuple(
        term.elbo_sign * term.value
        for term in (*emission_terms, *positive_kl_terms)
    )
    grouped_total = float(math.fsum(grouped_signed_values))
    monolithic_total = _monolithic_total(
        law,
        paths=paths,
        q_moments=q_moments,
        p_moments=p_moments,
        per_path_emissions=per_path_emissions,
    )
    equality_tolerance = h7_law_equality_tolerance(
        raw_values=tuple(term.value for term in raw_terms),
        grouped_signed_values=grouped_signed_values,
        monolithic_total=monolithic_total,
    )
    return H7LawComponents(
        semantics=H7_LAW_SEMANTICS,
        fixture_id=law.fixture_id,
        complete_law_snapshot_sha256=law.snapshot_sha256,
        quadrature_order=H7_LAW_QUADRATURE_ORDER,
        source_assembly_profile=derive_h7_source_assembly_profile(law),
        entropy_ownership=entropy_ownership,
        raw_terms=raw_terms,
        emission_terms=emission_terms,
        positive_kl_terms=positive_kl_terms,
        raw_total=raw_total,
        grouped_total=grouped_total,
        monolithic_total=monolithic_total,
        raw_grouped_equality_residual=abs(raw_total - grouped_total),
        grouped_monolithic_equality_residual=abs(
            grouped_total - monolithic_total
        ),
        equality_tolerance=equality_tolerance,
    )


def _require_exact_law(law: object) -> H7CompleteLawSnapshot:
    if type(law) is not H7CompleteLawSnapshot:
        raise ValueError("law must be an exact H7CompleteLawSnapshot")
    law.__post_init__()
    return law


def _unique_hashes(*values: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _source_paths(law: H7CompleteLawSnapshot) -> tuple[_SourcePath, ...]:
    if law.fixture_id == "h7-v1":
        semantics = {
            "complete_law_snapshot_sha256": law.snapshot_sha256,
            "path_id": "matrix-singleton-path",
            "state_sources": (0, 1),
            "model_sources": (0, 1),
            "q_probability": 1.0,
            "p_probability": 1.0,
        }
        return (
            _SourcePath(
                path_id="matrix-singleton-path",
                state_sources=(0, 1),
                model_sources=(0, 1),
                q_probability=1.0,
                p_probability=1.0,
                operand_sha256=h7_owned_sha256(
                    "vfe4.h7.law-source-path-operand.v1",
                    semantics,
                ),
            ),
        )
    p_law = law.generative.scalar_source_law
    q_law = law.recognition.scalar_source_law
    if p_law is None or q_law is None:
        raise ValueError("H1 complete law lacks exact source snapshots")
    if p_law.ordered_paths != q_law.ordered_paths:
        raise ValueError("H1 source path inventories disagree")
    paths: list[_SourcePath] = []
    for declared in p_law.ordered_paths:
        q_probability = 1.0
        p_probability = 1.0
        for index in range(2):
            state_source = declared.a[index]
            model_source = declared.b[index]
            q_model = q_law.model_source_probabilities[index].value()
            q_state = (
                q_law.state_source_probabilities_given_model_source[index]
                .value()
            )
            p_model = p_law.model_source_priors[index].value()
            p_state = p_law.state_source_priors[index].value()
            q_probability *= float(q_model[model_source]) * float(
                q_state[model_source, state_source]
            )
            p_probability *= float(p_model[model_source]) * float(
                p_state[state_source]
            )
        if q_probability > 0.0 and p_probability <= 0.0:
            raise ValueError("recognition source mass lies outside model support")
        semantics = {
            "complete_law_snapshot_sha256": law.snapshot_sha256,
            "path_id": declared.path_id,
            "state_sources": declared.a,
            "model_sources": declared.b,
            "q_probability": q_probability,
            "p_probability": p_probability,
            "generative_source_law_sha256": p_law.source_law_sha256,
            "recognition_source_law_sha256": q_law.source_law_sha256,
        }
        paths.append(
            _SourcePath(
                path_id=declared.path_id,
                state_sources=declared.a,
                model_sources=declared.b,
                q_probability=q_probability,
                p_probability=p_probability,
                operand_sha256=h7_owned_sha256(
                    "vfe4.h7.law-source-path-operand.v1",
                    semantics,
                ),
            )
        )
    if not math.isclose(
        math.fsum(path.q_probability for path in paths),
        1.0,
        rel_tol=0.0,
        abs_tol=64.0 * _FLOAT64_EPSILON,
    ):
        raise ValueError("recognition source paths do not normalize")
    if not math.isclose(
        math.fsum(path.p_probability for path in paths),
        1.0,
        rel_tol=0.0,
        abs_tol=64.0 * _FLOAT64_EPSILON,
    ):
        raise ValueError("generative source paths do not normalize")
    return tuple(paths)


def _joint_moments(
    law: H7CompleteLawSnapshot,
    path: _SourcePath,
    *,
    role: Literal["q", "p"],
) -> _JointMoments:
    if role == "q":
        initial = law.recognition.initial_joint

        def model_selector(
            receiver_t: int,
            source_j: int,
        ) -> H7AffineComponentSnapshot:
            return _recognition_model(
                law.recognition,
                receiver_t,
                source_j,
            )

        def state_selector(
            receiver_t: int,
            state_j: int,
            model_j: int,
        ) -> H7AffineComponentSnapshot:
            return _recognition_state(
                law.recognition,
                receiver_t,
                state_j,
                model_j,
            )
    else:
        initial = law.generative.initial_joint

        def model_selector(
            receiver_t: int,
            source_j: int,
        ) -> H7AffineComponentSnapshot:
            return _generative_transition(
                law.generative,
                "model",
                receiver_t,
                source_j,
            )

        def state_selector(
            receiver_t: int,
            state_j: int,
            _model_j: int,
        ) -> H7AffineComponentSnapshot:
            return _generative_transition(
                law.generative,
                "state",
                receiver_t,
                state_j,
            )
    moments = _propagate_joint_moments(
        initial=initial,
        model_selector=model_selector,
        state_selector=state_selector,
        path=path,
    )
    operand_sha256 = h7_owned_sha256(
        "vfe4.h7.law-assembled-joint-moments.v1",
        {
            "complete_law_snapshot_sha256": law.snapshot_sha256,
            "role": role,
            "path_operand_sha256": path.operand_sha256,
            "mean": _tensor_operand_sha256(moments[0]),
            "covariance": _tensor_operand_sha256(moments[1]),
        },
    )
    return _JointMoments(
        mean=moments[0],
        covariance=moments[1],
        operand_sha256=operand_sha256,
    )


def _propagate_joint_moments(
    *,
    initial: H7GaussianComponentSnapshot,
    model_selector: Callable[[int, int], H7AffineComponentSnapshot],
    state_selector: Callable[[int, int, int], H7AffineComponentSnapshot],
    path: _SourcePath,
) -> tuple[Tensor, Tensor]:
    initial_mean = initial.mean.value()
    initial_covariance = initial.covariance.value()
    dimension = initial_mean.numel() // 2
    total_dimension = 6 * dimension
    mean = torch.zeros(
        total_dimension,
        dtype=torch.float64,
        device=initial_mean.device,
    )
    covariance = torch.zeros(
        (total_dimension, total_dimension),
        dtype=torch.float64,
        device=initial_covariance.device,
    )
    mean[: 2 * dimension] = initial_mean
    covariance[: 2 * dimension, : 2 * dimension] = initial_covariance
    active = list(range(2 * dimension))
    for receiver_t in (1, 2):
        state_source = path.state_sources[receiver_t - 1]
        model_source = path.model_sources[receiver_t - 1]
        model = model_selector(receiver_t, model_source)
        model_target = _block_indices("m", receiver_t, dimension)
        model_parent = _block_indices("m", model_source, dimension)
        _insert_affine_moments(
            mean,
            covariance,
            active=tuple(active),
            target=model_target,
            parent_blocks=((model_parent, model.parent_map.value()),),
            offset=model.offset.value(),
            noise_covariance=model.receiver_law.covariance.value(),
        )
        active.extend(model_target)

        state = state_selector(receiver_t, state_source, model_source)
        state_target = _block_indices("z", receiver_t, dimension)
        state_parent = _block_indices("z", state_source, dimension)
        model_map = state.same_receiver_model_map
        if model_map is None:
            raise ValueError("state recognition conditional lacks its model map")
        _insert_affine_moments(
            mean,
            covariance,
            active=tuple(active),
            target=state_target,
            parent_blocks=(
                (state_parent, state.parent_map.value()),
                (model_target, model_map.value()),
            ),
            offset=state.offset.value(),
            noise_covariance=state.receiver_law.covariance.value(),
        )
        active.extend(state_target)
    torch.linalg.cholesky(covariance)
    return mean, covariance


def _insert_affine_moments(
    mean: Tensor,
    covariance: Tensor,
    *,
    active: tuple[int, ...],
    target: tuple[int, ...],
    parent_blocks: tuple[tuple[tuple[int, ...], Tensor], ...],
    offset: Tensor,
    noise_covariance: Tensor,
) -> None:
    linear = torch.zeros(
        (len(target), mean.numel()),
        dtype=torch.float64,
        device=mean.device,
    )
    for indices, matrix in parent_blocks:
        linear[:, list(indices)] += matrix
    target_mean = linear @ mean + offset
    cross = linear @ covariance[:, list(active)]
    target_covariance = linear @ covariance @ linear.T + noise_covariance
    mean[list(target)] = target_mean
    covariance[list(target), :] = 0.0
    covariance[:, list(target)] = 0.0
    target_rows = torch.tensor(
        target,
        dtype=torch.long,
        device=mean.device,
    ).unsqueeze(1)
    active_columns = torch.tensor(
        active,
        dtype=torch.long,
        device=mean.device,
    ).unsqueeze(0)
    covariance[target_rows, active_columns] = cross
    covariance[active_columns.T, target_rows.T] = cross.T
    target_columns = torch.tensor(
        target,
        dtype=torch.long,
        device=mean.device,
    ).unsqueeze(0)
    covariance[target_rows, target_columns] = target_covariance


def _block_indices(
    channel: Literal["z", "m"],
    population_label: int,
    dimension: int,
) -> tuple[int, ...]:
    start = 2 * population_label * dimension
    if channel == "m":
        start += dimension
    return tuple(range(start, start + dimension))


def _initial_values(law: H7CompleteLawSnapshot) -> _InitialValues:
    q = law.recognition.initial_joint
    p = law.generative.initial_joint
    q_mean = q.mean.value()
    q_covariance = q.covariance.value()
    p_mean = p.mean.value()
    p_covariance = p.covariance.value()
    entropy = _gaussian_entropy(q_covariance)
    raw_log_p = _expected_log_gaussian(
        q_mean,
        q_covariance,
        p_mean,
        p_covariance,
    )
    positive_kl = _nonnegative_kl(
        _gaussian_kl(
            q_mean,
            q_covariance,
            p_mean,
            p_covariance,
        ),
        "initial joint KL",
        scale=max(1.0, abs(raw_log_p), abs(entropy)),
    )
    return _InitialValues(
        raw_log_p=raw_log_p,
        entropy=entropy,
        positive_kl=positive_kl,
        raw_operands=_unique_hashes(
            law.snapshot_sha256,
            q.component_sha256,
            p.component_sha256,
        ),
        entropy_operands=_unique_hashes(
            law.snapshot_sha256,
            q.component_sha256,
            q.covariance.snapshot_sha256,
        ),
        kl_operands=_unique_hashes(
            law.snapshot_sha256,
            q.component_sha256,
            p.component_sha256,
        ),
    )


def _source_factor_values(
    law: H7CompleteLawSnapshot,
    *,
    partition: Literal["model_source", "state_source"],
    receiver_t: Literal[1, 2],
) -> _FactorValues:
    if law.fixture_id == "h1-v1":
        p_law = law.generative.scalar_source_law
        q_law = law.recognition.scalar_source_law
        if p_law is None or q_law is None:
            raise ValueError("H1 source factor lacks exact source laws")
        if partition == "model_source":
            q_snapshot = q_law.model_source_probabilities[receiver_t - 1]
            p_snapshot = p_law.model_source_priors[receiver_t - 1]
            q_rows = (q_snapshot.value(),)
            conditioning_weights = (1.0,)
            raw_operands = _unique_hashes(
                law.snapshot_sha256,
                q_law.source_law_sha256,
                p_law.source_law_sha256,
                q_snapshot.snapshot_sha256,
                p_snapshot.snapshot_sha256,
            )
            entropy_operands = _unique_hashes(
                law.snapshot_sha256,
                q_law.source_law_sha256,
                q_snapshot.snapshot_sha256,
            )
        else:
            q_snapshot = (
                q_law.state_source_probabilities_given_model_source[
                    receiver_t - 1
                ]
            )
            p_snapshot = p_law.state_source_priors[receiver_t - 1]
            q_rows = tuple(q_snapshot.value()[index] for index in range(
                q_snapshot.shape[0]
            ))
            q_model_snapshot = q_law.model_source_probabilities[
                receiver_t - 1
            ]
            conditioning_weights = tuple(
                float(value) for value in q_model_snapshot.value().tolist()
            )
            raw_operands = _unique_hashes(
                law.snapshot_sha256,
                q_law.source_law_sha256,
                p_law.source_law_sha256,
                q_model_snapshot.snapshot_sha256,
                q_snapshot.snapshot_sha256,
                p_snapshot.snapshot_sha256,
            )
            entropy_operands = _unique_hashes(
                law.snapshot_sha256,
                q_law.source_law_sha256,
                q_model_snapshot.snapshot_sha256,
                q_snapshot.snapshot_sha256,
            )
        p_probabilities = p_snapshot.value()
        raw_log_p, entropy, positive_kl = _categorical_factor_values(
            q_rows=q_rows,
            conditioning_weights=conditioning_weights,
            p_probabilities=p_probabilities,
        )
        return _FactorValues(
            raw_log_p=raw_log_p,
            entropy=entropy,
            positive_kl=positive_kl,
            raw_operands=raw_operands,
            entropy_operands=entropy_operands,
            kl_operands=raw_operands,
        )

    p_context = law.generative.source_context
    if p_context is None:
        raise ValueError("H7 generative source context is absent")
    bank = "model" if partition == "model_source" else "state"
    p_row = _find_source_row(p_context.scorer_rows, bank, receiver_t)
    q_row = _find_source_row(law.recognition.source_rows, bank, receiver_t)
    if p_row.support != q_row.support:
        raise ValueError("H7 source supports disagree between p and q")
    raw_log_p, entropy, positive_kl = _categorical_factor_values(
        q_rows=(q_row.probabilities.value(),),
        conditioning_weights=(1.0,),
        p_probabilities=p_row.probabilities.value(),
    )
    raw_operands = _unique_hashes(
        law.snapshot_sha256,
        p_context.context_sha256,
        p_row.row_sha256,
        q_row.row_sha256,
        p_row.probabilities.snapshot_sha256,
        q_row.probabilities.snapshot_sha256,
    )
    return _FactorValues(
        raw_log_p=raw_log_p,
        entropy=entropy,
        positive_kl=positive_kl,
        raw_operands=raw_operands,
        entropy_operands=_unique_hashes(
            law.snapshot_sha256,
            q_row.row_sha256,
            q_row.probabilities.snapshot_sha256,
        ),
        kl_operands=raw_operands,
    )


def _categorical_factor_values(
    *,
    q_rows: tuple[Tensor, ...],
    conditioning_weights: tuple[float, ...],
    p_probabilities: Tensor,
) -> tuple[float, float, float]:
    if len(q_rows) != len(conditioning_weights):
        raise ValueError("categorical conditioning inventory disagrees")
    raw_contributions: list[float] = []
    entropy_contributions: list[float] = []
    kl_contributions: list[float] = []
    for weight, q_row in zip(
        conditioning_weights,
        q_rows,
        strict=True,
    ):
        if weight < 0.0:
            raise ValueError("categorical conditioning weight is negative")
        if q_row.shape != p_probabilities.shape:
            raise ValueError("categorical q/p row shapes disagree")
        for q_value, p_value in zip(
            q_row.tolist(),
            p_probabilities.tolist(),
            strict=True,
        ):
            q_probability = float(q_value)
            p_probability = float(p_value)
            if q_probability < 0.0 or p_probability < 0.0:
                raise ValueError("categorical probabilities must be nonnegative")
            if q_probability == 0.0 or weight == 0.0:
                continue
            if p_probability <= 0.0:
                raise ValueError("recognition source mass lies outside p support")
            weighted_q = weight * q_probability
            log_q = math.log(q_probability)
            log_p = math.log(p_probability)
            raw_contributions.append(weighted_q * log_p)
            entropy_contributions.append(-weighted_q * log_q)
            kl_contributions.append(weighted_q * (log_q - log_p))
    raw_log_p = float(math.fsum(raw_contributions))
    entropy = float(math.fsum(entropy_contributions))
    positive_kl = _nonnegative_kl(
        float(math.fsum(kl_contributions)),
        "categorical source KL",
        scale=max(1.0, abs(raw_log_p), abs(entropy)),
    )
    return raw_log_p, entropy, positive_kl


def _transition_factor_values(
    law: H7CompleteLawSnapshot,
    *,
    paths: tuple[_SourcePath, ...],
    q_moments: dict[str, _JointMoments],
    partition: Literal["model_transition", "state_transition"],
    receiver_t: Literal[1, 2],
) -> _FactorValues:
    bank: Literal["model", "state"] = (
        "model" if partition == "model_transition" else "state"
    )
    raw_contributions: list[float] = []
    entropy_contributions: list[float] = []
    kl_contributions: list[float] = []
    raw_operands: list[str] = [law.snapshot_sha256]
    entropy_operands: list[str] = [law.snapshot_sha256]
    kl_operands: list[str] = [law.snapshot_sha256]
    for path in paths:
        if path.q_probability <= 0.0:
            continue
        q_component, p_component = _transition_components(
            law,
            path=path,
            bank=bank,
            receiver_t=receiver_t,
        )
        raw_log_p, entropy, positive_kl = _affine_factor_values(
            q_component,
            p_component,
            q_moments[path.path_id],
            path=path,
            bank=bank,
            receiver_t=receiver_t,
        )
        weight = path.q_probability
        raw_contributions.append(weight * raw_log_p)
        entropy_contributions.append(weight * entropy)
        kl_contributions.append(weight * positive_kl)
        raw_operands.extend(
            (
                path.operand_sha256,
                q_moments[path.path_id].operand_sha256,
                q_component.component_sha256,
                p_component.component_sha256,
            )
        )
        entropy_operands.extend(
            (
                path.operand_sha256,
                q_moments[path.path_id].operand_sha256,
                q_component.component_sha256,
            )
        )
        kl_operands.extend(
            (
                path.operand_sha256,
                q_moments[path.path_id].operand_sha256,
                q_component.component_sha256,
                p_component.component_sha256,
            )
        )
    raw_log_p = float(math.fsum(raw_contributions))
    entropy = float(math.fsum(entropy_contributions))
    positive_kl = _nonnegative_kl(
        float(math.fsum(kl_contributions)),
        f"{partition}[{receiver_t}]",
        scale=max(1.0, abs(raw_log_p), abs(entropy)),
    )
    return _FactorValues(
        raw_log_p=raw_log_p,
        entropy=entropy,
        positive_kl=positive_kl,
        raw_operands=_unique_hashes(*raw_operands),
        entropy_operands=_unique_hashes(*entropy_operands),
        kl_operands=_unique_hashes(*kl_operands),
    )


def _transition_components(
    law: H7CompleteLawSnapshot,
    *,
    path: _SourcePath,
    bank: Literal["model", "state"],
    receiver_t: int,
) -> tuple[H7AffineComponentSnapshot, H7AffineComponentSnapshot]:
    state_source = path.state_sources[receiver_t - 1]
    model_source = path.model_sources[receiver_t - 1]
    if bank == "model":
        q_component = _recognition_model(
            law.recognition,
            receiver_t,
            model_source,
        )
        p_component = _generative_transition(
            law.generative,
            "model",
            receiver_t,
            model_source,
        )
    else:
        q_component = _recognition_state(
            law.recognition,
            receiver_t,
            state_source,
            model_source,
        )
        p_component = _generative_transition(
            law.generative,
            "state",
            receiver_t,
            state_source,
        )
    return q_component, p_component


def _affine_factor_values(
    q_component: H7AffineComponentSnapshot,
    p_component: H7AffineComponentSnapshot,
    q_moments: _JointMoments,
    *,
    path: _SourcePath,
    bank: Literal["model", "state"],
    receiver_t: int,
) -> tuple[float, float, float]:
    q_noise = q_component.receiver_law.covariance.value()
    p_noise = p_component.receiver_law.covariance.value()
    dimension = q_noise.shape[0]
    if p_noise.shape != q_noise.shape:
        raise ValueError("affine q/p receiver dimensions disagree")
    difference = torch.zeros(
        (dimension, q_moments.mean.numel()),
        dtype=torch.float64,
        device=q_moments.mean.device,
    )
    source = (
        path.model_sources[receiver_t - 1]
        if bank == "model"
        else path.state_sources[receiver_t - 1]
    )
    source_indices = _block_indices(
        "m" if bank == "model" else "z",
        source,
        dimension,
    )
    difference[:, list(source_indices)] = (
        q_component.parent_map.value() - p_component.parent_map.value()
    )
    if bank == "state":
        q_model_map = q_component.same_receiver_model_map
        p_model_map = p_component.same_receiver_model_map
        if q_model_map is None or p_model_map is None:
            raise ValueError("state q/p factor lacks its same-receiver model map")
        model_indices = _block_indices("m", receiver_t, dimension)
        difference[:, list(model_indices)] = (
            q_model_map.value() - p_model_map.value()
        )
    displacement_mean = (
        difference @ q_moments.mean
        + q_component.offset.value()
        - p_component.offset.value()
    )
    displacement_covariance = (
        difference @ q_moments.covariance @ difference.T
    )
    p_cholesky = torch.linalg.cholesky(p_noise)
    q_cholesky = torch.linalg.cholesky(q_noise)
    solved_q = torch.cholesky_solve(q_noise, p_cholesky)
    solved_displacement = torch.cholesky_solve(
        displacement_covariance,
        p_cholesky,
    )
    solved_mean = torch.cholesky_solve(
        displacement_mean.unsqueeze(1),
        p_cholesky,
    ).squeeze(1)
    expected_quadratic = (
        torch.trace(solved_displacement)
        + torch.dot(displacement_mean, solved_mean)
    )
    trace_q = torch.trace(solved_q)
    p_logdet = 2.0 * torch.log(torch.diagonal(p_cholesky)).sum()
    q_logdet = 2.0 * torch.log(torch.diagonal(q_cholesky)).sum()
    raw_log_p = -0.5 * (
        dimension * _LOG_2_PI
        + float(p_logdet)
        + float(trace_q)
        + float(expected_quadratic)
    )
    entropy = 0.5 * (
        dimension * (1.0 + _LOG_2_PI) + float(q_logdet)
    )
    positive_kl = 0.5 * (
        float(trace_q)
        + float(expected_quadratic)
        - dimension
        + float(p_logdet)
        - float(q_logdet)
    )
    return (
        float(raw_log_p),
        float(entropy),
        _nonnegative_kl(
            float(positive_kl),
            f"{bank} transition KL",
            scale=max(1.0, abs(raw_log_p), abs(entropy)),
        ),
    )


def _entropy_ownership(
    law: H7CompleteLawSnapshot,
    *,
    initial: _InitialValues,
    source_values: dict[
        tuple[str, int],
        _FactorValues,
    ],
    transition_values: dict[
        tuple[str, int],
        _FactorValues,
    ],
) -> H7ChronologicalEntropyOwnership:
    children = (
        H7EntropyChild(
            child_id="initial_joint",
            owner_receiver_t=1,
            value=initial.entropy,
            complete_law_operand_sha256s=initial.entropy_operands,
        ),
        *(
            H7EntropyChild(
                child_id=f"{partition}[{receiver_t}]",
                owner_receiver_t=receiver_t,
                value=(
                    source_values[(partition, receiver_t)].entropy
                    if partition in ("model_source", "state_source")
                    else transition_values[(partition, receiver_t)].entropy
                ),
                complete_law_operand_sha256s=(
                    source_values[
                        (partition, receiver_t)
                    ].entropy_operands
                    if partition in ("model_source", "state_source")
                    else transition_values[
                        (partition, receiver_t)
                    ].entropy_operands
                ),
            )
            for receiver_t in (1, 2)
            for partition in (
                "model_source",
                "model_transition",
                "state_source",
                "state_transition",
            )
        ),
    )
    owner_1_children = children[:5]
    owner_2_children = children[5:]
    return H7ChronologicalEntropyOwnership(
        complete_law_snapshot_sha256=law.snapshot_sha256,
        slots=(
            H7EntropySlotOwnership(
                raw_slot=("entropy", 1),
                children=owner_1_children,
                value=float(
                    math.fsum(child.value for child in owner_1_children)
                ),
            ),
            H7EntropySlotOwnership(
                raw_slot=("entropy", 2),
                children=owner_2_children,
                value=float(
                    math.fsum(child.value for child in owner_2_children)
                ),
            ),
        ),
    )


def _expected_log_emission(
    law: H7CompleteLawSnapshot,
    moments: _JointMoments,
    *,
    receiver_t: Literal[1, 2],
) -> float:
    decoder = _decoder(law, receiver_t)
    selected = law.recognition.context.observation_labels[receiver_t - 1]
    if law.fixture_id == "h1-v1":
        selected -= 1
    dimension = decoder.state_weight.shape[1]
    state_indices = _block_indices("z", receiver_t, dimension)
    model_indices = _block_indices("m", receiver_t, dimension)
    indices = (*state_indices, *model_indices)
    latent_mean = moments.mean[list(indices)]
    latent_covariance = moments.covariance[list(indices)][:, list(indices)]
    decoder_weight = torch.cat(
        (
            decoder.state_weight.value(),
            decoder.model_weight.value(),
        ),
        dim=1,
    )
    logits_mean = decoder_weight @ latent_mean + decoder.bias.value()
    logits_covariance = decoder_weight @ latent_covariance @ decoder_weight.T
    vocabulary = logits_mean.numel()
    if selected < 0 or selected >= vocabulary:
        raise ValueError("observation label is outside the decoder vocabulary")
    contrast = torch.zeros(
        (vocabulary - 1, vocabulary),
        dtype=torch.float64,
        device=logits_mean.device,
    )
    for row in range(vocabulary - 1):
        contrast[row, row] = 1.0
        contrast[row, vocabulary - 1] = -1.0
    contrast_mean = contrast @ logits_mean
    contrast_covariance = contrast @ logits_covariance @ contrast.T
    cholesky = torch.linalg.cholesky(contrast_covariance)
    nodes, weights = _probabilists_gauss_hermite_51(
        device=logits_mean.device,
    )
    grids = torch.meshgrid(
        *(nodes for _ in range(vocabulary - 1)),
        indexing="ij",
    )
    standards = torch.stack(
        tuple(grid.reshape(-1) for grid in grids),
        dim=1,
    )
    weight_grids = torch.meshgrid(
        *(weights for _ in range(vocabulary - 1)),
        indexing="ij",
    )
    quadrature_weights = torch.ones(
        standards.shape[0],
        dtype=torch.float64,
        device=logits_mean.device,
    )
    for grid in weight_grids:
        quadrature_weights *= grid.reshape(-1)
    contrasts = contrast_mean + standards @ cholesky.T
    augmented = torch.cat(
        (
            contrasts,
            torch.zeros(
                (contrasts.shape[0], 1),
                dtype=torch.float64,
                device=logits_mean.device,
            ),
        ),
        dim=1,
    )
    selected_values = augmented[:, selected] - torch.logsumexp(
        augmented,
        dim=1,
    )
    result = float(torch.sum(quadrature_weights * selected_values))
    if not math.isfinite(result):
        raise ValueError("expected log emission is nonfinite")
    return result


def _probabilists_gauss_hermite_51(
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Construct the frozen standard-normal GH-51 rule without NumPy BLAS."""

    order = H7_LAW_QUADRATURE_ORDER
    off_diagonal = torch.sqrt(
        torch.arange(
            1,
            order,
            dtype=torch.float64,
            device=device,
        )
    )
    jacobi = torch.diag(off_diagonal, diagonal=1) + torch.diag(
        off_diagonal,
        diagonal=-1,
    )
    nodes, eigenvectors = torch.linalg.eigh(jacobi)
    weights = eigenvectors[0, :].square()
    if (
        not bool(torch.isfinite(nodes).all())
        or not bool(torch.isfinite(weights).all())
        or abs(float(weights.sum()) - 1.0) > 64.0 * _FLOAT64_EPSILON
    ):
        raise ValueError("frozen GH-51 rule is invalid")
    return nodes, weights


def _monolithic_total(
    law: H7CompleteLawSnapshot,
    *,
    paths: tuple[_SourcePath, ...],
    q_moments: dict[str, _JointMoments],
    p_moments: dict[str, _JointMoments],
    per_path_emissions: dict[tuple[str, int], float],
) -> float:
    contributions: list[float] = []
    for path in paths:
        if path.q_probability <= 0.0:
            continue
        q = q_moments[path.path_id]
        p = p_moments[path.path_id]
        gaussian_ratio = -_gaussian_kl(
            q.mean,
            q.covariance,
            p.mean,
            p.covariance,
        )
        source_ratio = math.log(path.p_probability) - math.log(
            path.q_probability
        )
        emission = math.fsum(
            per_path_emissions[(path.path_id, receiver_t)]
            for receiver_t in (1, 2)
        )
        contributions.append(
            path.q_probability
            * math.fsum((gaussian_ratio, source_ratio, emission))
        )
    result = float(math.fsum(contributions))
    if not math.isfinite(result):
        raise ValueError("monolithic augmented-joint expectation is nonfinite")
    return result


def _decoder(
    law: H7CompleteLawSnapshot,
    receiver_t: int,
):
    matches = tuple(
        decoder
        for decoder in law.generative.decoders
        if decoder.receiver_t == receiver_t
    )
    if len(matches) != 1:
        raise ValueError("decoder inventory is incomplete or ambiguous")
    return matches[0]


def _generative_transition(
    generative: H7GenerativeSnapshot,
    bank: Literal["model", "state"],
    receiver_t: int,
    source_j: int,
) -> H7AffineComponentSnapshot:
    matches = tuple(
        component
        for component in generative.transitions
        if component.bank == bank
        and component.receiver_t == receiver_t
        and component.source_j == source_j
    )
    if len(matches) != 1:
        raise ValueError("generative transition lookup is incomplete or ambiguous")
    return matches[0]


def _recognition_model(
    recognition: H7RecognitionSnapshot,
    receiver_t: int,
    source_j: int,
) -> H7AffineComponentSnapshot:
    matches = tuple(
        component
        for component in recognition.model_conditionals
        if component.receiver_t == receiver_t
        and component.source_j == source_j
    )
    if len(matches) != 1:
        raise ValueError("recognition model lookup is incomplete or ambiguous")
    return matches[0]


def _recognition_state(
    recognition: H7RecognitionSnapshot,
    receiver_t: int,
    state_source_j: int,
    model_source_j: int,
) -> H7AffineComponentSnapshot:
    candidates = tuple(
        component
        for component in recognition.state_conditionals
        if component.receiver_t == receiver_t
    )
    if recognition.initial_joint.mean.shape == (2,):
        marker = f".a_{state_source_j}.b_{model_source_j}."
        matches = tuple(
            component
            for component in candidates
            if marker in component.component_id
        )
    else:
        matches = tuple(
            component
            for component in candidates
            if component.source_j == state_source_j
        )
    if len(matches) != 1:
        raise ValueError("recognition state lookup is incomplete or ambiguous")
    return matches[0]


def _find_source_row(
    rows: tuple[H7SourceScorerRowSnapshot, ...],
    bank: str,
    receiver_t: int,
) -> H7SourceScorerRowSnapshot:
    matches = tuple(
        row
        for row in rows
        if row.bank == bank and row.receiver_t == receiver_t
    )
    if len(matches) != 1:
        raise ValueError("source scorer row lookup is incomplete or ambiguous")
    return matches[0]


def _gaussian_entropy(covariance: Tensor) -> float:
    cholesky = torch.linalg.cholesky(covariance)
    dimension = covariance.shape[0]
    logdet = 2.0 * torch.log(torch.diagonal(cholesky)).sum()
    result = 0.5 * (
        dimension * (1.0 + _LOG_2_PI) + float(logdet)
    )
    if not math.isfinite(result):
        raise ValueError("Gaussian entropy is nonfinite")
    return float(result)


def _expected_log_gaussian(
    q_mean: Tensor,
    q_covariance: Tensor,
    p_mean: Tensor,
    p_covariance: Tensor,
) -> float:
    p_cholesky = torch.linalg.cholesky(p_covariance)
    displacement = (q_mean - p_mean).unsqueeze(1)
    trace_term = torch.trace(
        torch.cholesky_solve(q_covariance, p_cholesky)
    )
    quadratic_term = torch.sum(
        displacement * torch.cholesky_solve(displacement, p_cholesky)
    )
    p_logdet = 2.0 * torch.log(torch.diagonal(p_cholesky)).sum()
    result = -0.5 * (
        q_mean.numel() * _LOG_2_PI
        + float(p_logdet)
        + float(trace_term)
        + float(quadratic_term)
    )
    if not math.isfinite(result):
        raise ValueError("expected log Gaussian density is nonfinite")
    return float(result)


def _gaussian_kl(
    q_mean: Tensor,
    q_covariance: Tensor,
    p_mean: Tensor,
    p_covariance: Tensor,
) -> float:
    q_cholesky = torch.linalg.cholesky(q_covariance)
    p_cholesky = torch.linalg.cholesky(p_covariance)
    displacement = (q_mean - p_mean).unsqueeze(1)
    trace_term = torch.trace(
        torch.cholesky_solve(q_covariance, p_cholesky)
    )
    quadratic_term = torch.sum(
        displacement * torch.cholesky_solve(displacement, p_cholesky)
    )
    q_logdet = 2.0 * torch.log(torch.diagonal(q_cholesky)).sum()
    p_logdet = 2.0 * torch.log(torch.diagonal(p_cholesky)).sum()
    value = 0.5 * (
        trace_term
        + quadratic_term
        - q_mean.numel()
        + p_logdet
        - q_logdet
    )
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Gaussian KL is nonfinite")
    return result


def _nonnegative_kl(value: float, name: str, *, scale: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} is nonfinite")
    if value >= 0.0:
        return float(value)
    allowance = 1024.0 * _FLOAT64_EPSILON * max(1.0, scale)
    if abs(value) <= allowance:
        return 0.0
    raise ValueError(f"{name} is materially negative")


def _tensor_operand_sha256(value: Tensor) -> str:
    cpu = value.detach().to(device="cpu").contiguous()
    try:
        raw = cpu.numpy().tobytes(order="C")
    except (TypeError, RuntimeError):
        raw = bytes(cpu.view(torch.uint8).reshape(-1).tolist())
    return h7_owned_sha256(
        "vfe4.h7.law-derived-tensor-operand.v1",
        {
            "dtype": str(cpu.dtype),
            "shape": tuple(cpu.shape),
            "raw_bytes_sha256": hashlib.sha256(raw).hexdigest(),
        },
    )


__all__ = [
    "build_h7_law_components",
    "derive_h7_source_assembly_profile",
]
