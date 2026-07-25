"""Strict raw-byte adapters for the frozen H7 and scalar H1 fixtures."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast

import torch

from vfe4.types.h7 import (
    H7_CONTROL_IDS,
    H7_MATRIX_TRIAL_IDS,
    H7_SCALAR_TRIAL_IDS,
    H7AffineComponentSnapshot,
    H7CompleteLawSnapshot,
    H7DecoderSnapshot,
    H7DensityProbePair,
    H7Fixture,
    H7FrameProfile,
    H7GLPlus2Action,
    H7GaussianComponentSnapshot,
    H7GenerativeSnapshot,
    H7HistoryValueSnapshot,
    H7JacobianMetadataSnapshot,
    H7OwnedTensorSnapshot,
    H7RecognitionContextSnapshot,
    H7RecognitionSnapshot,
    H7ScalarGenerativeSourceLawSnapshot,
    H7ScalarProbeSetSnapshot,
    H7ScalarRecognitionSourceLawSnapshot,
    H7ScalarReplayAction,
    H7ScalarSourcePathSnapshot,
    H7SourceContextSnapshot,
    H7SourceCovectorSnapshot,
    H7SourceScorerRowSnapshot,
    H7TrialId,
    H7TrialSpec,
    canonical_h7_bytes,
    h7_owned_sha256,
)


H7_FIXTURE_PATH = Path(__file__).with_name("fixtures") / "h7_v1.json"
H7_DENSITY_PROBE_TABLE_PATH = (
    Path(__file__).with_name("fixtures") / "h7_density_probes_v1.json"
)
H1_FIXTURE_RAW_SHA256 = (
    "388e38cc8c16d8b5e2c61919c1e712a134d88fb0bbd8ec1f2939b9859c9a583b"
)
H7_FIXTURE_RAW_SHA256 = (
    "d2ed126c3deab3eafc7b94f81f13152be63eb854e3e62e03f1494dea163666d4"
)
H7_DENSITY_PROBE_SET_SHA256 = (
    "f002618a32270846c83fedf9888bc06a01d755019edc6421526aee33f89fb42f"
)
H7_DENSITY_PROBE_TABLE_RAW_SHA256 = (
    "4857af296e84a33f47964c3bca65e0d42967009aa5c79a52bcc98d6db04382c6"
)
H7_DENSITY_PROBE_EXPANSION = "all-actions-components-directions-v1"

_ROOT_FIELDS = frozenset(
    {
        "fixture_schema_version",
        "fixture_id",
        "group",
        "representations",
        "horizon",
        "dimensions",
        "continuous_order",
        "state_parent_sets",
        "model_parent_sets",
        "state_source_support",
        "model_source_support",
        "observation_label_base",
        "observation_labels",
        "frame_profiles",
        "actions",
        "generative",
        "recognition",
        "density_probes",
        "oracle",
    }
)
_CONTINUOUS_ORDER = (
    "z0[0]",
    "z0[1]",
    "m0[0]",
    "m0[1]",
    "z1[0]",
    "z1[1]",
    "m1[0]",
    "m1[1]",
    "z2[0]",
    "z2[1]",
    "m2[0]",
    "m2[1]",
)
_CHAIN = ((0,), (1,))
_ACTION_VALUES = {
    "diagonal": (
        ((1.2, 0.2), (-0.1, 0.9)),
        ((1.2, 0.2), (-0.1, 0.9)),
        ((1.2, 0.2), (-0.1, 0.9)),
    ),
    "internal": (
        ((1.25, 0.1), (0.05, 0.95)),
        ((0.85, -0.2), (0.1, 1.15)),
        ((1.05, 0.25), (-0.15, 0.9)),
    ),
    "fixed_decoder_stabilizer": (
        ((1.0, 0.0), (0.2, 1.1)),
        ((1.0, 0.0), (0.2, 1.1)),
        ((1.0, 0.0), (0.2, 1.1)),
    ),
}
_FRAME_VALUES = {
    "identity": (
        ((1.0, 0.0), (0.0, 1.0)),
        ((1.0, 0.0), (0.0, 1.0)),
        ((1.0, 0.0), (0.0, 1.0)),
    ),
    "nonidentity": (
        ((1.0, 0.0), (0.0, 1.0)),
        ((1.1, 0.15), (-0.05, 0.95)),
        ((0.9, -0.1), (0.2, 1.05)),
    ),
}
_PROBE_COMPONENTS = (
    ("p.initial_joint", "initial", 4, "initial_joint"),
    ("p.model.receiver_1", "model:1<-0", 2, "receiver_model"),
    ("p.model.receiver_2", "model:2<-1", 2, "receiver_model"),
    ("p.state.receiver_1", "state:1<-0", 2, "receiver_state"),
    ("p.state.receiver_2", "state:2<-1", 2, "receiver_state"),
    ("q.structured.initial_joint", "initial", 4, "initial_joint"),
    (
        "q.structured.model.receiver_1",
        "model:1<-0",
        2,
        "receiver_model",
    ),
    (
        "q.structured.model.receiver_2",
        "model:2<-1",
        2,
        "receiver_model",
    ),
    (
        "q.structured.state.receiver_1",
        "state:1<-0",
        2,
        "receiver_state",
    ),
    (
        "q.structured.state.receiver_2",
        "state:2<-1",
        2,
        "receiver_state",
    ),
    ("q.factorized.initial_joint", "initial", 4, "initial_joint"),
    (
        "q.factorized.model.receiver_1",
        "model:1<-0",
        2,
        "receiver_model",
    ),
    (
        "q.factorized.model.receiver_2",
        "model:2<-1",
        2,
        "receiver_model",
    ),
    (
        "q.factorized.state.receiver_1",
        "state:1<-0",
        2,
        "receiver_state",
    ),
    (
        "q.factorized.state.receiver_2",
        "state:2<-1",
        2,
        "receiver_state",
    ),
    ("p.global", "matrix-singleton-path", 12, "global"),
    ("q.structured.global", "matrix-singleton-path", 12, "global"),
    ("q.factorized.global", "matrix-singleton-path", 12, "global"),
)
_H1_SCALAR_PROBE_ANCHOR_PROFILE = "original-generative-conditional-global-mean-v1"
_H1_SCALAR_PATH_MEANS = (
    (0.2, -0.15, 0.090625, -0.0875, 0.2995, -0.17),
    (0.2, -0.15, 0.090625, -0.0875, 0.1975, -0.17),
    (0.2, -0.15, 0.090625, -0.0875, 0.2771, -0.106),
    (0.2, -0.15, 0.090625, -0.0875, 0.1751, -0.106),
)
_H1_SCALAR_PATH_PRIMES = (
    (
        (0.25, -0.1875, 0.11328125, -0.109375, 0.374375, -0.2125),
        (0.25, -0.1875, 0.11328125, -0.109375, 0.246875, -0.2125),
        (0.25, -0.1875, 0.11328125, -0.109375, 0.346375, -0.1325),
        (0.25, -0.1875, 0.11328125, -0.109375, 0.218875, -0.1325),
    ),
    (
        (0.16, -0.12, 0.0996875, -0.09625, 0.4193, -0.238),
        (0.16, -0.12, 0.0996875, -0.09625, 0.2765, -0.238),
        (0.16, -0.12, 0.0996875, -0.09625, 0.38794, -0.1484),
        (0.16, -0.12, 0.0996875, -0.09625, 0.24514, -0.1484),
    ),
)
_H1_SCALAR_GLOBAL_LOG_JACOBIAN_SHIFTS = (
    1.3388613078852587,
    0.4172777302226563,
)


def _reject_duplicate_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"nonfinite JSON constant {value!r} is forbidden")


def _parse_exact_json(data: bytes, *, expected_sha256: str) -> dict[str, object]:
    if type(data) is not bytes:
        raise ValueError("fixture data must be immutable raw bytes")
    observed = hashlib.sha256(data).hexdigest()
    if observed != expected_sha256:
        raise ValueError("fixture raw SHA-256 does not match the frozen bytes")
    try:
        value = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"fixture JSON could not be parsed: {exc}") from exc
    if type(value) is not dict:
        raise ValueError("fixture root must be an object")
    return value


def _fields(
    value: object, expected: frozenset[str] | set[str], location: str
) -> dict[str, object]:
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError(f"{location} fields do not match the frozen schema")
    return value


def _number(value: object, location: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{location} must be a JSON number, not a Boolean")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{location} must be finite")
    return result


def _integer(value: object, location: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{location} must be a JSON integer")
    return value


def _sequence(value: object, length: int | None, location: str) -> list[object]:
    if type(value) is not list or (length is not None and len(value) != length):
        raise ValueError(f"{location} has the wrong sequence length")
    return value


def _tensor(value: object, shape: tuple[int, ...], location: str) -> torch.Tensor:
    def convert(item: object, depth: int, prefix: str) -> object:
        if depth == len(shape):
            return _number(item, prefix)
        row = _sequence(item, shape[depth], prefix)
        return [
            convert(child, depth + 1, f"{prefix}[{index}]")
            for index, child in enumerate(row)
        ]

    return torch.tensor(convert(value, 0, location), dtype=torch.float64)


def _require_spd(value: torch.Tensor, location: str) -> None:
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError(f"{location} must be square")
    if not torch.equal(value, value.T):
        raise ValueError(f"{location} must be exactly symmetric")
    info = torch.linalg.cholesky_ex(value).info
    if bool(torch.any(info != 0).item()):
        raise ValueError(f"{location} must be positive definite")


def _precision(value: torch.Tensor, location: str) -> torch.Tensor:
    _require_spd(value, location)
    return torch.cholesky_inverse(torch.linalg.cholesky(value))


def _snapshot(value: torch.Tensor) -> H7OwnedTensorSnapshot:
    return H7OwnedTensorSnapshot.capture(value)


def _zero_jacobian_metadata(
    *,
    scope: Literal["generative", "recognition"],
    anchor: H7OwnedTensorSnapshot,
    receiver_component_ids: tuple[str, ...],
) -> H7JacobianMetadataSnapshot:
    if (
        type(receiver_component_ids) is not tuple
        or not receiver_component_ids
        or any(type(item) is not str or not item for item in receiver_component_ids)
        or len(set(receiver_component_ids)) != len(receiver_component_ids)
    ):
        raise ValueError(
            "fixture Jacobian metadata requires unique receiver component IDs"
        )
    anchor_value = anchor.value()
    if anchor_value.dtype != torch.float64 or not anchor_value.numel():
        raise ValueError("fixture Jacobian anchor must be a nonempty float64 tensor")
    zero = anchor_value.reshape(-1).sum() * 0.0
    return H7JacobianMetadataSnapshot.create(
        scope=scope,
        initial_logabsdet=_snapshot(zero),
        receiver_logabsdet={
            component_id: _snapshot(zero) for component_id in receiver_component_ids
        },
        global_logabsdet=_snapshot(zero),
        entropy_shift=(_snapshot(zero) if scope == "recognition" else None),
    )


def _gaussian(
    component_id: str,
    mean: torch.Tensor,
    covariance: torch.Tensor,
    *,
    receiver_t: int | None,
    source_j: int | None,
) -> H7GaussianComponentSnapshot:
    precision = _precision(covariance, f"{component_id}.covariance")
    return H7GaussianComponentSnapshot.create(
        component_id=component_id,
        receiver_t=receiver_t,
        source_j=source_j,
        mean=_snapshot(mean),
        covariance=_snapshot(covariance),
        precision=_snapshot(precision),
        information_vector=_snapshot(precision @ mean),
        second_moment=_snapshot(covariance + torch.outer(mean, mean)),
    )


def _right_solve(value: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.linalg.solve(right.T, value.T).T


def _links(
    frames: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[tuple[int, int], torch.Tensor]:
    return {
        (receiver, source): _right_solve(frames[receiver], frames[source])
        for receiver in range(3)
        for source in range(3)
        if receiver != source
    }


def _action_snapshots(
    raw_actions: Mapping[str, object],
) -> dict[str, H7GLPlus2Action]:
    if set(raw_actions) != set(_ACTION_VALUES):
        raise ValueError("actions must contain the exact three profiles")
    result: dict[str, H7GLPlus2Action] = {}
    for name, expected in _ACTION_VALUES.items():
        value = _tensor(raw_actions[name], (3, 2, 2), f"actions.{name}")
        if (
            tuple(
                tuple(tuple(float(x) for x in row) for row in matrix)
                for matrix in value.tolist()
            )
            != expected
        ):
            raise ValueError(f"actions.{name} changed from the frozen values")
        elements = cast(
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            tuple(value[index] for index in range(3)),
        )
        kind = "internal_product" if name == "internal" else "diagonal_base"
        result[name] = H7GLPlus2Action.create(elements=elements, kind=kind)
    return result


def _frame_snapshots(
    raw_frames: Mapping[str, object],
) -> tuple[
    dict[H7FrameProfile, tuple[H7OwnedTensorSnapshot, ...]],
    dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
]:
    if set(raw_frames) != set(_FRAME_VALUES):
        raise ValueError("frame_profiles must contain identity and nonidentity")
    owned: dict[H7FrameProfile, tuple[H7OwnedTensorSnapshot, ...]] = {}
    tensors: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for name, expected in _FRAME_VALUES.items():
        value = _tensor(raw_frames[name], (3, 2, 2), f"frame_profiles.{name}")
        observed = tuple(
            tuple(tuple(float(x) for x in row) for row in matrix)
            for matrix in value.tolist()
        )
        if observed != expected:
            raise ValueError(f"frame_profiles.{name} changed from the frozen values")
        elements = cast(
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            tuple(value[index] for index in range(3)),
        )
        for element in elements:
            if not bool((torch.linalg.det(element) > 0).item()):
                raise ValueError("population frames require positive determinant")
        tensors[name] = elements
        owned[cast(H7FrameProfile, name)] = tuple(_snapshot(item) for item in elements)
    return owned, tensors


def _history_snapshots(
    values: torch.Tensor, channel: str
) -> tuple[H7HistoryValueSnapshot, ...]:
    return tuple(
        H7HistoryValueSnapshot.create(
            channel=channel,
            population_label=index,
            value=_snapshot(values[index]),
        )
        for index in range(2)
    )


def _source_context(
    profile: Mapping[str, object],
) -> H7SourceContextSnapshot:
    expected_fields = {
        "profile_id",
        "law",
        "prefix_tokens",
        "alpha_bias",
        "alpha_token_scale",
        "z_history",
        "m_history",
        "r_z",
        "r_m",
    }
    root = _fields(profile, expected_fields, "generative.source_scorer_profile")
    if (
        root["profile_id"] != "h7-linear-history-source-v1"
        or root["law"] != "alpha(prefix)+r_z^T z_j+r_m^T m_j"
    ):
        raise ValueError("source scorer identity/law changed")
    prefix_tokens = tuple(
        _integer(item, f"prefix_tokens[{index}]")
        for index, item in enumerate(
            _sequence(root["prefix_tokens"], 2, "prefix_tokens")
        )
    )
    if prefix_tokens != (0, 2):
        raise ValueError("source scorer prefix changed")
    z_values = _tensor(root["z_history"], (2, 2), "z_history")
    m_values = _tensor(root["m_history"], (2, 2), "m_history")
    z_history = _history_snapshots(z_values, "z")
    m_history = _history_snapshots(m_values, "m")
    alpha_bias = _fields(root["alpha_bias"], {"model", "state"}, "alpha_bias")
    alpha_scale = _fields(
        root["alpha_token_scale"],
        {"model", "state"},
        "alpha_token_scale",
    )
    r_z = _fields(root["r_z"], {"model", "state"}, "r_z")
    r_m = _fields(root["r_m"], {"model", "state"}, "r_m")
    rows: list[H7SourceScorerRowSnapshot] = []
    for bank in ("model", "state"):
        biases = _tensor(alpha_bias[bank], (2,), f"alpha_bias.{bank}")
        scales = _tensor(alpha_scale[bank], (2,), f"alpha_scale.{bank}")
        z_covectors = _tensor(r_z[bank], (2, 2), f"r_z.{bank}")
        m_covectors = _tensor(r_m[bank], (2, 2), f"r_m.{bank}")
        for receiver_t in (1, 2):
            source_j = receiver_t - 1
            row_prefix = prefix_tokens[:receiver_t]
            prefix_bytes = json.dumps(row_prefix, separators=(",", ":")).encode("ascii")
            weighted_prefix = sum(
                (index + 1) * (token + 1) for index, token in enumerate(row_prefix)
            )
            bias = float(biases[receiver_t - 1])
            scale = float(scales[receiver_t - 1])
            prefix_term = bias + scale * weighted_prefix
            raw_score = (
                prefix_term
                + float(z_covectors[receiver_t - 1] @ z_values[source_j])
                + float(m_covectors[receiver_t - 1] @ m_values[source_j])
            )
            raw_scores = torch.tensor([raw_score], dtype=torch.float64)
            probabilities = torch.ones(1, dtype=torch.float64)
            z_covector = H7SourceCovectorSnapshot.create(
                bank=bank,
                channel="z",
                receiver_t=receiver_t,
                source_j=source_j,
                value=_snapshot(z_covectors[receiver_t - 1]),
            )
            m_covector = H7SourceCovectorSnapshot.create(
                bank=bank,
                channel="m",
                receiver_t=receiver_t,
                source_j=source_j,
                value=_snapshot(m_covectors[receiver_t - 1]),
            )
            row_semantic = {
                "bank": bank,
                "receiver_t": receiver_t,
                "source_j": source_j,
                "prefix_tokens": row_prefix,
                "prefix_term": prefix_term,
                "z_covector_sha256": z_covector.covector_sha256,
                "m_covector_sha256": m_covector.covector_sha256,
                "raw_score": raw_score,
                "support": (source_j,),
            }
            row_bytes = canonical_h7_bytes(row_semantic)
            rows.append(
                H7SourceScorerRowSnapshot.create(
                    bank=bank,
                    receiver_t=receiver_t,
                    source_j=source_j,
                    prefix_tokens=row_prefix,
                    prefix_bytes=prefix_bytes,
                    prefix_bytes_sha256=hashlib.sha256(prefix_bytes).hexdigest(),
                    alpha_bias=bias,
                    alpha_token_scale=scale,
                    prefix_term=prefix_term,
                    z_history=z_history,
                    m_history=m_history,
                    z_covector=z_covector,
                    m_covector=m_covector,
                    mask=(True,),
                    support=(source_j,),
                    raw_scores=_snapshot(raw_scores),
                    probabilities=_snapshot(probabilities),
                    source_row_raw_bytes=row_bytes,
                    row_raw_bytes_sha256=hashlib.sha256(row_bytes).hexdigest(),
                )
            )
    prefix_bytes = json.dumps(prefix_tokens, separators=(",", ":")).encode("ascii")
    scorer_sha = h7_owned_sha256(
        "vfe4.h7.source-scorer.v1",
        tuple(row.row_sha256 for row in rows),
    )
    return H7SourceContextSnapshot.create(
        prefix_tokens=prefix_tokens,
        prefix_bytes=prefix_bytes,
        prefix_bytes_sha256=hashlib.sha256(prefix_bytes).hexdigest(),
        z_history=z_history,
        m_history=m_history,
        scorer_rows=tuple(rows),
        source_scorer_profile="h7-linear-history-source-v1",
        source_scorer_sha256=scorer_sha,
    )


def _affine(
    *,
    component_id: str,
    bank: str,
    receiver_t: int,
    source_j: int,
    parent_map: torch.Tensor,
    model_map: torch.Tensor | None,
    offset: torch.Tensor,
    covariance: torch.Tensor,
) -> H7AffineComponentSnapshot:
    return H7AffineComponentSnapshot.create(
        component_id=component_id,
        bank=bank,
        receiver_t=receiver_t,
        source_j=source_j,
        parent_map=_snapshot(parent_map),
        same_receiver_model_map=(None if model_map is None else _snapshot(model_map)),
        offset=_snapshot(offset),
        receiver_law=_gaussian(
            f"{component_id}.receiver",
            offset,
            covariance,
            receiver_t=receiver_t,
            source_j=source_j,
        ),
    )


def _build_generative(
    raw: Mapping[str, object],
    frames: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    source_context: H7SourceContextSnapshot,
) -> H7GenerativeSnapshot:
    root = _fields(
        raw,
        {
            "initial_mean",
            "initial_covariance",
            "model_source_probabilities",
            "state_source_probabilities",
            "source_scorer_profile",
            "model_offsets",
            "model_receiver_covariances",
            "state_offsets",
            "state_receiver_covariances",
            "B",
            "decoder",
        },
        "generative",
    )
    initial = _gaussian(
        "p.initial_joint",
        _tensor(root["initial_mean"], (4,), "generative.initial_mean"),
        _tensor(
            root["initial_covariance"],
            (4, 4),
            "generative.initial_covariance",
        ),
        receiver_t=None,
        source_j=None,
    )
    model_offsets = _tensor(root["model_offsets"], (2, 2), "generative.model_offsets")
    model_covariances = _tensor(
        root["model_receiver_covariances"],
        (2, 2, 2),
        "generative.model_receiver_covariances",
    )
    state_offsets = _tensor(root["state_offsets"], (2, 2), "generative.state_offsets")
    state_covariances = _tensor(
        root["state_receiver_covariances"],
        (2, 2, 2),
        "generative.state_receiver_covariances",
    )
    morphisms = _tensor(root["B"], (2, 2, 2), "generative.B")
    links = _links(frames)
    transitions: list[H7AffineComponentSnapshot] = []
    for receiver_t in (1, 2):
        source_j = receiver_t - 1
        transitions.append(
            _affine(
                component_id=f"p.model.receiver_{receiver_t}",
                bank="model",
                receiver_t=receiver_t,
                source_j=source_j,
                parent_map=links[(receiver_t, source_j)],
                model_map=None,
                offset=model_offsets[receiver_t - 1],
                covariance=model_covariances[receiver_t - 1],
            )
        )
        transitions.append(
            _affine(
                component_id=f"p.state.receiver_{receiver_t}",
                bank="state",
                receiver_t=receiver_t,
                source_j=source_j,
                parent_map=links[(receiver_t, source_j)],
                model_map=morphisms[receiver_t - 1],
                offset=state_offsets[receiver_t - 1],
                covariance=state_covariances[receiver_t - 1],
            )
        )
    decoder_values = _sequence(root["decoder"], 2, "generative.decoder")
    decoders: list[H7DecoderSnapshot] = []
    for index, item in enumerate(decoder_values):
        decoder = _fields(item, {"W_z", "W_m", "bias"}, f"generative.decoder[{index}]")
        decoders.append(
            H7DecoderSnapshot.create(
                receiver_t=index + 1,
                state_weight=_snapshot(_tensor(decoder["W_z"], (3, 2), "decoder.W_z")),
                model_weight=_snapshot(_tensor(decoder["W_m"], (3, 2), "decoder.W_m")),
                bias=_snapshot(_tensor(decoder["bias"], (3,), "decoder.bias")),
                centered_stabilizer_class="transformed",
            )
        )
    support_semantic = {
        "model": root["model_source_probabilities"],
        "state": root["state_source_probabilities"],
        "support": _CHAIN,
    }
    return H7GenerativeSnapshot.create(
        frames=tuple(_snapshot(item) for item in frames),
        ordered_links={key: _snapshot(value) for key, value in links.items()},
        initial_joint=initial,
        transitions=tuple(transitions),
        source_context=source_context,
        scalar_source_law=None,
        decoders=tuple(decoders),
        support_sha256=h7_owned_sha256(
            "vfe4.h7.generative-support.v1", support_semantic
        ),
        jacobian=_zero_jacobian_metadata(
            scope="generative",
            anchor=initial.mean,
            receiver_component_ids=tuple(item.component_id for item in transitions),
        ),
    )


def _build_recognition_family(
    *,
    root: Mapping[str, object],
    source_rows: tuple[H7SourceScorerRowSnapshot, ...],
    factorized: bool,
    observation_labels: tuple[int, int],
) -> H7RecognitionSnapshot:
    factor = cast(Mapping[str, object], root["factorized_fixture"])
    if factorized:
        mean = _tensor(
            factor["initial_mean"], (4,), "recognition.factorized.initial_mean"
        )
        initial_covariance = torch.diag(
            _tensor(
                factor["initial_diagonal_covariance"],
                (4,),
                "recognition.factorized.initial_diagonal_covariance",
            )
        )
        model_covariances = tuple(
            torch.diag(row)
            for row in _tensor(
                factor["model_receiver_diagonal_covariances"],
                (2, 2),
                "recognition.factorized.model_covariances",
            )
        )
        state_covariances = tuple(
            torch.diag(row)
            for row in _tensor(
                factor["state_receiver_diagonal_covariances"],
                (2, 2),
                "recognition.factorized.state_covariances",
            )
        )
        origin = "factorized_diagonal_within_fiber"
        representation = "factorized_diagonal_within_fiber"
    else:
        mean = _tensor(root["initial_mean"], (4,), "recognition.initial_mean")
        initial_covariance = _tensor(
            root["initial_covariance"],
            (4, 4),
            "recognition.initial_covariance",
        )
        model_covariances = tuple(
            _tensor(
                root["model_receiver_covariances"],
                (2, 2, 2),
                "recognition.model_receiver_covariances",
            )[index]
            for index in range(2)
        )
        state_covariances = tuple(
            _tensor(
                root["state_receiver_covariances"],
                (2, 2, 2),
                "recognition.state_receiver_covariances",
            )[index]
            for index in range(2)
        )
        origin = "structured_full_block"
        representation = "structured_full_block"
    initial = _gaussian(
        f"q.{origin}.initial_joint",
        mean,
        initial_covariance,
        receiver_t=None,
        source_j=None,
    )
    model_maps = _tensor(
        root["model_parent_maps"],
        (2, 2, 2),
        "recognition.model_parent_maps",
    )
    model_offsets = _tensor(root["model_offsets"], (2, 2), "recognition.model_offsets")
    state_maps = _tensor(
        root["state_parent_maps"],
        (2, 2, 2),
        "recognition.state_parent_maps",
    )
    state_model_maps = _tensor(
        root["state_model_maps"],
        (2, 2, 2),
        "recognition.state_model_maps",
    )
    state_offsets = _tensor(root["state_offsets"], (2, 2), "recognition.state_offsets")
    model_conditionals = tuple(
        _affine(
            component_id=f"q.{origin}.model.receiver_{receiver_t}",
            bank="model",
            receiver_t=receiver_t,
            source_j=receiver_t - 1,
            parent_map=model_maps[receiver_t - 1],
            model_map=None,
            offset=model_offsets[receiver_t - 1],
            covariance=model_covariances[receiver_t - 1],
        )
        for receiver_t in (1, 2)
    )
    state_conditionals = tuple(
        _affine(
            component_id=f"q.{origin}.state.receiver_{receiver_t}",
            bank="state",
            receiver_t=receiver_t,
            source_j=receiver_t - 1,
            parent_map=state_maps[receiver_t - 1],
            model_map=state_model_maps[receiver_t - 1],
            offset=state_offsets[receiver_t - 1],
            covariance=state_covariances[receiver_t - 1],
        )
        for receiver_t in (1, 2)
    )
    context = H7RecognitionContextSnapshot.create(
        observation_labels=observation_labels,
        conditioning="smoothing",
    )
    return H7RecognitionSnapshot.create(
        origin_family=origin,
        representation=representation,
        initial_joint=initial,
        model_conditionals=model_conditionals,
        state_conditionals=state_conditionals,
        source_rows=source_rows,
        context=context,
        scalar_source_law=None,
        jacobian=_zero_jacobian_metadata(
            scope="recognition",
            anchor=initial.mean,
            receiver_component_ids=tuple(
                item.component_id for item in (*model_conditionals, *state_conditionals)
            ),
        ),
    )


def _build_recognitions(
    raw: Mapping[str, object],
    source_context: H7SourceContextSnapshot,
    observation_labels: tuple[int, int],
) -> tuple[H7RecognitionSnapshot, H7RecognitionSnapshot]:
    expected = {
        "family_id",
        "initial_mean",
        "initial_covariance",
        "model_source_probabilities",
        "state_source_probabilities_given_model_source",
        "model_parent_maps",
        "model_offsets",
        "model_receiver_covariances",
        "state_parent_maps",
        "state_model_maps",
        "state_offsets",
        "state_receiver_covariances",
        "factorized_fixture",
    }
    root = _fields(raw, expected, "recognition")
    factor = _fields(
        root["factorized_fixture"],
        {
            "family_id",
            "representation",
            "shared_fields",
            "initial_mean",
            "initial_diagonal_covariance",
            "model_receiver_diagonal_covariances",
            "state_receiver_diagonal_covariances",
            "generic_gl_plus_2_output_representation",
        },
        "recognition.factorized_fixture",
    )
    if (
        root["family_id"] != "structured-full-block-v1"
        or factor["family_id"] != "factorized-diagonal-within-fiber-v1"
        or factor["representation"] != "factorized_diagonal_within_fiber"
        or factor["generic_gl_plus_2_output_representation"]
        != "unrestricted_full_block_pushforward"
    ):
        raise ValueError("recognition family declarations changed")
    expected_shared = (
        "model_source_probabilities",
        "state_source_probabilities_given_model_source",
        "model_parent_maps",
        "model_offsets",
        "state_parent_maps",
        "state_model_maps",
        "state_offsets",
    )
    if tuple(factor["shared_fields"]) != expected_shared:
        raise ValueError("factorized shared-field declaration changed")
    source_rows = source_context.scorer_rows
    return (
        _build_recognition_family(
            root=root,
            source_rows=source_rows,
            factorized=False,
            observation_labels=observation_labels,
        ),
        _build_recognition_family(
            root=root,
            source_rows=source_rows,
            factorized=True,
            observation_labels=observation_labels,
        ),
    )


def _matrix_trial_specs(
    actions: Mapping[str, H7GLPlus2Action],
) -> tuple[H7TrialSpec, ...]:
    declarations = (
        (
            "matrix-identity-base-transformed",
            "positive_covariance",
            "complete_covariance",
            "identity",
            "transform",
            "diagonal",
        ),
        (
            "matrix-identity-internal-transformed",
            "positive_covariance",
            "complete_covariance",
            "identity",
            "transform",
            "internal",
        ),
        (
            "matrix-nonidentity-base-transformed",
            "positive_covariance",
            "complete_covariance",
            "nonidentity",
            "transform",
            "diagonal",
        ),
        (
            "matrix-nonidentity-internal-transformed",
            "positive_covariance",
            "complete_covariance",
            "nonidentity",
            "transform",
            "internal",
        ),
        (
            "matrix-fixed-decoder-centered-stabilizer",
            "positive_covariance",
            "centered_decoder_stabilizer_invariance",
            "nonidentity",
            "fixed",
            "fixed_decoder_stabilizer",
        ),
        (
            "matrix-fixed-decoder-outside-stabilizer",
            "expected_negative",
            "decisive_outside_stabilizer_change",
            "nonidentity",
            "fixed",
            "diagonal",
        ),
    )
    result = tuple(
        H7TrialSpec.create(
            trial_id=trial_id,
            role=role,
            expected_predicate=predicate,
            fixture_id="h7-v1",
            frame_profile=frame_profile,
            decoder_policy=decoder_policy,
            action=actions[action_name],
            action_sha256=actions[action_name].action_sha256,
        )
        for (
            trial_id,
            role,
            predicate,
            frame_profile,
            decoder_policy,
            action_name,
        ) in declarations
    )
    if tuple(item.trial_id for item in result) != H7_MATRIX_TRIAL_IDS:
        raise ValueError("matrix trial order drifted")
    return result


def h7_scalar_trial_specs() -> tuple[H7TrialSpec, H7TrialSpec]:
    """Construct the two separately typed scalar-regression trial specs."""

    base = H7ScalarReplayAction.create(
        elements=cast(
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            tuple(torch.tensor([[1.25]], dtype=torch.float64) for _ in range(3)),
        ),
        kind="diagonal_base",
    )
    internal = H7ScalarReplayAction.create(
        elements=cast(
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            tuple(
                torch.tensor([[item]], dtype=torch.float64) for item in (0.8, 1.1, 1.4)
            ),
        ),
        kind="internal_product",
    )
    return (
        H7TrialSpec.create(
            trial_id="scalar-base-transformed",
            role="scalar_regression",
            expected_predicate="complete_covariance",
            fixture_id="h1-v1",
            frame_profile="h1_v1",
            decoder_policy="transform",
            action=base,
            action_sha256=base.action_sha256,
        ),
        H7TrialSpec.create(
            trial_id="scalar-internal-transformed",
            role="scalar_regression",
            expected_predicate="complete_covariance",
            fixture_id="h1-v1",
            frame_profile="h1_v1",
            decoder_policy="transform",
            action=internal,
            action_sha256=internal.action_sha256,
        ),
    )


def _direction_ids(dimension: int) -> tuple[str, ...]:
    values = ["zero"]
    for index in range(dimension):
        values.extend((f"+e{index}", f"-e{index}"))
    return tuple(values)


def _load_probe_pairs(
    actions: Mapping[str, H7GLPlus2Action],
) -> tuple[H7DensityProbePair, ...]:
    table = _parse_exact_json(
        H7_DENSITY_PROBE_TABLE_PATH.read_bytes(),
        expected_sha256=H7_DENSITY_PROBE_TABLE_RAW_SHA256,
    )
    root = _fields(
        table,
        {"probe_table_schema", "probe_set_sha256", "records"},
        "density_probe_table",
    )
    if (
        root["probe_table_schema"] != "h7-density-probe-table-v1"
        or root["probe_set_sha256"] != H7_DENSITY_PROBE_SET_SHA256
    ):
        raise ValueError("density-probe table identity changed")
    raw_records = _sequence(root["records"], 486, "density_probe_table.records")
    pairs: list[H7DensityProbePair] = []
    row_index = 0
    for action_name in (
        "diagonal",
        "fixed_decoder_stabilizer",
        "internal",
    ):
        action_sha256 = actions[action_name].action_sha256
        for component_id, source_id, dimension, _scope in _PROBE_COMPONENTS:
            for direction_id in _direction_ids(dimension):
                location = f"density_probe_table.records[{row_index}]"
                record = _fields(
                    raw_records[row_index],
                    {
                        "row_index",
                        "probe_id",
                        "fixture_id",
                        "component_id",
                        "source_id",
                        "action_sha256",
                        "anchor_sha256",
                        "anchor_provenance",
                        "x",
                        "x_prime",
                        "initial_log_jacobian_shift",
                        "receiver_log_jacobian_shift",
                        "global_log_jacobian_shift",
                        "probe_sha256",
                    },
                    location,
                )
                expected_probe_id = f"{action_name}:{component_id}:{direction_id}"
                if (
                    _integer(record["row_index"], f"{location}.row_index") != row_index
                    or record["probe_id"] != expected_probe_id
                    or record["fixture_id"] != "h7-v1"
                    or record["component_id"] != component_id
                    or record["source_id"] != source_id
                    or record["action_sha256"] != action_sha256
                    or record["anchor_provenance"]
                    != "raw_fixture_component_mean_and_lower_cholesky_v1"
                ):
                    raise ValueError(
                        f"{location} changed from the frozen order/identity"
                    )
                pair = H7DensityProbePair.create(
                    probe_id=expected_probe_id,
                    fixture_id="h7-v1",
                    component_id=component_id,
                    source_id=source_id,
                    action_sha256=action_sha256,
                    anchor_sha256=cast(str, record["anchor_sha256"]),
                    anchor_provenance=cast(str, record["anchor_provenance"]),
                    x=_snapshot(_tensor(record["x"], (dimension,), f"{location}.x")),
                    x_prime=_snapshot(
                        _tensor(
                            record["x_prime"],
                            (dimension,),
                            f"{location}.x_prime",
                        )
                    ),
                    initial_log_jacobian_shift=_number(
                        record["initial_log_jacobian_shift"],
                        f"{location}.initial_log_jacobian_shift",
                    ),
                    receiver_log_jacobian_shift=_number(
                        record["receiver_log_jacobian_shift"],
                        f"{location}.receiver_log_jacobian_shift",
                    ),
                    global_log_jacobian_shift=_number(
                        record["global_log_jacobian_shift"],
                        f"{location}.global_log_jacobian_shift",
                    ),
                )
                if pair.probe_sha256 != record["probe_sha256"]:
                    raise ValueError(f"{location}.probe_sha256 does not bind the row")
                pairs.append(pair)
                row_index += 1
    if row_index != len(raw_records):
        raise ValueError("density-probe table contains unconsumed rows")
    return tuple(pairs)


def _validate_probe_inventory(raw: Mapping[str, object]) -> None:
    root = _fields(
        raw,
        {
            "probe_set_schema",
            "whitened_scale",
            "anchor_policy",
            "anchor_provenance",
            "pair_law",
            "direction_ids_by_dimension",
            "components",
        },
        "density_probes",
    )
    if (
        root["probe_set_schema"] != "h7-density-probe-pairs-v1"
        or _number(root["whitened_scale"], "whitened_scale") != 0.25
        or root["anchor_policy"] != "original_component_mean"
        or root["anchor_provenance"]
        != "raw_fixture_component_mean_and_lower_cholesky_v1"
        or root["pair_law"] != "x=anchor+L@(scale*direction);x_prime=G_component@x"
    ):
        raise ValueError("density-probe policy changed")
    directions = _fields(
        root["direction_ids_by_dimension"], {"2", "4", "12"}, "directions"
    )
    for dimension in (2, 4, 12):
        observed = tuple(directions[str(dimension)])
        expected = _direction_ids(dimension)
        if observed != expected:
            raise ValueError(f"density directions for dimension {dimension} changed")
    observed_components = tuple(
        (
            item["component_id"],
            item["source_id"],
            item["dimension"],
            item["shift_scope"],
        )
        for item in _sequence(root["components"], 18, "components")
        if type(item) is dict
    )
    if observed_components != _PROBE_COMPONENTS:
        raise ValueError("density-probe component inventory changed")


def parse_h7_fixture_bytes(data: bytes) -> H7Fixture:
    """Parse only the exact frozen H7-v1 raw bytes into owned records."""

    root = _parse_exact_json(data, expected_sha256=H7_FIXTURE_RAW_SHA256)
    _fields(root, _ROOT_FIELDS, "fixture")
    if (
        _integer(root["fixture_schema_version"], "fixture_schema_version") != 1
        or root["fixture_id"] != "h7-v1"
        or root["group"] != "GL+(2,R)"
        or _integer(root["horizon"], "horizon") != 2
        or _integer(root["observation_label_base"], "observation_label_base") != 0
    ):
        raise ValueError("H7 fixture identity changed")
    representations = _fields(
        root["representations"], {"state", "model"}, "representations"
    )
    if representations != {"state": "standard", "model": "standard"}:
        raise ValueError("H7 supports only standard representations")
    dimensions = _fields(root["dimensions"], {"d_z", "d_m", "D", "V"}, "dimensions")
    if tuple(dimensions[key] for key in ("d_z", "d_m", "D", "V")) != (
        2,
        2,
        12,
        3,
    ):
        raise ValueError("H7 dimensions changed")
    if tuple(root["continuous_order"]) != _CONTINUOUS_ORDER:
        raise ValueError("H7 continuous order changed")
    for name in (
        "state_parent_sets",
        "model_parent_sets",
        "state_source_support",
        "model_source_support",
    ):
        if tuple(tuple(row) for row in root[name]) != _CHAIN:
            raise ValueError(f"{name} must remain the frozen chain")
    observation_labels = tuple(
        _integer(item, f"observation_labels[{index}]")
        for index, item in enumerate(
            _sequence(root["observation_labels"], 2, "observation_labels")
        )
    )
    if observation_labels != (0, 2):
        raise ValueError("observation labels changed")
    oracle = _fields(
        root["oracle"], {"decimal_precision", "gauss_hermite_orders"}, "oracle"
    )
    if oracle != {"decimal_precision": 100, "gauss_hermite_orders": [41, 51]}:
        raise ValueError("H7 oracle settings changed")
    _validate_probe_inventory(cast(Mapping[str, object], root["density_probes"]))

    frame_profiles, frame_tensors = _frame_snapshots(
        cast(Mapping[str, object], root["frame_profiles"])
    )
    actions = _action_snapshots(cast(Mapping[str, object], root["actions"]))
    generative_raw = cast(Mapping[str, object], root["generative"])
    source_context = _source_context(
        cast(Mapping[str, object], generative_raw["source_scorer_profile"])
    )
    generative = _build_generative(
        generative_raw, frame_tensors["nonidentity"], source_context
    )
    recognitions = _build_recognitions(
        cast(Mapping[str, object], root["recognition"]),
        source_context,
        cast(tuple[int, int], observation_labels),
    )
    trial_specs = _matrix_trial_specs(actions)
    probe_pairs = _load_probe_pairs(actions)
    expected_probe_identity = h7_owned_sha256(
        "vfe4.h7.density-probe-set.v1",
        probe_pairs,
    )
    if expected_probe_identity != H7_DENSITY_PROBE_SET_SHA256:
        raise ValueError("density-probe-set source constant drifted")
    return H7Fixture.create(
        fixture_id="h7-v1",
        raw_fixture_sha256=H7_FIXTURE_RAW_SHA256,
        frame_profiles=frame_profiles,
        actions=actions,
        generative=generative,
        recognition_families=recognitions,
        matrix_trial_specs=trial_specs,
        density_probe_pairs=probe_pairs,
        density_probe_table_raw_sha256=(H7_DENSITY_PROBE_TABLE_RAW_SHA256),
        density_probe_set_sha256=H7_DENSITY_PROBE_SET_SHA256,
    )


def _adapt_h1(data: bytes) -> H7CompleteLawSnapshot:
    root = _parse_exact_json(data, expected_sha256=H1_FIXTURE_RAW_SHA256)
    if root.get("fixture_id") != "h1-v1":
        raise ValueError("H1 fixture identity changed")
    observation_label_base = _integer(
        root["observation_label_base"], "h1.observation_label_base"
    )
    observation_labels = tuple(
        _integer(item, f"h1.observation_labels[{index}]")
        for index, item in enumerate(
            _sequence(root["observation_labels"], 2, "h1.observation_labels")
        )
    )
    if observation_label_base != 1 or observation_labels != (1, 2):
        raise ValueError("H1 observation labels must remain one-based (1, 2)")
    path_declarations = (
        ("h1-path-0:a0-b0", (0, 0), (0, 0), (0, 0), (0, 0)),
        ("h1-path-1:a1-b0", (0, 1), (0, 0), (0, 0), (0, 1)),
        ("h1-path-2:a0-b1", (0, 0), (0, 1), (0, 1), (0, 2)),
        ("h1-path-3:a1-b1", (0, 1), (0, 1), (0, 1), (0, 3)),
    )
    ordered_paths = tuple(
        H7ScalarSourcePathSnapshot.create(
            path_id=path_id,
            a=a,
            b=b,
            model_kernel_selectors=model_selectors,
            state_kernel_selectors=state_selectors,
            observation_label_base=observation_label_base,
            observation_labels=observation_labels,
            decoder_row_indices=tuple(
                label - observation_label_base for label in observation_labels
            ),
        )
        for (
            path_id,
            a,
            b,
            model_selectors,
            state_selectors,
        ) in path_declarations
    )
    frames_raw = _tensor(root["frames"], (3,), "h1.frames")
    frames = cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple(item.reshape(1, 1) for item in frames_raw),
    )
    links = _links(frames)
    initial = _gaussian(
        "h1.p.initial_joint",
        _tensor(root["initial_joint"]["mean"], (2,), "h1.initial.mean"),
        _tensor(
            root["initial_joint"]["covariance"],
            (2, 2),
            "h1.initial.covariance",
        ),
        receiver_t=None,
        source_j=None,
    )
    model_offsets = _tensor(root["model_offsets"], (2,), "h1.model_offsets")
    state_offsets = _tensor(root["state_offsets"], (2,), "h1.state_offsets")
    model_variances = _tensor(root["model_variances"], (2,), "h1.model_variances")
    state_variances = _tensor(root["state_variances"], (2,), "h1.state_variances")
    slopes = _tensor(root["state_model_slopes"], (2,), "h1.state_model_slopes")
    supports = ((0,), (0, 1))
    transitions: list[H7AffineComponentSnapshot] = []
    for receiver_t in (1, 2):
        for source_j in supports[receiver_t - 1]:
            transitions.append(
                _affine(
                    component_id=f"h1.p.model.{receiver_t}<-{source_j}",
                    bank="model",
                    receiver_t=receiver_t,
                    source_j=source_j,
                    parent_map=links[(receiver_t, source_j)],
                    model_map=None,
                    offset=model_offsets[receiver_t - 1].reshape(1),
                    covariance=model_variances[receiver_t - 1].reshape(1, 1),
                )
            )
            transitions.append(
                _affine(
                    component_id=f"h1.p.state.{receiver_t}<-{source_j}",
                    bank="state",
                    receiver_t=receiver_t,
                    source_j=source_j,
                    parent_map=links[(receiver_t, source_j)],
                    model_map=slopes[receiver_t - 1].reshape(1, 1),
                    offset=state_offsets[receiver_t - 1].reshape(1),
                    covariance=state_variances[receiver_t - 1].reshape(1, 1),
                )
            )
    decoders = tuple(
        H7DecoderSnapshot.create(
            receiver_t=index + 1,
            state_weight=_snapshot(
                _tensor(item["w_z"], (3,), "h1.decoder.w_z").reshape(3, 1)
            ),
            model_weight=_snapshot(
                _tensor(item["w_m"], (3,), "h1.decoder.w_m").reshape(3, 1)
            ),
            bias=_snapshot(_tensor(item["bias"], (3,), "h1.decoder.bias")),
            centered_stabilizer_class="transformed",
        )
        for index, item in enumerate(root["decoder"])
    )
    model_source_priors_raw = _sequence(
        root["model_source_priors"], 2, "h1.model_source_priors"
    )
    state_source_priors_raw = _sequence(
        root["state_source_priors"], 2, "h1.state_source_priors"
    )
    generative_source_law = H7ScalarGenerativeSourceLawSnapshot.create(
        model_source_priors=(
            _snapshot(
                _tensor(
                    model_source_priors_raw[0],
                    (1,),
                    "h1.model_source_priors[0]",
                )
            ),
            _snapshot(
                _tensor(
                    model_source_priors_raw[1],
                    (2,),
                    "h1.model_source_priors[1]",
                )
            ),
        ),
        state_source_priors=(
            _snapshot(
                _tensor(
                    state_source_priors_raw[0],
                    (1,),
                    "h1.state_source_priors[0]",
                )
            ),
            _snapshot(
                _tensor(
                    state_source_priors_raw[1],
                    (2,),
                    "h1.state_source_priors[1]",
                )
            ),
        ),
        ordered_paths=ordered_paths,
    )
    generative = H7GenerativeSnapshot.create(
        frames=tuple(_snapshot(item) for item in frames),
        ordered_links={key: _snapshot(value) for key, value in links.items()},
        initial_joint=initial,
        transitions=tuple(transitions),
        source_context=None,
        scalar_source_law=generative_source_law,
        decoders=decoders,
        support_sha256=h7_owned_sha256(
            "vfe4.h7.h1-generative-support.v1",
            {
                "model_source_priors": root["model_source_priors"],
                "state_source_priors": root["state_source_priors"],
                "source_path_order": tuple(
                    {
                        "path_id": path.path_id,
                        "a": path.a,
                        "b": path.b,
                    }
                    for path in ordered_paths
                ),
                "observation_label_base": observation_label_base,
                "observation_labels": observation_labels,
            },
        ),
        jacobian=_zero_jacobian_metadata(
            scope="generative",
            anchor=initial.mean,
            receiver_component_ids=tuple(item.component_id for item in transitions),
        ),
    )
    recognition_raw = cast(Mapping[str, object], root["recognition"])
    recognition_initial = _gaussian(
        "h1.q.initial_joint",
        _tensor(recognition_raw["initial_mean"], (2,), "h1.q.initial_mean"),
        _tensor(
            recognition_raw["initial_covariance"],
            (2, 2),
            "h1.q.initial_covariance",
        ),
        receiver_t=None,
        source_j=None,
    )
    model_conditionals: list[H7AffineComponentSnapshot] = []
    for time, rows in enumerate(recognition_raw["model_kernels"], start=1):
        for source_j, row in enumerate(rows):
            model_conditionals.append(
                _affine(
                    component_id=f"h1.q.model.{time}<-{source_j}",
                    bank="model",
                    receiver_t=time,
                    source_j=source_j,
                    parent_map=torch.tensor(
                        [[_number(row["slope"], "h1.q.model.slope")]],
                        dtype=torch.float64,
                    ),
                    model_map=None,
                    offset=torch.tensor(
                        [_number(row["offset"], "h1.q.model.offset")],
                        dtype=torch.float64,
                    ),
                    covariance=torch.tensor(
                        [[_number(row["variance"], "h1.q.model.variance")]],
                        dtype=torch.float64,
                    ),
                )
            )
    state_conditionals: list[H7AffineComponentSnapshot] = []
    for time, rows in enumerate(recognition_raw["state_kernels"], start=1):
        observed_paths: list[tuple[int, int]] = []
        for index, row in enumerate(rows):
            a = _integer(row["a"], "h1.q.state.a") if time == 2 else 0
            b = _integer(row["b"], "h1.q.state.b") if time == 2 else 0
            observed_paths.append((a, b))
            state_conditionals.append(
                _affine(
                    component_id=(f"h1.q.state.{time}.a_{a}.b_{b}.row_{index}"),
                    bank="state",
                    receiver_t=time,
                    source_j=b,
                    parent_map=torch.tensor(
                        [[_number(row["z_slope"], "h1.q.state.z_slope")]],
                        dtype=torch.float64,
                    ),
                    model_map=torch.tensor(
                        [[_number(row["m_slope"], "h1.q.state.m_slope")]],
                        dtype=torch.float64,
                    ),
                    offset=torch.tensor(
                        [_number(row["offset"], "h1.q.state.offset")],
                        dtype=torch.float64,
                    ),
                    covariance=torch.tensor(
                        [[_number(row["variance"], "h1.q.state.variance")]],
                        dtype=torch.float64,
                    ),
                )
            )
        expected_paths = ((0, 0),) if time == 1 else ((0, 0), (1, 0), (0, 1), (1, 1))
        if tuple(observed_paths) != expected_paths:
            raise ValueError("H1 recognition state-kernel (a,b) order changed")
    context = H7RecognitionContextSnapshot.create(
        observation_labels=observation_labels,
        conditioning="smoothing",
    )
    model_source_probabilities_raw = _sequence(
        recognition_raw["model_source_probabilities"],
        2,
        "h1.q.model_source_probabilities",
    )
    state_source_probabilities_raw = _sequence(
        recognition_raw["state_source_probabilities_given_model_source"],
        2,
        "h1.q.state_source_probabilities_given_model_source",
    )
    recognition_source_law = H7ScalarRecognitionSourceLawSnapshot.create(
        model_source_probabilities=(
            _snapshot(
                _tensor(
                    model_source_probabilities_raw[0],
                    (1,),
                    "h1.q.model_source_probabilities[0]",
                )
            ),
            _snapshot(
                _tensor(
                    model_source_probabilities_raw[1],
                    (2,),
                    "h1.q.model_source_probabilities[1]",
                )
            ),
        ),
        state_source_probabilities_given_model_source=(
            _snapshot(
                _tensor(
                    state_source_probabilities_raw[0],
                    (1, 1),
                    "h1.q.state_source_probabilities[0]",
                )
            ),
            _snapshot(
                _tensor(
                    state_source_probabilities_raw[1],
                    (2, 2),
                    "h1.q.state_source_probabilities[1]",
                )
            ),
        ),
        ordered_paths=ordered_paths,
    )
    recognition = H7RecognitionSnapshot.create(
        origin_family="structured_full_block",
        representation="structured_full_block",
        initial_joint=recognition_initial,
        model_conditionals=tuple(model_conditionals),
        state_conditionals=tuple(state_conditionals),
        source_rows=(),
        context=context,
        scalar_source_law=recognition_source_law,
        jacobian=_zero_jacobian_metadata(
            scope="recognition",
            anchor=recognition_initial.mean,
            receiver_component_ids=tuple(
                item.component_id for item in (*model_conditionals, *state_conditionals)
            ),
        ),
    )
    scalar_trial_specs = h7_scalar_trial_specs()
    scalar_probe_pairs: list[H7DensityProbePair] = []
    for trial_index, trial_spec in enumerate(scalar_trial_specs):
        for path_index, path in enumerate(ordered_paths):
            x = _snapshot(
                torch.tensor(
                    _H1_SCALAR_PATH_MEANS[path_index],
                    dtype=torch.float64,
                )
            )
            x_prime = _snapshot(
                torch.tensor(
                    _H1_SCALAR_PATH_PRIMES[trial_index][path_index],
                    dtype=torch.float64,
                )
            )
            anchor_provenance = (
                f"{_H1_SCALAR_PROBE_ANCHOR_PROFILE};"
                f"raw_h1_sha256={H1_FIXTURE_RAW_SHA256};"
                f"source_id={path.path_id}"
            )
            anchor_sha256 = h7_owned_sha256(
                "vfe4.h7.scalar-density-anchor.v1",
                {
                    "raw_fixture_sha256": H1_FIXTURE_RAW_SHA256,
                    "source_id": path.path_id,
                    "anchor": x,
                },
            )
            scalar_probe_pairs.append(
                H7DensityProbePair.create(
                    probe_id=(
                        f"{trial_spec.trial_id}:h1.p.global.source_path:{path.path_id}"
                    ),
                    fixture_id="h1-v1",
                    component_id="h1.p.global.source_path",
                    source_id=path.path_id,
                    action_sha256=trial_spec.action_sha256,
                    anchor_sha256=anchor_sha256,
                    anchor_provenance=anchor_provenance,
                    x=x,
                    x_prime=x_prime,
                    initial_log_jacobian_shift=0.0,
                    receiver_log_jacobian_shift=0.0,
                    global_log_jacobian_shift=(
                        _H1_SCALAR_GLOBAL_LOG_JACOBIAN_SHIFTS[trial_index]
                    ),
                )
            )
    scalar_probe_set = H7ScalarProbeSetSnapshot.create(
        raw_fixture_sha256=H1_FIXTURE_RAW_SHA256,
        ordered_source_path_ids=tuple(path.path_id for path in ordered_paths),
        scalar_trial_action_sha256=tuple(
            spec.action_sha256 for spec in scalar_trial_specs
        ),
        anchor_provenance=_H1_SCALAR_PROBE_ANCHOR_PROFILE,
        probe_pairs=tuple(scalar_probe_pairs),
    )
    return H7CompleteLawSnapshot.create(
        fixture_id="h1-v1",
        generative=generative,
        recognition=recognition,
        raw_fixture_sha256=H1_FIXTURE_RAW_SHA256,
        scalar_probe_set=scalar_probe_set,
    )


def adapt_optional_h1_fixture_bytes(
    data: bytes | None,
    *,
    required_scalar_trials: tuple[H7TrialId, ...],
) -> H7CompleteLawSnapshot | None:
    """Adapt H1 bytes only under the exact scalar-trial requirement."""

    if type(required_scalar_trials) is not tuple:
        raise ValueError("required_scalar_trials must be an exact tuple")
    if required_scalar_trials == ():
        if data is not None:
            raise ValueError("unused H1 fixture bytes must be absent")
        return None
    if required_scalar_trials != H7_SCALAR_TRIAL_IDS:
        raise ValueError("required scalar trials must be the exact frozen pair")
    if data is None:
        raise ValueError("required H1 fixture bytes are missing")
    return _adapt_h1(data)


def h7_validation_config_mapping() -> dict[str, object]:
    """Return a fresh mutable rendering of the frozen H7 config section."""

    actions = {
        name: [[list(row) for row in matrix] for matrix in matrices]
        for name, matrices in _ACTION_VALUES.items()
    }
    trials = [
        {
            "trial_id": item[0],
            "role": item[1],
            "expected_predicate": item[2],
            "fixture_id": item[3],
            "frame_profile": item[4],
            "decoder_policy": item[5],
            "action_profile": item[6],
        }
        for item in (
            (
                "scalar-base-transformed",
                "scalar_regression",
                "complete_covariance",
                "h1-v1",
                "h1_v1",
                "transform",
                "scalar_base",
            ),
            (
                "scalar-internal-transformed",
                "scalar_regression",
                "complete_covariance",
                "h1-v1",
                "h1_v1",
                "transform",
                "scalar_internal",
            ),
            (
                "matrix-identity-base-transformed",
                "positive_covariance",
                "complete_covariance",
                "h7-v1",
                "identity",
                "transform",
                "diagonal",
            ),
            (
                "matrix-identity-internal-transformed",
                "positive_covariance",
                "complete_covariance",
                "h7-v1",
                "identity",
                "transform",
                "internal",
            ),
            (
                "matrix-nonidentity-base-transformed",
                "positive_covariance",
                "complete_covariance",
                "h7-v1",
                "nonidentity",
                "transform",
                "diagonal",
            ),
            (
                "matrix-nonidentity-internal-transformed",
                "positive_covariance",
                "complete_covariance",
                "h7-v1",
                "nonidentity",
                "transform",
                "internal",
            ),
            (
                "matrix-fixed-decoder-centered-stabilizer",
                "positive_covariance",
                "centered_decoder_stabilizer_invariance",
                "h7-v1",
                "nonidentity",
                "fixed",
                "fixed_decoder_stabilizer",
            ),
            (
                "matrix-fixed-decoder-outside-stabilizer",
                "expected_negative",
                "decisive_outside_stabilizer_change",
                "h7-v1",
                "nonidentity",
                "fixed",
                "diagonal",
            ),
        )
    ]
    return {
        "schema_version": "h7-validation-config-v1",
        "operation": "H7",
        "group": "GL+(2,R)",
        "representations": {"state": "standard", "model": "standard"},
        "actions": actions,
        "required_trials": trials,
        "required_control_ids": list(H7_CONTROL_IDS),
        "recognition_families": [
            "structured_full_block",
            "factorized_diagonal_within_fiber",
        ],
        "h1_fixture_raw_sha256": H1_FIXTURE_RAW_SHA256,
        "h7_fixture_raw_sha256": H7_FIXTURE_RAW_SHA256,
        "density_probe_table_raw_sha256": (H7_DENSITY_PROBE_TABLE_RAW_SHA256),
        "density_probe_set_sha256": H7_DENSITY_PROBE_SET_SHA256,
        "oracle_decimal_precision": 100,
        "gauss_hermite_orders": [41, 51],
        "group_norm_limit": 2.0,
        "group_inverse_norm_limit": 2.0,
        "spd_condition_limit": 1000.0,
        "predecessor_keys": ["h1_h5", "h1_prefix_prior", "h6_prefix"],
    }


def h7_trial_specs_from_config(
    raw: Mapping[str, object],
) -> tuple[H7TrialSpec, ...]:
    """Build the closed trial tuple from an already strict H7 config section."""

    def exact_structure_equal(value: object, expected: object) -> bool:
        if type(value) is not type(expected):
            return False
        if type(expected) is dict:
            observed_mapping = cast(dict[object, object], value)
            expected_mapping = cast(dict[object, object], expected)
            return observed_mapping.keys() == expected_mapping.keys() and all(
                exact_structure_equal(
                    observed_mapping[key],
                    expected_mapping[key],
                )
                for key in expected_mapping
            )
        if type(expected) in (list, tuple):
            observed_sequence = cast(list[object] | tuple[object, ...], value)
            expected_sequence = cast(
                list[object] | tuple[object, ...],
                expected,
            )
            return len(observed_sequence) == len(expected_sequence) and all(
                exact_structure_equal(observed, frozen)
                for observed, frozen in zip(
                    observed_sequence,
                    expected_sequence,
                    strict=True,
                )
            )
        return value == expected

    if not exact_structure_equal(raw, h7_validation_config_mapping()):
        raise ValueError("H7 config section differs from the frozen mapping")
    actions = _action_snapshots(cast(Mapping[str, object], raw["actions"]))
    return (*h7_scalar_trial_specs(), *_matrix_trial_specs(actions))


__all__ = [
    "H1_FIXTURE_RAW_SHA256",
    "H7_DENSITY_PROBE_EXPANSION",
    "H7_DENSITY_PROBE_SET_SHA256",
    "H7_DENSITY_PROBE_TABLE_PATH",
    "H7_DENSITY_PROBE_TABLE_RAW_SHA256",
    "H7_FIXTURE_PATH",
    "H7_FIXTURE_RAW_SHA256",
    "adapt_optional_h1_fixture_bytes",
    "h7_scalar_trial_specs",
    "h7_trial_specs_from_config",
    "h7_validation_config_mapping",
    "parse_h7_fixture_bytes",
]
