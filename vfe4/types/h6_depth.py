"""Frozen records for the nonblocking H6 two-layer composition probe."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, TypeAlias


Layer: TypeAlias = Literal[1, 2]
Channel: TypeAlias = Literal["state", "model"]
SourceConditioning: TypeAlias = Literal["causal_latent_prefix_only"]


def _finite(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite float")
    return value


def _positive(value: object, name: str) -> float:
    checked = _finite(value, name)
    if checked <= 0.0:
        raise ValueError(f"{name} must be positive")
    return checked


def _probability_row(
    values: object,
    length: int,
    name: str,
) -> tuple[float, ...]:
    if type(values) is not tuple or len(values) != length:
        raise ValueError(f"{name} must contain {length} probabilities")
    checked = tuple(_positive(value, name) for value in values)
    if not math.isclose(
        math.fsum(checked),
        1.0,
        rel_tol=0.0,
        abs_tol=2.0e-15,
    ):
        raise ValueError(f"{name} must sum to one")
    return checked


@dataclass(frozen=True, slots=True)
class GaussianMarginal:
    """One scalar mean-field recognition marginal."""

    mean: float
    variance: float

    def __post_init__(self) -> None:
        _finite(self.mean, "mean")
        _positive(self.variance, "variance")


@dataclass(frozen=True, slots=True)
class ScalarGaussianRegression:
    """One normalized scalar Gaussian conditional with named predictors."""

    intercept: float
    coefficients: tuple[tuple[str, float], ...]
    variance: float

    def __post_init__(self) -> None:
        _finite(self.intercept, "intercept")
        _positive(self.variance, "variance")
        if type(self.coefficients) is not tuple:
            raise ValueError("coefficients must be an immutable tuple")
        labels: list[str] = []
        for item in self.coefficients:
            if (
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not str
                or not item[0]
            ):
                raise ValueError(
                    "each coefficient must be a (nonempty label, float) tuple"
                )
            labels.append(item[0])
            _finite(item[1], f"coefficient {item[0]}")
        if len(labels) != len(set(labels)) or labels != sorted(labels):
            raise ValueError("coefficient labels must be unique and sorted")

    @property
    def predictor_labels(self) -> tuple[str, ...]:
        return tuple(label for label, _ in self.coefficients)


@dataclass(frozen=True, slots=True)
class ConditionalSourceRow:
    """One normalized target-blind row over an explicit causal parent set."""

    receiver_t: int
    parents: tuple[int, ...]
    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        if type(self.receiver_t) is not int or self.receiver_t < 1:
            raise ValueError("receiver_t must be a positive integer")
        if (
            type(self.parents) is not tuple
            or not self.parents
            or any(type(parent) is not int for parent in self.parents)
            or tuple(sorted(set(self.parents))) != self.parents
            or self.parents[0] < 0
            or self.parents[-1] >= self.receiver_t
        ):
            raise ValueError(
                "parents must be unique sorted nonnegative causal indices"
            )
        _probability_row(
            self.probabilities,
            len(self.parents),
            "source probabilities",
        )

    def probability(self, parent: int) -> float:
        try:
            index = self.parents.index(parent)
        except ValueError as exc:
            raise ValueError("parent is outside the explicit source support") from exc
        return self.probabilities[index]


@dataclass(frozen=True, slots=True)
class NormalizedConditionalSourceBank:
    """All normalized source rows for one layer/channel."""

    bank_id: str
    layer: Layer
    channel: Channel
    conditioning: SourceConditioning
    rows: tuple[ConditionalSourceRow, ...]

    def __post_init__(self) -> None:
        if type(self.bank_id) is not str or not self.bank_id:
            raise ValueError("bank_id must be nonempty")
        if self.layer not in (1, 2):
            raise ValueError("source-bank layer must be 1 or 2")
        if self.channel not in ("state", "model"):
            raise ValueError("source-bank channel must be state or model")
        if self.conditioning != "causal_latent_prefix_only":
            raise ValueError("source banks must be target-blind causal conditionals")
        if type(self.rows) is not tuple or not self.rows:
            raise ValueError("source bank must contain at least one row")
        if tuple(row.receiver_t for row in self.rows) != tuple(
            range(1, len(self.rows) + 1)
        ):
            raise ValueError("source rows must cover receiver_t=1..T in order")

    @property
    def horizon(self) -> int:
        return len(self.rows)

    def row(self, receiver_t: int) -> ConditionalSourceRow:
        if type(receiver_t) is not int or not 1 <= receiver_t <= self.horizon:
            raise ValueError("receiver_t is outside the source bank")
        return self.rows[receiver_t - 1]


@dataclass(frozen=True, slots=True)
class Depth2SourceBanks:
    """The four disjoint normalized source banks in the cascade."""

    layer1_state: NormalizedConditionalSourceBank
    layer1_model: NormalizedConditionalSourceBank
    layer2_state: NormalizedConditionalSourceBank
    layer2_model: NormalizedConditionalSourceBank

    def __post_init__(self) -> None:
        expected = (
            (self.layer1_state, 1, "state"),
            (self.layer1_model, 1, "model"),
            (self.layer2_state, 2, "state"),
            (self.layer2_model, 2, "model"),
        )
        horizons = set()
        identifiers = set()
        for bank, layer, channel in expected:
            if (
                type(bank) is not NormalizedConditionalSourceBank
                or bank.layer != layer
                or bank.channel != channel
            ):
                raise ValueError("source bank is assigned to the wrong layer/channel")
            horizons.add(bank.horizon)
            identifiers.add(bank.bank_id)
        if len(horizons) != 1:
            raise ValueError("all source banks must share one horizon")
        if len(identifiers) != 4:
            raise ValueError("all source banks require distinct identities")

    @property
    def horizon(self) -> int:
        return self.layer1_state.horizon

    def bank(
        self,
        layer: Layer,
        channel: Channel,
    ) -> NormalizedConditionalSourceBank:
        return {
            (1, "state"): self.layer1_state,
            (1, "model"): self.layer1_model,
            (2, "state"): self.layer2_state,
            (2, "model"): self.layer2_model,
        }[(layer, channel)]


@dataclass(frozen=True, slots=True)
class TopLayerThresholdEmission:
    """Normalized V-way emission selected by a top-layer affine half-space."""

    state_weight: float
    model_weight: float
    offset: float
    negative_probabilities: tuple[float, ...]
    nonnegative_probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        _finite(self.state_weight, "state_weight")
        _finite(self.model_weight, "model_weight")
        _finite(self.offset, "offset")
        if self.state_weight == 0.0 and self.model_weight == 0.0:
            raise ValueError("top-layer emission must depend on a latent channel")
        if (
            type(self.negative_probabilities) is not tuple
            or len(self.negative_probabilities) < 2
        ):
            raise ValueError("emission vocabulary must contain at least two tokens")
        _probability_row(
            self.negative_probabilities,
            len(self.negative_probabilities),
            "negative emission probabilities",
        )
        _probability_row(
            self.nonnegative_probabilities,
            len(self.negative_probabilities),
            "nonnegative emission probabilities",
        )
        if self.negative_probabilities == self.nonnegative_probabilities:
            raise ValueError("top-layer emission regions must be distinct")

    @property
    def vocabulary_size(self) -> int:
        return len(self.negative_probabilities)


_REGRESSION_LABELS: dict[str, tuple[str, ...]] = {
    "layer1_initial_model": (),
    "layer1_initial_state": ("m1_0",),
    "layer2_initial_model": ("m1_0", "z1_0"),
    "layer2_initial_state": ("m1_0", "m2_0", "z1_0"),
    "layer1_model_transition": ("m1_parent",),
    "layer1_state_transition": ("m1_t", "z1_parent"),
    "layer2_model_transition": ("m1_t", "m2_parent"),
    "layer2_state_transition": ("m2_t", "z1_t", "z2_parent"),
}


@dataclass(frozen=True, slots=True)
class Depth2CascadeSpec:
    """A normalized two-layer scalar cascade, not a trainable arm/config."""

    probe_law_id: Literal["H6-DEPTH2-CASCADE-v1"]
    scientific_disposition: Literal["nonblocking_composition_risk_probe"]
    horizon: int
    source_banks: Depth2SourceBanks
    layer1_parameter_owner: Literal["depth2-probe-layer1"]
    layer2_parameter_owner: Literal["depth2-probe-layer2"]
    layer1_initial_model: ScalarGaussianRegression
    layer1_initial_state: ScalarGaussianRegression
    layer2_initial_model: ScalarGaussianRegression
    layer2_initial_state: ScalarGaussianRegression
    layer1_model_transition: ScalarGaussianRegression
    layer1_state_transition: ScalarGaussianRegression
    layer2_model_transition: ScalarGaussianRegression
    layer2_state_transition: ScalarGaussianRegression
    emission: TopLayerThresholdEmission

    def __post_init__(self) -> None:
        if (
            self.probe_law_id != "H6-DEPTH2-CASCADE-v1"
            or self.scientific_disposition
            != "nonblocking_composition_risk_probe"
        ):
            raise ValueError("depth-2 cascade has the wrong scientific identity")
        if (
            type(self.horizon) is not int
            or self.horizon < 1
            or self.source_banks.horizon != self.horizon
        ):
            raise ValueError("cascade horizon must match every source bank")
        if self.layer1_parameter_owner == self.layer2_parameter_owner:
            raise ValueError("layer parameter ownership must be disjoint")
        for name, labels in _REGRESSION_LABELS.items():
            regression = getattr(self, name)
            if (
                type(regression) is not ScalarGaussianRegression
                or regression.predictor_labels != labels
            ):
                raise ValueError(
                    f"{name} predictors must equal the frozen cascade semantics"
                )
        if type(self.emission) is not TopLayerThresholdEmission:
            raise ValueError("cascade emission has the wrong type")

    @property
    def vocabulary_size(self) -> int:
        return self.emission.vocabulary_size


@dataclass(frozen=True, slots=True)
class Depth2RecognitionLaw:
    """Mean-field recognition over all continuous and source variables."""

    horizon: int
    state_marginals: tuple[
        tuple[GaussianMarginal, ...],
        tuple[GaussianMarginal, ...],
    ]
    model_marginals: tuple[
        tuple[GaussianMarginal, ...],
        tuple[GaussianMarginal, ...],
    ]
    source_posteriors: Depth2SourceBanks

    def __post_init__(self) -> None:
        if (
            type(self.horizon) is not int
            or self.horizon < 1
            or self.source_posteriors.horizon != self.horizon
        ):
            raise ValueError("recognition horizon must match its source banks")
        for name in ("state_marginals", "model_marginals"):
            layers = getattr(self, name)
            if type(layers) is not tuple or len(layers) != 2:
                raise ValueError(f"{name} must contain exactly two layers")
            for series in layers:
                if (
                    type(series) is not tuple
                    or len(series) != self.horizon + 1
                    or any(type(item) is not GaussianMarginal for item in series)
                ):
                    raise ValueError(
                        f"{name} layers must contain T+1 Gaussian marginals"
                    )

    def marginal(
        self,
        layer: Layer,
        channel: Channel,
        receiver_t: int,
    ) -> GaussianMarginal:
        if type(receiver_t) is not int or not 0 <= receiver_t <= self.horizon:
            raise ValueError("receiver_t is outside the recognition horizon")
        layers = (
            self.state_marginals
            if channel == "state"
            else self.model_marginals
        )
        return layers[layer - 1][receiver_t]


@dataclass(frozen=True, slots=True)
class Depth2CascadeProbe:
    """The frozen T=2, scalar, V=3 source-level composition probe."""

    probe_id: Literal["h6-depth2-t2-scalar-v3-v1"]
    cascade: Depth2CascadeSpec
    recognition: Depth2RecognitionLaw
    observed_tokens: tuple[int, int]

    def __post_init__(self) -> None:
        if self.probe_id != "h6-depth2-t2-scalar-v3-v1":
            raise ValueError("unsupported depth-2 probe identity")
        if (
            type(self.cascade) is not Depth2CascadeSpec
            or type(self.recognition) is not Depth2RecognitionLaw
            or self.cascade.horizon != 2
            or self.recognition.horizon != 2
            or self.cascade.vocabulary_size != 3
        ):
            raise ValueError("depth-2 probe must use T=2, scalar channels, V=3")
        if (
            type(self.observed_tokens) is not tuple
            or len(self.observed_tokens) != 2
            or any(
                type(token) is not int or not 0 <= token < 3
                for token in self.observed_tokens
            )
        ):
            raise ValueError("observed_tokens must contain two V=3 token IDs")
        for banks in (
            self.cascade.source_banks,
            self.recognition.source_posteriors,
        ):
            for layer in (1, 2):
                for channel in ("state", "model"):
                    for receiver_t in (1, 2):
                        expected = tuple(range(receiver_t))
                        if (
                            banks.bank(layer, channel).row(receiver_t).parents
                            != expected
                        ):
                            raise ValueError(
                                "probe parent sets must equal explicit range(t)"
                            )


def _bank(
    bank_id: str,
    layer: Layer,
    channel: Channel,
    second_row: tuple[float, float],
) -> NormalizedConditionalSourceBank:
    return NormalizedConditionalSourceBank(
        bank_id=bank_id,
        layer=layer,
        channel=channel,
        conditioning="causal_latent_prefix_only",
        rows=(
            ConditionalSourceRow(1, (0,), (1.0,)),
            ConditionalSourceRow(2, (0, 1), second_row),
        ),
    )


def build_tiny_depth2_probe() -> Depth2CascadeProbe:
    """Construct the immutable T=2/V=3 amendment probe."""

    prior_banks = Depth2SourceBanks(
        layer1_state=_bank("p-a1", 1, "state", (0.65, 0.35)),
        layer1_model=_bank("p-b1", 1, "model", (0.45, 0.55)),
        layer2_state=_bank("p-a2", 2, "state", (0.30, 0.70)),
        layer2_model=_bank("p-b2", 2, "model", (0.60, 0.40)),
    )
    cascade = Depth2CascadeSpec(
        probe_law_id="H6-DEPTH2-CASCADE-v1",
        scientific_disposition="nonblocking_composition_risk_probe",
        horizon=2,
        source_banks=prior_banks,
        layer1_parameter_owner="depth2-probe-layer1",
        layer2_parameter_owner="depth2-probe-layer2",
        layer1_initial_model=ScalarGaussianRegression(0.15, (), 0.80),
        layer1_initial_state=ScalarGaussianRegression(
            0.05, (("m1_0", 0.40),), 0.70
        ),
        layer2_initial_model=ScalarGaussianRegression(
            -0.10, (("m1_0", 0.20), ("z1_0", 0.25)), 0.90
        ),
        layer2_initial_state=ScalarGaussianRegression(
            0.12,
            (("m1_0", -0.15), ("m2_0", 0.35), ("z1_0", 0.30)),
            0.65,
        ),
        layer1_model_transition=ScalarGaussianRegression(
            0.02, (("m1_parent", 0.55),), 0.75
        ),
        layer1_state_transition=ScalarGaussianRegression(
            -0.08, (("m1_t", 0.30), ("z1_parent", 0.45)), 0.60
        ),
        layer2_model_transition=ScalarGaussianRegression(
            0.10, (("m1_t", 0.25), ("m2_parent", 0.50)), 0.85
        ),
        layer2_state_transition=ScalarGaussianRegression(
            -0.03,
            (("m2_t", 0.30), ("z1_t", 0.20), ("z2_parent", 0.40)),
            0.70,
        ),
        emission=TopLayerThresholdEmission(
            state_weight=1.0,
            model_weight=0.35,
            offset=-0.10,
            negative_probabilities=(0.55, 0.30, 0.15),
            nonnegative_probabilities=(0.15, 0.25, 0.60),
        ),
    )
    recognition_banks = Depth2SourceBanks(
        layer1_state=_bank("q-a1", 1, "state", (0.52, 0.48)),
        layer1_model=_bank("q-b1", 1, "model", (0.40, 0.60)),
        layer2_state=_bank("q-a2", 2, "state", (0.25, 0.75)),
        layer2_model=_bank("q-b2", 2, "model", (0.70, 0.30)),
    )
    recognition = Depth2RecognitionLaw(
        horizon=2,
        state_marginals=(
            (
                GaussianMarginal(0.10, 0.90),
                GaussianMarginal(0.20, 0.80),
                GaussianMarginal(-0.10, 0.75),
            ),
            (
                GaussianMarginal(-0.05, 0.85),
                GaussianMarginal(0.30, 0.70),
                GaussianMarginal(0.40, 0.65),
            ),
        ),
        model_marginals=(
            (
                GaussianMarginal(0.05, 0.70),
                GaussianMarginal(0.15, 0.65),
                GaussianMarginal(0.25, 0.60),
            ),
            (
                GaussianMarginal(0.20, 0.75),
                GaussianMarginal(-0.10, 0.80),
                GaussianMarginal(0.10, 0.70),
            ),
        ),
        source_posteriors=recognition_banks,
    )
    return Depth2CascadeProbe(
        probe_id="h6-depth2-t2-scalar-v3-v1",
        cascade=cascade,
        recognition=recognition,
        observed_tokens=(2, 1),
    )


__all__ = [
    "Channel",
    "ConditionalSourceRow",
    "Depth2CascadeProbe",
    "Depth2CascadeSpec",
    "Depth2RecognitionLaw",
    "Depth2SourceBanks",
    "GaussianMarginal",
    "Layer",
    "NormalizedConditionalSourceBank",
    "ScalarGaussianRegression",
    "TopLayerThresholdEmission",
    "build_tiny_depth2_probe",
]
