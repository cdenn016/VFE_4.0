"""Static Task-6 scientific preconditions without Task-10 readiness."""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from typing import Literal

from vfe4.artifacts.manifest import ClosedManifestIdentity
from vfe4.types.results import GateStatus
from vfe4.types.training import (
    A0ArchitectureProfile,
    A0FormulaRecord,
    EndpointInventory,
    ScientificPreconditionProfile,
    SyntheticFixtureTokenCacheIdentity,
    SyntheticFixtureTokenizerSpec,
    TrainingSparsityCertificate,
    WT103ExperimentProfile,
    owned_sha256,
)

from .factories import WT103FactorySetIdentity


_EXPECTED_ARTIFACT_SCHEMAS = {
    "h5": "h5-update-result-v1",
    "h6_prefix": "h6-prefix-certificate-set-v2",
    "h6_prediction": "h6-prediction-result-v3",
    "h7": "h7-gate-result-v1",
    "h8": "h8-sparse-scale-v5",
}
_EXPECTED_ARTIFACT_ORDER = tuple(_EXPECTED_ARTIFACT_SCHEMAS)
_PERTURBATION_TARGETS = (0, 1, 50_256)
_PERTURBATION_SUFFIXES = ((), (7,), (50_256, 9))
_PERTURBATION_TRAVERSALS = ("cold", "forward", "reverse")


def _sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _git_head(value: object, name: str = "git_head") -> str:
    if (
        type(value) is not str
        or len(value) not in (40, 64)
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a concrete hex object id")
    return value


@dataclass(frozen=True, slots=True)
class PredictorPerturbationObservation:
    """One target/suffix/cache-order perturbation at a fixed causal prefix."""

    prefix_tokens: tuple[int, ...]
    current_target: int
    suffix_tokens: tuple[int, ...]
    cache_traversal: Literal["cold", "forward", "reverse"]
    prediction_sha256: str
    cache_key_sha256: str

    def __post_init__(self) -> None:
        for name in ("prefix_tokens", "suffix_tokens"):
            value = getattr(self, name)
            if (
                type(value) is not tuple
                or any(
                    type(item) is not int or not 0 <= item < 50_257
                    for item in value
                )
            ):
                raise ValueError(f"{name} must contain bounded GPT-2 IDs")
        if (
            type(self.current_target) is not int
            or not 0 <= self.current_target < 50_257
            or self.cache_traversal not in _PERTURBATION_TRAVERSALS
        ):
            raise ValueError("predictor perturbation metadata is invalid")
        _sha256(self.prediction_sha256, "prediction_sha256")
        _sha256(self.cache_key_sha256, "cache_key_sha256")


@dataclass(frozen=True, slots=True)
class WT103PredictorSafetyCertificate:
    """Bounded synthetic target-blindness/static-audit result."""

    schema_version: Literal["wt103-predictor-safety-v1"]
    authority: Literal["nonproduction_synthetic_smoke"]
    vocabulary_size: Literal[50257]
    predictor_module: str
    predictor_qualname: str
    signature_sha256: str
    static_source_sha256: str
    tokenizer_spec_sha256: str
    cache_sha256: str
    perturbation_set_sha256: str
    perturbation_count: int
    status: GateStatus
    obligations: tuple[str, ...]
    production_token_authorized: Literal[False]
    certificate_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in tuple(self.__dataclass_fields__)[:-1]
        }

    def __post_init__(self) -> None:
        if (
            self.schema_version != "wt103-predictor-safety-v1"
            or self.authority != "nonproduction_synthetic_smoke"
            or type(self.vocabulary_size) is not int
            or self.vocabulary_size != 50_257
            or type(self.predictor_module) is not str
            or not self.predictor_module
            or type(self.predictor_qualname) is not str
            or not self.predictor_qualname
            or type(self.perturbation_count) is not int
            or self.perturbation_count <= 0
            or type(self.status) is not GateStatus
            or type(self.obligations) is not tuple
            or any(type(item) is not str or not item for item in self.obligations)
            or self.production_token_authorized is not False
        ):
            raise ValueError("predictor safety certificate is invalid")
        for name in (
            "signature_sha256",
            "static_source_sha256",
            "tokenizer_spec_sha256",
            "cache_sha256",
            "perturbation_set_sha256",
        ):
            _sha256(getattr(self, name), name)
        if (
            (self.status is GateStatus.PASS and self.obligations)
            or (self.status is not GateStatus.PASS and not self.obligations)
        ):
            raise ValueError("predictor safety status/obligations disagree")
        expected = owned_sha256(
            "vfe4.wt103.predictor-safety-certificate.v1",
            self.semantic_payload(),
        )
        _sha256(self.certificate_sha256, "certificate_sha256")
        if self.certificate_sha256 != expected:
            raise ValueError("predictor safety certificate hash does not match")


def certify_wt103_predictor_safety(
    *,
    predictor_type: type,
    tokenizer: SyntheticFixtureTokenizerSpec,
    cache: SyntheticFixtureTokenCacheIdentity,
    observations: tuple[PredictorPerturbationObservation, ...],
) -> WT103PredictorSafetyCertificate:
    """Audit the existing PriorPredictor call boundary on bounded fixtures."""

    if type(tokenizer) is not SyntheticFixtureTokenizerSpec:
        raise ValueError("tokenizer must be an exact synthetic fixture type")
    if type(cache) is not SyntheticFixtureTokenCacheIdentity:
        raise ValueError("cache must be an exact synthetic fixture cache")
    tokenizer.__post_init__()
    cache.__post_init__()
    if cache.tokenizer != tokenizer:
        raise ValueError("synthetic cache does not bind the tokenizer")
    if type(predictor_type) is not type:
        raise ValueError("predictor_type must be an explicit implementation type")
    method = getattr(predictor_type, "next_token_log_probs", None)
    if not callable(method):
        raise ValueError("predictor implementation lacks next_token_log_probs")

    obligations: list[str] = []
    signature = inspect.signature(method)
    parameters = tuple(signature.parameters.values())
    if (
        tuple(item.name for item in parameters)
        != ("self", "prefix_tokens", "estimator_rng", "cache")
        or parameters[-1].default is not None
        or any(
            item.kind
            not in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
            for item in parameters
        )
    ):
        obligations.append("prior_predictor_signature_changed")
    signature_sha256 = owned_sha256(
        "vfe4.wt103.predictor-signature.v1",
        str(signature),
    )
    try:
        source = inspect.getsource(predictor_type)
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError):
        source = "<source-unavailable>"
        tree = None
        obligations.append("predictor_static_source_unavailable")
    if tree is not None:
        forbidden_names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
            and (
                "target" in node.id.lower()
                or "suffix" in node.id.lower()
            )
        }
        forbidden_imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
            if alias.name.startswith(("vfe4.data.production", "vfe4.training"))
        }
        if forbidden_names:
            obligations.append("predictor_static_target_or_suffix_taint")
        if forbidden_imports:
            obligations.append("predictor_static_forbidden_import")
    static_source_sha256 = owned_sha256(
        "vfe4.wt103.predictor-static-source.v1",
        source,
    )

    if (
        type(observations) is not tuple
        or any(
            type(item) is not PredictorPerturbationObservation
            for item in observations
        )
        or not observations
    ):
        raise ValueError("observations must be exact predictor perturbations")
    for item in observations:
        item.__post_init__()
    prefixes = {item.prefix_tokens for item in observations}
    expected_grid = {
        (target, suffix, traversal)
        for target in _PERTURBATION_TARGETS
        for suffix in _PERTURBATION_SUFFIXES
        for traversal in _PERTURBATION_TRAVERSALS
    }
    observed_grid = {
        (item.current_target, item.suffix_tokens, item.cache_traversal)
        for item in observations
    }
    if len(prefixes) != 1 or observed_grid != expected_grid:
        obligations.append("bounded_predictor_perturbation_grid_incomplete")
    prediction_hashes = {item.prediction_sha256 for item in observations}
    cache_hashes = {item.cache_key_sha256 for item in observations}
    witnessed_leak = len(prediction_hashes) != 1 or len(cache_hashes) != 1
    if witnessed_leak:
        obligations.append("target_suffix_or_cache_order_changes_prediction")
    status = (
        GateStatus.FAIL
        if witnessed_leak
        else GateStatus.INCONCLUSIVE
        if obligations
        else GateStatus.PASS
    )
    payload = {
        "schema_version": "wt103-predictor-safety-v1",
        "authority": "nonproduction_synthetic_smoke",
        "vocabulary_size": 50_257,
        "predictor_module": predictor_type.__module__,
        "predictor_qualname": predictor_type.__qualname__,
        "signature_sha256": signature_sha256,
        "static_source_sha256": static_source_sha256,
        "tokenizer_spec_sha256": tokenizer.spec_sha256,
        "cache_sha256": cache.cache_sha256,
        "perturbation_set_sha256": owned_sha256(
            "vfe4.wt103.predictor-perturbations.v1",
            observations,
        ),
        "perturbation_count": len(observations),
        "status": status,
        "obligations": tuple(dict.fromkeys(obligations)),
        "production_token_authorized": False,
    }
    return WT103PredictorSafetyCertificate(
        **payload,
        certificate_sha256=owned_sha256(
            "vfe4.wt103.predictor-safety-certificate.v1",
            payload,
        ),
    )  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class StaticScientificArtifactRef:
    """Manifest-bound PASS identity for one exact predecessor generation."""

    schema_version: Literal["wt103-static-scientific-artifact-ref-v1"]
    kind: Literal["h5", "h6_prefix", "h6_prediction", "h7", "h8"]
    result_schema: str
    git_head: str
    dirty_digest: str
    result_sha256: str
    manifest: ClosedManifestIdentity
    status: GateStatus
    reference_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in tuple(self.__dataclass_fields__)[:-1]
        }

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != "wt103-static-scientific-artifact-ref-v1"
            or self.kind not in _EXPECTED_ARTIFACT_SCHEMAS
            or self.result_schema
            != _EXPECTED_ARTIFACT_SCHEMAS[self.kind]
            or self.status is not GateStatus.PASS
        ):
            raise ValueError("static predecessor schema/status is ineligible")
        _git_head(self.git_head)
        _sha256(self.dirty_digest, "dirty_digest")
        _sha256(self.result_sha256, "result_sha256")
        if type(self.manifest) is not ClosedManifestIdentity:
            raise ValueError("predecessor requires an exact closed manifest")
        self.manifest.manifest.__post_init__()
        for entry in self.manifest.entries:
            entry.__post_init__()
        reopened = ClosedManifestIdentity.create(
            manifest=self.manifest.manifest,
            entries=self.manifest.entries,
        )
        if (
            reopened.identity_sha256 != self.manifest.identity_sha256
            or self.result_sha256
            not in tuple(entry.sha256 for entry in self.manifest.entries)
        ):
            raise ValueError("closed manifest does not bind the result")
        expected = owned_sha256(
            "vfe4.wt103.static-scientific-artifact-ref.v1",
            self.semantic_payload(),
        )
        _sha256(self.reference_sha256, "reference_sha256")
        if self.reference_sha256 != expected:
            raise ValueError("predecessor reference hash does not match")

    @classmethod
    def create(
        cls,
        *,
        kind: str,
        result_schema: str,
        git_head: str,
        dirty_digest: str,
        result_sha256: str,
        manifest: ClosedManifestIdentity,
        status: GateStatus,
    ) -> "StaticScientificArtifactRef":
        payload = {
            "schema_version": "wt103-static-scientific-artifact-ref-v1",
            "kind": kind,
            "result_schema": result_schema,
            "git_head": git_head,
            "dirty_digest": dirty_digest,
            "result_sha256": result_sha256,
            "manifest": manifest,
            "status": status,
        }
        return cls(
            **payload,
            reference_sha256=owned_sha256(
                "vfe4.wt103.static-scientific-artifact-ref.v1",
                payload,
            ),
        )  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class StaticScientificPreconditionRecord:
    """One-way Task-6 closure record; never a Task-10 readiness token."""

    schema_version: Literal["wt103-static-scientific-preconditions-v1"]
    git_head: str
    dirty_digest: str
    h6_prediction_schema: Literal["h6-prediction-result-v3"]
    h8_schema: Literal["h8-sparse-scale-v5"]
    profile_sha256: str
    architecture_sha256: str
    formula_sha256: str
    factory_set_sha256: str
    endpoint_inventory_sha256: str
    objective_sha256: str
    update_policy_sha256: str
    snapshot_policy_sha256: str
    estimator_protocol_sha256: str
    predecessor_reference_sha256s: tuple[str, ...]
    predictor_safety_sha256: str
    training_sparsity_sha256: str
    status: GateStatus
    obligations: tuple[str, ...]
    production_readiness_token_issued: Literal[False]
    record_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in tuple(self.__dataclass_fields__)[:-1]
        }

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != "wt103-static-scientific-preconditions-v1"
            or self.h6_prediction_schema != "h6-prediction-result-v3"
            or self.h8_schema != "h8-sparse-scale-v5"
            or type(self.status) is not GateStatus
            or type(self.obligations) is not tuple
            or any(type(item) is not str or not item for item in self.obligations)
            or self.production_readiness_token_issued is not False
        ):
            raise ValueError("static scientific precondition record is invalid")
        _git_head(self.git_head)
        for name in (
            "dirty_digest",
            "profile_sha256",
            "architecture_sha256",
            "formula_sha256",
            "factory_set_sha256",
            "endpoint_inventory_sha256",
            "objective_sha256",
            "update_policy_sha256",
            "snapshot_policy_sha256",
            "estimator_protocol_sha256",
            "predictor_safety_sha256",
            "training_sparsity_sha256",
        ):
            _sha256(getattr(self, name), name)
        if (
            type(self.predecessor_reference_sha256s) is not tuple
            or len(self.predecessor_reference_sha256s) != 5
        ):
            raise ValueError("predecessor reference inventory must contain five")
        for value in self.predecessor_reference_sha256s:
            _sha256(value, "predecessor_reference_sha256")
        if (
            (self.status is GateStatus.PASS and self.obligations)
            or (self.status is not GateStatus.PASS and not self.obligations)
        ):
            raise ValueError("static precondition status/obligations disagree")
        expected = owned_sha256(
            "vfe4.wt103.static-scientific-preconditions.v1",
            self.semantic_payload(),
        )
        _sha256(self.record_sha256, "record_sha256")
        if self.record_sha256 != expected:
            raise ValueError("static precondition record hash does not match")


def validate_static_scientific_preconditions(
    *,
    profile: WT103ExperimentProfile,
    scientific_profile: ScientificPreconditionProfile,
    architecture: A0ArchitectureProfile,
    formula: A0FormulaRecord,
    factory_set: WT103FactorySetIdentity,
    endpoint_inventory: EndpointInventory,
    objective_sha256: str,
    update_policy_sha256: str,
    snapshot_policy_sha256: str,
    estimator_protocol_sha256: str,
    predecessor_references: tuple[StaticScientificArtifactRef, ...],
    predictor_safety: WT103PredictorSafetyCertificate,
    training_sparsity: TrainingSparsityCertificate,
    h6_byte_evidence: None,
    h8_allocation_evidence: None,
    capacity_evidence: None,
) -> StaticScientificPreconditionRecord:
    """Fail closed on exact static science while excluding Task-10 authority."""

    if h6_byte_evidence is not None:
        raise ValueError("H6 byte evidence cannot establish GPT-2 safety")
    if h8_allocation_evidence is not None:
        raise ValueError("H8 allocation evidence cannot establish training sparsity")
    if capacity_evidence is not None:
        raise ValueError("capacity evidence belongs only to Task 10")
    exact_types = (
        (profile, WT103ExperimentProfile),
        (scientific_profile, ScientificPreconditionProfile),
        (architecture, A0ArchitectureProfile),
        (formula, A0FormulaRecord),
        (factory_set, WT103FactorySetIdentity),
        (endpoint_inventory, EndpointInventory),
        (predictor_safety, WT103PredictorSafetyCertificate),
        (training_sparsity, TrainingSparsityCertificate),
    )
    if any(type(value) is not expected for value, expected in exact_types):
        raise ValueError("static preconditions require exact typed inputs")
    for value, _ in exact_types:
        value.__post_init__()
    for name, value in (
        ("objective_sha256", objective_sha256),
        ("update_policy_sha256", update_policy_sha256),
        ("snapshot_policy_sha256", snapshot_policy_sha256),
        ("estimator_protocol_sha256", estimator_protocol_sha256),
    ):
        _sha256(value, name)
    if (
        type(predecessor_references) is not tuple
        or tuple(item.kind for item in predecessor_references)
        != _EXPECTED_ARTIFACT_ORDER
        or any(
            type(item) is not StaticScientificArtifactRef
            for item in predecessor_references
        )
    ):
        raise ValueError("predecessor references must retain exact order/types")
    for reference in predecessor_references:
        reference.__post_init__()

    obligations: list[str] = []
    candidate_revisions = {
        (item.git_head, item.dirty_digest)
        for item in predecessor_references
    }
    candidate_revisions.add(
        (training_sparsity.git_head, training_sparsity.dirty_digest)
    )
    if len(candidate_revisions) != 1:
        obligations.append("predecessors_not_same_revision")
    if (
        scientific_profile.h6_prediction_authority
        != "native_executable_v3"
        or scientific_profile.h6_prediction_schema
        != "h6-prediction-result-v3"
        or scientific_profile.h8_schema != "h8-sparse-scale-v5"
        or scientific_profile.h8_config_schema
        != "h8-validation-config-v3"
        or scientific_profile.h8_parent_child_protocol
        != "vfe4.h8.parent-child-protocol.v3"
    ):
        obligations.append("scientific_precondition_profile_is_stale")
    if (
        architecture.formula_sha256 != formula.formula_sha256
        or factory_set.arm_spec_sha256s
        != tuple(
            item.arm_spec_sha256 for item in endpoint_inventory.arms
        )
        or estimator_protocol_sha256
        != endpoint_inventory.estimator_protocol_sha256
    ):
        obligations.append("static_scientific_identity_mismatch")
    if (
        training_sparsity.status is not GateStatus.PASS
        or training_sparsity.profile_sha256 != profile.profile_sha256
        or training_sparsity.factory_set_sha256
        != factory_set.factory_set_sha256
        or training_sparsity.endpoint_inventory_sha256
        != endpoint_inventory.endpoint_inventory_sha256
    ):
        obligations.append("training_sparsity_not_exact_pass")
    if predictor_safety.status is not GateStatus.PASS:
        obligations.append("predictor_safety_not_pass")
    status = (
        GateStatus.INCONCLUSIVE if obligations else GateStatus.PASS
    )
    revision = (
        training_sparsity.git_head,
        training_sparsity.dirty_digest,
    )
    payload = {
        "schema_version": "wt103-static-scientific-preconditions-v1",
        "git_head": revision[0],
        "dirty_digest": revision[1],
        "h6_prediction_schema": "h6-prediction-result-v3",
        "h8_schema": "h8-sparse-scale-v5",
        "profile_sha256": profile.profile_sha256,
        "architecture_sha256": architecture.architecture_sha256,
        "formula_sha256": formula.formula_sha256,
        "factory_set_sha256": factory_set.factory_set_sha256,
        "endpoint_inventory_sha256": (
            endpoint_inventory.endpoint_inventory_sha256
        ),
        "objective_sha256": objective_sha256,
        "update_policy_sha256": update_policy_sha256,
        "snapshot_policy_sha256": snapshot_policy_sha256,
        "estimator_protocol_sha256": estimator_protocol_sha256,
        "predecessor_reference_sha256s": tuple(
            item.reference_sha256 for item in predecessor_references
        ),
        "predictor_safety_sha256": predictor_safety.certificate_sha256,
        "training_sparsity_sha256": training_sparsity.certificate_sha256,
        "status": status,
        "obligations": tuple(obligations),
        "production_readiness_token_issued": False,
    }
    return StaticScientificPreconditionRecord(
        **payload,
        record_sha256=owned_sha256(
            "vfe4.wt103.static-scientific-preconditions.v1",
            payload,
        ),
    )  # type: ignore[arg-type]


__all__ = [
    "PredictorPerturbationObservation",
    "StaticScientificArtifactRef",
    "StaticScientificPreconditionRecord",
    "WT103PredictorSafetyCertificate",
    "certify_wt103_predictor_safety",
    "validate_static_scientific_preconditions",
]
