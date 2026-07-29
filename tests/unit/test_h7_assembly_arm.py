from __future__ import annotations

import dataclasses
import hashlib
import math

import pytest
import torch

import vfe4.training.arms as arms
from vfe4.generative import FixedSourcePrior
from vfe4.predictive import canonical_model_state_sha256
from vfe4.training.h7_assembly import (
    H7FixedSourceAssemblySpec,
    build_h7_fixed_a5_arm,
)
from vfe4.types.h6 import (
    ArmConfig,
    ArmId,
    CapacityAllocation,
    CausalDag,
    CausalDagRow,
    H6LanguageStructure,
    VocabularyIdentity,
    ZeroDimensionalBase,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _a5_config(*, horizon: int = 2) -> ArmConfig:
    return ArmConfig.create(
        arm=ArmId.A5,
        config_id="h6-a5-structured-fixed-exact-complete-latent-smoothing-v1",
        vocabulary=VocabularyIdentity(
            vocabulary_id="h7-assembly-test-v1",
            size=258,
            tokenizer_spec_sha256=_sha("h7 assembly tokenizer"),
        ),
        horizon=horizon,
        latent_enabled=True,
        state_channel_enabled=True,
        model_channel_enabled=True,
        source_mode="categorical",
        map_mode="shared_vertex_coboundary",
        recognition_family="structured",
        recognition_conditioning="smoothing",
        prior_variant="fixed",
        mixture_mode="exact",
        objective_kind="complete_elbo",
        capacity_allocation=CapacityAllocation.create(
            emission_width=4,
            latent_width=2,
            recognition_width=4,
            prior_context_width=None,
        ),
    )


def _structure(
    second_receiver_parents: tuple[int, ...],
) -> H6LanguageStructure:
    dag = CausalDag.create(
        node_labels=(0, 1, 2),
        rows=(
            CausalDagRow(1, (0,)),
            CausalDagRow(2, second_receiver_parents),
        ),
    )
    return H6LanguageStructure.create(
        base=ZeroDimensionalBase.create(),
        dag=dag,
        receiver_labels=(1, 2),
    )


def test_h1_spec_hash_binds_structure_probabilities_and_exact_logits() -> None:
    spec = H7FixedSourceAssemblySpec.from_h1(_structure((0, 1)))

    assert spec.fixture_id == "h1-v1"
    assert spec.structure_sha256 == spec.structure.structure_sha256
    assert spec.state_source_probabilities == ((1.0,), (0.55, 0.45))
    assert spec.model_source_probabilities == ((1.0,), (0.35, 0.65))
    assert spec.state_logits == (
        (0.0,),
        (math.log(0.55) - math.log(0.45), 0.0),
    )
    assert spec.model_logits == (
        (0.0,),
        (math.log(0.35) - math.log(0.65), 0.0),
    )
    assert len(spec.source_specification_sha256) == 64

    with pytest.raises(ValueError, match="source specification hash"):
        dataclasses.replace(
            spec,
            source_specification_sha256=_sha("tampered H1 source specification"),
        )


def test_h1_factory_issues_exact_registered_fixed_a5_before_capture() -> None:
    config = _a5_config()
    spec = H7FixedSourceAssemblySpec.from_h1(_structure((0, 1)))

    built = build_h7_fixed_a5_arm(config, spec)

    assert type(built) is arms.BuiltArm
    assert arms._require_factory_issued_built_arm(built) is built
    assert type(built.model) is arms.LatentLanguageArmModel
    prior = built.model.source_prior
    assert type(prior) is FixedSourcePrior
    assert prior.structure is spec.structure
    assert prior.fixture_sha256 == spec.source_specification_sha256
    torch.testing.assert_close(
        prior.state_source_log_probs(receiver_t=1).log_probs.value(),
        torch.tensor([0.0], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        prior.state_source_log_probs(receiver_t=2).log_probs.value(),
        torch.log(torch.tensor([0.55, 0.45], dtype=torch.float64)),
        rtol=0.0,
        atol=2.0 * math.ulp(1.0),
    )
    torch.testing.assert_close(
        prior.model_source_log_probs(receiver_t=2).log_probs.value(),
        torch.log(torch.tensor([0.35, 0.65], dtype=torch.float64)),
        rtol=0.0,
        atol=2.0 * math.ulp(1.0),
    )
    role_names = {record.qualified_name for record in built.parameter_roles}
    assert "model.source_prior.state_source_free_logits.0" in role_names
    assert "model.source_prior.model_source_free_logits.0" in role_names
    assert built.proposal.model_state_sha256 == canonical_model_state_sha256(
        built.model
    )


def test_h7_v1_factory_uses_singleton_second_rows_and_rejects_dense_a5() -> None:
    singleton_structure = _structure((1,))
    spec = H7FixedSourceAssemblySpec.from_h7_v1(singleton_structure)

    built = build_h7_fixed_a5_arm(_a5_config(), spec)

    prior = built.model.source_prior
    assert type(prior) is FixedSourcePrior
    assert tuple(row.parents for row in prior.structure.dag.rows) == ((0,), (1,))
    assert torch.equal(
        prior.state_source_log_probs(receiver_t=2).log_probs.value(),
        torch.tensor([-torch.inf, 0.0], dtype=torch.float64),
    )
    assert torch.equal(
        prior.model_source_log_probs(receiver_t=2).log_probs.value(),
        torch.tensor([-torch.inf, 0.0], dtype=torch.float64),
    )
    role_names = {record.qualified_name for record in built.parameter_roles}
    assert not any("source_free_logits" in name for name in role_names)

    with pytest.raises(ValueError, match="H7-v1 source rows"):
        H7FixedSourceAssemblySpec.from_h7_v1(_structure((0, 1)))


def test_h7_factory_captures_replacement_before_boundary_and_registration() -> None:
    config = _a5_config()
    built = build_h7_fixed_a5_arm(
        config,
        H7FixedSourceAssemblySpec.from_h7_v1(_structure((1,))),
    )
    issued_state_sha256 = built.proposal.model_state_sha256

    ordinary = arms.build_a5(config)
    ordinary_prior = ordinary.model.source_prior
    assert type(ordinary_prior) is FixedSourcePrior
    assert tuple(row.parents for row in ordinary_prior.structure.dag.rows) == (
        (0,),
        (0, 1),
    )
    assert torch.equal(
        ordinary_prior.state_source_log_probs(receiver_t=2).log_probs.value(),
        torch.log(torch.tensor([0.5, 0.5], dtype=torch.float64)),
    )

    built.model.source_prior = ordinary_prior
    assert canonical_model_state_sha256(built.model) != issued_state_sha256
    with pytest.raises(ValueError, match="changed after its issuance"):
        arms._require_factory_issued_built_arm(built)
    with pytest.raises(ValueError, match="model state changed"):
        built.proposal.assert_current_state()


def test_h7_factory_rejects_non_two_horizon_before_construction() -> None:
    source_spec = H7FixedSourceAssemblySpec.from_h7_v1(_structure((1,)))

    with pytest.raises(ValueError, match="horizon must be exactly 2"):
        build_h7_fixed_a5_arm(_a5_config(horizon=3), source_spec)
