"""Focused static mutants for the frozen H6 source-support rules."""

from pathlib import Path

from vfe4.types.h6 import ArmId, EvidenceStatus, PrefixCaseKey
from vfe4.validation.h6_static_audit import audit_h6_static_source


def test_mask_audit_rejects_normalization_and_support_mutants(
    tmp_path: Path,
) -> None:
    case_key = PrefixCaseKey(
        arm=ArmId.A5,
        predictor_config_sha256="a" * 64,
        estimator_sha256="b" * 64,
        model_family_sha256="c" * 64,
        vocabulary_sha256="d" * 64,
        data_safety_sha256="e" * 64,
        git_head="1" * 40,
        dirty_digest="f" * 64,
    )
    unsafe_root = tmp_path / "unsafe"
    unsafe_source = unsafe_root / "vfe4" / "generative" / "source_priors.py"
    unsafe_source.parent.mkdir(parents=True)
    unsafe_source.write_text(
        """
def masked_log_softmax_from_parents(logits, declared_parents, receiver_t):
    if declared_parents == ():
        return logits
    if len(declared_parents) == 0:
        return logits * 0.0
    declared_parents = declared_parents + (receiver_t, receiver_t + 1)
    probabilities = logits.softmax(dim=-1)
    support_mask = tuple(parent <= receiver_t + 1 for parent in declared_parents)
    post_softmax_masked = probabilities * support_mask
    return renormalize_source_row(post_softmax_masked)


def renormalize_source_row(values):
    return values / values.sum(dim=-1, keepdim=True)


def literal_noncausal_source_parents(receiver_t):
    declared_parents = (receiver_t, receiver_t + 1)
    return declared_parents


class ModelSourceBank:
    def source_log_probabilities(self, logits):
        return logits.log_softmax(dim=-1)
""".lstrip(),
        encoding="utf-8",
    )
    unsafe_language = unsafe_root / "vfe4" / "generative" / "language.py"
    unsafe_language.write_text(
        """
def inline_post_softmax_mask(unmasked_logits, support_mask):
    return unmasked_logits.softmax(dim=-1) * support_mask


class StateSourceBank:
    def forward(self, logits):
        return logits.softmax(dim=-1)
""".lstrip(),
        encoding="utf-8",
    )
    unsafe_report = audit_h6_static_source(
        repo_root=unsafe_root,
        exact_case_keys=(case_key,),
    )
    assert unsafe_report.status is EvidenceStatus.FAIL
    unsafe_rule_ids = "\n".join(
        finding.rule_id for finding in unsafe_report.findings
    )
    assert "post_softmax_mask" in unsafe_rule_ids
    assert "duplicate_normalizer" in unsafe_rule_ids
    assert "direct_source_softmax" in unsafe_rule_ids
    assert "noncausal_parent" in unsafe_rule_ids
    assert "all_invalid_fallback" in unsafe_rule_ids

    safe_root = tmp_path / "safe"
    safe_source = safe_root / "vfe4" / "generative" / "source_priors.py"
    safe_source.parent.mkdir(parents=True)
    safe_source.write_text(
        """
class AllInvalidSourceRowError(ValueError):
    pass


def masked_log_softmax_from_parents(logits, declared_parents, receiver_t):
    if not declared_parents:
        raise AllInvalidSourceRowError("source row has no declared parents")
    if any(parent >= receiver_t for parent in declared_parents):
        raise ValueError("source parents must be strictly causal")
    support_mask = logits.new_zeros(logits.shape, dtype=bool)
    for parent in declared_parents:
        support_mask[..., parent] = True
    if not support_mask.any():
        raise AllInvalidSourceRowError("source row has no valid parent")
    masked_logits = logits.masked_fill(~support_mask, float("-inf"))
    return masked_logits.log_softmax(dim=-1)


def state_source_bank(logits, declared_parents, receiver_t):
    return masked_log_softmax_from_parents(
        logits, declared_parents, receiver_t
    )


def safe_exclusive_parent_range(start, receiver_t):
    return tuple(range(start + 1, receiver_t))
""".lstrip(),
        encoding="utf-8",
    )
    safe_report = audit_h6_static_source(
        repo_root=safe_root,
        exact_case_keys=(case_key,),
    )
    assert safe_report.status is EvidenceStatus.PASS
    assert safe_report.obligations == ()
    assert all(
        finding.status is not EvidenceStatus.FAIL
        for finding in safe_report.findings
    )
