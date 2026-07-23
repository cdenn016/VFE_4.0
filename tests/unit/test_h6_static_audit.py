from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from vfe4.types import ArmId, EvidenceStatus, PrefixCaseKey
from vfe4.types.h6 import canonical_json_bytes
from vfe4.validation.h6_static_audit import audit_h6_static_source


def test_h6_static_import_signature_count_inventory_and_access_rules(
    tmp_path: Path,
) -> None:
    case_key = PrefixCaseKey(
        arm=ArmId.A0,
        predictor_config_sha256="1" * 64,
        estimator_sha256="2" * 64,
        model_family_sha256="3" * 64,
        vocabulary_sha256="4" * 64,
        data_safety_sha256="5" * 64,
        git_head="6" * 40,
        dirty_digest="7" * 64,
    )
    exact_case_keys = (case_key,)

    def source_tree(name: str, files: dict[str, str]) -> Path:
        root = tmp_path / name
        for relative_path, source in files.items():
            path = root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8", newline="\n")
        return root

    def findings_for(report: object, rule_id: str) -> tuple[object, ...]:
        findings = getattr(report, "findings")
        return tuple(item for item in findings if item.rule_id == rule_id)

    def require_failure(report: object, rule_id: str, path_suffix: str) -> tuple[object, ...]:
        assert getattr(report, "status") is EvidenceStatus.FAIL
        findings = findings_for(report, rule_id)
        assert findings
        assert any(item.status is EvidenceStatus.FAIL for item in findings)
        assert all(item.message for item in findings)
        assert any(
            str(item.path).replace("\\", "/").endswith(path_suffix)
            for item in findings
        )
        return findings

    def owned_hash(domain: str, payload: object) -> str:
        return hashlib.sha256(
            domain.encode("ascii") + b"\x00" + canonical_json_bytes(payload)
        ).hexdigest()

    safe_predictor_source = (
        "class CausalPrefix:\n"
        "    pass\n\n"
        "class EstimatorStream:\n"
        "    pass\n\n"
        "class PrefixCache:\n"
        "    pass\n\n"
        "class PriorPrediction:\n"
        "    pass\n\n"
        "class SafePredictor:\n"
        "    def next_token_log_probs(\n"
        "        self,\n"
        "        prefix_tokens: CausalPrefix,\n"
        "        estimator_rng: EstimatorStream,\n"
        "        cache: PrefixCache | None = None,\n"
        "    ) -> PriorPrediction:\n"
        "        return PriorPrediction()\n"
    )
    complete_sink_inventory_source = (
        "H6_STATIC_SINK_INVENTORY = (\n"
        "    'source_prior_logits',\n"
        "    'transition_parameters',\n"
        "    'emission_logits',\n"
        "    'estimator_proposals',\n"
        "    'estimator_weights',\n"
        "    'predictor_outputs',\n"
        "    'cache_keys',\n"
        "    'cache_values',\n"
        "    'training_inputs',\n"
        "    'tuning_inputs',\n"
        "    'analysis_inputs',\n"
        "    'public_preprocessing_returns',\n"
        ")\n"
    )
    safe_access_source = (
        "def _require_readiness(store, readiness):\n"
        "    return readiness\n\n"
        "def _validate(store, opening):\n"
        "    return opening\n\n"
        "def _materialize_train(store, readiness):\n"
        "    _require_readiness(store, readiness)\n"
        "    return ()\n\n"
        "def _open_test(store, opening):\n"
        "    validated = _validate(store, opening)\n"
        "    return validated\n\n"
        "(materialize_prediction_train, open_test_for_scoring) = (\n"
        "    _materialize_train,\n"
        "    _open_test,\n"
        ")\n"
    )

    safe_root = source_tree(
        "safe-literal-reflection",
        {
            "vfe4/predictive/prior.py": (
                safe_predictor_source
                + "\ndef literal_feature_probe(value):\n"
                "    return getattr(value, 'known_field', None)\n"
            ),
        },
    )
    safe_report = audit_h6_static_source(safe_root, exact_case_keys)
    assert safe_report.status is EvidenceStatus.PASS
    assert not findings_for(safe_report, "reflection")

    recognition_root = source_tree(
        "recognition-import",
        {
            "vfe4/predictive/prior.py": (
                "from vfe4.recognition.language import H6LanguageRecognitionLaw\n\n"
                "def build_prior(recognition: H6LanguageRecognitionLaw):\n"
                "    return recognition\n"
            ),
        },
    )
    recognition_report = audit_h6_static_source(
        recognition_root,
        exact_case_keys,
    )
    require_failure(
        recognition_report,
        "recognition_import",
        "vfe4/predictive/prior.py",
    )

    signature_root = source_tree(
        "predictor-signature",
        {
            "vfe4/predictive/prior.py": (
                "class TargetReadingPredictor:\n"
                "    def next_token_log_probs(\n"
                "        self,\n"
                "        prefix_tokens,\n"
                "        estimator_rng,\n"
                "        cache=None,\n"
                "        target_tokens=None,\n"
                "        **kwargs,\n"
                "    ):\n"
                "        return target_tokens, kwargs\n"
            ),
        },
    )
    signature_report = audit_h6_static_source(signature_root, exact_case_keys)
    signature_findings = require_failure(
        signature_report,
        "predictor_signature",
        "vfe4/predictive/prior.py",
    )
    signature_messages = " ".join(
        item.message.lower() for item in signature_findings
    )
    assert "target" in signature_messages
    assert "kwargs" in signature_messages or "variadic" in signature_messages
    forged_checks = (signature_report.checks[0], *safe_report.checks[1:])
    forged_payload = {
        "schema_version": safe_report.schema_version,
        "source_manifest_sha256": safe_report.source_manifest_sha256,
        "rules_sha256": safe_report.rules_sha256,
        "case_key_manifest_sha256": safe_report.case_key_manifest_sha256,
        "checks": tuple(check.check_sha256 for check in forged_checks),
        "findings": (),
        "status": EvidenceStatus.PASS.value,
        "obligations": (),
    }
    with pytest.raises((TypeError, ValueError)):
        type(safe_report)(
            safe_report.schema_version,
            safe_report.source_manifest_sha256,
            safe_report.rules_sha256,
            safe_report.case_key_manifest_sha256,
            forged_checks,
            (),
            EvidenceStatus.PASS,
            (),
            owned_hash("vfe4.h6.static-audit-report.v1", forged_payload),
        )

    count_root = source_tree(
        "base-counts",
        {
            "vfe4/generative/source_priors.py": (
                "SMALL_BASE_MASK_CASE_COUNT = 167\n"
                "WIKITEXT2_BASE_MASK_CASE_COUNT = 16_383\n"
                "H6_MASK_BASE_COUNTS = (167, 16_383)\n"
            ),
            "vfe4/predictive/prior.py": (
                "class RequiredCachePredictor:\n"
                "    def next_token_log_probs(\n"
                "        self,\n"
                "        prefix_tokens,\n"
                "        estimator_rng,\n"
                "        cache,\n"
                "    ):\n"
                "        return prefix_tokens\n"
            ),
        },
    )
    count_report = audit_h6_static_source(count_root, exact_case_keys)
    count_findings = require_failure(
        count_report,
        "base_count",
        "vfe4/generative/source_priors.py",
    )
    count_messages = " ".join(item.message for item in count_findings)
    normalized_count_messages = count_messages.replace(",", "").replace("_", "")
    assert "168" in normalized_count_messages
    assert "16384" in normalized_count_messages
    cache_default_findings = require_failure(
        count_report,
        "predictor_signature",
        "vfe4/predictive/prior.py",
    )
    cache_default_messages = " ".join(
        item.message.lower() for item in cache_default_findings
    )
    assert "cache" in cache_default_messages
    assert "default" in cache_default_messages or "none" in cache_default_messages

    inventory_root = source_tree(
        "sink-inventory",
        {
            "vfe4/validation/static_inventory.py": (
                "H6_STATIC_SINK_INVENTORY = (\n"
                "    'source_prior_logits',\n"
                "    'emission_logits',\n"
                ")\n"
            ),
        },
    )
    inventory_report = audit_h6_static_source(inventory_root, exact_case_keys)
    require_failure(
        inventory_report,
        "sink_inventory",
        "vfe4/validation/static_inventory.py",
    )

    reflection_root = source_tree(
        "reflection",
        {
            "vfe4/predictive/dispatch.py": (
                "def unresolved_dispatch(predictor, method_name, arguments):\n"
                "    method = getattr(predictor, method_name)\n"
                "    return method(*arguments)\n"
            ),
        },
    )
    reflection_report = audit_h6_static_source(reflection_root, exact_case_keys)
    assert reflection_report.status is EvidenceStatus.INCONCLUSIVE
    reflection_findings = findings_for(reflection_report, "reflection")
    assert reflection_findings
    assert all(
        finding.status is EvidenceStatus.INCONCLUSIVE
        for finding in reflection_findings
    )
    assert reflection_report.obligations
    assert not any(
        finding.status is EvidenceStatus.FAIL
        for finding in reflection_report.findings
    )

    access_root = source_tree(
        "access",
        {
            "vfe4/data/access.py": (
                "def _unseal_test_bytes(store):\n"
                "    return store.test_bytes\n\n"
                "def _materialize_train(store, readiness):\n"
                "    return store.train_bytes\n\n"
                "def _open_test(store, opening):\n"
                "    return store.test_bytes\n\n"
                "def materialize_prediction_train(store, readiness):\n"
                "    return store.train_bytes\n"
            ),
            "vfe4/training/runner.py": (
                "from vfe4.data.access import (\n"
                "    _unseal_test_bytes,\n"
                "    materialize_prediction_train,\n"
                ")\n\n"
                "def train_before_readiness(store):\n"
                "    test_bytes = _unseal_test_bytes(store)\n"
                "    train_bytes = materialize_prediction_train(store, None)\n"
                "    return test_bytes, train_bytes\n"
            ),
        },
    )
    access_report = audit_h6_static_source(access_root, exact_case_keys)
    require_failure(
        access_report,
        "private_unsealer",
        "vfe4/training/runner.py",
    )
    require_failure(
        access_report,
        "pre_readiness_access",
        "vfe4/training/runner.py",
    )
    require_failure(
        access_report,
        "pre_readiness_access",
        "vfe4/data/access.py",
    )
    require_failure(
        access_report,
        "durable_opening",
        "vfe4/data/access.py",
    )

    missing_count_root = source_tree(
        "production-missing-count",
        {
            "vfe4/data/access.py": safe_access_source,
            "vfe4/predictive/prior.py": safe_predictor_source,
            "vfe4/predictive/cache.py": "class PrefixCache:\n    pass\n",
            "vfe4/generative/source_priors.py": complete_sink_inventory_source,
            "vfe4/numerics/categorical.py": (
                "def masked_log_softmax_from_parents(\n"
                "    logits, declared_parents, receiver_t\n"
                "):\n"
                "    if not declared_parents:\n"
                "        raise ValueError('empty support')\n"
                "    return logits\n"
            ),
        },
    )
    missing_count_report = audit_h6_static_source(
        missing_count_root,
        exact_case_keys,
    )
    assert missing_count_report.status is not EvidenceStatus.PASS
    assert findings_for(missing_count_report, "base_count")
    assert not any(
        finding.status is EvidenceStatus.FAIL
        for finding in missing_count_report.findings
        if finding.rule_id in {"pre_readiness_access", "durable_opening", "reflection"}
    )

    missing_marker_files = {
        "vfe4/data/access.py": safe_access_source,
        "vfe4/predictive/prior.py": safe_predictor_source,
        "vfe4/generative/source_priors.py": (
            complete_sink_inventory_source
            + "\nH6_MASK_BASE_COUNTS = (168, 16_384)\n"
        ),
        "vfe4/numerics/categorical.py": (
            "def masked_log_softmax_from_parents(\n"
            "    logits, declared_parents, receiver_t\n"
            "):\n"
            "    if not declared_parents:\n"
            "        raise ValueError('empty support')\n"
            "    return logits\n"
        ),
    }
    missing_marker_root = source_tree(
        "production-missing-marker",
        missing_marker_files,
    )
    missing_marker_report = audit_h6_static_source(
        missing_marker_root,
        exact_case_keys,
    )
    assert missing_marker_report.status is not EvidenceStatus.PASS
    missing_marker_text = " ".join(
        " ".join(
            (
                finding.rule_id,
                finding.message,
                str(finding.path).replace("\\", "/"),
            )
        )
        for finding in missing_marker_report.findings
    )
    assert (
        "vfe4/predictive/cache.py" in missing_marker_text
        or (
            "production" in missing_marker_text.lower()
            and "marker" in missing_marker_text.lower()
        )
    )

    real_repo_root = Path(__file__).resolve().parents[2]
    real_report = audit_h6_static_source(real_repo_root, exact_case_keys)
    assert real_report.status in {
        EvidenceStatus.PASS,
        EvidenceStatus.INCONCLUSIVE,
    }
    assert not any(
        finding.status is EvidenceStatus.FAIL
        for finding in real_report.findings
    )
    if real_report.status is EvidenceStatus.INCONCLUSIVE:
        assert real_report.obligations
