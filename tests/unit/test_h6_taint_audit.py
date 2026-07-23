from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from vfe4.types import ArmId, EvidenceStatus, PrefixCaseKey
from vfe4.validation.h6_static_audit import audit_h6_static_source


def test_h6_taint_cache_and_split_capability_rules(tmp_path: Path) -> None:
    key = PrefixCaseKey(
        arm=ArmId.A0,
        predictor_config_sha256="a" * 64,
        estimator_sha256="b" * 64,
        model_family_sha256="c" * 64,
        vocabulary_sha256="d" * 64,
        data_safety_sha256="e" * 64,
        git_head="1" * 40,
        dirty_digest="f" * 64,
    )

    safe_sources = {
        "vfe4/predictive/prior.py": """
            def source_prior_logits(prefix_tensor):
                causal_state = prefix_tensor
                return causal_state


            def emission_logits(causal_state):
                return causal_state


            def predictor_output(prefix_tensor):
                prior_state = source_prior_logits(prefix_tensor)
                return emission_logits(prior_state)
        """,
        "vfe4/predictive/cache.py": """
            def cache_key(
                source_sha256,
                predictor_config_sha256,
                model_state_sha256,
                estimator_sha256,
                prefix_sha256,
            ):
                return (
                    source_sha256,
                    predictor_config_sha256,
                    model_state_sha256,
                    estimator_sha256,
                    prefix_sha256,
                )


            def cache_value(causal_filter_state, counter_position):
                return {
                    "causal_filter_state": causal_filter_state,
                    "counter_position": counter_position,
                }
        """,
        "vfe4/data/preprocess.py": """
            def blinded_preprocess(
                sealed_train_sha256,
                validation_sha256,
                sealed_test_sha256,
            ):
                return {
                    "train_sha256": sealed_train_sha256,
                    "validation_sha256": validation_sha256,
                    "test_sha256": sealed_test_sha256,
                }
        """,
        "vfe4/training/runner.py": """
            def training_step(materialized_train_tokens):
                return len(materialized_train_tokens)
        """,
        "vfe4/tuning/select.py": """
            def tune_candidate(validation_metrics):
                return min(validation_metrics)
        """,
    }

    def write_tree(
        name: str, overrides: dict[str, str] | None = None
    ) -> Path:
        root = tmp_path / name
        sources = dict(safe_sources)
        if overrides is not None:
            sources.update(overrides)
        for relative_path, source in sources.items():
            destination = root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(dedent(source), encoding="utf-8", newline="\n")
        return root

    def finding_text(report: object) -> str:
        findings = getattr(report, "findings")
        rows = (
            f"{getattr(finding, 'rule_id', '')} "
            f"{getattr(finding, 'message', finding)}"
            for finding in findings
        )
        return "\n".join(rows).lower()

    safe_report = audit_h6_static_source(
        repo_root=write_tree("safe"),
        exact_case_keys=(key,),
    )
    assert safe_report.status is EvidenceStatus.PASS
    assert safe_report.findings == ()
    assert safe_report.obligations == ()

    flow_report = audit_h6_static_source(
        repo_root=write_tree(
            "tainted-flows",
            {
                "vfe4/predictive/prior.py": """
                    def source_prior_logits(prefix_tensor, target_tensor):
                        copied_target = target_tensor
                        return copied_target


                    def emission_logits(prefix_tensor, suffix_tensor):
                        copied_suffix = suffix_tensor
                        return copied_suffix
                """,
                "vfe4/predictive/cache.py": """
                    def cache_key(
                        source_sha256,
                        predictor_config_sha256,
                        model_state_sha256,
                        estimator_sha256,
                        prefix_sha256,
                        target_tensor,
                    ):
                        return (
                            source_sha256,
                            predictor_config_sha256,
                            model_state_sha256,
                            estimator_sha256,
                            prefix_sha256,
                            target_tensor,
                        )


                    def cache_value(
                        causal_filter_state,
                        counter_position,
                        recognition_activation,
                        target_tensor,
                    ):
                        return {
                            "causal_filter_state": causal_filter_state,
                            "counter_position": counter_position,
                            "recognition_activation": recognition_activation,
                            "target_tensor": target_tensor,
                        }
                """,
            },
        ),
        exact_case_keys=(key,),
    )
    flow_text = finding_text(flow_report)
    assert flow_report.status is EvidenceStatus.FAIL
    assert "target_dataflow" in flow_text
    assert "cache_target_data" in flow_text
    assert "target" in flow_text
    assert "suffix" in flow_text
    assert "recognition" in flow_text

    cache_schema_report = audit_h6_static_source(
        repo_root=write_tree(
            "cache-schema",
            {
                "vfe4/predictive/cache.py": """
                    def cache_key(opaque_key):
                        return opaque_key


                    def cache_value(
                        causal_filter_state,
                        counter_position,
                        debug_payload,
                    ):
                        return {
                            "causal_filter_state": causal_filter_state,
                            "counter_position": counter_position,
                            "debug_payload": debug_payload,
                        }
                """,
            },
        ),
        exact_case_keys=(key,),
    )
    cache_schema_text = finding_text(cache_schema_report)
    assert cache_schema_report.status is EvidenceStatus.FAIL
    assert "cache_target_data" in cache_schema_text
    assert "source_sha256" in cache_schema_text
    assert "predictor_config_sha256" in cache_schema_text
    assert "model_state_sha256" in cache_schema_text
    assert "estimator_sha256" in cache_schema_text
    assert "prefix_sha256" in cache_schema_text
    assert "debug_payload" in cache_schema_text

    split_report = audit_h6_static_source(
        repo_root=write_tree(
            "sealed-splits",
            {
                "vfe4/training/runner.py": """
                    def optimizer_step(batch):
                        return batch


                    def training_step(sealed_train_bytes):
                        return optimizer_step(sealed_train_bytes)
                """,
                "vfe4/tuning/select.py": """
                    def score_candidate(candidate_data):
                        return candidate_data


                    def tune_candidate(split_store):
                        sealed_test_bytes = split_store["sealed_test"]
                        return score_candidate(sealed_test_bytes)
                """,
            },
        ),
        exact_case_keys=(key,),
    )
    split_text = finding_text(split_report)
    assert split_report.status is EvidenceStatus.FAIL
    assert "split_dataflow" in split_text
    assert "sealed_train" in split_text
    assert "sealed_test" in split_text

    preprocessing_report = audit_h6_static_source(
        repo_root=write_tree(
            "preprocessing-escape",
            {
                "vfe4/data/preprocess.py": """
                    def blinded_preprocess(sealed_test_bytes):
                        public_result = {"raw": sealed_test_bytes}
                        return public_result
                """,
            },
        ),
        exact_case_keys=(key,),
    )
    preprocessing_text = finding_text(preprocessing_report)
    assert preprocessing_report.status is EvidenceStatus.FAIL
    assert "preprocessing_escape" in preprocessing_text
    assert "sealed_test" in preprocessing_text

    reflection_report = audit_h6_static_source(
        repo_root=write_tree(
            "unresolved-reflection",
            {
                "vfe4/predictive/dispatch.py": """
                    def unresolved_predictor(
                        prefix_tensor,
                        predictor_object,
                        method_name,
                        expression,
                    ):
                        dynamic_method = getattr(predictor_object, method_name)
                        dynamic_result = dynamic_method(prefix_tensor)
                        if expression:
                            return eval(expression)
                        return dynamic_result
                """,
            },
        ),
        exact_case_keys=(key,),
    )
    reflection_findings = finding_text(reflection_report)
    reflection_obligations = "\n".join(reflection_report.obligations).lower()
    assert reflection_report.status is EvidenceStatus.INCONCLUSIVE
    assert "target_dataflow" not in reflection_findings
    assert "reflection" in reflection_obligations
    assert "getattr" in reflection_obligations
    assert "eval" in reflection_obligations
