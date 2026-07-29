from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "train_vfe4.py"
OPERATIONS = (
    "idle",
    "synthetic_smoke",
    "source_lock",
    "readiness",
    "train",
    "resume",
)


def _load(name: str) -> object:
    spec = importlib.util.spec_from_file_location(name, LAUNCHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def test_wt103_launcher_is_one_dictionary_click_surface_and_import_safe() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=LAUNCHER.name)
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imports.isdisjoint(
        {"argparse", "click", "typer", "hydra", "tiktoken", "torch"}
    )
    assert not any(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
        and node.attr in {"environ", "getenv"}
        for node in ast.walk(tree)
    )
    assignments = tuple(
        node
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "CONFIG"
    )
    guards = tuple(
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
    )
    assert len(assignments) == len(guards) == 1

    launcher = _load("train_vfe4_wt103_import")
    assert launcher.OPERATIONS == OPERATIONS
    assert launcher.CONFIG["training"]["operation"] == "idle"
    assert set(launcher.CONFIG) == {
        "launcher_schema",
        "training",
        "paths",
        "authorization",
    }
    assert launcher.CONFIG["paths"]["cache_root"] == str(
        Path.home() / ".cache" / "vfe4" / "wikitext103"
    )
    assert launcher.CONFIG["paths"]["run_root"]
    assert launcher.CONFIG["paths"]["source_record_path"]
    idle = launcher.main(copy.deepcopy(launcher.CONFIG))
    assert idle.status == "IDLE"
    assert idle.operation == "idle"


def test_launcher_import_stays_pure_under_transitive_live_import_blockers() -> None:
    script = f"""
import builtins
import importlib.util
import sys
from pathlib import Path

blocked = (
    "torch",
    "tiktoken",
    "importlib.metadata",
    "urllib.request",
)
original = builtins.__import__

def guarded(name, *args, **kwargs):
    if any(name == item or name.startswith(item + ".") for item in blocked):
        raise AssertionError("blocked live import: " + name)
    return original(name, *args, **kwargs)

builtins.__import__ = guarded
path = Path({str(LAUNCHER)!r})
spec = importlib.util.spec_from_file_location("task12_pure_import", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert module.CONFIG["training"]["operation"] == "idle"
assert "torch" not in sys.modules
assert "tiktoken" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_synthetic_smoke_exercises_all_arms_resume_metrics_and_terminal_ids(
    tmp_path: Path,
) -> None:
    launcher = _load("train_vfe4_wt103_smoke")
    config = copy.deepcopy(launcher.CONFIG)
    config["training"]["operation"] = "synthetic_smoke"
    config["paths"]["cache_root"] = str(tmp_path / "cache")
    config["paths"]["run_root"] = str(tmp_path / "runs")
    config["paths"]["source_record_path"] = str(
        tmp_path / "unavailable-production-source.json"
    )
    config["paths"]["resume_experiment_index_path"] = str(
        tmp_path / "unavailable-resume-index.json"
    )
    config["paths"]["smoke_run_id"] = "task12-focused-smoke"

    result = launcher.main(config)

    assert result.status == "COMPLETED"
    assert result.operation == "synthetic_smoke"
    smoke = result.payload
    assert smoke.authority == "nonproduction_synthetic_smoke"
    assert smoke.production_readiness_eligible is False
    assert smoke.heldout_test_opened is False
    assert tuple(row.arm_id for row in smoke.arm_results) == (
        "WT103-A0-AR-v1",
        "WT103-A5-PARENT-SPECIFIC-PREFIX-COMPLETE-v1",
        "WT103-A5-FIXED-COMPLETE-v1",
        "WT103-A5-PARENT-SPECIFIC-PREFIX-EMISSION-v1",
        "WT103-A5-NOLATENT-v1",
    )
    assert tuple(row.constructor_id for row in smoke.arm_results) == (
        "build_wt103_a0",
        "build_wt103_a5_parent_specific",
        "build_wt103_a5_fixed",
        "build_wt103_a5_parent_specific",
        "build_wt103_a5_nolatent",
    )
    assert all(row.accepted_update for row in smoke.arm_results)
    assert all(row.validation_completed for row in smoke.arm_results)
    assert all(
        row.authority == "nonproduction_synthetic_smoke"
        for row in smoke.arm_results
    )
    assert all(row.terminal_checkpoint_role == "terminal_scoring" for row in smoke.arm_results)
    assert all(row.metrics_jsonl_sha256 and row.metrics_csv_sha256 for row in smoke.arm_results)
    assert all(
        Path(row.run_manifest_path).is_file()
        for row in smoke.arm_results
    )
    expected_execution_paths = (
        (
            "wt103_a0_decoder_cross_entropy",
            "absent",
            "exact_autoregressive",
            "score_prior_nll",
        ),
        (
            "language_generative_complete_elbo",
            "parent_specific_pooled_prefix",
            "weighted_smc",
            "score_prior_nll",
        ),
        (
            "language_generative_complete_elbo",
            "fixed",
            "weighted_smc",
            "score_prior_nll",
        ),
        (
            "language_generative_emission_only_non_elbo",
            "parent_specific_pooled_prefix",
            "weighted_smc",
            "score_prior_nll",
        ),
        (
            "mean_pooled_prefix_cross_entropy",
            "absent",
            "exact_autoregressive",
            "score_prior_nll",
        ),
    )
    assert tuple(
        (
            row.execution_trace.forward_path,
            row.execution_trace.prior_path,
            row.execution_trace.scorer_path,
            row.execution_trace.evaluator_path,
        )
        for row in smoke.arm_results
    ) == expected_execution_paths
    assert all(row.execution_trace.counted_targets == 3 for row in smoke.arm_results)
    assert all(
        row.execution_trace.score_trace_sha256
        and row.execution_trace.nll_totals_sha256
        and row.execution_trace.forward_evidence_sha256
        for row in smoke.arm_results
    )
    assert (
        smoke.arm_results[1].execution_trace.source_factor_sha256
        != smoke.arm_results[2].execution_trace.source_factor_sha256
    )
    assert sum(row.resume_exercised for row in smoke.arm_results) == 1
    assert smoke.resume_checkpoint_role == "resume_only"
    assert smoke.resume_identity_before_sha256 == (
        smoke.resume_identity_after_sha256
    )
    assert smoke.resume_oracle_passed is True
    assert (
        smoke.resume_uninterrupted_terminal_scientific_state_sha256
        == smoke.resume_resumed_terminal_scientific_state_sha256
    )
    assert (
        smoke.resume_uninterrupted_metrics_jsonl_sha256
        == smoke.resume_resumed_metrics_jsonl_sha256
    )
    assert smoke.resume_uninterrupted_next_predictions_equal is True
    runtime = smoke.runtime_observation
    assert runtime.execution_mode == "isolated_subprocess"
    assert runtime.worker_process_id != runtime.parent_process_id
    assert runtime.intraop_threads == runtime.interop_threads == 1
    assert dict(runtime.thread_environment) == {
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
    assert runtime.cuda_visible_devices == "-1"
    assert runtime.cuda_available is False
    assert runtime.cuda_initialized_on_entry is False
    assert runtime.cuda_initialized_on_exit is False
    assert smoke.experiment_index_stage == "pretest"
    assert Path(smoke.experiment_index_path).is_file()


@pytest.mark.parametrize("operation", ("readiness", "train", "resume"))
def test_production_modes_fail_before_reservation_without_source_lock(
    tmp_path: Path,
    operation: str,
) -> None:
    launcher = _load(f"train_vfe4_wt103_fail_closed_{operation}")
    config = copy.deepcopy(launcher.CONFIG)
    config["training"]["operation"] = operation
    config["paths"]["cache_root"] = str(tmp_path / "cache")
    config["paths"]["run_root"] = str(tmp_path / "runs")
    config["paths"]["source_record_path"] = str(
        tmp_path / "missing-source-record.json"
    )
    config["paths"]["resume_experiment_index_path"] = str(
        tmp_path / "missing-index.json"
    )
    config["authorization"] = (
        launcher.PRODUCTION_AUTHORIZATION
        if operation in ("train", "resume")
        else None
    )

    with pytest.raises(
        launcher.TrainingLaunchError,
        match="finalized production source",
    ):
        launcher.main(config)
    assert not (tmp_path / "runs").exists()


@dataclass
class _HermeticProductionDriver:
    events: list[str]
    source: object
    readiness_result: object
    operation_result: object

    def source_lock(self, *, training: object, paths: object) -> object:
        del training, paths
        self.events.append("source_lock")
        return self.source

    def reopen_source_lock(
        self,
        *,
        training: object,
        paths: object,
    ) -> object:
        del training, paths
        self.events.append("reopen_source_lock")
        return self.source

    def readiness(
        self,
        *,
        training: object,
        paths: object,
        source_lock: object,
    ) -> object:
        del training, paths
        assert source_lock is self.source
        self.events.append("readiness")
        return self.readiness_result

    def train(
        self,
        *,
        training: object,
        paths: object,
        source_lock: object,
        readiness: object,
    ) -> object:
        del training, paths
        assert source_lock is self.source
        assert readiness is self.readiness_result
        self.events.append("train")
        return self.operation_result

    def resume(
        self,
        *,
        training: object,
        paths: object,
        source_lock: object,
        readiness: object,
    ) -> object:
        del training, paths
        assert source_lock is self.source
        assert readiness is self.readiness_result
        self.events.append("resume")
        return self.operation_result


@pytest.mark.parametrize(
    ("operation", "expected_events"),
    (
        ("source_lock", ("source_lock",)),
        ("readiness", ("reopen_source_lock", "readiness")),
        ("train", ("reopen_source_lock", "readiness", "train")),
        ("resume", ("reopen_source_lock", "readiness", "resume")),
    ),
)
def test_authorized_production_operations_dispatch_concrete_typed_sequence(
    tmp_path: Path,
    operation: str,
    expected_events: tuple[str, ...],
) -> None:
    launcher = _load(f"train_vfe4_wt103_dispatch_{operation}")
    config = copy.deepcopy(launcher.CONFIG)
    config["training"]["operation"] = operation
    config["paths"]["cache_root"] = str(tmp_path / "cache")
    config["paths"]["run_root"] = str(tmp_path / "runs")
    source_path = tmp_path / "source-record.json"
    config["paths"]["source_record_path"] = str(source_path)
    config["paths"]["resume_experiment_index_path"] = str(
        tmp_path / "runs" / "experiment-index.json"
    )
    if operation != "source_lock":
        source_path.write_bytes(b"typed source seam")
    config["authorization"] = (
        launcher.SOURCE_LOCK_AUTHORIZATION
        if operation == "source_lock"
        else (
            launcher.PRODUCTION_AUTHORIZATION
            if operation in ("train", "resume")
            else None
        )
    )
    source = object()
    readiness = object()
    operation_result = object()
    driver = _HermeticProductionDriver(
        events=[],
        source=source,
        readiness_result=readiness,
        operation_result=operation_result,
    )

    result = launcher.main(config, driver=driver)

    assert result.status == "COMPLETED"
    assert tuple(driver.events) == expected_events
    assert result.payload is (
        source
        if operation == "source_lock"
        else readiness
        if operation == "readiness"
        else operation_result
    )


def test_wrong_authorization_fails_before_default_driver_or_live_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _load("train_vfe4_wt103_auth_before_driver")
    config = copy.deepcopy(launcher.CONFIG)
    config["training"]["operation"] = "source_lock"
    config["paths"]["cache_root"] = str(tmp_path / "cache")
    config["paths"]["run_root"] = str(tmp_path / "runs")
    config["paths"]["source_record_path"] = str(tmp_path / "source.json")
    config["paths"]["resume_experiment_index_path"] = str(
        tmp_path / "runs" / "index.json"
    )
    config["authorization"] = "wrong"
    monkeypatch.setattr(
        launcher,
        "_default_driver",
        lambda: (_ for _ in ()).throw(AssertionError("driver constructed")),
    )

    with pytest.raises(PermissionError, match="exact explicit authorization"):
        launcher.main(config)


class _InjectedProductionTokenizer:
    distribution_name = "tiktoken"
    distribution_version = "0.12.0"
    encoding_name = "gpt2"
    vocabulary_size = 50_257
    eot_token_id = 50_256
    regex_pattern_sha256 = hashlib.sha256(b"fixture-regex").hexdigest()
    mergeable_ranks_sha256 = hashlib.sha256(b"fixture-ranks").hexdigest()
    special_tokens_sha256 = hashlib.sha256(b"fixture-special").hexdigest()
    golden_vectors_sha256 = hashlib.sha256(b"fixture-golden").hexdigest()
    regex_engine_distribution_name = "regex"
    regex_engine_distribution_version = "2026.1.1"
    regex_engine_distribution_record_sha256 = hashlib.sha256(
        b"fixture-regex-record"
    ).hexdigest()

    def __init__(self, distribution_record_sha256: str) -> None:
        from vfe4.types.training import production_tokenizer_tables_sha256

        self.distribution_record_sha256 = distribution_record_sha256
        self.tokenizer_tables_sha256 = production_tokenizer_tables_sha256(
            regex_pattern_sha256=self.regex_pattern_sha256,
            regex_engine_distribution_name=(
                self.regex_engine_distribution_name
            ),
            regex_engine_distribution_version=(
                self.regex_engine_distribution_version
            ),
            regex_engine_distribution_record_sha256=(
                self.regex_engine_distribution_record_sha256
            ),
            mergeable_ranks_sha256=self.mergeable_ranks_sha256,
            special_tokens_sha256=self.special_tokens_sha256,
            golden_vectors_sha256=self.golden_vectors_sha256,
        )

    def encode_ordinary(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int]) -> str:
        return bytes(token_ids).decode("utf-8", errors="strict")

    def split_regex_pieces(self, text: str) -> tuple[str, ...]:
        return tuple(text)

    def encode_single_piece(self, piece: str) -> list[int]:
        return list(piece.encode("utf-8"))

    def decode_token_bytes(self, token_ids: list[int]) -> bytes:
        return bytes(token_ids)


def _injected_wt103_archive() -> bytes:
    members = {
        "wikitext-103-raw/wiki.train.raw": b"train text\n",
        "wikitext-103-raw/wiki.valid.raw": b"validation text\n",
        "wikitext-103-raw/wiki.test.raw": b"test text\n",
    }
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", allowZip64=False) as archive:
        directory = zipfile.ZipInfo("wikitext-103-raw/")
        directory.external_attr = (0o40755 << 16) | 0x10
        directory.compress_type = zipfile.ZIP_STORED
        archive.writestr(directory, b"")
        for name, payload in members.items():
            info = zipfile.ZipInfo(name)
            info.external_attr = 0o100600 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, payload)
    return stream.getvalue()


class _InjectedWt103HttpClient:
    def __init__(self) -> None:
        self.archive = _injected_wt103_archive()
        self.source_page = (
            b"<html><body><p>Released under a Creative Commons "
            b'<a href="https://creativecommons.org/licenses/by-sa/4.0/">'
            b"Attribution-ShareAlike license</a>.</p></body></html>"
        )
        self.calls: list[str] = []

    def fetch(self, url: str, *, maximum_bytes: int) -> object:
        from vfe4.data.wikitext103 import (
            BoundedHttpObservation,
            HttpRedirectObservation,
            WIKITEXT103_ARCHIVE_REQUEST_URL,
            WIKITEXT103_SOURCE_PAGE_REQUEST_URL,
        )

        self.calls.append(url)
        if url == WIKITEXT103_ARCHIVE_REQUEST_URL:
            assert len(self.archive) <= maximum_bytes
            return BoundedHttpObservation.create(
                request_url=url,
                final_url=url,
                redirect_chain=(),
                status_code=200,
                headers=(("content-type", "application/zip"),),
                body=self.archive,
            )
        if url == WIKITEXT103_SOURCE_PAGE_REQUEST_URL:
            assert len(self.source_page) <= maximum_bytes
            final_url = "https://www.salesforce.com/research/wikitext/"
            return BoundedHttpObservation.create(
                request_url=url,
                final_url=final_url,
                redirect_chain=(
                    HttpRedirectObservation(
                        status_code=301,
                        location=final_url,
                        resolved_url=final_url,
                    ),
                ),
                status_code=200,
                headers=(("content-type", "text/html; charset=utf-8"),),
                body=self.source_page,
            )
        raise AssertionError(f"unexpected source-lock URL: {url}")


class _RecordingDurabilityBackend:
    def __init__(
        self,
        delegate: object,
        *,
        inconsistent_token_identity: bool = False,
    ) -> None:
        self.delegate = delegate
        self.writes: list[Path] = []
        self.stream_writes: list[Path] = []
        self.stream_chunk_sizes: list[tuple[str, tuple[int, ...]]] = []
        self.inconsistent_token_identity = inconsistent_token_identity

    def probe(self, root: Path) -> object:
        return self.delegate.probe(root)

    @staticmethod
    def _reject_corpus_bytes(path: Path) -> None:
        if path.suffix in (".raw", ".int32le"):
            raise AssertionError(
                "corpus payload reached a whole-bytes publication API"
            )

    def create_exclusive(self, path: Path, payload: bytes) -> object:
        self._reject_corpus_bytes(path)
        result = self.delegate.create_exclusive(path, payload)
        self.writes.append(path)
        return result

    def replace_durable(self, path: Path, payload: bytes) -> object:
        self._reject_corpus_bytes(path)
        result = self.delegate.replace_durable(path, payload)
        self.writes.append(path)
        return result

    def publish_bytes(self, path: Path, payload: bytes) -> None:
        self._reject_corpus_bytes(path)
        self.delegate.publish_bytes(path, payload)
        self.writes.append(path)

    def publish_content_addressed_stream(
        self,
        directory: Path,
        chunks: object,
        *,
        suffix: str,
        chunk_size_limit: int,
        reopen_block_size: int = 1_048_576,
    ) -> object:
        from vfe4.artifacts.durability import DurableFileIdentity

        sizes: list[int] = []

        def recorded_chunks():
            for chunk in chunks:
                assert type(chunk) is bytes
                assert len(chunk) <= chunk_size_limit
                sizes.append(len(chunk))
                yield chunk

        identity = self.delegate.publish_content_addressed_stream(
            directory,
            recorded_chunks(),
            suffix=suffix,
            chunk_size_limit=chunk_size_limit,
            reopen_block_size=reopen_block_size,
        )
        self.stream_writes.append(
            directory / f"{identity.sha256}{suffix}"
        )
        self.stream_chunk_sizes.append((suffix, tuple(sizes)))
        if self.inconsistent_token_identity and suffix == ".int32le":
            return DurableFileIdentity.create_verified(
                operation="content_addressed",
                size_bytes=identity.size_bytes,
                sha256="0" * 64,
                volume_identity=identity.volume_identity,
            )
        return identity


def _injected_source_lock_case(
    tmp_path: Path,
    *,
    inconsistent_token_identity: bool = False,
) -> tuple[object, object, Path, object, _RecordingDurabilityBackend, object]:
    from vfe4.artifacts.environment import (
        DistributionIdentity,
        parse_lock_input_manifest,
    )
    from vfe4.training import production

    launcher = _load("train_vfe4_wt103_injected_source_lock")
    config = copy.deepcopy(launcher.CONFIG)
    config["training"]["operation"] = "source_lock"
    config["paths"]["cache_root"] = str(tmp_path / "cache")
    config["paths"]["run_root"] = str(tmp_path / "runs")
    config["paths"]["source_record_path"] = str(
        tmp_path / "tracked" / "source-record.json"
    )
    config["paths"]["resume_experiment_index_path"] = str(
        tmp_path / "runs" / "index.json"
    )
    config["authorization"] = launcher.SOURCE_LOCK_AUTHORIZATION
    training, paths, _ = launcher._resolve_launcher(config)

    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    lock_input = (ROOT / "requirements-wt103.lock-input.json").read_bytes()
    (repository_root / "requirements-wt103.lock-input.json").write_bytes(
        lock_input
    )
    (repository_root / "requirements-wt103.lock").write_bytes(
        (ROOT / "requirements-wt103.lock").read_bytes()
    )
    manifest = parse_lock_input_manifest(lock_input)
    installed = tuple(
        DistributionIdentity(
            name=requirement.name,
            version=requirement.version,
            record_sha256=hashlib.sha256(
                f"injected:{requirement.name}".encode("ascii")
            ).hexdigest(),
        )
        for requirement in manifest.requirements
    )
    tokenizer_record = next(
        item.record_sha256 for item in installed if item.name == "tiktoken"
    )
    client = _InjectedWt103HttpClient()
    durability = _RecordingDurabilityBackend(
        production._platform_backend(),
        inconsistent_token_identity=inconsistent_token_identity,
    )
    dependencies = production.ProductionSourceLockDependencies(
        http_client=client,
        tokenizer=_InjectedProductionTokenizer(tokenizer_record),
        installed_distributions=installed,
        pytorch_version="2.8.0+cu128",
        sdpa_api_sha256=hashlib.sha256(b"injected-sdpa").hexdigest(),
        flash_backend_sha256=hashlib.sha256(
            b"injected-flash"
        ).hexdigest(),
        repository_root=repository_root,
        durability_backend=durability,
    )
    return (
        training,
        paths,
        repository_root,
        client,
        durability,
        dependencies,
    )


def test_source_lock_uses_only_injected_source_and_tokenizer_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vfe4.training import production

    (
        training,
        paths,
        repository_root,
        client,
        durability,
        dependencies,
    ) = _injected_source_lock_case(
        tmp_path,
    )
    monkeypatch.setattr(
        production,
        "_live_source_lock_dependencies",
        lambda: (_ for _ in ()).throw(
            AssertionError("live dependency discovery attempted")
        ),
    )

    source_lock = production.run_source_lock(
        training=training,
        paths=paths,
        dependencies=dependencies,
    )

    assert tuple(client.calls) == (
        production.WIKITEXT103_ARCHIVE_REQUEST_URL,
        production.WIKITEXT103_SOURCE_PAGE_REQUEST_URL,
    )
    assert source_lock.finalized_source.freeze_completeness is True
    assert tuple(item.split for item in source_lock.token_caches) == (
        "train",
        "validation",
        "test",
    )
    matching = source_lock.a0_matching
    assert type(matching) is production.A0SourceLockMatchingAssessment
    assert matching.status.value == "inconclusive"
    assert matching.selected_hidden_width is None
    assert matching.primary_parameters.parameter_count == 8_599_703
    assert (
        matching.primary_parameters.model_parameter_count
        == 4_367_583
    )
    assert (
        matching.primary_parameters.recognition_parameter_count
        == 4_232_120
    )
    assert matching.primary_semantic_train_flops is None
    assert matching.primary_flop_ledger.semantic_train_flops is None
    assert all(not row.parameter_eligible for row in matching.rows)
    readiness = production.run_readiness(
        training=training,
        paths=paths,
        source_lock=source_lock,
    )
    assert readiness.status.value == "inconclusive"
    assert readiness.readiness is None
    assert readiness.readiness_token is None
    assert readiness.obligations
    assert all(
        item.startswith("capacity_matching:")
        for item in readiness.obligations
    )
    assert paths.source_record_path.is_file()
    assert durability.writes[-1] == paths.source_record_path
    assert tuple(path.suffix for path in durability.stream_writes).count(
        ".raw"
    ) == 3
    assert tuple(path.suffix for path in durability.stream_writes).count(
        ".int32le"
    ) == 3
    assert all(
        sizes and all(0 < size <= 1_048_576 for size in sizes)
        for _, sizes in durability.stream_chunk_sizes
    )
    assert all(
        path.suffix not in (".raw", ".int32le")
        for path in durability.writes
    )
    assert (
        production._reopen_source_lock(
            training=training,
            paths=paths,
            repository_root=repository_root,
        )
        == source_lock
    )
    regular_nonlink_bytes = production._regular_nonlink_bytes

    def reject_whole_corpus_reads(
        path: Path,
        *,
        expected_size: int | None = None,
        expected_sha256: str | None = None,
    ) -> bytes:
        if path.suffix in (".raw", ".int32le"):
            raise AssertionError(
                "corpus artifact reached the whole-file read helper"
            )
        return regular_nonlink_bytes(
            path,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        )

    monkeypatch.setattr(
        production,
        "_regular_nonlink_bytes",
        reject_whole_corpus_reads,
    )
    assert (
        production._reopen_source_lock(
            training=training,
            paths=paths,
            repository_root=repository_root,
        )
        == source_lock
    )
    for split in ("train", "validation"):
        windows, schedule = production.open_production_training_split(
            source_lock=source_lock,
            cache_root=paths.cache_root,
            split=split,
        )
        assert windows.split == split
        assert schedule.split == split
        windows.tokens._mmap.close()

    def mutate_without_resizing(payload: bytes) -> bytes:
        assert payload
        index = len(payload) // 2
        return (
            payload[:index]
            + bytes((payload[index] ^ 1,))
            + payload[index + 1 :]
        )

    raw_member = source_lock.finalized_source.members[0]
    raw_path = (
        paths.cache_root
        / "production-source-lock"
        / "staging"
        / "staged"
        / "splits"
        / raw_member.split
        / f"{raw_member.payload_sha256}.raw"
    )
    raw_payload = raw_path.read_bytes()
    raw_path.write_bytes(mutate_without_resizing(raw_payload))
    try:
        with pytest.raises(
            production.ProductionOperationError,
            match="production corpus validation failed",
        ):
            production._reopen_source_lock(
                training=training,
                paths=paths,
                repository_root=repository_root,
            )
    finally:
        raw_path.write_bytes(raw_payload)

    token_cache = source_lock.token_caches[0]
    token_path = paths.cache_root.joinpath(
        *Path(token_cache.cache_relative_path).parts
    )
    token_payload = token_path.read_bytes()
    token_path.write_bytes(mutate_without_resizing(token_payload))
    try:
        with pytest.raises(
            production.ProductionOperationError,
            match="production corpus validation failed",
        ):
            production._reopen_source_lock(
                training=training,
                paths=paths,
                repository_root=repository_root,
            )
        memmap_calls: list[Path] = []

        def reject_memmap(path: Path, *args: object, **kwargs: object) -> None:
            memmap_calls.append(path)
            raise AssertionError("mutated token payload reached np.memmap")

        monkeypatch.setattr(production.np, "memmap", reject_memmap)
        with pytest.raises(
            production.ProductionOperationError,
            match="production corpus validation failed",
        ):
            production.open_production_training_split(
                source_lock=source_lock,
                cache_root=paths.cache_root,
                split="train",
            )
        assert memmap_calls == []
    finally:
        token_path.write_bytes(token_payload)
    assert (
        production._reopen_source_lock(
            training=training,
            paths=paths,
            repository_root=repository_root,
        )
        == source_lock
    )


def test_source_lock_rejects_inconsistent_token_stream_identity(
    tmp_path: Path,
) -> None:
    from vfe4.training import production

    (
        training,
        paths,
        _,
        _,
        durability,
        dependencies,
    ) = _injected_source_lock_case(
        tmp_path,
        inconsistent_token_identity=True,
    )

    with pytest.raises(
        production.ProductionOperationError,
        match="bounded token publication failed",
    ):
        production.run_source_lock(
            training=training,
            paths=paths,
            dependencies=dependencies,
        )

    assert any(
        path.suffix == ".int32le"
        for path in durability.stream_writes
    )
    assert not (
        paths.cache_root
        / "production-source-lock"
        / "finalized-source-bundle.json"
    ).exists()
    assert not paths.source_record_path.exists()


@pytest.mark.parametrize(
    "forgery",
    ("raw_parent", "cache_tokenizer"),
)
def test_source_lock_reopen_rejects_canonical_cache_parentage_forgery(
    tmp_path: Path,
    forgery: str,
) -> None:
    from vfe4.artifacts.provenance import (
        production_token_cache_set_sha256,
    )
    from vfe4.training import production
    from vfe4.types.training import ProductionTokenCacheIdentity

    (
        training,
        paths,
        repository_root,
        _,
        _,
        dependencies,
    ) = _injected_source_lock_case(tmp_path)
    source_lock = production.run_source_lock(
        training=training,
        paths=paths,
        dependencies=dependencies,
    )
    bundle = asdict(source_lock)

    if forgery == "raw_parent":
        cache = bundle["token_caches"][0]
        cache["raw_parent_sha256"] = hashlib.sha256(
            b"unrelated raw parent"
        ).hexdigest()
        cache["record_sha256"] = production.owned_sha256(
            "vfe4.wt103.production-token-cache-record.v1",
            {
                name: value
                for name, value in cache.items()
                if name != "record_sha256"
            },
        )
    else:
        original = source_lock.tokenizer
        alternate = type(original).create_verified(
            distribution_record_sha256=hashlib.sha256(
                b"alternate tiktoken distribution"
            ).hexdigest(),
            regex_pattern_sha256=original.regex_pattern_sha256,
            regex_engine_distribution_name=(
                original.regex_engine_distribution_name
            ),
            regex_engine_distribution_version=(
                original.regex_engine_distribution_version
            ),
            regex_engine_distribution_record_sha256=(
                original.regex_engine_distribution_record_sha256
            ),
            mergeable_ranks_sha256=original.mergeable_ranks_sha256,
            special_tokens_sha256=original.special_tokens_sha256,
            golden_vectors_sha256=original.golden_vectors_sha256,
            tokenizer_tables_sha256=original.tokenizer_tables_sha256,
        )
        identities = []
        for cache in bundle["token_caches"]:
            identity = ProductionTokenCacheIdentity.create(
                tokenizer=alternate,
                split=cache["split"],
                payload_sha256=cache["payload_sha256"],
            )
            identities.append(identity)
            cache["tokenizer"] = asdict(alternate)
            cache["cache_identity"] = asdict(identity)
            cache["record_sha256"] = production.owned_sha256(
                "vfe4.wt103.production-token-cache-record.v1",
                {
                    name: value
                    for name, value in cache.items()
                    if name != "record_sha256"
                },
            )
        finalized = bundle["finalized_source"]
        finalized["production_token_cache_set_sha256"] = (
            production_token_cache_set_sha256(tuple(identities))
        )
        finalized["record_sha256"] = production.owned_sha256(
            "vfe4.wt103.finalized-source-record.v1",
            {
                name: value
                for name, value in finalized.items()
                if name != "record_sha256"
            },
        )

    bundle["source_lock_sha256"] = production.owned_sha256(
        "vfe4.wt103.production-source-lock.v1",
        {
            name: value
            for name, value in bundle.items()
            if name != "source_lock_sha256"
        },
    )
    bundle_path = (
        paths.cache_root
        / "production-source-lock"
        / "finalized-source-bundle.json"
    )
    bundle_path.write_bytes(
        production.canonical_json_bytes_generic(bundle) + b"\n"
    )
    paths.source_record_path.write_bytes(
        production.canonical_json_bytes_generic(bundle["finalized_source"])
        + b"\n"
    )

    with pytest.raises(
        production.ProductionOperationError,
        match="source-lock bundle cross-links disagree",
    ):
        production._reopen_source_lock(
            training=training,
            paths=paths,
            repository_root=repository_root,
        )


def test_dependency_lock_writer_rejects_a_stale_source_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vfe4.artifacts.environment import (
        LockInputManifest,
        parse_lock_input_manifest,
    )
    from vfe4.training import production

    payload = (
        ROOT / "requirements-wt103.lock-input.json"
    ).read_bytes()
    manifest = parse_lock_input_manifest(payload)
    assert (
        production._validate_lock_writer_source(manifest)
        == manifest.writer_code_sha256
    )
    stale = LockInputManifest.create(
        writer_code_sha256="0" * 64,
        target_python_version=manifest.target_python_version,
        requirements=manifest.requirements,
    )
    with pytest.raises(
        production.ProductionOperationError,
        match="lock-writer source identity changed",
    ):
        production._validate_lock_writer_source(stale)

    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    (
        repository_root / "requirements-wt103.lock-input.json"
    ).write_bytes(payload)
    events: list[str] = []

    def reject_before_resolution(_manifest: object) -> None:
        events.append("source_check")
        raise production.ProductionOperationError("source check sentinel")

    monkeypatch.setattr(
        production,
        "_validate_lock_writer_source",
        reject_before_resolution,
    )
    with pytest.raises(
        production.ProductionOperationError,
        match="source check sentinel",
    ):
        production._resolve_dependency_lock(
            SimpleNamespace(repository_root=repository_root)
        )
    assert events == ["source_check"]


def test_production_plan_is_canonical_and_tuning_selection_rejects_duplicates(
    tmp_path: Path,
) -> None:
    from vfe4.artifacts.durability import canonical_json_bytes_generic
    from vfe4.training import production_attempt

    launcher = _load("train_vfe4_wt103_attempt_plan")
    config = copy.deepcopy(launcher.CONFIG)
    config["paths"]["cache_root"] = str(tmp_path / "cache")
    config["paths"]["run_root"] = str(tmp_path / "runs")
    config["paths"]["source_record_path"] = str(
        tmp_path / "source-record.json"
    )
    config["paths"]["resume_experiment_index_path"] = str(
        tmp_path / "runs" / "production-experiment-index.json"
    )
    training, _, _ = launcher._resolve_launcher(config)
    source_lock = SimpleNamespace(source_lock_sha256="1" * 64)
    readiness = SimpleNamespace(
        result_sha256="2" * 64,
        readiness_token=SimpleNamespace(token_sha256="3" * 64),
    )

    plan = production_attempt._experiment_plan_document(
        training=training,
        source_lock=source_lock,
        readiness=readiness,
    )
    reopened = json.loads(
        canonical_json_bytes_generic(plan).decode("utf-8")
    )

    assert reopened == plan
    assert len(plan["tuning_attempts"]) == 60
    assert len(
        {
            row["attempt_sha256"]
            for row in plan["tuning_attempts"]
        }
    ) == 60
    assert production_attempt._validate_experiment_plan(
        reopened,
        training=training,
        source_lock=source_lock,
        readiness=readiness,
    ) == plan["experiment_plan_sha256"]

    rows: list[dict[str, object]] = []
    for attempt in production_attempt._attempt_inventory(training, None):
        outcome = production_attempt.ProductionAttemptOutcome.create(
            attempt_sha256=attempt.attempt_sha256,
            validation_nll_sum=100.0,
            validation_counted_targets=100,
            validation_nll_per_token=1.0,
            accepted_updates=1,
            terminal_checkpoint_identity_sha256=None,
            metrics_jsonl_sha256="4" * 64,
            metrics_csv_sha256="5" * 64,
        )
        rows.append(
            production_attempt._outcome_document(attempt, outcome)
        )
    selected = production_attempt._select_hyperparameters(
        {"completed_outcomes": rows},
        training,
    )
    assert set(selected.values()) == {(1.0e-4, 0.0)}

    with pytest.raises(
        production_attempt.ProductionOperationError,
        match="duplicate",
    ):
        production_attempt._select_hyperparameters(
            {"completed_outcomes": [*rows, copy.deepcopy(rows[0])]},
            training,
        )


def test_launcher_rejects_overlapping_or_v3_roots_before_dispatch(
    tmp_path: Path,
) -> None:
    launcher = _load("train_vfe4_wt103_path_safety")
    base = copy.deepcopy(launcher.CONFIG)
    base["training"]["operation"] = "synthetic_smoke"
    base["paths"]["source_record_path"] = str(tmp_path / "source.json")
    base["paths"]["resume_experiment_index_path"] = str(
        tmp_path / "runs" / "index.json"
    )

    overlapping = copy.deepcopy(base)
    overlapping["paths"]["cache_root"] = str(tmp_path / "owned")
    overlapping["paths"]["run_root"] = str(tmp_path / "owned" / "runs")
    with pytest.raises(
        launcher.TrainingLaunchError,
        match="cache_root and run_root must be disjoint",
    ):
        launcher.main(overlapping)

    v3 = copy.deepcopy(base)
    v3["paths"]["cache_root"] = str(tmp_path / "V3-Transformer" / "cache")
    v3["paths"]["run_root"] = str(tmp_path / "runs")
    with pytest.raises(
        launcher.TrainingLaunchError,
        match="V3",
    ):
        launcher.main(v3)


def test_launcher_rejects_exact_legacy_tokenized_cache_root_before_dispatch(
    tmp_path: Path,
) -> None:
    launcher = _load("train_vfe4_wt103_legacy_cache_safety")
    base = copy.deepcopy(launcher.CONFIG)
    base["training"]["operation"] = "source_lock"
    base["authorization"] = launcher.SOURCE_LOCK_AUTHORIZATION
    base["paths"]["cache_root"] = str(tmp_path / "cache")
    base["paths"]["run_root"] = str(tmp_path / "runs")
    base["paths"]["source_record_path"] = str(tmp_path / "source.json")
    base["paths"]["resume_experiment_index_path"] = str(
        tmp_path / "runs" / "index.json"
    )
    legacy_root = Path.home() / ".cache" / "tokenized_cache"
    mixed_case_descendant = Path(str(legacy_root).upper()) / "nested" / "data"
    device_namespace_paths = (
        "\\\\.\\" + str(legacy_root),
        "\\\\.\\" + str(legacy_root / "nested" / "data"),
        "\\\\?\\" + str(legacy_root),
        "\\\\?\\" + str(legacy_root / "nested" / "data"),
        "//./" + str(legacy_root).replace("\\", "/"),
        "//./" + str(legacy_root / "nested" / "data").replace("\\", "/"),
        "//?/" + str(legacy_root).replace("\\", "/"),
        "//?/" + str(legacy_root / "nested" / "data").replace("\\", "/"),
    )

    class _DispatchForbiddenDriver(_HermeticProductionDriver):
        def source_lock(
            self,
            *,
            training: object,
            paths: object,
        ) -> object:
            del training, paths
            raise AssertionError("launcher reached the injected driver")

    for field in (
        "cache_root",
        "run_root",
        "source_record_path",
        "resume_experiment_index_path",
    ):
        forbidden_cases = (
            (legacy_root, "legacy V3 token cache"),
            (mixed_case_descendant, "legacy V3 token cache"),
            *(
                (device_path, "Windows device namespace")
                for device_path in device_namespace_paths
            ),
        )
        for forbidden_path, expected_message in forbidden_cases:
            case = copy.deepcopy(base)
            case["paths"][field] = str(forbidden_path)
            driver = _DispatchForbiddenDriver(
                events=[],
                source=object(),
                readiness_result=object(),
                operation_result=object(),
            )

            with pytest.raises(
                launcher.TrainingLaunchError,
                match=expected_message,
            ):
                launcher.main(case, driver=driver)


def test_launcher_rejects_forbidden_final_path_before_filesystem_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _load("train_vfe4_wt103_lexical_admission_order")
    config = copy.deepcopy(launcher.CONFIG)
    config["training"]["operation"] = "source_lock"
    config["authorization"] = launcher.SOURCE_LOCK_AUTHORIZATION
    config["paths"]["cache_root"] = str(tmp_path / "cache")
    config["paths"]["run_root"] = str(tmp_path / "runs")
    config["paths"]["source_record_path"] = str(tmp_path / "source.json")
    config["paths"]["resume_experiment_index_path"] = str(
        Path.home() / ".cache" / "tokenized_cache" / "final-field"
    )

    def _fail_lstat(path: Path) -> object:
        raise AssertionError(f"filesystem metadata reached for {path}")

    monkeypatch.setattr(Path, "lstat", _fail_lstat)

    with pytest.raises(
        launcher.TrainingLaunchError,
        match="legacy V3 token cache",
    ):
        launcher.main(config)
