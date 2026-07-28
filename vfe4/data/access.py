"""Closure-owned H6 split access and durable held-out-test opening."""

from __future__ import annotations

import errno
import hashlib
import os
import stat
import threading
import unicodedata
import uuid
import weakref
from dataclasses import dataclass
from pathlib import Path

from vfe4.types.h6 import (
    DataIdentity,
    DurableTestOpeningCapability,
    ExperimentIdentity,
    FrozenBatchSchedule,
    H6PredictionReadinessToken,
    ValidatedTestOpening,
    ValidationSafetyFixture,
    VocabularyIdentity,
)

from .byte_tokenizer import ByteTokenizerV1
from .windows import CausalWindows, build_causal_windows, frozen_batch_schedule
from .wikitext2 import AuthenticatedReopenedDataIdentity, BlindedCorpusStore


class OpeningCapabilityError(RuntimeError):
    """The blinded-data readiness or one-shot opening contract failed."""


H6_TEST_RESERVATION_DURABILITY = "process-crash-durable-no-power-loss-claim"
H6_V3_RESERVATION_AUTHORITY_CHECKS = (
    "authenticated_store",
    "destination_absent",
    "readiness_plan_cross_binding",
    "experiment_identity",
    "vocabulary_equality",
    "complete_reservation_bytes",
)


def _exclusive_write_complete_file(path: Path, content: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _install_staged_file_no_replace(source: Path, destination: Path) -> None:
    """Atomically install one complete sibling file without replacement."""

    if os.name == "nt":
        # Windows rename is no-replace when the destination already exists.
        os.rename(source, destination)
        return
    # Same-directory hard-link installation is atomic and fails with EEXIST.
    os.link(source, destination)
    source.unlink()


def _fsync_parent_directory_best_effort(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _install_complete_reservation_file_no_replace(
    destination: Path,
    canonical_bytes: bytes,
) -> None:
    """Stage complete bytes, then atomically expose the authoritative name."""

    if type(destination) is not Path or type(canonical_bytes) is not bytes:
        raise OpeningCapabilityError(
            "reservation destination and canonical bytes are required"
        )
    temporary = destination.with_name(f".reservation-stage-{uuid.uuid4().hex}")
    installed = False
    try:
        _exclusive_write_complete_file(temporary, canonical_bytes)
        _install_staged_file_no_replace(temporary, destination)
        installed = True
        _fsync_parent_directory_best_effort(destination.parent)
    except OSError as exc:
        raise OpeningCapabilityError(
            f"reservation atomic install failed: {exc}"
        ) from exc
    finally:
        if not installed:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _revalidate_blinded_data_identity_for_readiness(
    directory: Path,
    *,
    expected_archive_sha256: str,
    expected_data_identity_sha256: str,
    expected_access_policy_sha256: str,
) -> DataIdentity:
    """Privately reconstruct the typed identity for the readiness boundary."""

    from .wikitext2 import _rehydrate_blinded_data_identity

    return _rehydrate_blinded_data_identity(
        directory,
        expected_archive_sha256=expected_archive_sha256,
        expected_data_identity_sha256=expected_data_identity_sha256,
        expected_access_policy_sha256=expected_access_policy_sha256,
    )


@dataclass(frozen=True)
class MaterializedPredictionData:
    """Readiness-gated train and ordinary-validation causal windows."""

    data_identity_sha256: str
    train: CausalWindows
    validation: CausalWindows

    def __post_init__(self) -> None:
        if (
            type(self.data_identity_sha256) is not str
            or len(self.data_identity_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.data_identity_sha256
            )
        ):
            raise ValueError("data_identity_sha256 must be lowercase SHA-256 hex")
        if (
            type(self.train) is not CausalWindows
            or self.train.split != "train"
            or type(self.validation) is not CausalWindows
            or self.validation.split != "validation"
        ):
            raise ValueError(
                "materialized data must contain train and validation windows"
            )
        self.train.__post_init__()
        self.validation.__post_init__()

    def schedule_for_pass(self, zero_based_pass_index: int) -> FrozenBatchSchedule:
        self.__post_init__()
        return frozen_batch_schedule(
            window_count=len(self.train),
            zero_based_pass_index=zero_based_pass_index,
        )


@dataclass(frozen=True)
class H6TrainingDataV3:
    """One authenticated v3 train-only materialization."""

    data_identity_sha256: str
    readiness_sha256: str
    plan_sha256: str
    matching_set_sha256: str
    runtime_identity_sha256: str
    windows: CausalWindows
    vocabulary: VocabularyIdentity

    def __post_init__(self) -> None:
        for name in (
            "data_identity_sha256",
            "readiness_sha256",
            "plan_sha256",
            "matching_set_sha256",
            "runtime_identity_sha256",
        ):
            value = getattr(self, name)
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be lowercase SHA-256 hex")
        if type(self.windows) is not CausalWindows or self.windows.split != "train":
            raise ValueError("v3 training data must contain only train windows")
        self.windows.__post_init__()
        if type(self.vocabulary) is not VocabularyIdentity:
            raise ValueError("v3 training data requires an exact vocabulary identity")
        self.vocabulary.__post_init__()


def _build_access_api():
    """Build the API while retaining every path and authority only in closures."""

    proof_domain = b"VFE4-H6-DURABLE-TEST-OPENING-PROOF-V1\x00"
    proof_suffix = b"RESERVED\x00"
    marker_domain = b"VFE4-H6-TEST-OPENING-MARKER-V1\x00"
    issuer_authority = object()
    validated_authority = object()
    validation_v3_authority = object()
    train_v3_authority = object()
    registry_lock = threading.RLock()
    registry: dict[int, tuple[weakref.ReferenceType[BlindedCorpusStore], object]] = {}
    validation_v3_registry: weakref.WeakKeyDictionary[object, object] = (
        weakref.WeakKeyDictionary()
    )
    train_v3_registry: weakref.WeakKeyDictionary[object, object] = (
        weakref.WeakKeyDictionary()
    )

    source_file = Path(__file__).resolve(strict=True)
    production_anchor = source_file.parents[2]
    production_anchor_stat = os.lstat(production_anchor)
    production_anchor_identity = (
        production_anchor_stat.st_dev,
        production_anchor_stat.st_ino,
    )
    production_marker_root = (
        production_anchor / ".vfe4" / "h6-test-opening-reservations"
    )

    @dataclass
    class StoreAccessState:
        sealed_directory: Path
        directory_identity: tuple[int, int]
        split_paths: tuple[Path, Path, Path]
        split_file_identities: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
        marker_mode: str
        synthetic_anchor: Path | None
        synthetic_anchor_identity: tuple[int, int] | None
        opening: object | None = None

    @dataclass
    class RegisteredOpeningProof:
        reservation_path: Path
        reservation_root_identity: tuple[int, int]
        reservation_file_identity: tuple[int, int]
        canonical_bytes: bytes
        marker_bytes_sha256: str
        proof_identity_sha256: str
        readiness_sha256: str
        experiment_identity_sha256: str
        data_identity_sha256: str
        sealed_test_sha256: str
        access_policy_sha256: str
        capability: object
        is_v3_reservation: bool = False
        consumed: bool = False

    class DurableOpening:
        __slots__ = ("_proof_identity_sha256", "__weakref__")

        def __init__(self, proof_identity_sha256: str, authority: object) -> None:
            if authority is not issuer_authority:
                raise TypeError("durable test-opening capabilities are issuer-only")
            self._proof_identity_sha256 = proof_identity_sha256

        @property
        def proof_identity_sha256(self) -> str:
            return self._proof_identity_sha256

        def __copy__(self):
            raise TypeError("durable test-opening capabilities cannot be copied")

        def __deepcopy__(self, memo):
            del memo
            raise TypeError("durable test-opening capabilities cannot be copied")

        def __reduce__(self):
            raise TypeError("durable test-opening capabilities cannot be serialized")

        def __reduce_ex__(self, protocol):
            del protocol
            raise TypeError("durable test-opening capabilities cannot be serialized")

        def __repr__(self) -> str:
            return "<opaque H6 durable test-opening capability>"

    class ValidatedOpening:
        __slots__ = ("_proof_identity_sha256",)

        def __init__(self, proof_identity_sha256: str, authority: object) -> None:
            if authority is not validated_authority:
                raise TypeError("validated test openings are validator-only")
            self._proof_identity_sha256 = proof_identity_sha256

        @property
        def proof_identity_sha256(self) -> str:
            return self._proof_identity_sha256

        def __reduce__(self):
            raise TypeError("validated test openings cannot be serialized")

        def __reduce_ex__(self, protocol):
            del protocol
            raise TypeError("validated test openings cannot be serialized")

        def __repr__(self) -> str:
            return "<opaque H6 validated test opening>"

    class ValidationCapabilityV3:
        __slots__ = ("__weakref__",)

        def __init__(self, authority: object) -> None:
            if authority is not validation_v3_authority:
                raise TypeError("v3 validation capabilities are access-issued only")

        def __copy__(self):
            raise TypeError("v3 validation capabilities cannot be copied")

        def __deepcopy__(self, memo):
            del memo
            raise TypeError("v3 validation capabilities cannot be copied")

        def __reduce__(self):
            raise TypeError("v3 validation capabilities cannot be serialized")

        def __reduce_ex__(self, protocol):
            del protocol
            raise TypeError("v3 validation capabilities cannot be serialized")

        def __repr__(self) -> str:
            return "<opaque H6 v3 validation capability>"

    class TrainCapabilityV3:
        __slots__ = ("__weakref__",)

        def __init__(self, authority: object) -> None:
            if authority is not train_v3_authority:
                raise TypeError("v3 train capabilities are access-issued only")

        def __copy__(self):
            raise TypeError("v3 train capabilities cannot be copied")

        def __deepcopy__(self, memo):
            del memo
            raise TypeError("v3 train capabilities cannot be copied")

        def __reduce__(self):
            raise TypeError("v3 train capabilities cannot be serialized")

        def __reduce_ex__(self, protocol):
            del protocol
            raise TypeError("v3 train capabilities cannot be serialized")

        def __repr__(self) -> str:
            return "<opaque H6 v3 train capability>"

    @dataclass(frozen=True)
    class AuthorizedValidationV3:
        windows: CausalWindows
        vocabulary: object
        readiness_sha256: str
        experiment_config_sha256: str
        plan_sha256: str
        matching_set_sha256: str
        data_identity_sha256: str
        runtime_identity_sha256: str

    @dataclass(frozen=True)
    class AuthorizedTrainV3:
        data: H6TrainingDataV3

    def _is_redirect(path: Path, path_stat: os.stat_result) -> bool:
        if stat.S_ISLNK(path_stat.st_mode):
            return True
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        if reparse_flag and (
            getattr(path_stat, "st_file_attributes", 0) & reparse_flag
        ):
            return True
        is_junction = getattr(path, "is_junction", None)
        return bool(callable(is_junction) and is_junction())

    def _require_directory(
        path: Path,
        *,
        expected_identity: tuple[int, int] | None = None,
        require_canonical_spelling: bool = True,
    ) -> tuple[int, int]:
        try:
            path_stat = os.lstat(path)
        except OSError as exc:
            raise OpeningCapabilityError(
                f"managed directory is unavailable: {exc}"
            ) from exc
        if not stat.S_ISDIR(path_stat.st_mode) or _is_redirect(path, path_stat):
            raise OpeningCapabilityError("managed directory cannot be a redirect")
        if require_canonical_spelling:
            resolved = path.resolve(strict=True)
            if os.path.normcase(os.fspath(resolved)) != os.path.normcase(
                os.fspath(path)
            ):
                raise OpeningCapabilityError("managed directory path is redirected")
        identity = (path_stat.st_dev, path_stat.st_ino)
        if expected_identity is not None and identity != expected_identity:
            raise OpeningCapabilityError("managed directory identity changed")
        return identity

    def _require_regular_file(path: Path) -> tuple[int, int]:
        try:
            path_stat = os.lstat(path)
        except OSError as exc:
            raise OpeningCapabilityError(f"sealed split is unavailable: {exc}") from exc
        if not stat.S_ISREG(path_stat.st_mode) or _is_redirect(path, path_stat):
            raise OpeningCapabilityError(
                "sealed split must be a non-redirected regular file"
            )
        return path_stat.st_dev, path_stat.st_ino

    def _register(
        store: BlindedCorpusStore,
        sealed_directory: Path,
        *,
        marker_mode: str,
        refuse_existing_reservation: bool = False,
    ) -> None:
        if type(store) is not BlindedCorpusStore:
            raise OpeningCapabilityError("registration requires an exact blinded store")
        try:
            store.__post_init__()
        except ValueError as exc:
            raise OpeningCapabilityError("blinded store failed validation") from exc
        if (
            not isinstance(sealed_directory, Path)
            or not sealed_directory.is_absolute()
            or marker_mode not in ("production", "synthetic")
            or type(refuse_existing_reservation) is not bool
        ):
            raise OpeningCapabilityError("registration path or mode is invalid")
        directory = sealed_directory.resolve(strict=True)
        directory_identity = _require_directory(directory)
        split_paths = (
            directory / "sealed" / "wiki.train.raw",
            directory / "sealed" / "wiki.valid.raw",
            directory / "sealed" / "wiki.test.raw",
        )
        split_identities = tuple(_require_regular_file(path) for path in split_paths)
        data_identity = store.data_identity
        expected_raw_hashes = (
            data_identity.train_raw_sha256,
            data_identity.validation_raw_sha256,
            data_identity.test_raw_sha256,
        )
        handle_hashes = (
            store.sealed_train_handle.sealed_content_sha256,
            store.sealed_validation_handle.sealed_content_sha256,
            store.sealed_test_handle.sealed_content_sha256,
        )
        if handle_hashes != expected_raw_hashes:
            raise OpeningCapabilityError(
                "sealed handles do not match the data identity"
            )

        synthetic_anchor: Path | None = None
        synthetic_anchor_identity: tuple[int, int] | None = None
        if marker_mode == "synthetic":
            synthetic_anchor = directory.parent.parent
            synthetic_anchor_identity = _require_directory(synthetic_anchor)

        state = StoreAccessState(
            directory,
            directory_identity,
            split_paths,
            split_identities,  # type: ignore[arg-type]
            marker_mode,
            synthetic_anchor,
            synthetic_anchor_identity,
        )
        if refuse_existing_reservation:
            _refuse_existing_reservation(store, state)
        key = id(store)

        def _forget(reference: weakref.ReferenceType[BlindedCorpusStore]) -> None:
            with registry_lock:
                current = registry.get(key)
                if current is not None and current[0] is reference:
                    registry.pop(key, None)

        reference = weakref.ref(store, _forget)
        with registry_lock:
            current = registry.get(key)
            if current is not None and current[0]() is store:
                raise OpeningCapabilityError("blinded store is already registered")
            registry[key] = (reference, state)

    def _register_production(store: BlindedCorpusStore, sealed_directory: Path) -> None:
        _register(store, sealed_directory, marker_mode="production")

    def _register_synthetic(store: BlindedCorpusStore, sealed_directory: Path) -> None:
        _register(store, sealed_directory, marker_mode="synthetic")

    def _register_reopened(
        store: BlindedCorpusStore,
        sealed_directory: Path,
    ) -> None:
        """Reopened stores always inherit the production reservation authority."""

        _register(
            store,
            sealed_directory,
            marker_mode="production",
            refuse_existing_reservation=True,
        )

    def _state_for(store: BlindedCorpusStore) -> StoreAccessState:
        if type(store) is not BlindedCorpusStore:
            raise OpeningCapabilityError("access requires an exact blinded store")
        try:
            store.__post_init__()
        except ValueError as exc:
            raise OpeningCapabilityError("blinded store failed validation") from exc
        with registry_lock:
            registered = registry.get(id(store))
            if registered is None or registered[0]() is not store:
                raise OpeningCapabilityError("blinded store is not registered")
            state = registered[1]
        if type(state) is not StoreAccessState:
            raise OpeningCapabilityError("blinded store registry is invalid")
        _require_directory(
            state.sealed_directory,
            expected_identity=state.directory_identity,
        )
        return state

    def _require_readiness(
        store: BlindedCorpusStore, readiness: H6PredictionReadinessToken
    ) -> None:
        if type(readiness) is not H6PredictionReadinessToken:
            raise OpeningCapabilityError("exact H6 Prediction readiness is required")
        try:
            readiness.__post_init__()
        except ValueError as exc:
            raise OpeningCapabilityError(
                "Prediction readiness failed validation"
            ) from exc
        data_identity = store.data_identity
        retained_readiness_identity = object.__getattribute__(
            readiness,
            "_data_identity",
        )
        exact_identity_matches = (
            type(data_identity) is DataIdentity
            and retained_readiness_identity == data_identity
        )
        restricted_identity_matches = (
            type(data_identity) is not DataIdentity
            and retained_readiness_identity.data_identity_sha256
            == data_identity.data_identity_sha256
            and retained_readiness_identity.access_policy_sha256
            == data_identity.access_policy_sha256
        )
        if (
            readiness.status != "PASS"
            or readiness.data_identity_sha256 != store.data_identity_sha256
            or readiness.access_policy_sha256 != data_identity.access_policy_sha256
            or not (exact_identity_matches or restricted_identity_matches)
        ):
            raise OpeningCapabilityError(
                "Prediction readiness does not match the store data identity"
            )

    def _read_exact_file(
        path: Path,
        *,
        expected_identity: tuple[int, int],
        expected_length: int,
        expected_sha256: str,
    ) -> bytes:
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise OpeningCapabilityError(
                f"sealed split cannot be opened: {exc}"
            ) from exc
        try:
            opened_stat = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened_stat.st_mode)
                or (opened_stat.st_dev, opened_stat.st_ino) != expected_identity
                or opened_stat.st_size != expected_length
            ):
                raise OpeningCapabilityError("sealed split identity or size changed")
            chunks: list[bytes] = []
            total = 0
            while total <= expected_length:
                chunk = os.read(descriptor, min(65536, expected_length + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
            content = b"".join(chunks)
        finally:
            os.close(descriptor)
        if (
            len(content) != expected_length
            or hashlib.sha256(content).hexdigest() != expected_sha256
        ):
            raise OpeningCapabilityError("sealed split content changed")
        return content

    def _read_split(
        store: BlindedCorpusStore, state: StoreAccessState, split: str
    ) -> CausalWindows:
        split_index = {"train": 0, "validation": 1, "test": 2}.get(split)
        if split_index is None:
            raise OpeningCapabilityError("unsupported sealed split")
        data_identity = store.data_identity
        token_identity = (
            data_identity.train_tokens,
            data_identity.validation_tokens,
            data_identity.test_tokens,
        )[split_index]
        expected_raw_sha256 = (
            data_identity.train_raw_sha256,
            data_identity.validation_raw_sha256,
            data_identity.test_raw_sha256,
        )[split_index]
        raw = _read_exact_file(
            state.split_paths[split_index],
            expected_identity=state.split_file_identities[split_index],
            expected_length=token_identity.token_count - 2,
            expected_sha256=expected_raw_sha256,
        )
        tokenizer = ByteTokenizerV1()
        tokens = tokenizer.encode(raw)
        observed_token_identity = tokenizer.storage_identity(tokens)
        if (
            observed_token_identity.storage_schema != token_identity.storage_schema
            or observed_token_identity.token_count != token_identity.token_count
            or observed_token_identity.byte_length != token_identity.byte_length
            or observed_token_identity.encoded_token_sha256
            != token_identity.encoded_token_sha256
        ):
            raise OpeningCapabilityError("sealed split token identity changed")
        return build_causal_windows(tokens, split=split)  # type: ignore[arg-type]

    def _validation_fixture(store: BlindedCorpusStore) -> ValidationSafetyFixture:
        _state_for(store)
        fixture = store.frozen_validation_fixture
        fixture.__post_init__()
        return fixture

    def _materialize_train(
        store: BlindedCorpusStore, readiness: H6PredictionReadinessToken
    ) -> MaterializedPredictionData:
        state = _state_for(store)
        _require_readiness(store, readiness)
        train = _read_split(store, state, "train")
        validation = _read_split(store, state, "validation")
        return MaterializedPredictionData(
            store.data_identity_sha256,
            train,
            validation,
        )

    def _issue_validation_v3(
        store: BlindedCorpusStore,
        readiness: object,
        plan: object,
    ) -> object:
        from vfe4.training.h6_experiment_v3 import H6ExperimentPlanV3
        from vfe4.types.h6_prediction_v3 import H6PredictionV3ReadinessToken

        state = _state_for(store)
        if (
            state.marker_mode != "production"
            or type(store.data_identity) is not AuthenticatedReopenedDataIdentity
        ):
            raise OpeningCapabilityError(
                "v3 validation requires authenticated reopened-store provenance"
            )
        if type(readiness) is not H6PredictionV3ReadinessToken:
            raise OpeningCapabilityError("exact H6 Prediction v3 readiness is required")
        if type(plan) is not H6ExperimentPlanV3:
            raise OpeningCapabilityError("exact H6 experiment plan v3 is required")
        try:
            readiness.__post_init__()
            plan.__post_init__()
        except ValueError as exc:
            raise OpeningCapabilityError(
                "v3 validation authority failed validation"
            ) from exc
        data_identity = store.data_identity
        if (
            readiness.status != "PASS"
            or readiness.data_identity_sha256 != store.data_identity_sha256
            or readiness.access_policy_sha256 != data_identity.access_policy_sha256
            or plan.readiness_sha256 != readiness.readiness_sha256
            or plan.experiment_config_sha256 != readiness.experiment_config_sha256
            or plan.matching_set_sha256 != readiness.matching_set_sha256
            or plan.matching_policy_sha256 != readiness.matching_policy_sha256
            or plan.git_head != readiness.git_head
            or plan.dirty_digest != readiness.dirty_digest
            or plan.training_schedule.schedule_sha256
            != readiness.training_schedule_sha256
            or plan.training_schedule.runtime_identity_sha256
            != readiness.runtime_identity_sha256
            or any(
                attempt.attempt_spec.data_identity_sha256 != store.data_identity_sha256
                for attempt in plan.attempts
            )
        ):
            raise OpeningCapabilityError(
                "v3 validation store/readiness/plan authority drift"
            )
        windows = _read_split(store, state, "validation")
        vocabulary = ByteTokenizerV1().vocabulary_identity
        if any(config.vocabulary != vocabulary for config in plan.endpoint_configs):
            raise OpeningCapabilityError(
                "v3 plan vocabulary does not match the authenticated store"
            )
        capability = ValidationCapabilityV3(validation_v3_authority)
        with registry_lock:
            validation_v3_registry[capability] = AuthorizedValidationV3(
                windows=windows,
                vocabulary=vocabulary,
                readiness_sha256=readiness.readiness_sha256,
                experiment_config_sha256=readiness.experiment_config_sha256,
                plan_sha256=plan.plan_sha256,
                matching_set_sha256=readiness.matching_set_sha256,
                data_identity_sha256=store.data_identity_sha256,
                runtime_identity_sha256=readiness.runtime_identity_sha256,
            )
        return capability

    def _issue_train_v3(
        store: BlindedCorpusStore,
        readiness: object,
        plan: object,
    ) -> object:
        from vfe4.training.h6_experiment_v3 import H6ExperimentPlanV3
        from vfe4.types.h6_prediction_v3 import H6PredictionV3ReadinessToken

        state = _state_for(store)
        if (
            state.marker_mode != "production"
            or type(store.data_identity) is not AuthenticatedReopenedDataIdentity
        ):
            raise OpeningCapabilityError(
                "v3 training requires authenticated reopened-store provenance"
            )
        if type(readiness) is not H6PredictionV3ReadinessToken:
            raise OpeningCapabilityError("exact H6 Prediction v3 readiness is required")
        if type(plan) is not H6ExperimentPlanV3:
            raise OpeningCapabilityError("exact H6 experiment plan v3 is required")
        try:
            readiness.__post_init__()
            plan.__post_init__()
        except ValueError as exc:
            raise OpeningCapabilityError(
                "v3 training authority failed validation"
            ) from exc
        data_identity = store.data_identity
        if (
            readiness.status != "PASS"
            or readiness.data_identity_sha256 != store.data_identity_sha256
            or readiness.access_policy_sha256 != data_identity.access_policy_sha256
            or plan.readiness_sha256 != readiness.readiness_sha256
            or plan.experiment_config_sha256 != readiness.experiment_config_sha256
            or plan.matching_set_sha256 != readiness.matching_set_sha256
            or plan.matching_policy_sha256 != readiness.matching_policy_sha256
            or plan.git_head != readiness.git_head
            or plan.dirty_digest != readiness.dirty_digest
            or plan.training_schedule.schedule_sha256
            != readiness.training_schedule_sha256
            or plan.training_schedule.runtime_identity_sha256
            != readiness.runtime_identity_sha256
            or any(
                attempt.attempt_spec.data_identity_sha256
                != store.data_identity_sha256
                for attempt in plan.attempts
            )
        ):
            raise OpeningCapabilityError(
                "v3 train store/readiness/plan authority drift"
            )
        vocabulary = ByteTokenizerV1().vocabulary_identity
        if any(config.vocabulary != vocabulary for config in plan.endpoint_configs):
            raise OpeningCapabilityError(
                "v3 plan vocabulary does not match the authenticated store"
            )
        data = H6TrainingDataV3(
            data_identity_sha256=store.data_identity_sha256,
            readiness_sha256=readiness.readiness_sha256,
            plan_sha256=plan.plan_sha256,
            matching_set_sha256=readiness.matching_set_sha256,
            runtime_identity_sha256=readiness.runtime_identity_sha256,
            windows=_read_split(store, state, "train"),
            vocabulary=vocabulary,
        )
        capability = TrainCapabilityV3(train_v3_authority)
        with registry_lock:
            train_v3_registry[capability] = AuthorizedTrainV3(data=data)
        return capability

    def _consume_validation_v3(
        capability: object,
        *,
        plan: object,
    ) -> object:
        from vfe4.training.h6_experiment_v3 import H6ExperimentPlanV3

        if type(capability) is not ValidationCapabilityV3:
            raise OpeningCapabilityError(
                "exact access-issued v3 validation capability is required"
            )
        if type(plan) is not H6ExperimentPlanV3:
            raise OpeningCapabilityError("exact H6 experiment plan v3 is required")
        try:
            plan.__post_init__()
        except ValueError as exc:
            raise OpeningCapabilityError("v3 validation plan is stale") from exc
        with registry_lock:
            authorized = validation_v3_registry.get(capability)
        if type(authorized) is not AuthorizedValidationV3:
            raise OpeningCapabilityError(
                "v3 validation capability is forged or expired"
            )
        if (
            authorized.plan_sha256 != plan.plan_sha256
            or authorized.experiment_config_sha256 != plan.experiment_config_sha256
            or authorized.readiness_sha256 != plan.readiness_sha256
            or authorized.matching_set_sha256 != plan.matching_set_sha256
            or authorized.runtime_identity_sha256
            != plan.training_schedule.runtime_identity_sha256
        ):
            raise OpeningCapabilityError("v3 validation capability authority drift")
        return authorized

    def _consume_train_v3(
        capability: object,
        *,
        plan: object,
    ) -> H6TrainingDataV3:
        from vfe4.training.h6_experiment_v3 import H6ExperimentPlanV3

        if type(capability) is not TrainCapabilityV3:
            raise OpeningCapabilityError(
                "exact access-issued v3 train capability is required"
            )
        if type(plan) is not H6ExperimentPlanV3:
            raise OpeningCapabilityError("exact H6 experiment plan v3 is required")
        try:
            plan.__post_init__()
        except ValueError as exc:
            raise OpeningCapabilityError("v3 training plan is stale") from exc
        with registry_lock:
            authorized = train_v3_registry.get(capability)
            if type(authorized) is not AuthorizedTrainV3:
                raise OpeningCapabilityError(
                    "v3 train capability is forged, expired, or consumed"
                )
            data = authorized.data
            data.__post_init__()
            if (
                data.plan_sha256 != plan.plan_sha256
                or data.readiness_sha256 != plan.readiness_sha256
                or data.matching_set_sha256 != plan.matching_set_sha256
                or data.runtime_identity_sha256
                != plan.training_schedule.runtime_identity_sha256
            ):
                raise OpeningCapabilityError("v3 train capability authority drift")
            del train_v3_registry[capability]
        return data

    def _create_child_directory(parent: Path, name: str) -> Path:
        child = parent / name
        try:
            child.mkdir(mode=0o700)
        except FileExistsError:
            pass
        except OSError as exc:
            raise OpeningCapabilityError(
                f"opening reservation directory cannot be created: {exc}"
            ) from exc
        _require_directory(child)
        return child

    def _marker_root(state: StoreAccessState) -> tuple[Path, tuple[int, int]]:
        if state.marker_mode == "production":
            _require_directory(
                production_anchor,
                expected_identity=production_anchor_identity,
            )
            policy_root = _create_child_directory(production_anchor, ".vfe4")
            root = _create_child_directory(policy_root, "h6-test-opening-reservations")
            if root != production_marker_root:
                raise OpeningCapabilityError("production opening root changed")
        elif state.marker_mode == "synthetic":
            anchor = state.synthetic_anchor
            anchor_identity = state.synthetic_anchor_identity
            if anchor is None or anchor_identity is None:
                raise OpeningCapabilityError("synthetic opening anchor is missing")
            _require_directory(anchor, expected_identity=anchor_identity)
            root = _create_child_directory(
                anchor, ".vfe4-h6-synthetic-opening-reservations"
            )
        else:
            raise OpeningCapabilityError("unknown opening reservation mode")
        return root, _require_directory(root)

    def _reservation_path(
        store: BlindedCorpusStore, state: StoreAccessState
    ) -> tuple[Path, tuple[int, int]]:
        root, root_identity = _marker_root(state)
        return root / _reservation_filename(store), root_identity

    def _reservation_filename(store: BlindedCorpusStore) -> str:
        access_policy_sha256 = store.data_identity.access_policy_sha256
        marker_sha256 = hashlib.sha256(
            marker_domain
            + bytes.fromhex(store.data_identity_sha256)
            + bytes.fromhex(access_policy_sha256)
        ).hexdigest()
        return marker_sha256 + ".reservation.bin"

    def _refuse_existing_reservation(
        store: BlindedCorpusStore,
        state: StoreAccessState,
    ) -> None:
        roots: list[Path] = []
        if state.marker_mode == "production":
            _require_directory(
                production_anchor,
                expected_identity=production_anchor_identity,
            )
            policy_root = production_anchor / ".vfe4"
            if os.path.lexists(policy_root):
                _require_directory(policy_root)
                root = production_marker_root
                if os.path.lexists(root):
                    _require_directory(root)
                    roots.append(root)

            legacy_synthetic_anchor = state.sealed_directory.parent.parent
            _require_directory(legacy_synthetic_anchor)
            legacy_synthetic_root = (
                legacy_synthetic_anchor / ".vfe4-h6-synthetic-opening-reservations"
            )
            if os.path.lexists(legacy_synthetic_root):
                _require_directory(legacy_synthetic_root)
                roots.append(legacy_synthetic_root)
        elif state.marker_mode == "synthetic":
            anchor = state.synthetic_anchor
            anchor_identity = state.synthetic_anchor_identity
            if anchor is None or anchor_identity is None:
                raise OpeningCapabilityError("synthetic opening anchor is missing")
            _require_directory(anchor, expected_identity=anchor_identity)
            root = anchor / ".vfe4-h6-synthetic-opening-reservations"
            if not os.path.lexists(root):
                return
            _require_directory(root)
            roots.append(root)
        else:
            raise OpeningCapabilityError("unknown opening reservation mode")
        if any(os.path.lexists(root / _reservation_filename(store)) for root in roots):
            raise OpeningCapabilityError(
                "opening reservation already exists or was consumed"
            )

    def _proof_bytes(
        reservation_path: Path,
        *,
        readiness_sha256: str,
        experiment_identity_sha256: str,
        data_identity_sha256: str,
        sealed_test_sha256: str,
        access_policy_sha256: str,
    ) -> bytes:
        normalized_path = unicodedata.normalize(
            "NFC", str(reservation_path.resolve(strict=False))
        ).replace("\\", "/")
        path_bytes = normalized_path.encode("utf-8")
        if len(path_bytes) > 0xFFFFFFFF:
            raise OpeningCapabilityError("opening reservation path is too long")
        return (
            proof_domain
            + len(path_bytes).to_bytes(4, "little")
            + path_bytes
            + bytes.fromhex(readiness_sha256)
            + bytes.fromhex(experiment_identity_sha256)
            + bytes.fromhex(data_identity_sha256)
            + bytes.fromhex(sealed_test_sha256)
            + bytes.fromhex(access_policy_sha256)
            + proof_suffix
        )

    def _exclusive_reserve(
        reservation_path: Path,
        canonical_bytes: bytes,
        root_identity: tuple[int, int],
    ) -> tuple[int, int]:
        _require_directory(
            reservation_path.parent,
            expected_identity=root_identity,
        )
        _install_complete_reservation_file_no_replace(
            reservation_path,
            canonical_bytes,
        )
        _require_directory(
            reservation_path.parent,
            expected_identity=root_identity,
        )
        file_identity = _require_regular_file(reservation_path)
        if reservation_path.stat().st_size != len(canonical_bytes):
            raise OpeningCapabilityError("opening reservation length changed")
        return file_identity

    def _reserve_and_issue(
        *,
        store: BlindedCorpusStore,
        state: StoreAccessState,
        readiness_sha256: str,
        experiment_identity: ExperimentIdentity,
    ) -> DurableTestOpeningCapability:
        with registry_lock:
            reservation_path, root_identity = _reservation_path(store, state)
            if state.opening is not None:
                raise FileExistsError(
                    errno.EEXIST,
                    "opening reservation already exists",
                    reservation_path,
                )
            canonical_bytes = _proof_bytes(
                reservation_path,
                readiness_sha256=readiness_sha256,
                experiment_identity_sha256=(
                    experiment_identity.experiment_identity_sha256
                ),
                data_identity_sha256=store.data_identity_sha256,
                sealed_test_sha256=(store.sealed_test_handle.sealed_content_sha256),
                access_policy_sha256=store.data_identity.access_policy_sha256,
            )
            file_identity = _exclusive_reserve(
                reservation_path,
                canonical_bytes,
                root_identity,
            )
            proof_identity_sha256 = hashlib.sha256(canonical_bytes).hexdigest()
            capability = DurableOpening(proof_identity_sha256, issuer_authority)
            state.opening = RegisteredOpeningProof(
                reservation_path,
                root_identity,
                file_identity,
                bytes(canonical_bytes),
                proof_identity_sha256,
                proof_identity_sha256,
                readiness_sha256,
                experiment_identity.experiment_identity_sha256,
                store.data_identity_sha256,
                store.sealed_test_handle.sealed_content_sha256,
                store.data_identity.access_policy_sha256,
                capability,
                False,
            )
            return capability

    def _issue(
        *,
        store: BlindedCorpusStore,
        readiness: H6PredictionReadinessToken,
        experiment_identity: ExperimentIdentity,
    ) -> DurableTestOpeningCapability:
        state = _state_for(store)
        _require_readiness(store, readiness)
        if type(experiment_identity) is not ExperimentIdentity:
            raise OpeningCapabilityError("exact ExperimentIdentity is required")
        try:
            experiment_identity.__post_init__()
        except ValueError as exc:
            raise OpeningCapabilityError(
                "ExperimentIdentity failed validation"
            ) from exc
        if (
            experiment_identity.sealed_data_sha256 != store.data_identity_sha256
            or experiment_identity.access_policy_sha256
            != store.data_identity.access_policy_sha256
        ):
            raise OpeningCapabilityError(
                "ExperimentIdentity does not match the store data identity"
            )
        return _reserve_and_issue(
            store=store,
            state=state,
            readiness_sha256=readiness.readiness_sha256,
            experiment_identity=experiment_identity,
        )

    def _issue_v3(
        *,
        store: BlindedCorpusStore,
        readiness: object,
        plan: object,
        experiment_identity: ExperimentIdentity,
        reservation: object,
    ) -> DurableTestOpeningCapability:
        from vfe4.training.h6_test_transaction_v3 import H6TestReservationV3
        from vfe4.training.h6_experiment_v3 import H6ExperimentPlanV3
        from vfe4.types.h6_prediction_v3 import H6PredictionV3ReadinessToken

        state = _state_for(store)
        if (
            state.marker_mode != "production"
            or type(store.data_identity) is not AuthenticatedReopenedDataIdentity
        ):
            raise OpeningCapabilityError(
                "v3 test opening requires authenticated reopened-store provenance"
            )
        if type(readiness) is not H6PredictionV3ReadinessToken:
            raise OpeningCapabilityError("exact H6 Prediction v3 readiness is required")
        if type(plan) is not H6ExperimentPlanV3:
            raise OpeningCapabilityError("exact H6 experiment plan v3 is required")
        if type(experiment_identity) is not ExperimentIdentity:
            raise OpeningCapabilityError("exact ExperimentIdentity is required")
        if type(reservation) is not H6TestReservationV3:
            raise OpeningCapabilityError(
                "exact complete H6 test reservation v3 is required"
            )
        try:
            readiness.__post_init__()
            plan.__post_init__()
            experiment_identity.__post_init__()
            reservation.__post_init__()
        except ValueError as exc:
            raise OpeningCapabilityError(
                "v3 test-opening authority failed validation"
            ) from exc
        data_identity = store.data_identity
        if (
            readiness.status != "PASS"
            or readiness.data_identity_sha256 != store.data_identity_sha256
            or readiness.access_policy_sha256 != data_identity.access_policy_sha256
            or plan.readiness_sha256 != readiness.readiness_sha256
            or plan.experiment_config_sha256 != readiness.experiment_config_sha256
            or plan.matching_set_sha256 != readiness.matching_set_sha256
            or plan.matching_policy_sha256 != readiness.matching_policy_sha256
            or plan.git_head != readiness.git_head
            or plan.dirty_digest != readiness.dirty_digest
            or plan.training_schedule.schedule_sha256
            != readiness.training_schedule_sha256
            or plan.training_schedule.runtime_identity_sha256
            != readiness.runtime_identity_sha256
            or any(
                attempt.attempt_spec.data_identity_sha256 != store.data_identity_sha256
                for attempt in plan.attempts
            )
            or experiment_identity.sealed_data_sha256 != store.data_identity_sha256
            or experiment_identity.access_policy_sha256
            != data_identity.access_policy_sha256
            or reservation.experiment_config_sha256
            != readiness.experiment_config_sha256
            or reservation.readiness_sha256 != readiness.readiness_sha256
            or reservation.plan_sha256 != plan.plan_sha256
            or reservation.experiment_identity_sha256
            != experiment_identity.experiment_identity_sha256
            or reservation.data_identity_sha256 != store.data_identity_sha256
            or reservation.sealed_test_sha256
            != store.sealed_test_handle.sealed_content_sha256
            or reservation.test_inventory_sha256
            != store.sealed_test_handle.handle_sha256
            or reservation.access_policy_sha256 != data_identity.access_policy_sha256
        ):
            raise OpeningCapabilityError(
                "v3 test-opening store/readiness/plan authority drift"
            )
        vocabulary = ByteTokenizerV1().vocabulary_identity
        if any(config.vocabulary != vocabulary for config in plan.endpoint_configs):
            raise OpeningCapabilityError(
                "v3 plan vocabulary does not match the authenticated store"
            )
        canonical_bytes = reservation.canonical_bytes()
        marker_bytes_sha256 = hashlib.sha256(canonical_bytes).hexdigest()
        with registry_lock:
            reservation_path, root_identity = _reservation_path(store, state)
            if state.opening is not None or os.path.lexists(reservation_path):
                raise FileExistsError(
                    errno.EEXIST,
                    "opening reservation already exists",
                    reservation_path,
                )
            file_identity = _exclusive_reserve(
                reservation_path,
                canonical_bytes,
                root_identity,
            )
            # The complete authoritative reservation exists before a
            # capability is registered or returned. A process crash in this
            # narrow gap leaves a recoverable RESERVED record, not test access.
            capability = DurableOpening(
                reservation.opening_proof_sha256,
                issuer_authority,
            )
            state.opening = RegisteredOpeningProof(
                reservation_path,
                root_identity,
                file_identity,
                bytes(canonical_bytes),
                marker_bytes_sha256,
                reservation.opening_proof_sha256,
                readiness.readiness_sha256,
                experiment_identity.experiment_identity_sha256,
                store.data_identity_sha256,
                store.sealed_test_handle.sealed_content_sha256,
                store.data_identity.access_policy_sha256,
                capability,
                True,
            )
            return capability

    def _v3_reservation_destination(store: BlindedCorpusStore) -> Path:
        state = _state_for(store)
        if (
            state.marker_mode != "production"
            or type(store.data_identity) is not AuthenticatedReopenedDataIdentity
        ):
            raise OpeningCapabilityError(
                "v3 reservation destination requires authenticated reopened store"
            )
        with registry_lock:
            reservation_path, _ = _reservation_path(store, state)
            if state.opening is not None or os.path.lexists(reservation_path):
                raise FileExistsError(
                    errno.EEXIST,
                    "opening reservation already exists",
                    reservation_path,
                )
            return reservation_path

    def _v3_registered_reservation_path(
        store: BlindedCorpusStore,
        opening: DurableTestOpeningCapability,
    ) -> Path:
        state = _state_for(store)
        with registry_lock:
            proof = state.opening
            if (
                type(proof) is not RegisteredOpeningProof
                or not proof.is_v3_reservation
                or type(opening) is not DurableOpening
                or opening is not proof.capability
            ):
                raise OpeningCapabilityError(
                    "exact registered v3 opening capability is required"
                )
            return proof.reservation_path

    def _read_registered_proof(proof: RegisteredOpeningProof) -> bytes:
        _require_directory(
            proof.reservation_path.parent,
            expected_identity=proof.reservation_root_identity,
        )
        return _read_exact_file(
            proof.reservation_path,
            expected_identity=proof.reservation_file_identity,
            expected_length=len(proof.canonical_bytes),
            expected_sha256=proof.marker_bytes_sha256,
        )

    def _validate(
        store: BlindedCorpusStore,
        opening: DurableTestOpeningCapability,
    ) -> ValidatedTestOpening:
        state = _state_for(store)
        with registry_lock:
            proof = state.opening
            if type(proof) is not RegisteredOpeningProof:
                raise OpeningCapabilityError("no opening proof is registered")
            if proof.consumed:
                raise OpeningCapabilityError("opening capability has been consumed")
            if type(opening) is not DurableOpening or opening is not proof.capability:
                raise OpeningCapabilityError("opening capability is forged")
            if opening.proof_identity_sha256 != proof.proof_identity_sha256:
                raise OpeningCapabilityError("opening capability identity changed")
            if (
                proof.data_identity_sha256 != store.data_identity_sha256
                or proof.sealed_test_sha256
                != store.sealed_test_handle.sealed_content_sha256
                or proof.access_policy_sha256
                != store.data_identity.access_policy_sha256
            ):
                raise OpeningCapabilityError("registered opening identities changed")
            observed = _read_registered_proof(proof)
            if proof.is_v3_reservation:
                from vfe4.training.h6_test_transaction_v3 import (
                    H6TestReservationV3,
                )

                try:
                    reservation = H6TestReservationV3.from_canonical_bytes(observed)
                except (TypeError, ValueError) as exc:
                    raise OpeningCapabilityError(
                        "authoritative v3 reservation cannot be reopened"
                    ) from exc
                proof_is_valid = (
                    reservation.opening_proof_sha256 == proof.proof_identity_sha256
                    and reservation.readiness_sha256 == proof.readiness_sha256
                    and reservation.experiment_identity_sha256
                    == proof.experiment_identity_sha256
                    and reservation.data_identity_sha256 == proof.data_identity_sha256
                    and reservation.sealed_test_sha256 == proof.sealed_test_sha256
                    and reservation.access_policy_sha256 == proof.access_policy_sha256
                )
            else:
                reconstructed = _proof_bytes(
                    proof.reservation_path,
                    readiness_sha256=proof.readiness_sha256,
                    experiment_identity_sha256=(proof.experiment_identity_sha256),
                    data_identity_sha256=proof.data_identity_sha256,
                    sealed_test_sha256=proof.sealed_test_sha256,
                    access_policy_sha256=proof.access_policy_sha256,
                )
                proof_is_valid = reconstructed == proof.canonical_bytes
            if (
                not proof_is_valid
                or observed != proof.canonical_bytes
                or hashlib.sha256(observed).hexdigest() != proof.marker_bytes_sha256
            ):
                raise OpeningCapabilityError("durable opening proof changed")
            proof.consumed = True
            return ValidatedOpening(
                proof.proof_identity_sha256,
                validated_authority,
            )

    def _open_test(
        store: BlindedCorpusStore,
        opening: DurableTestOpeningCapability,
    ) -> CausalWindows:
        windows, _ = _open_test_with_receipt(store, opening)
        return windows

    def _open_test_with_receipt(
        store: BlindedCorpusStore,
        opening: DurableTestOpeningCapability,
    ) -> tuple[CausalWindows, ValidatedTestOpening]:
        state = _state_for(store)
        validated = _validate(store, opening)
        if type(validated) is not ValidatedOpening:
            raise OpeningCapabilityError("test opening was not privately validated")
        return _read_split(store, state, "test"), validated

    def _validated_opening_identity(
        opening: ValidatedTestOpening,
    ) -> str:
        if type(opening) is not ValidatedOpening:
            raise OpeningCapabilityError("validated test-opening receipt is forged")
        return opening.proof_identity_sha256

    return (
        _register_production,
        _register_synthetic,
        _register_reopened,
        _validation_fixture,
        _materialize_train,
        _issue_train_v3,
        _consume_train_v3,
        _issue_validation_v3,
        _consume_validation_v3,
        _issue,
        _issue_v3,
        _v3_reservation_destination,
        _v3_registered_reservation_path,
        _validate,
        _open_test,
        _open_test_with_receipt,
        _validated_opening_identity,
    )


(
    _register_production_blinded_store,
    _register_synthetic_blinded_store,
    _register_reopened_blinded_store_v3,
    materialize_validation_safety_fixture,
    materialize_prediction_train,
    issue_h6_train_capability_v3,
    open_train_for_training_v3,
    issue_h6_validation_capability_v3,
    _consume_h6_validation_capability_v3,
    reserve_and_issue_durable_test_opening_capability,
    reserve_and_issue_durable_test_opening_capability_v3,
    h6_test_reservation_destination_v3,
    registered_h6_test_reservation_path_v3,
    validate_durable_test_opening_capability,
    open_test_for_scoring,
    open_test_for_scoring_with_receipt,
    validated_test_opening_identity,
) = _build_access_api()
del _build_access_api


__all__ = [
    "H6_TEST_RESERVATION_DURABILITY",
    "H6_V3_RESERVATION_AUTHORITY_CHECKS",
    "H6TrainingDataV3",
    "MaterializedPredictionData",
    "OpeningCapabilityError",
    "issue_h6_train_capability_v3",
    "materialize_prediction_train",
    "materialize_validation_safety_fixture",
    "open_train_for_training_v3",
    "issue_h6_validation_capability_v3",
    "open_test_for_scoring",
    "open_test_for_scoring_with_receipt",
    "h6_test_reservation_destination_v3",
    "registered_h6_test_reservation_path_v3",
    "reserve_and_issue_durable_test_opening_capability",
    "reserve_and_issue_durable_test_opening_capability_v3",
    "validate_durable_test_opening_capability",
    "validated_test_opening_identity",
]
