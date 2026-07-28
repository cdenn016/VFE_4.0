from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import vfe4.training.h6_training_attempt_v3 as attempt_v3
from test_h6_training_attempt_v3 import (
    _tiny_attempt_authority_v3,  # noqa: F401
)
from vfe4.training.h6_execution_v3 import bind_h6_executable_attempt_v3

pytestmark = pytest.mark.filterwarnings(
    "ignore:Failed to find (cuobjdump|nvdisasm)\\.exe:UserWarning"
)


class _IncompatibleRecoveryModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.unexpected = nn.Parameter(torch.ones(1, dtype=torch.float64))


def test_orphan_adoption_hydrates_canonical_state_before_progress_publication(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorities, training_data, runtime = request.getfixturevalue(
        "_tiny_attempt_authority_v3"
    )
    endpoint_config_id = (
        "h6-a5-structured-parent-specific-prefix-exact-complete-latent-smoothing-v2"
    )
    planned = next(
        attempt
        for attempt in authorities.plan.tuning_attempts  # type: ignore[union-attr]
        if attempt.endpoint_config_id == endpoint_config_id
    )
    executable = bind_h6_executable_attempt_v3(
        authorities=authorities,  # type: ignore[arg-type]
        planned_attempt=planned,
    )
    maximum_bytes = 256 * 1024 * 1024
    checkpoint_path = (
        tmp_path / f"{planned.endpoint_config_id}.orphan-preflight.h6v3"
    ).resolve()
    progress_path = attempt_v3.h6_training_attempt_progress_path_v3(checkpoint_path)
    original_publish = attempt_v3._publish_progress_catalog_v3
    publication_calls = 0

    def lose_before_first_catalog(
        *args: object,
        **kwargs: object,
    ) -> object:
        nonlocal publication_calls
        publication_calls += 1
        if publication_calls == 1:
            raise RuntimeError("simulated catalog publication loss")
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(
        attempt_v3,
        "_publish_progress_catalog_v3",
        lose_before_first_catalog,
    )
    with pytest.raises(RuntimeError, match="catalog publication loss"):
        attempt_v3._execute_new_training_attempt_v3(
            executable=executable,
            training_data=training_data,
            runtime=runtime,
            checkpoint_path=checkpoint_path,
            maximum_checkpoint_bytes=maximum_bytes,
        )
    assert publication_calls == 1
    assert not progress_path.exists()

    original_fresh = attempt_v3._fresh_cpu_training_modules_v3
    built, _model, recognition = original_fresh(
        executable=executable,
        runtime=runtime,
    )
    assert recognition is not None

    def incompatible_fresh_modules(
        *,
        executable: object,
        runtime: object,
    ) -> tuple[object, nn.Module, nn.Module]:
        del executable, runtime
        return built, _IncompatibleRecoveryModel(), recognition

    monkeypatch.setattr(
        attempt_v3,
        "_fresh_cpu_training_modules_v3",
        incompatible_fresh_modules,
    )

    def forbidden_progress_publication(
        *args: object,
        **kwargs: object,
    ) -> object:
        del args, kwargs
        raise AssertionError("incompatible orphan state reached progress publication")

    monkeypatch.setattr(
        attempt_v3,
        "_publish_progress_catalog_v3",
        forbidden_progress_publication,
    )
    with pytest.raises(ValueError, match="module inventory"):
        attempt_v3.recover_h6_training_attempt_v3(
            executable=executable,
            runtime=runtime,
            checkpoint_path=checkpoint_path,
            maximum_checkpoint_bytes=maximum_bytes,
        )

    assert not progress_path.exists()
