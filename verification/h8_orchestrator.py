"""Pure parent-request planning for the frozen H8 protocol."""

from __future__ import annotations

from vfe4.types.h8 import (
    H8_NEGATIVE_CONTROL_IDS,
    H8_PRODUCTION_SEEDS,
    H8ChildRequest,
)


def build_h8_child_request_plan(
    *,
    config_sha256: str,
    protocol_sha256: str,
) -> tuple[H8ChildRequest, ...]:
    """Return the immutable, ordered 30-request H8 child plan."""

    return (
        *(
            H8ChildRequest(
                mode="production",
                seed=seed,
                repetition=repetition,
                config_sha256=config_sha256,
                protocol_sha256=protocol_sha256,
                control_id=None,
            )
            for seed in H8_PRODUCTION_SEEDS
            for repetition in range(5)
        ),
        *(
            H8ChildRequest(
                mode="profiler",
                seed=seed,
                repetition=None,
                config_sha256=config_sha256,
                protocol_sha256=protocol_sha256,
                control_id=None,
            )
            for seed in H8_PRODUCTION_SEEDS
        ),
        *(
            H8ChildRequest(
                mode="negative_control",
                seed=H8_PRODUCTION_SEEDS[0],
                repetition=None,
                config_sha256=config_sha256,
                protocol_sha256=protocol_sha256,
                control_id=control_id,
            )
            for control_id in H8_NEGATIVE_CONTROL_IDS
        ),
    )


__all__ = ["build_h8_child_request_plan"]
