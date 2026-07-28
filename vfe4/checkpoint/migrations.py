"""Closed checkpoint migration registry with permanent V3 exclusion."""

from __future__ import annotations

import dataclasses
from types import MappingProxyType
from typing import Literal, Mapping

from vfe4.types.training import owned_sha256

from .schema import (
    CHECKPOINT_ENVELOPE_SCHEMA,
    CheckpointMigrationError,
    identifier_is_v3,
)


_LOWER_HEX = frozenset("0123456789abcdef")


def _require_sha256(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise CheckpointMigrationError(
            f"migration {field} must be a lowercase SHA-256 hash"
        )
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class MigrationProfile:
    """Evidence required before a future explicit VFE4 schema migration."""

    source_schema_sha256: str
    destination_schema_sha256: str
    transform_code_sha256: str
    information_loss: Literal["lossless", "lossy"]
    independent_test_sha256: str
    profile_sha256: str

    def __post_init__(self) -> None:
        for field in (
            "source_schema_sha256",
            "destination_schema_sha256",
            "transform_code_sha256",
            "independent_test_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        if self.information_loss not in ("lossless", "lossy"):
            raise CheckpointMigrationError(
                "migration information-loss declaration is invalid"
            )
        expected = owned_sha256(
            "vfe4.wt103.checkpoint-migration-profile.v1",
            {
                "source_schema_sha256": self.source_schema_sha256,
                "destination_schema_sha256": self.destination_schema_sha256,
                "transform_code_sha256": self.transform_code_sha256,
                "information_loss": self.information_loss,
                "independent_test_sha256": self.independent_test_sha256,
            },
        )
        _require_sha256(self.profile_sha256, field="profile_sha256")
        if self.profile_sha256 != expected:
            raise CheckpointMigrationError(
                "migration profile hash does not match its code/loss/test identity"
            )

    @classmethod
    def create(
        cls,
        *,
        source_schema_sha256: str,
        destination_schema_sha256: str,
        transform_code_sha256: str,
        information_loss: Literal["lossless", "lossy"],
        independent_test_sha256: str,
    ) -> "MigrationProfile":
        payload = {
            "source_schema_sha256": source_schema_sha256,
            "destination_schema_sha256": destination_schema_sha256,
            "transform_code_sha256": transform_code_sha256,
            "information_loss": information_loss,
            "independent_test_sha256": independent_test_sha256,
        }
        return cls(
            **payload,
            profile_sha256=owned_sha256(
                "vfe4.wt103.checkpoint-migration-profile.v1",
                payload,
            ),
        )


MIGRATION_PROFILES: Mapping[
    tuple[str, str],
    MigrationProfile,
] = MappingProxyType({})


def select_migration(
    *,
    source_schema: str,
    destination_schema: str = CHECKPOINT_ENVELOPE_SCHEMA,
) -> MigrationProfile | None:
    if type(source_schema) is not str or type(destination_schema) is not str:
        raise CheckpointMigrationError("migration schemas must be exact strings")
    if identifier_is_v3(source_schema):
        raise CheckpointMigrationError(
            "V3 is permanently rejected and is never a VFE4 migration source"
        )
    if source_schema == destination_schema:
        return None
    profile = MIGRATION_PROFILES.get((source_schema, destination_schema))
    if profile is None:
        raise CheckpointMigrationError(
            "no migration profile exists for this checkpoint schema"
        )
    profile.__post_init__()
    return profile


__all__ = [
    "MIGRATION_PROFILES",
    "MigrationProfile",
    "select_migration",
]
