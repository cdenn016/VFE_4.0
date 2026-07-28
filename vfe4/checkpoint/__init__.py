"""Safe, exact WikiText-103 checkpoint publication and restoration."""

from vfe4.types.training import (
    CheckpointBundle,
    LoadedCheckpoint,
    WT103CheckpointIdentity,
)

from .io import FreshCheckpointTarget, load_checkpoint, save_checkpoint
from .migrations import (
    MIGRATION_PROFILES,
    MigrationProfile,
    select_migration,
)
from .schema import (
    CHECKPOINT_ENVELOPE_SCHEMA,
    CheckpointCompatibilityError,
    CheckpointError,
    CheckpointMigrationError,
    CheckpointSchemaError,
    CheckpointSecurityError,
    ResumeContract,
    make_checkpoint_bundle,
    make_checkpoint_identity,
    require_terminal_scoring,
)
from .serialization import (
    SCIENTIFIC_STATE_KEYS,
    TensorInventoryEntry,
)


__all__ = [
    "CHECKPOINT_ENVELOPE_SCHEMA",
    "MIGRATION_PROFILES",
    "SCIENTIFIC_STATE_KEYS",
    "CheckpointBundle",
    "CheckpointCompatibilityError",
    "CheckpointError",
    "CheckpointMigrationError",
    "CheckpointSchemaError",
    "CheckpointSecurityError",
    "FreshCheckpointTarget",
    "LoadedCheckpoint",
    "MigrationProfile",
    "ResumeContract",
    "TensorInventoryEntry",
    "WT103CheckpointIdentity",
    "load_checkpoint",
    "make_checkpoint_bundle",
    "make_checkpoint_identity",
    "require_terminal_scoring",
    "save_checkpoint",
    "select_migration",
]
