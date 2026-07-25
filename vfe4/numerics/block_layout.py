"""Strict block-chain layout arithmetic for the bounded H8 reference.

This module is deliberately tensor-free.  It computes and validates the only
population coordinate layout admitted by H8 before any allocation is made.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


H8_MAX_STORAGE_SCALARS = 411_200
H8_REFERENCE_HORIZON = 128
H8_REFERENCE_CHANNEL_DIMENSION = 20
BlockKind = Literal["diagonal", "lower_adjacent"]
Channel = Literal["z", "m"]


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class BlockId:
    """One canonically oriented stored block.

    Adjacent blocks are stored only in lower orientation ``(t, t-1)``.  The
    corresponding upper block is a transpose at its point of use, never a
    second stored block.
    """

    kind: BlockKind
    row: int
    column: int

    def __post_init__(self) -> None:
        if self.kind not in ("diagonal", "lower_adjacent"):
            raise ValueError("kind must be diagonal or lower_adjacent")
        if type(self.row) is not int or self.row < 0:
            raise ValueError("row must be a nonnegative integer")
        if type(self.column) is not int or self.column < 0:
            raise ValueError("column must be a nonnegative integer")
        if self.kind == "diagonal" and self.row != self.column:
            raise ValueError("diagonal block IDs require row == column")
        if self.kind == "lower_adjacent" and self.row != self.column + 1:
            raise ValueError(
                "lower-adjacent block IDs require row == column + 1"
            )

    @classmethod
    def diagonal(cls, index: int) -> "BlockId":
        return cls(kind="diagonal", row=index, column=index)

    @classmethod
    def lower(cls, target: int) -> "BlockId":
        return cls(kind="lower_adjacent", row=target, column=target - 1)


@dataclass(frozen=True, slots=True)
class BlockChainLayout:
    """Population-major ``[z_t, m_t]`` block-chain coordinates."""

    horizon: int
    d_z: int
    d_m: int

    def __post_init__(self) -> None:
        _positive_int(self.horizon, "horizon")
        _positive_int(self.d_z, "d_z")
        _positive_int(self.d_m, "d_m")
        if self.band_storage_scalar_count > H8_MAX_STORAGE_SCALARS:
            raise ValueError(
                "block precision/factor storage exceeds the frozen H8 cap"
            )

    @property
    def population_size(self) -> int:
        """Number of population slices, ``N=T+1``."""

        return self.horizon + 1

    @property
    def block_size(self) -> int:
        """Combined local population width, ``b=d_z+d_m``."""

        return self.d_z + self.d_m

    @property
    def dimension(self) -> int:
        """Dense-equivalent dimension, retained as arithmetic only."""

        return self.population_size * self.block_size

    @property
    def diagonal_scalar_count(self) -> int:
        return self.population_size * self.block_size * self.block_size

    @property
    def lower_scalar_count(self) -> int:
        return self.horizon * self.block_size * self.block_size

    @property
    def band_storage_scalar_count(self) -> int:
        return self.diagonal_scalar_count + self.lower_scalar_count

    @property
    def information_scalar_count(self) -> int:
        return self.population_size * self.block_size

    @property
    def dense_scalar_count(self) -> int:
        return self.dimension * self.dimension

    @property
    def diagonal_block_ids(self) -> tuple[BlockId, ...]:
        return tuple(
            BlockId.diagonal(index) for index in range(self.population_size)
        )

    @property
    def lower_block_ids(self) -> tuple[BlockId, ...]:
        return tuple(
            BlockId.lower(target) for target in range(1, self.population_size)
        )

    @property
    def stored_block_ids(self) -> tuple[BlockId, ...]:
        """Canonical storage order: all diagonal, then all lower-adjacent."""

        return self.diagonal_block_ids + self.lower_block_ids

    def block_slice(self, population: int) -> slice:
        """Return a bounded-oracle coordinate slice without allocating it."""

        self._require_population(population)
        start = population * self.block_size
        return slice(start, start + self.block_size)

    def state_slice(self, population: int) -> slice:
        self._require_population(population)
        start = population * self.block_size
        return slice(start, start + self.d_z)

    def model_slice(self, population: int) -> slice:
        self._require_population(population)
        start = population * self.block_size + self.d_z
        return slice(start, start + self.d_m)

    def flatten_coordinate(
        self,
        population: int,
        channel: Channel,
        offset: int,
    ) -> int:
        """Map ``(t, channel, local_index)`` to population-major position."""

        self._require_population(population)
        if channel not in ("z", "m"):
            raise ValueError("channel must be z or m")
        width = self.d_z if channel == "z" else self.d_m
        if type(offset) is not int or not 0 <= offset < width:
            raise ValueError("offset is outside the selected channel")
        channel_start = 0 if channel == "z" else self.d_z
        return population * self.block_size + channel_start + offset

    def coordinate_id(self, flat_index: int) -> tuple[int, Channel, int]:
        """Invert one flat coordinate without constructing a global vector."""

        if (
            type(flat_index) is not int
            or flat_index < 0
            or flat_index >= self.dimension
        ):
            raise ValueError("flat_index is outside the layout")
        population, local = divmod(flat_index, self.block_size)
        if local < self.d_z:
            return population, "z", local
        return population, "m", local - self.d_z

    def require_block_id(self, block: object) -> BlockId:
        if type(block) is not BlockId:
            raise ValueError("block must be a BlockId")
        if block.row >= self.population_size:
            raise ValueError("block row is outside the layout")
        if block.column >= self.population_size:
            raise ValueError("block column is outside the layout")
        return block

    def require_complete_stored_blocks(
        self, blocks: object
    ) -> tuple[BlockId, ...]:
        if type(blocks) is not tuple:
            raise ValueError("blocks must be a tuple in canonical order")
        checked = tuple(self.require_block_id(block) for block in blocks)
        if len(set(checked)) != len(checked):
            raise ValueError("blocks must not contain duplicate IDs")
        if checked != self.stored_block_ids:
            raise ValueError(
                "blocks must contain exactly the canonical stored block IDs"
            )
        return checked

    def require_block_vector_shape(
        self,
        shape: object,
        *,
        name: str = "block vector",
    ) -> tuple[int, int]:
        """Validate a native ``[N,b]`` vector shape without allocating it."""

        expected = (self.population_size, self.block_size)
        if type(shape) is not tuple or shape != expected:
            raise ValueError(f"{name} shape must be exactly [N,b]")
        return expected

    def require_block_matrix_shape(
        self,
        shape: object,
        *,
        adjacent: bool,
        name: str = "block matrix",
    ) -> tuple[int, int, int]:
        """Validate diagonal or lower-adjacent native block storage."""

        if type(adjacent) is not bool:
            raise ValueError("adjacent must be a bool")
        leading = self.horizon if adjacent else self.population_size
        expected = (leading, self.block_size, self.block_size)
        if type(shape) is not tuple or shape != expected:
            category = "lower-adjacent" if adjacent else "diagonal"
            raise ValueError(f"{name} must use exact {category} block storage")
        return expected

    def require_rhs_shape(
        self,
        shape: object,
        *,
        name: str = "right-hand side",
    ) -> tuple[int, ...]:
        """Accept only ``[N,b]`` or ``[N,b,r]`` with ``1<=r<=b``."""

        if type(shape) is not tuple or len(shape) not in (2, 3):
            raise ValueError(f"{name} must be [N,b] or [N,b,r]")
        if any(type(item) is not int or item <= 0 for item in shape):
            raise ValueError(f"{name} dimensions must be positive integers")
        if shape[:2] != (self.population_size, self.block_size):
            raise ValueError(f"{name} must preserve explicit [N,b] axes")
        if len(shape) == 3 and shape[2] > self.block_size:
            raise ValueError(f"{name} width cannot exceed b")
        return shape

    def require_sample_shape(
        self,
        shape: object,
        *,
        name: str = "sample noise",
    ) -> tuple[int, int]:
        """H8 admits exactly one block-shaped sample/noise path."""

        return self.require_block_vector_shape(shape, name=name)

    def require_local_square_shape(
        self,
        shape: object,
        *,
        name: str = "local matrix",
    ) -> tuple[int, int]:
        expected = (self.block_size, self.block_size)
        if type(shape) is not tuple or shape != expected:
            raise ValueError(f"{name} must be exactly [b,b]")
        return expected

    def require_bounded_storage_scalar_count(
        self,
        scalar_count: object,
        *,
        name: str,
    ) -> int:
        """Fail before allocation when one category exceeds the H8 cap."""

        if (
            type(scalar_count) is not int
            or scalar_count < 0
            or scalar_count > H8_MAX_STORAGE_SCALARS
        ):
            raise ValueError(f"{name} exceeds the frozen H8 scalar cap")
        return scalar_count

    def _require_population(self, population: object) -> int:
        if (
            type(population) is not int
            or population < 0
            or population >= self.population_size
        ):
            raise ValueError("population is outside the layout")
        return population


__all__ = [
    "BlockChainLayout",
    "BlockId",
    "BlockKind",
    "Channel",
    "H8_MAX_STORAGE_SCALARS",
    "H8_REFERENCE_CHANNEL_DIMENSION",
    "H8_REFERENCE_HORIZON",
]
