"""Deterministic H4 neutral Gaussian generator and structural H3 adapter."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from typing import Literal

import numpy as np

from vfe4.types.h3 import H3Fixture
from vfe4.types.h4 import (
    H4AffineGaussianFactor,
    H4NeutralProblem,
    H4RawDraw,
    canonical_h4_problem_bytes,
    h4_problem_digest,
)


def make_h4_problem(*, seed: int, kind: Literal["coupled", "zero_control"], horizon: Literal[7, 15, 31], d_z: Literal[4] = 4, d_m: Literal[4] = 4) -> H4NeutralProblem:
    if type(seed) is not int or seed <= 0 or kind not in ("coupled", "zero_control") or horizon not in (7, 15, 31) or (d_z, d_m) != (4, 4):
        raise ValueError("H4 scaled generator arguments are frozen")
    dimension = (horizon + 1) * 8
    coordinates = tuple(f"{prefix}[{t},{i}]" for t in range(horizon + 1) for prefix in ("z", "m") for i in range(4))
    schedule: list[H4AffineGaussianFactor] = [_initial_factor(dimension)]
    rng = np.random.Generator(np.random.PCG64(seed))
    for time in range(1, horizon + 1):
        raw_a_m = rng.standard_normal((4, 4)); raw_a_z = rng.standard_normal((4, 4)); raw_b = rng.standard_normal((4, 4))
        c_m = rng.uniform(-.25, .25, size=4); c_z = rng.uniform(-.25, .25, size=4)
        r_m = rng.uniform(.5, 1.5, size=4); r_z = rng.uniform(.5, 1.5, size=4)
        raw_g = rng.standard_normal((8, 8)); offset = rng.uniform(-.25, .25, size=8)
        observation_noise = rng.uniform(.75, 1.25, size=8); observed_target = rng.uniform(-1., 1., size=8)
        active_a_m = _clip(raw_a_m); joined = _clip(np.concatenate((raw_a_z, raw_b), axis=1)); active_a_z, active_b = joined[:, :4], joined[:, 4:]
        if kind == "zero_control": active_a_m = np.zeros_like(active_a_m); active_a_z = np.zeros_like(active_a_z); active_b = np.zeros_like(active_b)
        base = 11 * (time - 1)
        m_draws = (_draw(base, f"A_m[{time}]", raw_a_m), _draw(base + 3, f"c_m[{time}]", c_m), _draw(base + 5, f"R_m[{time}]", r_m))
        z_draws = (_draw(base + 1, f"A_z[{time}]", raw_a_z), _draw(base + 2, f"B[{time}]", raw_b), _draw(base + 4, f"c_z[{time}]", c_z), _draw(base + 6, f"R_z[{time}]", r_z))
        obs_draws = (_draw(base + 7, f"G[{time}]", raw_g), _draw(base + 8, f"observation_offset[{time}]", offset), _draw(base + 9, f"observation_noise[{time}]", observation_noise), _draw(base + 10, f"observed_target[{time}]", observed_target))
        z_prev, m_prev, z_now, m_now = _block(time - 1), _block(time - 1, "m"), _block(time), _block(time, "m")
        m_matrix = np.zeros((4, dimension)); m_matrix[:, m_now] = np.eye(4); m_matrix[:, m_prev] = -active_a_m
        z_matrix = np.zeros((4, dimension)); z_matrix[:, z_now] = np.eye(4); z_matrix[:, z_prev] = -active_a_z; z_matrix[:, m_now] = -active_b
        H = np.eye(8) + .05 * raw_g / max(1.0, float(np.linalg.norm(raw_g, 2)))
        observation_matrix = np.zeros((8, dimension)); observation_matrix[:, (*z_now, *m_now)] = H
        schedule.extend((
            H4AffineGaussianFactor(f"m_transition[{time}]", "transition", time, tuple(m_now), tuple(m_prev), _tuples(m_matrix), _tuple(c_m), _tuples(np.diag(r_m)), m_draws),
            H4AffineGaussianFactor(f"z_transition[{time}]", "transition", time, tuple(z_now), (*z_prev, *m_now), _tuples(z_matrix), _tuple(c_z), _tuples(np.diag(r_z)), z_draws),
            H4AffineGaussianFactor(f"observation[{time}]", "observation", time, (), (*z_now, *m_now), _tuples(observation_matrix), _tuple(observed_target - offset), _tuples(np.diag(observation_noise)), obs_draws),
        ))
    temporary = H4NeutralProblem(f"h4-{kind}-T{horizon}-dz4-dm4-seed{seed}-v1", "scaled_pcg64", seed, kind, horizon, 4, 4, dimension, coordinates, tuple(schedule), "0" * 64)
    return replace(temporary, canonical_sha256=h4_problem_digest(temporary))


def h4_anchor_from_h3(fixture: H3Fixture) -> H4NeutralProblem:
    if not isinstance(fixture, H3Fixture):
        raise ValueError("fixture must be H3Fixture")
    groups = (fixture.initial_factors, fixture.transition_factors, fixture.observation_factors)
    factors: list[H4AffineGaussianFactor] = []
    for group_index, group in enumerate(groups):
        for position, record in enumerate(group):
            if group_index == 0:
                role, time, normalized, parents = "initial", 0, (position,), ()
            elif group_index == 1:
                role, time = "transition", 1
                normalized, parents = ((3,), (1,)) if position == 0 else ((2,), (0, 3))
            else:
                role, time, normalized, parents = "observation", 1, (), (2 + position,)
            factors.append(H4AffineGaussianFactor(record.factor_id, role, time, normalized, parents, (record.row,), (record.target,), ((record.variance,),), ()))
    temporary = H4NeutralProblem(f"h4-anchor-{fixture.fixture_id}", "h3_anchor", 0, fixture.kind, 1, 1, 1, 4, fixture.continuous_order, tuple(factors), "0" * 64)
    return replace(temporary, canonical_sha256=h4_problem_digest(temporary))


def canonical_h4_gaussian(problem: H4NeutralProblem) -> tuple[np.ndarray, np.ndarray, float, float]:
    J = np.zeros((problem.dimension, problem.dimension)); h = np.zeros(problem.dimension); c = 0.0
    for factor in problem.factor_schedule:
        A, b, R = np.asarray(factor.matrix), np.asarray(factor.target), np.asarray(factor.covariance)
        inverse = np.linalg.inv(R)
        J += A.T @ inverse @ A; h += A.T @ inverse @ b
        c -= .5 * (b @ inverse @ b + len(b) * math.log(2.0 * math.pi) + np.linalg.slogdet(R)[1])
    np.linalg.cholesky(J)
    log_z = c + .5 * h @ np.linalg.solve(J, h) - .5 * np.linalg.slogdet(J)[1] + problem.dimension / 2.0 * math.log(2.0 * math.pi)
    return J, h, float(c), float(log_z)


def parse_h4_problem_bytes(data: bytes) -> H4NeutralProblem:
    if type(data) is not bytes: raise ValueError("data must be bytes")
    try:
        envelope = json.loads(data.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_constant)
        _fields(envelope, ("schema_version","canonical_sha256","problem"))
        if envelope["schema_version"] != "h4-neutral-problem-v1": raise ValueError("invalid H4 schema")
        core = envelope["problem"]; _fields(core, ("problem_id","source_kind","seed","kind","horizon","d_z","d_m","dimension","coordinate_order","factor_schedule"))
        if type(core["coordinate_order"]) is not list or type(core["factor_schedule"]) is not list: raise ValueError("invalid H4 sequences")
        factors = tuple(_parse_factor(item) for item in core["factor_schedule"])
        problem = H4NeutralProblem(core["problem_id"], core["source_kind"], core["seed"], core["kind"], core["horizon"], core["d_z"], core["d_m"], core["dimension"], tuple(core["coordinate_order"]), factors, envelope["canonical_sha256"])
        if h4_problem_digest(problem) != problem.canonical_sha256 or canonical_h4_problem_bytes(problem) != data: raise ValueError("noncanonical H4 bytes")
        if problem.source_kind == "scaled_pcg64":
            replay = make_h4_problem(seed=problem.seed, kind=problem.kind, horizon=problem.horizon)
            if problem != replay:
                raise ValueError("scaled H4 bytes do not replay the frozen PCG64 provenance")
        return problem
    except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid H4 canonical JSON: {exc}") from exc


def _initial_factor(dimension: int) -> H4AffineGaussianFactor:
    matrix = np.zeros((8, dimension)); matrix[:, :8] = np.eye(8)
    return H4AffineGaussianFactor("initial_joint", "initial", 0, tuple(range(8)), (), _tuples(matrix), (0.0,) * 8, _tuples(np.eye(8)), ())
def _block(time: int, kind: str = "z") -> tuple[int, ...]:
    start = time * 8 + (4 if kind == "m" else 0); return tuple(range(start, start + 4))
def _clip(value: np.ndarray) -> np.ndarray: return value * min(1.0, .65 / float(np.linalg.norm(value, 2)))
def _tuple(value: np.ndarray) -> tuple[float, ...]: return tuple(float(x) for x in value.reshape(-1))
def _tuples(value: np.ndarray) -> tuple[tuple[float, ...], ...]: return tuple(tuple(float(x) for x in row) for row in value)
def _draw(index: int, name: str, value: np.ndarray) -> H4RawDraw: return H4RawDraw(index, name, tuple(value.shape), _tuple(value))
def _parse_factor(raw: object) -> H4AffineGaussianFactor:
    _fields(raw, ("factor_id","role","time_index","normalized_coordinate_indices","parent_coordinate_indices","matrix","target","covariance","raw_draws"))
    if any(type(raw[name]) is not list for name in ("normalized_coordinate_indices","parent_coordinate_indices","matrix","target","covariance","raw_draws")): raise ValueError("invalid factor sequences")
    draws = tuple(_parse_draw(item) for item in raw["raw_draws"])
    return H4AffineGaussianFactor(raw["factor_id"], raw["role"], raw["time_index"], tuple(raw["normalized_coordinate_indices"]), tuple(raw["parent_coordinate_indices"]), tuple(tuple(row) for row in raw["matrix"]), tuple(raw["target"]), tuple(tuple(row) for row in raw["covariance"]), draws)

def _parse_draw(raw: object) -> H4RawDraw:
    _fields(raw, ("draw_index","name","shape","values"))
    if type(raw["shape"]) is not list or type(raw["values"]) is not list: raise ValueError("invalid draw sequences")
    return H4RawDraw(raw["draw_index"], raw["name"], tuple(raw["shape"]), tuple(raw["values"]))
def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError("duplicate JSON key")
        result[key] = value
    return result
def _constant(value): raise ValueError("nonfinite JSON constant")
def _fields(value, keys):
    if type(value) is not dict or set(value) != set(keys): raise ValueError("unexpected JSON object fields")

__all__ = ["canonical_h4_gaussian", "h4_anchor_from_h3", "make_h4_problem", "parse_h4_problem_bytes"]
