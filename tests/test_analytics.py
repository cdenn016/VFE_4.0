import torch

from vfe4.analytics.coherence import coherent_partition
from vfe4.analytics.holonomy import fixed_sector_rank
from vfe4.analytics.coarse import discarded_conditional_kl


def test_partition_and_fixed_sector() -> None:
    src, dst = torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])
    labels = coherent_partition(torch.tensor([0.1, 0.9, 0.2]), src, dst, 0.5, 4)
    assert labels[0] == labels[1] and labels[2] == labels[3] and labels[1] != labels[2]
    I = torch.eye(2, dtype=torch.float64)
    R = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=I.dtype)
    src, dst = torch.tensor([1, 2, 0]), torch.tensor([0, 1, 2])
    assert fixed_sector_rank(torch.stack([I, I, I]), src, dst, 3) == 2
    assert fixed_sector_rank(torch.stack([I, I, R]), src, dst, 3) == 0


def test_conditional_kl_simple_independent_case() -> None:
    mu_q = torch.tensor([1.0, 2.0], dtype=torch.float64)
    mu_p = torch.zeros(2, dtype=mu_q.dtype)
    covariance = torch.eye(2, dtype=mu_q.dtype)
    S = torch.tensor([[1.0, 0.0]], dtype=mu_q.dtype)
    value = discarded_conditional_kl(mu_q, covariance, mu_p, covariance, S)
    torch.testing.assert_close(value, torch.tensor(2.0, dtype=mu_q.dtype))


def _ring(n: int, K: int, std: float, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from vfe4.geometry.supports import agent_lattice_support, build_copy_table
    _, copies, _ = build_copy_table(agent_lattice_support(n, (n,), 'window', 1, None, None), True)
    generator = torch.Generator().manual_seed(seed)
    omega = std * torch.randn(len(copies.is_self), K, K, generator=generator, dtype=torch.float64)
    T = torch.where(copies.is_self[:, None, None], torch.eye(K, dtype=torch.float64), torch.linalg.matrix_exp(omega))
    return T, copies.source_agent, copies.receiver_agent


def test_fixed_sector_rank_on_deep_trees() -> None:
    # Rooted path products on a 128-ring reached ~e^100 and once reported rank 1 here.
    T, src, dst = _ring(128, 3, 1.0, 0)
    assert fixed_sector_rank(T, src, dst, 128) == 0
    n, K = 256, 3
    generator = torch.Generator().manual_seed(1)
    g = torch.linalg.matrix_exp(0.4 * torch.randn(n, K, K, generator=generator, dtype=torch.float64))
    src = torch.arange(n)
    dst = (src + 1) % n
    assert fixed_sector_rank(g[dst] @ torch.linalg.inv(g[src]), src, dst, n) == K


def test_reciprocity_excludes_self_rows_and_context_choice() -> None:
    from vfe4.config import load_config
    from vfe4.presets import resolve_preset
    from vfe4.model import build_model, initialize_sections
    from vfe4.analytics.observables import analytics_context, collect_observables
    config = load_config(resolve_preset('tiny_point', {'links': {'initial_twist_std': 0.5}}))
    model = build_model(config)
    state = initialize_sections(model, config)
    generator = torch.Generator().manual_seed(2)
    state.beta = torch.rand(state.beta.shape, generator=generator, dtype=config.dtype)
    bank = next(b for b in model.records if b.kind == 'communication' and b.channel == 0)
    beta = state.beta[bank.candidate_id]
    pairs = {(int(r), int(s)): float(w) for r, s, w in zip(bank.receiver, bank.sender, beta)}
    expected = [abs(w - pairs[(s, r)]) for (r, s), w in pairs.items() if r != s and (s, r) in pairs]
    observed = collect_observables(state, model)['routing_reciprocity_reference_mean_abs']
    assert abs(observed - sum(expected) / len(expected)) < 1e-12
    spatial = build_model(load_config(resolve_preset('tiny_spatial_3d', {})))
    counts = torch.bincount(spatial.incidence.context_id, minlength=spatial.domain.contexts.shape[0])
    assert counts[analytics_context(spatial)] == counts.max()
