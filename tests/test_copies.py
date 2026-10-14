import torch

from vfe4.geometry.supports import agent_lattice_support, build_copy_table


def test_window_prior_and_independent_mutual_copies() -> None:
    support = agent_lattice_support(8, (8,), 'window', 2, None, None)
    degrees = support.row_offset.diff().tolist()
    assert degrees == [4] * 8
    for row in torch.split(support.log_pi.exp(), degrees):
        torch.testing.assert_close(row, torch.full((4,), 0.25, dtype=row.dtype))
    support = agent_lattice_support(4, (4,), 'window', 1, None, None)
    rows, copies, loops = build_copy_table(support, True)
    a = (copies.receiver_agent == 0) & (copies.source_agent == 1)
    b = (copies.receiver_agent == 1) & (copies.source_agent == 0)
    assert copies.primitive_id[a].item() != copies.primitive_id[b].item()
    assert (copies.primitive_id[copies.is_self] == -1).all()
    assert loops.offset.numel() - 1 == 8
    for row in torch.split(rows.log_pi.exp(), rows.row_offset.diff().tolist()):
        torch.testing.assert_close(row.sum(), torch.tensor(1.0, dtype=row.dtype))


def test_window_stencil_matches_all_pairs() -> None:
    from math import prod
    for shape in ((3,), (4,), (5, 4), (3, 2, 5)):
        n = prod(shape)
        coords = torch.stack(torch.meshgrid(*[torch.arange(k) for k in shape], indexing='ij'), -1).reshape(n, -1)
        size = torch.tensor(shape)
        delta = torch.remainder(coords[None] - coords[:, None] + size // 2, size) - size // 2
        distance = delta.abs().sum(-1)
        for family in ('window', 'directed_window'):
            for radius in (1, 2):
                mask = (distance > 0) & (distance <= radius)
                if family == 'directed_window':
                    first = (delta != 0).long().argmax(-1)
                    mask &= delta.gather(-1, first.unsqueeze(-1)).squeeze(-1) > 0
                receiver, source = mask.nonzero(as_tuple=True)
                support = agent_lattice_support(n, shape, family, radius, None, None)
                assert torch.equal(support.receiver_agent, receiver) and torch.equal(support.source_agent, source)


def test_spatial_decay_is_per_grid_spacing() -> None:
    from vfe4.config import load_config
    from vfe4.presets import resolve_preset
    from vfe4.geometry.base import build_domain
    from vfe4.geometry.sections import build_incidence
    from vfe4.geometry.supports import overlap_support
    config = load_config(resolve_preset('tiny_spatial_2d', {'communication': {'decay': 1.0}}))
    domain = build_domain(config)
    incidence = build_incidence(domain, config)
    support = overlap_support(incidence, domain, config)
    distance = domain.manifold.distance(incidence.centers[support.receiver_agent], incidence.centers[support.source_agent])
    torch.testing.assert_close(support.log_weights, -distance / domain.manifold.spacing.min())
