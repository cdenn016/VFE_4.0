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
