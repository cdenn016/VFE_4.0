"""Numerical failures are reported, never repaired or converted into success."""

from dataclasses import dataclass

import torch

from .inference import StepReport
from .state import SectionState


@dataclass
class MonitorStatus:
    ok: bool
    message: str


class NumericalMonitor:
    def check(self, state: SectionState, report: StepReport) -> MonitorStatus:
        for channel, laws in (('belief', state.belief), ('model', state.model)):
            for bucket in laws.buckets:
                finite = torch.isfinite(bucket.mu).flatten(1).all(-1) & torch.isfinite(bucket.chol).flatten(1).all(-1)
                positive = (bucket.chol.diagonal(dim1=-2, dim2=-1) > 0).flatten(1).all(-1)
                bad = (~(finite & positive)).nonzero().flatten()
                if bad.numel():
                    agent = int(bucket.agent_ids[bad[0]])
                    raise ValueError(f'Numerical failure at sweep {state.sweep}, agent {agent}, channel {channel}')
        if not torch.isfinite(state.beta).all() or not torch.isfinite(state.log_pi).all():
            raise ValueError(f'Nonfinite source row at sweep {state.sweep}')
        return MonitorStatus(True, 'finite SPD factors')
