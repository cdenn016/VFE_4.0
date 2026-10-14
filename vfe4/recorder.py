"""Detached scalar history; recording owns no model computation graph."""

from dataclasses import dataclass, field

from .inference import StepReport
from .objective import VFEReport


@dataclass
class RecorderState:
    rows: list[dict[str, float | int | str]] = field(default_factory=list)


class MetricRecorder:
    def __init__(self, state: RecorderState | None = None) -> None:
        self.state = RecorderState() if state is None else state

    def record(self, sweep: int, report: StepReport, vfe: VFEReport,
               observables: dict[str, float]) -> dict[str, float | int | str]:
        row: dict[str, float | int | str] = {'sweep': sweep, 'vfe_total': float(vfe.total.detach()),
            'normalizer_status': vfe.normalizer_status, 'estimator_error': vfe.estimator_error,
            'accepted': int(report.accepted), 'retries': report.retries,
            'objective_paired_before': report.objective_before, 'objective_paired_after': report.objective_after,
            'spd_status': report.min_spd_status}
        row.update({f'vfe_{name}': float(value.detach()) for name, value in vfe.terms.items()})
        row.update(observables)
        self.state.rows.append(row)
        return row
