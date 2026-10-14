"""Click-to-run orchestration and same-model checkpoint continuation."""

from dataclasses import dataclass
from pathlib import Path
import csv
import os
import subprocess
import time

import torch

from .config import SimulationConfig
from .inference import make_schedule, step, StepReport
from .io import RunWriter, load_checkpoint, save_checkpoint
from .model import SimulationModel, build_model, initialize_sections
from .monitor import NumericalMonitor
from .objective import evaluate_vfe
from .recorder import MetricRecorder, RecorderState
from .state import SectionState


@dataclass
class RunResult:
    run_directory: Path
    final_state: SectionState
    completed_sweeps: int


def ensure_cuda_capacity(config: SimulationConfig) -> None:
    if config.device.type != 'cuda':
        return
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    query = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, check=True, creationflags=flags)
    graphics = ('brave.exe', 'chrome.exe', 'msedge.exe', 'firefox.exe', 'dwm.exe', 'codex.exe')
    processes = []
    for row in csv.reader(query.stdout.splitlines()):
        if len(row) != 2:
            continue
        pid, process_name = (item.strip() for item in row)
        if pid == str(os.getpid()):
            continue
        if process_name == '[Insufficient Permissions]' and os.name == 'nt':
            detail = subprocess.run(['tasklist.exe', '/FI', f'PID eq {int(pid)}', '/FO', 'CSV', '/NH'],
                                    capture_output=True, text=True, creationflags=flags)
            entries = list(csv.reader(detail.stdout.splitlines()))
            if entries and len(entries[0]) > 1 and entries[0][1].strip() == pid:
                process_name = entries[0][0]
        if Path(process_name).name.lower() not in graphics:
            processes.append(f'{pid}, {process_name}')
    if processes:
        raise RuntimeError('Other GPU compute processes are active; CUDA run deferred: ' + '; '.join(processes))
    free, total = torch.cuda.mem_get_info(config.device)
    from .preflight import storage_preflight
    estimate = storage_preflight(config)['predicted_peak_bytes']
    budget = min(free, int(config['runtime']['max_vram_gb'] * 1024**3))
    if estimate > budget:
        raise MemoryError(f'Local Gaussian work estimate {estimate} exceeds available/configured CUDA budget {budget}')


def _execute(model: SimulationModel, state: SectionState, writer: RunWriter,
             recorder: MetricRecorder, sweeps: int) -> RunResult:
    config = model.config
    monitor = NumericalMonitor()
    schedule = make_schedule(model, config)
    start = time.perf_counter()
    analytics_cache: dict[str, float] = {}
    from .analytics.observables import collect_observables
    from .latent_links import independent_diagnostics
    viewer = None
    stopped = False
    if config['output']['viewer']:
        from .viewer import SimulationViewer
        viewer = SimulationViewer()
    try:
        if not recorder.state.rows:
            initial = evaluate_vfe(state, model)
            value = float(initial.total)
            recorder.record(state.sweep, StepReport(value, value, True), initial,
                            {**collect_observables(state, model, analytics_cache), **independent_diagnostics(state, model)})
        for _ in range(sweeps):
            if viewer is not None and not viewer.ready():
                stopped = True
                break
            sweep_start = time.perf_counter()
            state, report = step(state, model, schedule, config)
            monitor.check(state, report)
            observables = ({**collect_observables(state, model, analytics_cache), **independent_diagnostics(state, model)}
                           if state.sweep % config['analytics']['every'] == 0 else {})
            observables['sweep_seconds'] = time.perf_counter() - sweep_start
            recorder.record(state.sweep, report, evaluate_vfe(state, model), observables)
            if viewer is not None and (state.sweep % config['output']['snapshot_every'] == 0 or viewer.paused):
                viewer.update(state, model, report.objective_after)
            if state.sweep % config['output']['checkpoint_every'] == 0:
                save_checkpoint(writer.run_dir / 'checkpoint.pt', model, state, recorder.state)
                writer.write_metrics(recorder.state)
        writer.write_metrics(recorder.state)
        save_checkpoint(writer.run_dir / 'final_state.pt', model, state, recorder.state)
        memory = ({'peak_allocated_bytes': torch.cuda.max_memory_allocated(config.device),
                   'peak_reserved_bytes': torch.cuda.max_memory_reserved(config.device)} if config.device.type == 'cuda' else {})
        writer.summary(status='stopped' if stopped else 'completed', sweeps=state.sweep, elapsed_seconds=time.perf_counter() - start,
                       objective=float(evaluate_vfe(state, model).total),
                       normalizer_status=evaluate_vfe(state, model).normalizer_status, **memory)
        if config['output']['figures']:
            from .visualization import render_run
            render_run(writer.run_dir, model, state, recorder.state)
    except Exception as error:
        writer.write_metrics(recorder.state)
        writer.summary(status='failed', sweep=state.sweep, error=str(error), error_type=type(error).__name__)
        save_checkpoint(writer.run_dir / 'failed_state.pt', model, state, recorder.state)
        raise
    return RunResult(writer.run_dir, state, state.sweep)


def run_simulation(config: SimulationConfig) -> RunResult:
    ensure_cuda_capacity(config)
    if config.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(config.device)
    model = build_model(config)
    state = initialize_sections(model, config)
    writer = RunWriter.create(config)
    from .io import write_json
    from .preflight import storage_preflight
    preflight = storage_preflight(config)
    if config.device.type == 'cuda':
        free, total = torch.cuda.mem_get_info(config.device)
        preflight.update(free_cuda_bytes=free, total_cuda_bytes=total)
    write_json(writer.run_dir / 'preflight.json', preflight)
    writer.save_model(model)
    return _execute(model, state, writer, MetricRecorder(), config['inference']['sweeps'])


def resume_simulation(path: Path, additional_sweeps: int) -> RunResult:
    model, state, recorder = load_checkpoint(path)
    ensure_cuda_capacity(model.config)
    writer = RunWriter.create(model.config)
    writer.save_model(model)
    return _execute(model, state, writer, MetricRecorder(recorder), additional_sweeps)
