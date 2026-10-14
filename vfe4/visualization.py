"""Publication artifacts in the fixture/reference frame, detached from inference."""

from pathlib import Path

import matplotlib
import numpy as np
import torch

from .model import SimulationModel
from .recorder import RecorderState
from .state import SectionState


def field_snapshot(state: SectionState, model: SimulationModel, channel: str = 'belief') -> dict[str, np.ndarray]:
    K, S, M = model.config['bundle']['K'], model.incidence.agent_id.numel(), model.domain.contexts.shape[0]
    mu, Sigma = getattr(state, channel).marginals(S, K)
    if not model.domain.grid_shape:
        return {'mean': mu[:, 0].detach().cpu().numpy(), 'truth': model.truth[:, 0].detach().cpu().numpy(),
                'std': Sigma[:, 0, 0].sqrt().detach().cpu().numpy()}
    count = torch.bincount(model.incidence.context_id, minlength=M).to(mu.dtype)
    mean = mu.new_zeros(M).index_add_(0, model.incidence.context_id, mu[:, 0]) / count.clamp_min(1)
    truth = mu.new_zeros(M).index_add_(0, model.incidence.context_id, model.truth[:, 0]) / count.clamp_min(1)
    variance = mu.new_zeros(M).index_add_(0, model.incidence.context_id, Sigma[:, 0, 0]) / count.clamp_min(1)
    mean[count == 0], truth[count == 0], variance[count == 0] = torch.nan, torch.nan, torch.nan
    shape = model.domain.grid_shape
    return {'mean': mean.reshape(shape).detach().cpu().numpy(), 'truth': truth.reshape(shape).detach().cpu().numpy(),
            'std': variance.sqrt().reshape(shape).detach().cpu().numpy()}


def render_run(run_dir: Path, model: SimulationModel, state: SectionState, recorder: RecorderState) -> list[Path]:
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 300})
    directory = run_dir / 'figures'
    directory.mkdir(exist_ok=True)
    paths = []

    def save(fig: object, name: str) -> None:
        for extension in ('png', 'pdf', 'svg'):
            path = directory / f'{name}.{extension}'
            fig.savefig(path, bbox_inches='tight')
            paths.append(path)
        plt.close(fig)

    rows = recorder.rows
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    x = [row['sweep'] for row in rows]
    for term in [key for key in rows[0] if key.startswith('vfe_')]:
        ax.plot(x, [row.get(term, np.nan) for row in rows], label=term.removeprefix('vfe_').replace('_', ' '))
    label = ('Relative sampled VFE contribution (nats; log Z omitted)' if state.links.mode == 'latent_gaussian' and model.config['links']['mutual_loop_coefficient'] and model.loops.weight.numel()
             else 'Sampled VFE contribution (nats)' if state.links.mode == 'latent_gaussian' else 'Collective VFE contribution (nats)')
    ax.set(xlabel='Inference sweep', ylabel=label)
    ax.legend(fontsize=8, ncol=2)
    save(fig, 'collective-vfe')
    data = field_snapshot(state, model)
    truth, mean, std = data['truth'], data['mean'], data['std']
    if mean.ndim == 3:
        section = mean.shape[2] // 2
        mean, truth, std = mean[:, :, section], truth[:, :, section], std[:, :, section]
        title = f'Reference field component 0; z index {section}'
    else:
        title = 'Reference field component 0'
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.5), constrained_layout=True)
    field_limit = max(float(np.nanmax(np.abs(truth))), float(np.nanmax(np.abs(mean))), 1e-12)
    for ax, image, label in zip(axes, (truth, mean, mean - truth), ('Truth fixture', 'Recognition mean', 'Error')):
        if image.ndim == 1:
            ax.plot(np.arange(image.size), image, linewidth=1)
            ax.set_xlabel('Agent index')
        else:
            limit = max(float(np.nanmax(np.abs(image))), 1e-12) if label == 'Error' else field_limit
            im = ax.imshow(image.T, origin='lower', extent=(0, float(model.domain.periods[0]), 0, float(model.domain.periods[1])), cmap='coolwarm', vmin=-limit, vmax=limit)
            fig.colorbar(im, ax=ax, shrink=0.75)
            ax.set(xlabel='Context axis 0', ylabel='Context axis 1')
        ax.set_title(label)
    fig.suptitle(title)
    save(fig, 'field-error')
    fig, ax = plt.subplots(figsize=(5.3, 4.0))
    if std.ndim == 1:
        ax.plot(std)
        ax.set(xlabel='Agent index', ylabel='Posterior standard deviation')
    else:
        im = ax.imshow(std.T, origin='lower', cmap='viridis')
        fig.colorbar(im, ax=ax, label='Marginal standard deviation')
        ax.set(xlabel='Context grid axis 0', ylabel='Context grid axis 1')
    ax.set_title('Uncertainty in the reference frame, component 0')
    save(fig, 'uncertainty-reference')
    from .analytics.attention import row_statistics
    from .analytics.coherence import coherent_partition
    statistics = row_statistics(state.beta, model.rows)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), constrained_layout=True)
    for ax, key in zip(axes, ('row_entropy', 'effective_degree')):
        values = statistics[key].detach().cpu().numpy()
        ax.hist(values, bins=min(30, max(3, len(values))), color='#2f6384')
        ax.set(xlabel=key.replace('_', ' ').capitalize(), ylabel='Receiving rows')
    save(fig, 'source-row-distributions')
    if model.domain.grid_shape:
        fig, ax = plt.subplots(figsize=(5.5, 4.1))
        selected_agents = range(min(8, model.config['population']['n_agents']))
        for agent in selected_agents:
            context = model.incidence.context_id[model.incidence.agent_id == agent]
            points = model.domain.contexts[context].detach().cpu().numpy()
            ax.scatter(points[:, 0], points[:, 1], s=12, alpha=0.6, label=str(agent))
        ax.set(xlabel='Context axis 0', ylabel='Context axis 1', title='Supports of the first eight agents (xy projection)')
        ax.legend(title='Physical agent', fontsize=7, ncol=2)
        save(fig, 'agent-supports')
    bank = next((b for b in model.records if b.kind == 'communication' and b.channel == 0), None)
    if bank is not None:
        vertices = (model.incidence.context_id == 0).nonzero().flatten()
        selected = model.incidence.context_id[bank.receiver] == 0
        rec, src = bank.receiver[selected], bank.sender[selected]
        local_r, local_s = torch.searchsorted(vertices, rec), torch.searchsorted(vertices, src)
        weights = state.beta[bank.candidate_id[selected]]
        agents = model.incidence.agent_id[vertices]
        if model.domain.grid_shape:
            positions = model.incidence.centers[agents, :2].detach().cpu().numpy()
        else:
            shape = model.config['population']['lattice_shape']
            a = agents.detach().cpu().numpy()
            positions = np.column_stack((a // shape[-1], a % shape[-1])) if len(shape) > 1 else np.column_stack((a, np.zeros_like(a)))
        A, B = model.record_maps(bank, state)
        all_mu = state.belief.marginals(model.incidence.agent_id.numel(), model.config['bundle']['K'])[0]
        norm = torch.linalg.vector_norm((A[selected] @ all_mu[rec, :, None] - B[selected] @ all_mu[src, :, None]).squeeze(-1), dim=-1)
        fig, ax = plt.subplots(figsize=(6.4, 4.3))
        for i in range(len(vertices)):
            candidates = (local_r == i).nonzero().flatten()
            if len(candidates):
                winner = candidates[weights[candidates].argmax()]
                j = int(local_s[winner])
                if i != j:
                    ax.annotate('', xy=positions[i], xytext=positions[j], arrowprops={'arrowstyle': '->', 'color': '#697d89', 'alpha': 0.4, 'lw': 0.7})
        ax.scatter(positions[:, 0], positions[:, 1], s=20, color='#224f69')
        ax.set(title='Most probable belief source per row; context 0', xlabel='Agent center/lattice axis 0', ylabel='Agent center/lattice axis 1')
        save(fig, 'selected-routing-context0')
        for resolution in model.config['analytics']['coherence_resolutions']:
            labels = coherent_partition(norm, local_s, local_r, resolution, len(vertices)).detach().cpu().numpy()
            fig, ax = plt.subplots(figsize=(6.4, 4.3))
            ax.scatter(positions[:, 0], positions[:, 1], c=labels, cmap='tab20', s=25)
            ax.set(title=f'Mean-field coherent blocks, context 0; resolution {resolution:g}', xlabel='Agent center/lattice axis 0', ylabel='Agent center/lattice axis 1')
            ax.title.set_fontsize(10)
            save(fig, f'coherent-blocks-context0-a{resolution:g}')
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), constrained_layout=True)
    for ax, key, label in zip(axes, ('fixed_sector_rank_context0', 'holonomy_mean_defect_reference'),
                              ('Fixed-sector rank, context 0 (held links)', 'Mean based-loop defect, reference metric')):
        ax.plot(x, [row.get(key, np.nan) for row in rows], marker='.', markersize=3)
        ax.set(xlabel='Inference sweep', ylabel=label)
    save(fig, 'transport-observables')
    return paths


def render_aggregate(directory: Path, data: dict[str, list], x_key: str, metric: str) -> list[Path]:
    """Seed-level mean and bootstrap interval for one explicitly selected model."""
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(data[x_key], data['mean'], color='#285f80')
    ax.fill_between(data[x_key], data['ci_low'], data['ci_high'], alpha=0.25, color='#285f80')
    ax.set(xlabel=x_key.replace('_', ' ').capitalize(), ylabel=metric.replace('_', ' '), title='Seed mean and 95% bootstrap interval')
    paths = []
    for extension in ('png', 'pdf', 'svg'):
        path = directory / f'aggregate-{metric}.{extension}'
        fig.savefig(path, bbox_inches='tight', dpi=300)
        paths.append(path)
    plt.close(fig)
    return paths
