"""Report the configured finite design and conservative storage before CUDA allocation."""

from copy import deepcopy

import torch

from .config import SimulationConfig, load_config
from .geometry.base import build_domain
from .geometry.sections import build_incidence
from .geometry.supports import agent_lattice_support, overlap_support, build_copy_table


def storage_preflight(config: SimulationConfig) -> dict[str, int | float | str]:
    raw = deepcopy(config.raw)
    raw['runtime']['device'] = 'cpu'
    cpu = load_config(raw)
    domain = build_domain(cpu)
    incidence = build_incidence(domain, cpu)
    comm = cpu['communication']
    support = (agent_lattice_support(cpu['population']['n_agents'], tuple(cpu['population']['lattice_shape']),
                comm['prior_family'], comm['radius'], comm['decay'], comm['alpha'])
               if comm['source_support'] == 'agent_lattice' else overlap_support(incidence, domain, cpu))
    rows, copies, loops = build_copy_table(support, comm['include_self'])
    N, K, P = cpu['population']['n_agents'], cpu['bundle']['K'], int(incidence.patch_offset.diff().max())
    S, E = incidence.agent_id.numel(), rows.copy_id.numel() * len(comm['channels'])
    size = torch.empty((), dtype=cpu.dtype).element_size()
    patch_elements = N * (P * K)**2 if cpu['recognition']['covariance_structure'] == 'full_patch' else N * P * K * K
    work_elements = 4 * patch_elements + 8 * N * (P * K)**2 + 32 * E * K * K + 10 * S * K * K
    if cpu['links']['mode'] == 'latent_gaussian':
        C = int((~copies.is_self).sum()) * (1 if cpu['links']['shared_channels'] else 2)
        work_elements += 3 * cpu['links']['samples'] * C * K * K
        work_elements += 24 * cpu['links']['samples'] * min(C, cpu['links']['copy_chunk_size']) * K**4
    report = {'n_agents': N, 'incidences_S': S, 'candidate_edges_E_all_channels': E,
              'max_row_degree': int(rows.row_offset.diff().max()), 'max_patch_dimension': P * K,
              'covariance_structure': cpu['recognition']['covariance_structure'],
              'predicted_peak_bytes': work_elements * size, 'estimate': 'conservative work estimate, not measured peak',
              'configured_budget_bytes': int(cpu['runtime']['max_vram_gb'] * 1024**3)}
    return report
