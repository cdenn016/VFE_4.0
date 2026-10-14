"""Proper Gaussian roots/bridges and normalized records (c04, c06, Supplement G)."""

from dataclasses import dataclass
import math
from typing import Optional

import torch

from .config import SimulationConfig
from .state import CandidateRows, CopyTable, Domain, Incidence, LinkState, LoopTable, SectionState
from .geometry.base import build_domain
from .geometry.sections import build_incidence, initialize_gaussians
from .geometry.supports import agent_lattice_support, build_copy_table, overlap_support
from .geometry.transport import ConstantFrameField, build_frame_field
from .observations import synthetic_truth


@dataclass
class GaussianRecords:
    kind: str
    channel: int
    receiver: torch.Tensor
    sender: torch.Tensor
    query: torch.Tensor             # (F,R,K)
    key: torch.Tensor               # (F,R,K), before independent twists
    observed: torch.Tensor          # (F,R)
    precision: torch.Tensor         # (F,R,R)
    candidate_id: torch.Tensor      # -1 for nonselective records

    def normalizer(self) -> torch.Tensor:
        return 0.5 * (self.query.shape[-2] * math.log(2 * math.pi) - torch.linalg.slogdet(self.precision)[1])


@dataclass
class SimulationModel:
    config: SimulationConfig
    domain: Domain
    incidence: Incidence
    frame_field: ConstantFrameField
    rows: CandidateRows
    copies: CopyTable
    loops: LoopTable
    channel_edge: torch.Tensor
    records: tuple[GaussianRecords, ...]
    initial_links: LinkState
    truth: torch.Tensor             # fixture sidecar, read only by diagnostics/output
    link_anchor: Optional[torch.Tensor] = None     # (C,K*K) generator prior mean alpha_e
    fixture_twist: Optional[torch.Tensor] = None   # (C,K*K) data-generating twist, sidecar only

    def __getstate__(self) -> dict:
        return {key: value for key, value in self.__dict__.items() if key != '_transport_cache'}

    def record_primitives(self, bank: GaussianRecords, state: SectionState) -> torch.Tensor:
        primitive = self.copies.primitive_id[self.rows.copy_id[bank.candidate_id]]
        count = state.links.omega_mean.shape[0]
        if not self.config['links']['shared_channels']:
            primitive = torch.where(primitive >= 0, primitive + bank.channel * (count // 2), primitive)
        return primitive

    def transport_moments(self, state: SectionState) -> tuple[torch.Tensor, torch.Tensor]:
        from .latent_links import sampled_generators
        links = state.links
        K = self.config['bundle']['K']
        R = K if self.config['communication']['comparison'] == 'full_state' else self.config['communication']['comparison_dim']
        omega = (sampled_generators(links, self.config['links']['samples']) if links.mode == 'latent_gaussian'
                 else links.omega_mean.unsqueeze(0))
        tensors = (links.omega_mean, links.precision, links.standard_normals)
        versions = tuple(None if x is None else x._version for x in tensors)
        cached = getattr(self, '_transport_cache', None)
        if cached is not None and all(a is b for a, b in zip(cached[0], tensors)) and cached[1:3] == (versions, links.mode):
            return cached[3], cached[4]
        G = torch.diag(omega.new_tensor([1 / self.config['communication']['tau']] * R + [0.] * (K - R)))
        mean, gram = [], []
        for start in range(0, omega.shape[1], self.config['links']['copy_chunk_size']):
            # Strided sample/copy slices must be contiguous for Torch's matrix exponential.
            matrices = omega[:, start:start + self.config['links']['copy_chunk_size']].reshape(omega.shape[0], -1, K, K)
            T = torch.matrix_exp(matrices.contiguous())  # (samples, copies_in_chunk, K, K)
            mean.append(T.mean(0))
            gram.append((T.mT @ G @ T).mean(0))
        mean_T = torch.cat(mean) if mean else omega.new_empty(0, K, K)
        mean_gram = torch.cat(gram) if gram else omega.new_empty(0, K, K)
        self._transport_cache = (tensors, versions, links.mode, mean_T, mean_gram)
        return mean_T, mean_gram

    def record_maps(self, bank: GaussianRecords, state: SectionState) -> tuple[torch.Tensor, torch.Tensor]:
        if bank.kind != 'communication':
            return bank.query, bank.key
        K = self.config['bundle']['K']
        primitive = self.record_primitives(bank, state)
        count = state.links.omega_mean.shape[0]
        if count:
            twist = self.transport_moments(state)[0][primitive.clamp_min(0)]
            twist = torch.where((primitive >= 0)[:, None, None], twist,
                                torch.eye(K, device=twist.device, dtype=twist.dtype))
        else:
            twist = torch.eye(K, device=bank.key.device, dtype=bank.key.dtype).expand(bank.key.shape[0], K, K)
        return bank.query, bank.key @ twist

    def record_statistics(self, bank: GaussianRecords, state: SectionState) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        A, B = self.record_maps(bank, state)
        gram = B.mT @ bank.precision @ B
        if bank.kind == 'communication' and state.links.mode == 'latent_gaussian' and state.links.omega_mean.numel():
            primitive = self.record_primitives(bank, state)
            gram = torch.where((primitive >= 0)[:, None, None],
                               self.transport_moments(state)[1][primitive.clamp_min(0)], gram)
        return A, B, gram

    def cross_covariance(self, state: SectionState, bank: GaussianRecords) -> torch.Tensor:
        K = self.config['bundle']['K']
        result = bank.query.new_zeros(bank.receiver.numel(), K, K)
        laws = state.belief if bank.channel == 0 else state.model
        src = bank.sender.clamp_min(0)
        same = (bank.sender >= 0) & (self.incidence.agent_id[bank.receiver] == self.incidence.agent_id[src])
        for bucket in laws.buckets:
            belongs = torch.isin(self.incidence.agent_id[bank.receiver], bucket.agent_ids) & same
            indices = belongs.nonzero().flatten()
            if not indices.numel():
                continue
            agents = self.incidence.agent_id[bank.receiver[indices]]
            position = torch.searchsorted(bucket.agent_ids, agents)
            r = self.incidence.local_node_id[bank.receiver[indices]]
            s = self.incidence.local_node_id[src[indices]]
            P = bucket.incidence_ids.shape[1]
            covariance = bucket.covariance().reshape(-1, P, K, P, K)
            result[indices] = covariance[position, r, :, s, :]
        return result

    def bank_costs(self, state: SectionState, bank: GaussianRecords) -> torch.Tensor:
        r"""E[-log N(o; A k_i-B k_j, Gamma)] with the complete residual covariance.

        The quadratic is E[r]^T Gamma^{-1} E[r] plus
        tr(Gamma^{-1} Cov(r)), including repeated variables and uncertain B.
        """
        K, S = self.config['bundle']['K'], self.incidence.agent_id.numel()
        laws = state.belief if bank.channel == 0 else state.model
        mu, Sigma = laws.marginals(S, K)
        A, B, gram = self.record_statistics(bank, state)
        source = bank.sender.clamp_min(0)
        delta = (A @ mu[bank.receiver, :, None] - B @ mu[source, :, None]).squeeze(-1) - bank.observed
        cross = self.cross_covariance(state, bank)
        covariance = A @ Sigma[bank.receiver] @ A.mT + B @ Sigma[source] @ B.mT
        covariance = covariance - A @ cross @ B.mT - B @ cross.mT @ A.mT
        quadratic = torch.einsum('fi,fij,fj->f', delta, bank.precision, delta)
        trace = torch.einsum('fij,fji->f', bank.precision, covariance)
        second_moment = Sigma[source] + mu[source, :, None] * mu[source, None, :]
        correction = torch.einsum('fij,fji->f', gram - B.mT @ bank.precision @ B, second_moment)
        return bank.normalizer() + 0.5 * (quadratic + trace + correction)

    def record_costs(self, state: SectionState) -> torch.Tensor:
        costs = state.beta.new_empty(state.beta.shape)
        for bank in self.records:
            if bank.kind == 'communication':
                costs[bank.candidate_id] = self.bank_costs(state, bank)
        return costs


def _spatial_pairs(domain: Domain, incidence: Incidence) -> tuple[torch.Tensor, torch.Tensor]:
    context, agents = incidence.context_id, incidence.agent_id
    S, M = context.numel(), domain.contexts.shape[0]
    keys = agents * M + context
    sorted_key, order = keys.sort()
    receivers, senders = [], []
    stride = M
    for n in domain.grid_shape:
        stride //= n
        coordinate = (context // stride) % n
        neighbor = context + ((coordinate + 1) % n - coordinate) * stride
        target = agents * M + neighbor
        pos = torch.searchsorted(sorted_key, target)
        safe = pos.clamp_max(S - 1)
        valid = (pos < S) & (sorted_key[safe] == target)
        if n == 2:
            valid &= context < neighbor
        senders.append(torch.arange(S, device=context.device)[valid])
        receivers.append(order[safe[valid]])
    return torch.cat(receivers), torch.cat(senders)


def build_model(config: SimulationConfig) -> SimulationModel:
    domain = build_domain(config)
    incidence = build_incidence(domain, config)
    frame_field = build_frame_field(domain, config)
    S, K, device, dtype = incidence.agent_id.numel(), config['bundle']['K'], config.device, config.dtype
    comm = config['communication']
    if comm['source_support'] == 'agent_lattice':
        support = agent_lattice_support(config['population']['n_agents'], tuple(config['population']['lattice_shape']),
            comm['prior_family'], comm['radius'], comm['decay'], comm['alpha'], device, dtype)
    else:
        support = overlap_support(incidence, domain, config)
    base_rows, copies, loops = build_copy_table(support, comm['include_self'])
    channel_numbers = [0 if name == 'belief' else 1 for name in comm['channels']]
    if not channel_numbers or any(name not in ('belief', 'model') for name in comm['channels']):
        raise ValueError('communication.channels must contain belief and/or model')
    E = base_rows.copy_id.numel()
    row_offsets = torch.cat([base_rows.row_offset[:-1] + i * E for i in range(len(channel_numbers))] +
                            [base_rows.row_offset.new_tensor([E * len(channel_numbers)])])
    rows = CandidateRows(row_offsets, base_rows.receiver_incidence.repeat(len(channel_numbers)),
        base_rows.source_incidence.repeat(len(channel_numbers)), base_rows.copy_id.repeat(len(channel_numbers)),
        base_rows.log_pi.repeat(len(channel_numbers)),
        torch.cat([base_rows.row_id + i * S for i in range(len(channel_numbers))]),
        torch.tensor(channel_numbers, device=device).repeat_interleave(S))
    channel_edge = rows.channel[rows.row_id]
    primitives = int((~copies.is_self).sum())
    if not config['links']['shared_channels']:
        primitives *= 2
    generator = torch.Generator(device=device).manual_seed(config['runtime']['seed'] + 2)
    # The fixture twist generates the records. Held mode declares it a known generative
    # parameter; latent recognition starts at the prior anchor and never reads it.
    fixture_twist = config['links']['initial_twist_std'] * torch.randn(primitives, K * K, device=device, dtype=dtype, generator=generator)
    anchor = fixture_twist.clone() if config['links']['prior_anchor'] == 'fixture_twist' else torch.zeros_like(fixture_twist)
    omega = fixture_twist.clone() if config['links']['mode'] == 'held' else anchor.clone()
    if config['links']['generator_precision'] == 'diagonal':
        precision = torch.full_like(omega, config['links']['prior_std']**-2)
    else:
        precision = torch.eye(K * K, device=device, dtype=dtype).expand(primitives, -1, -1).clone() / config['links']['prior_std']**2
    links = LinkState(config['links']['mode'], omega, precision, generator.get_state())
    truth = synthetic_truth(domain, incidence, config, generator)
    banks = []
    eye = torch.eye(K, device=device, dtype=dtype)
    selected = (torch.rand(S, device=device, generator=generator) < config['observations']['private_fraction']).nonzero().flatten()
    private_std = config['observations']['private_std']
    observed = truth[selected] + private_std * torch.randn(selected.numel(), K, device=device, dtype=dtype, generator=generator)
    banks.append(GaussianRecords('private', 0, selected, torch.full_like(selected, -1),
        eye.expand(selected.numel(), K, K), torch.zeros(selected.numel(), K, K, device=device, dtype=dtype),
        observed, (eye / private_std**2).expand(selected.numel(), K, K), torch.full_like(selected, -1)))
    if config['spatial']['enabled'] and domain.grid_shape:
        rec, src = _spatial_pairs(domain, incidence)
        for channel in (0, 1):
            banks.append(GaussianRecords('spatial', channel, rec, src, eye.expand(rec.numel(), K, K),
                eye.expand(rec.numel(), K, K), torch.zeros(rec.numel(), K, device=device, dtype=dtype),
                (eye / config['spatial']['noise_std']**2).expand(rec.numel(), K, K), torch.full_like(rec, -1)))
    R = K if comm['comparison'] == 'full_state' else comm['comparison_dim']
    if not isinstance(R, int) or R < 1 or R > K:
        raise ValueError('comparison_dim must be a positive integer no larger than K')
    comparison = eye[:R]
    for i, channel in enumerate(channel_numbers):
        ids = torch.arange(E, device=device) + i * E
        rec, src = rows.receiver_incidence[ids], rows.source_incidence[ids]
        A, B = comparison.expand(E, R, K), comparison.expand(E, R, K)
        bank = GaussianRecords('communication', channel, rec, src, A, B,
            torch.zeros(E, R, device=device, dtype=dtype),
            (torch.eye(R, device=device, dtype=dtype) / comm['tau']).expand(E, R, R), ids)
        # One secret source and one record per receiving row, broadcast across its candidates.
        gumbel = -torch.log(-torch.log(torch.rand(E, device=device, dtype=dtype, generator=generator)))
        score = base_rows.log_pi + gumbel
        maxima = torch.segment_reduce(score, 'max', lengths=base_rows.row_offset.diff())
        candidate = torch.arange(E, device=device)
        winners = torch.segment_reduce(torch.where(score == maxima[base_rows.row_id], candidate, E).to(dtype),
                                       'min', lengths=base_rows.row_offset.diff()).long()
        # Twist draws are known fixed model parameters in held mode, not hidden peer beliefs.
        fixed_links = LinkState('held', fixture_twist, precision, generator.get_state())
        temp = SectionState(None, None, rows.log_pi.exp(), rows.log_pi.clone(), fixed_links)
        _, actual_B = SimulationModel(config, domain, incidence, frame_field, rows, copies, loops,
                                     channel_edge, (), links, truth).record_maps(bank, temp)
        values = (A[winners] @ truth[rec[winners], :, None] - actual_B[winners] @ truth[src[winners], :, None]).squeeze(-1)
        values = values + comm['tau']**0.5 * torch.randn(S, R, device=device, dtype=dtype, generator=generator)
        bank.observed = values[base_rows.row_id]
        banks.append(bank)
    links.rng_state = generator.get_state()
    return SimulationModel(config, domain, incidence, frame_field, rows, copies, loops, channel_edge, tuple(banks), links, truth,
                           anchor, fixture_twist)


def initialize_sections(model: SimulationModel, config: SimulationConfig) -> SectionState:
    from copy import deepcopy
    std = config['generative']['model_prior_std']
    return SectionState(initialize_gaussians(model.incidence, config, (std**2 + config['generative']['bridge_std']**2)**0.5),
                        initialize_gaussians(model.incidence, config, std), model.rows.log_pi.exp().clone(),
                        model.rows.log_pi.clone(), deepcopy(model.initial_links))

