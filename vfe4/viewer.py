"""Inspect captured sections, spatial fields, and one source row."""

import numpy as np

from .dashboard import DashboardSnapshot, capture_dashboard, reference_image, routing_view
from .model import SimulationModel
from .section_view import SectionImage, section_image
from .state import SectionState


class SimulationViewer:
    def __init__(self, inspection_only: bool = False) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, RadioButtons, TextBox
        self.plt = plt
        self.figure, plots = plt.subplots(2, 3, figsize=(16, 9.5))
        self.axes = [plots[0, 0], plots[0, 1]]
        self.population_ax, self.reference_axes = plots[0, 2], plots[1]
        self.figure.subplots_adjust(left=0.25, right=0.96, bottom=0.25, top=0.87,
                                    wspace=0.5, hspace=0.75)
        self.inspection_only = inspection_only
        self.paused, self.stopped, self.once = inspection_only, False, False
        self.history: list[tuple[int, float]] = []
        self.snapshot: DashboardSnapshot | None = None
        self.last_section_image: SectionImage | None = None
        self.last_reference_image = None
        self.last_routing = None
        self._model: SimulationModel | None = None
        self._colorbar = None
        self._reference_colorbars = [None, None, None]
        self._routing_colorbar = None
        self.selection_agent, self.selection_component, self.selection_slice = 0, 0, 0
        self.selection_context = 0
        self.selection_channel, self.selection_statistic = 'belief', 'mean'
        self.agent_input = TextBox(self.figure.add_axes([0.085, 0.77, 0.105, 0.045]), 'Agent', initial='0')
        self.component_input = TextBox(self.figure.add_axes([0.085, 0.66, 0.105, 0.045]), 'Component', initial='0')
        self.agent_input.label.set_fontsize(9)
        self.component_input.label.set_fontsize(9)
        self.agent_range = self.figure.text(0.085, 0.745, '', fontsize=8)
        self.component_range = self.figure.text(0.085, 0.635, '', fontsize=8)
        self.context_input = TextBox(self.figure.add_axes([0.31, 0.13, 0.065, 0.035]), 'Context ID', initial='0')
        self.context_range = self.figure.text(0.385, 0.14, '', fontsize=8)
        self.channel_buttons = RadioButtons(self.figure.add_axes([0.025, 0.46, 0.165, 0.125]), ('belief', 'model'))
        self.statistic_buttons = RadioButtons(self.figure.add_axes([0.025, 0.255, 0.165, 0.135]), ('Mean', 'Std. deviation'))
        self.channel_buttons.ax.set_title('Channel', fontsize=10)
        self.statistic_buttons.ax.set_title('Display', fontsize=10)
        for control in (self.channel_buttons, self.statistic_buttons):
            for label in control.labels:
                label.set_fontsize(9)
        self.slice_slider = None
        self.figure.text(0.025, 0.21, 'Gray images = uncovered\nGray centers = absent at context\nClick a center to select an agent', fontsize=8)
        self.status_text = self.figure.text(0.025, 0.15, '', color='firebrick', fontsize=8)
        self.agent_input.on_submit(self._select_agent)
        self.component_input.on_submit(self._select_component)
        self.context_input.on_submit(self._select_context)
        self.channel_buttons.on_clicked(self._select_channel)
        self.statistic_buttons.on_clicked(self._select_statistic)
        self.buttons = []
        for index, (label, action) in enumerate((('Pause/resume', self.toggle), ('One sweep', self.single), ('Stop', self.stop))):
            button = Button(self.figure.add_axes([0.27 + index * 0.23, 0.055, 0.21, 0.065]), label)
            button.on_clicked(action)
            self.buttons.append(button)
        if inspection_only:
            for button in self.buttons[:2]:
                button.set_active(False)
                button.label.set_color('0.6')
            self.buttons[2].label.set_text('Close')
        self.figure.canvas.mpl_connect('close_event', self.stop)
        self.figure.canvas.mpl_connect('button_press_event', self._click_population)
        plt.show(block=False)

    def toggle(self, event: object) -> None:
        if not self.inspection_only:
            self.paused = not self.paused

    def single(self, event: object) -> None:
        if not self.inspection_only:
            self.paused, self.once = True, True

    def stop(self, event: object) -> None:
        if self.stopped:
            return
        self.stopped = True
        if self.inspection_only:
            self.plt.close(self.figure)

    def ready(self) -> bool:
        if self.inspection_only:
            return False
        while self.paused and not self.once and not self.stopped:
            self.plt.pause(0.05)
        self.once = False
        return not self.stopped

    def _select_integer(self, text: str, name: str, limit: int) -> bool:
        try:
            value = int(text)
            if not 0 <= value < limit:
                raise ValueError
        except ValueError:
            self.status_text.set_text(f'{name.capitalize()} must be 0..{limit - 1}')
            self.figure.canvas.draw_idle()
            return False
        setattr(self, f'selection_{name}', value)
        self.status_text.set_text('')
        return True

    def _select_agent(self, text: str) -> None:
        if self._model is not None and self._select_integer(text, 'agent', self._model.config['population']['n_agents']):
            self._center_slice()
            self._context_near_agent()
            self._redraw()

    def _select_component(self, text: str) -> None:
        if self._model is not None and self._select_integer(text, 'component', self._model.config['bundle']['K']):
            self._redraw()

    def _select_channel(self, label: str) -> None:
        self.selection_channel = label
        self._redraw()

    def _select_statistic(self, label: str) -> None:
        self.selection_statistic = 'mean' if label == 'Mean' else 'std'
        self._draw_section()

    def _select_slice(self, value: float) -> None:
        self.selection_slice = int(value)
        self._context_near_agent()
        self._redraw()

    def _set_context_text(self) -> None:
        self.context_input.eventson = False
        self.context_input.set_val(str(self.selection_context))
        self.context_input.eventson = True

    def _context_near_agent(self) -> None:
        model = self._model
        if not model.domain.grid_shape:
            self.selection_context = 0
        else:
            point = self.population_positions[self.selection_agent].copy()
            if len(model.domain.grid_shape) == 3:
                point[2] = self.selection_slice * self._periods[2] / (2 * model.domain.grid_shape[2])
            indices = np.floor(point * np.array(model.domain.grid_shape) / self._periods + 0.5).astype(int)
            self.selection_context = int(np.ravel_multi_index(indices % model.domain.grid_shape,
                                                              model.domain.grid_shape))
        self._set_context_text()

    def _select_context(self, text: str) -> None:
        if self._model is None or not self._select_integer(text, 'context', len(self._contexts)):
            return
        if self.slice_slider is not None:
            self.selection_slice = 2 * int(np.unravel_index(self.selection_context, self._model.domain.grid_shape)[2])
            self.slice_slider.eventson = False
            self.slice_slider.set_val(self.selection_slice)
            self.slice_slider.eventson = True
        self._redraw()

    def _click_population(self, event: object) -> None:
        if self._model is None or event.inaxes is not self.population_ax or event.button != 1:
            return
        point = np.array([event.xdata, event.ydata], dtype=float)
        if self._model.domain.grid_shape:
            if len(self._periods) == 3:
                point = np.append(point, self.selection_slice * self._periods[2] /
                                  (2 * self._model.domain.grid_shape[2]))
            delta = (self.population_positions - point + self._periods / 2) % self._periods - self._periods / 2
        else:
            delta = self.population_positions - point
        self.agent_input.set_val(str(int(np.argmin(np.sum(delta**2, axis=1)))))

    def _center_slice(self) -> None:
        if self.slice_slider is not None:
            model = self._model
            count = 2 * model.domain.grid_shape[2]
            self.selection_slice = round(float(model.incidence.centers[self.selection_agent, 2]) * count /
                                         float(model.domain.periods[2])) % count
            self.slice_slider.eventson = False
            self.slice_slider.set_val(self.selection_slice)
            self.slice_slider.eventson = True

    def _draw_section(self) -> None:
        if self.snapshot is None:
            return
        image = section_image(self.snapshot, self._model, self.selection_agent, self.selection_channel,
                              self.selection_component, self.selection_statistic, self.selection_slice)
        self.last_section_image = image
        ax = self.axes[1]
        ax.clear()
        label = 'Mean' if self.selection_statistic == 'mean' else 'Std. deviation'
        ax.set_title(f'Agent {self.selection_agent}, {self.selection_channel}, component {self.selection_component}\n'
                     f'{label} (agent frame)' + (f'; z = {image.z:g}' if image.z is not None else ''), fontsize=10)
        if image.extent is None:
            ax.plot([0], image.values, marker='o')
            ax.set(xticks=[0], xticklabels=['c*'], xlabel='Base context', ylabel=f'{label} (agent frame)')
        else:
            cmap = self.plt.get_cmap('coolwarm' if self.selection_statistic == 'mean' else 'viridis').copy()
            cmap.set_bad('0.85')
            valid = image.values.compressed()
            scale = max(float(abs(valid).max()), 1e-12) if valid.size else 1.0
            artist = ax.imshow(image.values.T, origin='lower', extent=image.extent, cmap=cmap,
                               vmin=-scale if self.selection_statistic == 'mean' else 0, vmax=scale)
            ax.set(xlabel='Base x', ylabel='Base y')
            if not valid.size:
                ax.text(0.5, 0.5, 'No support on this slice', transform=ax.transAxes, ha='center', va='center')
            if self._colorbar is None:
                self._colorbar = self.figure.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)
            else:
                self._colorbar.update_normal(artist)
            self._colorbar.set_label(label, fontsize=9)
        self.figure.canvas.draw_idle()

    def _draw_reference(self) -> None:
        data = reference_image(self.snapshot, self._model, self.selection_channel,
                               self.selection_component, self.selection_slice)
        self.last_reference_image = data
        labels = ('Truth fixture', 'Population mean', 'Mean minus truth')
        values = (data.truth, data.mean, data.error)
        shared = np.concatenate([x.compressed() for x in values[:2]])
        shared_limit = max(float(np.abs(shared).max()), 1e-12) if shared.size else 1.
        suffix = f'\nz={data.z:g}' if data.z is not None else ''
        if data.z is not None and self.selection_slice % 2:
            suffix += ' (nodal interpolant)'
        for index, (ax, value, label) in enumerate(zip(self.reference_axes, values, labels)):
            ax.clear()
            ax.set_title(f'{label} (reference)\n{self.selection_channel}, component {self.selection_component}' + suffix, fontsize=9)
            if data.extent is None:
                ax.plot(np.arange(value.size), value, marker='o', markersize=3)
                ax.set(xlabel='Agent index', ylabel='Reference component')
            else:
                supported = value.compressed()
                limit = (max(float(np.abs(supported).max()), 1e-12) if supported.size else 1.) if index == 2 else shared_limit
                cmap = self.plt.get_cmap('coolwarm').copy()
                cmap.set_bad('0.85')
                artist = ax.imshow(value.T, origin='lower', extent=data.extent, cmap=cmap, vmin=-limit, vmax=limit)
                ax.set(xlabel='Base x', ylabel='Base y')
                if self._reference_colorbars[index] is None:
                    self._reference_colorbars[index] = self.figure.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)
                else:
                    self._reference_colorbars[index].update_normal(artist)
                self._reference_colorbars[index].ax.tick_params(labelsize=8)

    def _draw_population(self) -> None:
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        ax, model = self.population_ax, self._model
        ax.clear()
        positions = self.population_positions[:, :2]
        active = np.unique(self._incidence_agents[self._incidence_contexts == self.selection_context])
        ax.scatter(*positions.T, s=12, c='0.8', zorder=2)
        ax.scatter(*positions[active].T, s=17, c='#287d91', zorder=3)
        support = self._contexts[self._incidence_contexts[self._incidence_agents == self.selection_agent]]
        if model.domain.grid_shape:
            if len(model.domain.grid_shape) == 3:
                support = support[np.isclose(support[:, 2], self._contexts[self.selection_context, 2])]
            ax.scatter(*support[:, :2].T, marker='s', s=22, c='#bedde2', alpha=0.7, zorder=1)
            ax.scatter(*self._contexts[self.selection_context, :2], marker='x', s=45, c='black', zorder=5)
        routing = routing_view(self.snapshot, model, self.selection_agent, self.selection_channel, self.selection_context)
        self.last_routing = routing
        cmap = self.plt.get_cmap('viridis')
        for source, probability in zip(routing.source_agents, routing.probabilities):
            if source == self.selection_agent:
                continue
            start, end = positions[source], positions[self.selection_agent]
            if model.domain.grid_shape:
                delta = (end - start + self._periods[:2] / 2) % self._periods[:2] - self._periods[:2] / 2
                shifts = np.array([(x, y) for x in (-1, 0, 1) for y in (-1, 0, 1)]) * self._periods[:2]
            else:
                delta, shifts = end - start, np.zeros((1, 2))
            for shift in shifts:
                arrow = ax.annotate('', xy=start + delta + shift, xytext=start + shift,
                                    arrowprops={'arrowstyle': '->', 'color': cmap(float(probability)),
                                                'lw': 0.4 + 3 * float(probability)}, annotation_clip=False)
                arrow.arrow_patch.set_clip_path(ax.patch)
        ax.scatter(*positions[self.selection_agent], s=95, facecolors='none', edgecolors='#c63632', linewidths=1.5, zorder=6)
        self_weight = routing.probabilities[routing.source_agents == self.selection_agent].sum()
        status = f'self={self_weight:.3f}' if routing.status == 'active' else routing.status.replace('_', ' ')
        projection = 'xy projection' if len(model.domain.grid_shape) == 3 else 'centers'
        ax.set_title(f'{len(positions)} agents: {projection}\nAgent {self.selection_agent}, {self.selection_channel}, context {self.selection_context}\nIncoming source probabilities; {status}', fontsize=9)
        if model.domain.grid_shape:
            ax.set(xlim=(-0.02 * self._periods[0], 1.02 * self._periods[0]),
                   ylim=(-0.02 * self._periods[1], 1.02 * self._periods[1]), xlabel='Base x', ylabel='Base y', aspect='equal')
        else:
            ax.set(xlabel='Agent lattice axis 0', ylabel='Agent lattice axis 1')
            ax.margins(0.15)
        if self._routing_colorbar is None:
            self._routing_colorbar = self.figure.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=cmap),
                                                         ax=ax, fraction=0.046, pad=0.04)
            self._routing_colorbar.set_label('Source probability', fontsize=8)
            self._routing_colorbar.ax.tick_params(labelsize=8)

    def _redraw(self) -> None:
        if self.snapshot is not None:
            self._draw_section()
            self._draw_reference()
            self._draw_population()
            self.figure.canvas.draw_idle()

    def update(self, state: SectionState, model: SimulationModel, objective: float | None,
               recorded_history: list[tuple[int, float]] | None = None) -> None:
        self.snapshot = capture_dashboard(state)
        if self._model is None:
            self._model = model
            self._contexts = model.domain.contexts.detach().cpu().numpy()
            self._periods = np.array(model.config['manifold']['periods'])
            self._incidence_agents = model.incidence.agent_id.detach().cpu().numpy()
            self._incidence_contexts = model.incidence.context_id.detach().cpu().numpy()
            if model.domain.grid_shape:
                self.population_positions = model.incidence.centers.detach().cpu().numpy()
            else:
                agents = np.arange(model.config['population']['n_agents'])
                shape = model.config['population']['lattice_shape']
                self.population_positions = (np.column_stack((agents // shape[-1], agents % shape[-1])) if len(shape) > 1
                                             else np.column_stack((agents, np.zeros_like(agents))))
            self.agent_range.set_text(f'0..{model.config["population"]["n_agents"] - 1}')
            self.component_range.set_text(f'0..{model.config["bundle"]["K"] - 1}')
            self.context_range.set_text(f'0..{len(self._contexts) - 1}')
            if len(model.domain.grid_shape) == 3:
                from matplotlib.widgets import Slider
                self.slice_slider = Slider(self.figure.add_axes([0.67, 0.135, 0.235, 0.025]), 'z slice',
                                           0, 2 * model.domain.grid_shape[2] - 1, valinit=0, valstep=1, valfmt='%d')
                self.slice_slider.on_changed(self._select_slice)
                self._center_slice()
            self._context_near_agent()
        if recorded_history is not None:
            self.history = list(recorded_history)
        elif objective is not None:
            self.history.append((state.sweep, objective))
        self.axes[0].clear()
        if self.history:
            self.axes[0].plot(*zip(*self.history), marker='o', markersize=3)
        else:
            self.axes[0].text(0.5, 0.5, 'Saved VFE history unavailable', transform=self.axes[0].transAxes,
                              ha='center', va='center')
        label = 'Collective VFE (nats)'
        if state.links.mode == 'latent_gaussian':
            relative = model.config['links']['mutual_loop_coefficient'] and model.loops.weight.numel()
            label = 'Relative sampled VFE (nats; log Z omitted)' if relative else 'Sampled VFE (nats)'
        self.axes[0].set(xlabel='Inference sweep', ylabel=label)
        self.figure.suptitle(f'Multi-agent spatial dashboard: snapshot {self.snapshot.sweep}', fontsize=13)
        self._redraw()
        self.plt.pause(0.001)
