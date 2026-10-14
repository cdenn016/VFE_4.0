"""Inspect captured agent sections without changing inference or the objective."""

from .model import SimulationModel
from .section_view import SectionImage, SectionSnapshot, capture_sections, section_image
from .state import SectionState


class SimulationViewer:
    def __init__(self) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, RadioButtons, TextBox
        self.plt = plt
        self.figure, self.axes = plt.subplots(1, 2, figsize=(12, 6.5))
        self.figure.subplots_adjust(left=0.26, right=0.95, bottom=0.32, top=0.85, wspace=0.35)
        self.paused, self.stopped, self.once = False, False, False
        self.history: list[tuple[int, float]] = []
        self.snapshot: SectionSnapshot | None = None
        self.last_section_image: SectionImage | None = None
        self._model: SimulationModel | None = None
        self._colorbar = None
        self.selection_agent, self.selection_component, self.selection_slice = 0, 0, 0
        self.selection_channel, self.selection_statistic = 'belief', 'mean'
        self.agent_input = TextBox(self.figure.add_axes([0.085, 0.77, 0.105, 0.045]), 'Agent', initial='0')
        self.component_input = TextBox(self.figure.add_axes([0.085, 0.66, 0.105, 0.045]), 'Component', initial='0')
        self.agent_input.label.set_fontsize(9)
        self.component_input.label.set_fontsize(9)
        self.agent_range = self.figure.text(0.085, 0.745, '', fontsize=8)
        self.component_range = self.figure.text(0.085, 0.635, '', fontsize=8)
        self.channel_buttons = RadioButtons(self.figure.add_axes([0.025, 0.46, 0.165, 0.125]), ('belief', 'model'))
        self.statistic_buttons = RadioButtons(self.figure.add_axes([0.025, 0.255, 0.165, 0.135]), ('Mean', 'Std. deviation'))
        self.channel_buttons.ax.set_title('Channel', fontsize=10)
        self.statistic_buttons.ax.set_title('Display', fontsize=10)
        for control in (self.channel_buttons, self.statistic_buttons):
            for label in control.labels:
                label.set_fontsize(9)
        self.slice_slider = None
        self.figure.text(0.025, 0.21, 'Gray = outside agent support', fontsize=8)
        self.status_text = self.figure.text(0.025, 0.15, '', color='firebrick', fontsize=8)
        self.agent_input.on_submit(self._select_agent)
        self.component_input.on_submit(self._select_component)
        self.channel_buttons.on_clicked(self._select_channel)
        self.statistic_buttons.on_clicked(self._select_statistic)
        self.buttons = []
        for index, (label, action) in enumerate((('Pause/resume', self.toggle), ('One sweep', self.single), ('Stop', self.stop))):
            button = Button(self.figure.add_axes([0.27 + index * 0.23, 0.055, 0.21, 0.065]), label)
            button.on_clicked(action)
            self.buttons.append(button)
        self.figure.canvas.mpl_connect('close_event', self.stop)
        plt.show(block=False)

    def toggle(self, event: object) -> None:
        self.paused = not self.paused

    def single(self, event: object) -> None:
        self.paused, self.once = True, True

    def stop(self, event: object) -> None:
        self.stopped = True

    def ready(self) -> bool:
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
            self._draw_section()

    def _select_component(self, text: str) -> None:
        if self._model is not None and self._select_integer(text, 'component', self._model.config['bundle']['K']):
            self._draw_section()

    def _select_channel(self, label: str) -> None:
        self.selection_channel = label
        self._draw_section()

    def _select_statistic(self, label: str) -> None:
        self.selection_statistic = 'mean' if label == 'Mean' else 'std'
        self._draw_section()

    def _select_slice(self, value: float) -> None:
        self.selection_slice = int(value)
        self._draw_section()

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

    def update(self, state: SectionState, model: SimulationModel, objective: float) -> None:
        self.snapshot = capture_sections(state)
        if self._model is None:
            self._model = model
            self.agent_range.set_text(f'0..{model.config["population"]["n_agents"] - 1}')
            self.component_range.set_text(f'0..{model.config["bundle"]["K"] - 1}')
            if len(model.domain.grid_shape) == 3:
                from matplotlib.widgets import Slider
                self.slice_slider = Slider(self.figure.add_axes([0.67, 0.22, 0.235, 0.025]), 'z slice',
                                           0, 2 * model.domain.grid_shape[2] - 1, valinit=0, valstep=1, valfmt='%d')
                self.slice_slider.on_changed(self._select_slice)
                self._center_slice()
        self.history.append((state.sweep, objective))
        self.axes[0].clear()
        self.axes[0].plot(*zip(*self.history), marker='o', markersize=3)
        label = 'Collective VFE (nats)'
        if state.links.mode == 'latent_gaussian':
            relative = model.config['links']['mutual_loop_coefficient'] and model.loops.weight.numel()
            label = 'Relative sampled VFE (nats; log Z omitted)' if relative else 'Sampled VFE (nats)'
        self.axes[0].set(xlabel='Inference sweep', ylabel=label)
        self.figure.suptitle(f'Multi-agent section inspector: snapshot {self.snapshot.sweep}', fontsize=13)
        self._draw_section()
        self.plt.pause(0.001)
