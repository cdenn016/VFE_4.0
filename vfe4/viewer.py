"""Detached snapshot viewer; controls affect orchestration, not the objective."""

from .model import SimulationModel
from .state import SectionState
from .visualization import field_snapshot


class SimulationViewer:
    def __init__(self) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button
        self.plt = plt
        self.figure, self.axes = plt.subplots(1, 2, figsize=(10, 4))
        self.figure.subplots_adjust(bottom=0.30)
        self.paused, self.stopped, self.once = False, False, False
        self.history: list[tuple[int, float]] = []
        self.buttons = []
        for index, (label, action) in enumerate((('Pause/resume', self.toggle), ('One sweep', self.single), ('Stop', self.stop))):
            button = Button(self.figure.add_axes([0.15 + index * 0.24, 0.06, 0.21, 0.08]), label)
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

    def update(self, state: SectionState, model: SimulationModel, objective: float) -> None:
        self.history.append((state.sweep, objective))
        image = field_snapshot(state, model)['mean']
        if image.ndim == 3:
            image = image[:, :, image.shape[2] // 2]
        for ax in self.axes:
            ax.clear()
        self.axes[0].plot(*zip(*self.history), marker='o', markersize=3)
        label = 'Collective VFE (nats)'
        if state.links.mode == 'latent_gaussian':
            relative = model.config['links']['mutual_loop_coefficient'] and model.loops.weight.numel()
            label = 'Relative sampled VFE (nats; log Z omitted)' if relative else 'Sampled VFE (nats)'
        self.axes[0].set(xlabel='Inference sweep', ylabel=label)
        if image.ndim == 1:
            self.axes[1].plot(image)
        else:
            self.axes[1].imshow(image.T, origin='lower', cmap='coolwarm')
        self.axes[1].set_title('Recognition component 0, reference frame')
        self.figure.canvas.draw_idle()
        self.plt.pause(0.001)
