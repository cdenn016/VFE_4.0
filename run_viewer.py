"""Edit CONFIG, select C:/anaconda/python.exe, and press Run to inspect saved data."""

from pathlib import Path
import os


CONFIG = {
    'checkpoint': 'runs/your-run/final_state.pt',  # Checkpoint file or run directory.
    'device': 'cpu',                             # Display on 'cpu' or explicit 'cuda'.
    'cpu_threads': 4,                            # Integer from 1 through 4.
}


def main() -> None:
    """Open the configured checkpoint in a bounded-thread Qt inspector."""
    threads = CONFIG['cpu_threads']
    if type(threads) is not int or not 1 <= threads <= 4:
        raise ValueError('cpu_threads must be an integer from 1 through 4')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = str(threads)
    import torch
    torch.set_num_threads(threads)
    import matplotlib
    matplotlib.use('QtAgg')
    from vfe4.inspection import open_saved_run

    checkpoint = Path(CONFIG['checkpoint']).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = Path(__file__).resolve().parent / checkpoint
    open_saved_run(checkpoint, device=CONFIG['device'])


if __name__ == '__main__':
    main()
