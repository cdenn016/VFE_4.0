"""Private isolated-process entry point for the Task 12 synthetic smoke."""

from __future__ import annotations

import pickle
import sys
import traceback


def main() -> int:
    try:
        request = pickle.loads(sys.stdin.buffer.read())
        if type(request) is not dict or set(request) != {
            "config",
            "cache_root",
            "run_root",
            "smoke_run_id",
            "parent_process_id",
        }:
            raise ValueError("smoke worker request is malformed")

        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        from vfe4.training.smoke import (
            _run_wt103_synthetic_smoke_in_process,
        )

        result = _run_wt103_synthetic_smoke_in_process(**request)
        response = (True, result)
        exit_code = 0
    except BaseException as exc:
        response = (
            False,
            type(exc).__name__,
            str(exc),
            traceback.format_exc(),
        )
        exit_code = 2
    sys.stdout.buffer.write(pickle.dumps(response, protocol=5))
    sys.stdout.buffer.flush()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
