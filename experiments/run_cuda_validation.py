"""Click-to-run tiny CUDA lane, with explicit admission and JUnit evidence."""

import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

from vfe4.config import load_config
from vfe4.presets import resolve_preset
from vfe4.simulation import ensure_cuda_capacity
from vfe4.io import write_json


def run_validation() -> None:
    status = ROOT / 'docs/implementation/cuda-status.json'
    try:
        ensure_cuda_capacity(load_config(resolve_preset('tiny_point', {'runtime': {'device': 'cuda'}})))
    except RuntimeError as error:
        write_json(status, {'status': 'deferred' if 'compute processes' in str(error) else 'failed',
            'checked_utc': datetime.now(timezone.utc).isoformat(), 'reason': str(error), 'tests_executed': 0})
        raise
    os.environ['VFE4_TEST_DEVICE'] = 'cuda'
    output = ROOT / 'docs/implementation/cuda.xml'
    result = pytest.main([str(ROOT / 'tests/test_cuda.py'), '-q', f'--junitxml={output}'])
    suites = list(ET.parse(output).getroot().iter('testsuite'))
    if result or not suites or sum(int(suite.get('tests', '0')) for suite in suites) < 9 or any(int(suite.get('skipped', '0')) for suite in suites):
        write_json(status, {'status': 'failed', 'checked_utc': datetime.now(timezone.utc).isoformat(), 'exit_code': result})
        raise RuntimeError('CUDA lane did not execute all expected tests successfully')
    write_json(status, {'status': 'completed', 'checked_utc': datetime.now(timezone.utc).isoformat(),
                        'tests_executed': sum(int(suite.get('tests', '0')) for suite in suites), 'junit': str(output)})


if __name__ == '__main__':
    run_validation()
