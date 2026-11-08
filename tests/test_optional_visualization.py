"""Rendering is optional locally, but installed explicitly in GitHub validation."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_bev_render_tests_skip_without_matplotlib_but_nonvisual_tests_still_run():
    pytest.importorskip("torch")
    pytest.importorskip("PIL.Image")
    root = Path(__file__).resolve().parents[1]
    script = """
import importlib.abc
import sys

class WithoutMatplotlib(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'matplotlib' or fullname.startswith('matplotlib.'):
            raise ModuleNotFoundError('matplotlib deliberately unavailable', name='matplotlib')

sys.meta_path.insert(0, WithoutMatplotlib())
import pytest
raise SystemExit(pytest.main([
    '-q', '-p', 'no:cacheprovider',
    'tests/test_bev_training.py::test_cpu_inspection_no_mutation_and_real_plot',
    'tests/test_bev_training.py::test_notebook_cells_execute_on_cpu_and_training_is_opt_in',
    'tests/test_bev_training.py::test_configuration_roundtrip_and_defaults',
]))
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=root,
        env=dict(os.environ, PYTHONPATH=str(root / "src"), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"),
        capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed, 2 skipped" in result.stdout
