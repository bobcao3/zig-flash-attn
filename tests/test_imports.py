import subprocess
import sys
from pathlib import Path


def test_public_imports_and_viewer_help_without_torch_or_cuda():
    source = Path(__file__).resolve().parents[1] / "src"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import importlib.abc
import sys

class RejectGpuImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'cuda'):
            raise ModuleNotFoundError(f'{fullname} must not load during public imports')

sys.meta_path.insert(0, RejectGpuImports())
sys.path.insert(0, sys.argv[1])

import zigfa
import zigfa.torch
from zigfa import InterSequenceMetadata, flash_attn_func, prepare_inter_sequence_metadata
from zigfa.cli.run_trace_viewer import main

try:
    main(['--help'])
except SystemExit as error:
    assert error.code == 0
else:
    raise AssertionError('--help did not exit')
""",
            str(source),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
