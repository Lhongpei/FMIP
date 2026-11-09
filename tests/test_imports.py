"""Exercise public entry points in fresh interpreters without path workarounds."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PackageImportTests(unittest.TestCase):
    def run_python(self, arguments, cwd):
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        completed = subprocess.run(
            [sys.executable, *arguments],
            cwd=cwd,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        return completed

    def test_package_and_solver_imports_share_model_class(self):
        self.run_python(["-c", """
import sys
from fmip.dataset import GraphMILP
from fmip.pl_gmip_model import GMIPModel
from fmip.utils import gp_utils
from downstream import gurobi_apollo, gurobi_mvb, gurobi_neuraldiving, gurobi_predsearch

assert gp_utils.GMIPModel is GMIPModel
for module in (gurobi_apollo, gurobi_mvb, gurobi_neuraldiving, gurobi_predsearch):
    assert module.GMIPModel is GMIPModel
assert 'pl_gmip_model' not in sys.modules
assert 'pl_meta_model' not in sys.modules
assert 'dataset' not in sys.modules
"""], ROOT)

    def test_training_script_help_works_outside_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            completed = self.run_python([str(ROOT / "fmip" / "train.py"), "--help"], directory)
        self.assertIn("--inference_steps", completed.stdout)


if __name__ == "__main__":
    unittest.main()
