"""Exercise complete solver entry points, instance selection, and result files."""

import os
from pathlib import Path
import pickle
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import gurobipy as gp
import pandas as pd
import torch

from downstream import gurobi_apollo as apollo
from downstream import gurobi_mvb as mvb
from downstream import gurobi_neuraldiving as diving
from downstream import gurobi_predsearch as predsearch
from test_training_inference import make_model


ROOT = Path(__file__).resolve().parents[1]
SOLVERS = (
    (apollo.ApolloOptimizer, 'process_instance_apollo', True),
    (predsearch.PredsearchOptimizer, 'process_instance_predsearch', True),
    (mvb.MVBOptimizer, 'process_instance_mvb', False),
    (diving.NeuralDivingOptimizer, 'process_instance_sampled', False),
)


class SolverBatchEntryTests(unittest.TestCase):
    def make_fixture(self, directory, count=12):
        directory = Path(directory)
        problems = directory / 'toy' / 'problem'
        problems.mkdir(parents=True)
        names = [f'case{i:02d}.lp' for i in range(count)]
        for name in names:
            (problems / name).touch()
        split = directory / 'selected.pkl'
        split.write_bytes(pickle.dumps({'test': names}))
        return problems, split, names

    def optimizer(self, cls, method, problems, split, limit=None):
        optimizer = cls.__new__(cls)
        optimizer.problem_root = str(problems)
        optimizer.problem_set = 'toy'
        optimizer.policy = SimpleNamespace(args=SimpleNamespace(dataset_cache=None))
        optimizer.split_path = split
        optimizer.instance_limit = limit
        optimizer.result_dict = {}
        optimizer.solving_history = {}
        setattr(optimizer, method, MagicMock(return_value={'objective': 0.}))
        return optimizer

    def run_solver(self, optimizer, parameterized):
        return optimizer.run_all([[0., 0., 0., 1]]) if parameterized else optimizer.run_all()

    def test_explicit_split_runs_every_selected_instance(self):
        with TemporaryDirectory() as directory:
            problems, split, names = self.make_fixture(directory)
            for cls, method, parameterized in SOLVERS:
                with self.subTest(solver=cls.__name__):
                    optimizer = self.optimizer(cls, method, problems, split)
                    self.run_solver(optimizer, parameterized)
                    self.assertEqual([call.args[0] for call in getattr(optimizer, method).call_args_list], names)

    def test_explicit_limit_is_the_only_reason_to_truncate(self):
        with TemporaryDirectory() as directory:
            problems, split, names = self.make_fixture(directory)
            for cls, method, parameterized in SOLVERS:
                with self.subTest(solver=cls.__name__):
                    optimizer = self.optimizer(cls, method, problems, split, limit=3)
                    self.run_solver(optimizer, parameterized)
                    self.assertEqual([call.args[0] for call in getattr(optimizer, method).call_args_list], names[:3])

    def test_missing_or_duplicate_instances_fail_before_any_solver_runs(self):
        with TemporaryDirectory() as directory:
            problems, split, names = self.make_fixture(directory)
            for invalid in ([names[0], names[0]], [names[0], 'missing.lp']):
                split.write_bytes(pickle.dumps({'test': invalid}))
                for cls, method, parameterized in SOLVERS:
                    with self.subTest(solver=cls.__name__, invalid=invalid):
                        optimizer = self.optimizer(cls, method, problems, split)
                        with self.assertRaises((ValueError, FileNotFoundError)):
                            self.run_solver(optimizer, parameterized)
                        getattr(optimizer, method).assert_not_called()

    def test_dataset_local_and_checkpoint_cache_splits_are_discovered(self):
        with TemporaryDirectory() as directory:
            problems, split, names = self.make_fixture(directory, count=1)
            for cls, method, parameterized in SOLVERS:
                for use_cache in (False, True):
                    with self.subTest(solver=cls.__name__, use_cache=use_cache):
                        local = problems.parent / 'split.pkl'
                        cache = Path(directory) / 'cache'
                        selected = cache / 'toy' / 'split.pkl' if use_cache else local
                        selected.parent.mkdir(parents=True, exist_ok=True)
                        selected.write_bytes(split.read_bytes())
                        optimizer = self.optimizer(cls, method, problems, None)
                        optimizer.policy.args.dataset_cache = str(cache) if use_cache else None
                        self.run_solver(optimizer, parameterized)
                        self.assertEqual(getattr(optimizer, method).call_args.args[0], names[0])
                        selected.unlink()


class ResultFileTests(unittest.TestCase):
    def test_csv_and_history_use_distinct_paths_for_all_supported_names(self):
        for cls in (mvb.MVBOptimizer, diving.NeuralDivingOptimizer):
            for name in ('results.csv', 'results.CSV', 'results'):
                for path_object in (False, True):
                    with self.subTest(solver=cls.__name__, name=name, path_object=path_object), TemporaryDirectory() as directory:
                        optimizer = cls.__new__(cls)
                        optimizer.result_dict = {'case.lp': {'objective': 3.}}
                        optimizer.solving_history = {'case.lp': {'time': [1.]}}
                        path = Path(directory) / name
                        optimizer.save_results(path if path_object else str(path))
                        self.assertEqual(pd.read_csv(path, index_col=0).loc['case.lp', 'objective'], 3.)
                        history = path.with_name(path.stem + '_history.pkl')
                        self.assertEqual(pickle.loads(history.read_bytes()), optimizer.solving_history)


class SolverImportTests(unittest.TestCase):
    def test_script_paths_import_without_pythonpath_from_another_directory(self):
        environment = os.environ.copy()
        environment.pop('PYTHONPATH', None)
        for filename in ('gurobi_apollo.py', 'gurobi_mvb.py', 'gurobi_neuraldiving.py', 'gurobi_predsearch.py'):
            with self.subTest(script=filename), TemporaryDirectory() as directory:
                path = ROOT / 'downstream' / filename
                code = 'import runpy, sys; runpy.run_path(sys.argv[1], run_name="entry_import_test")'
                result = subprocess.run([sys.executable, '-c', code, str(path)], cwd=directory,
                                        env=environment, text=True, capture_output=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class MVBBudgetTests(unittest.TestCase):
    def test_zero_time_limit_is_forwarded_to_solver(self):
        optimizer = mvb.MVBOptimizer.__new__(mvb.MVBOptimizer)
        optimizer.timelimit = 0.
        optimizer.set_threads = 1
        optimizer._get_variable_info = lambda model: ([], [], [])
        model = MagicMock(SolCount=0)
        args = SimpleNamespace(fixratio=0., fixthresh=1.1, tmvb=0.9, psucceed_low=0.9,
                               psucceed_up=0.9, upCut=True, lowCut=True, ratio_involve=False,
                               ratio_low=0.8, ratio_up=0., robust=False)
        with patch.object(mvb, 'MVB') as solver:
            solver.return_value.getMultiVarBranch.return_value = model
            optimizer.mvb_experiment(MagicMock(), [], args)
        self.assertIn(('Timelimit', 0.), [call.args for call in model.setParam.call_args_list])


class RealBatchPipelineTests(unittest.TestCase):
    def test_checkpoint_to_twelve_real_solves_and_saved_results(self):
        try:
            environment = gp.Env(params={'OutputFlag': 0, 'Threads': 1})
        except gp.GurobiError as error:
            self.skipTest(f'Gurobi environment unavailable: {error}')
        with environment, TemporaryDirectory() as directory:
            root = Path(directory)
            problems = root / 'toy' / 'problem'
            problems.mkdir(parents=True)
            names = [f'case{i:02d}.lp' for i in range(12)]
            for index, name in enumerate(names):
                with gp.Model(env=environment) as solver:
                    x = solver.addVar(vtype=gp.GRB.BINARY, name='x')
                    y = solver.addVar(vtype=gp.GRB.BINARY, name='y')
                    solver.addConstr(x + y >= 1)
                    solver.setObjective(x + 2 * y + index)
                    solver.update()
                    solver.write(str(problems / name))
            split = root / 'selected.pkl'
            split.write_bytes(pickle.dumps({'test': names}))
            model = make_model(discrete=True).eval()
            checkpoint = root / 'model.ckpt'
            torch.save({'state_dict': model.state_dict(),
                        'hyper_parameters': {'param_args': vars(model.args)}}, checkpoint)
            optimizer = predsearch.PredsearchOptimizer(directory, 'toy', checkpoint,
                                                       set_threads=1, timelimit=2., split_path=split)
            parameters = [0., 0., 0.]
            results = optimizer.run_all([parameters])[str(parameters)]
            self.assertEqual(sorted(results), names)
            for index, name in enumerate(names):
                self.assertAlmostEqual(results[name]['predsearch_obj'], 1. + index)
            prefix = root / 'results' / 'predsearch'
            optimizer.save_all_results(prefix)
            saved = pd.read_csv(f'{prefix}_{parameters}.csv', index_col=0)
            self.assertEqual(len(saved), 12)
            history = pickle.loads(Path(f'{prefix}_{parameters}_history.pkl').read_bytes())
            self.assertEqual(sorted(history), names)


if __name__ == '__main__':
    unittest.main()
