"""Regression checks for LP arithmetic and sampled downstream searches."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import gurobipy as gp
import numpy as np
import torch

from downstream import gurobi_neuraldiving as diving
from fmip.utils.lp_utils import calculate_Aty, dual_infer_primal_basis, mask_and_divide_dominance


class LPArithmeticTests(unittest.TestCase):
    def test_dual_product_preserves_coefficient_precision_and_gradient(self):
        matrix = torch.diag(torch.tensor([2., 3.])).to_sparse()
        values = torch.tensor([1., 2.], dtype=torch.float16, requires_grad=True)
        result = calculate_Aty(values, matrix)
        self.assertEqual(result.dtype, torch.float32)
        torch.testing.assert_close(result, torch.tensor([2., 6.]))
        result.square().sum().backward()
        torch.testing.assert_close(values.grad, torch.tensor([8., 36.], dtype=torch.float16))

    def test_basis_recovery_ranks_absolute_complementarity_residual(self):
        matrix = torch.tensor([[0., 1.]])
        dual = torch.tensor([0.9])
        coefficients = torch.tensor([1., 0.85])
        basis = dual_infer_primal_basis(dual, matrix, coefficients)
        # The first variable has a large negative residual, not a zero residual.
        self.assertEqual(basis.tolist(), [0, 1, 2])

    def test_lp_basis_probability_projection_normalizes_subnormal_mass(self):
        probabilities = torch.tensor([[1e-7, 0., 0.], [0., 0., 0.]], dtype=torch.float16)
        features = torch.tensor([[0., 1., 0., 1., 1., 0.], [0., 0., 0., 0., 0., 0.]])
        projected = mask_and_divide_dominance(probabilities, features)
        torch.testing.assert_close(projected, torch.tensor([[1., 0., 0.], [0., 1., 0.]], dtype=torch.float16))

    @unittest.skipUnless(torch.cuda.is_available(), 'requires a GPU Slurm allocation')
    def test_dual_sparse_product_under_cuda_autocast(self):
        matrix = torch.diag(torch.tensor([2., 3.], device='cuda')).to_sparse()
        values = torch.tensor([1., 2.], device='cuda', requires_grad=True)
        with torch.autocast('cuda', dtype=torch.float16):
            result = calculate_Aty(values, matrix)
            loss = result.square().sum()
        self.assertEqual(result.dtype, torch.float32)
        loss.backward()
        torch.testing.assert_close(values.grad, torch.tensor([8., 36.], device='cuda'))


def make_diving_optimizer():
    optimizer = diving.NeuralDivingOptimizer.__new__(diving.NeuralDivingOptimizer)
    optimizer.set_threads = 1
    optimizer.timelimit = 2
    optimizer.sense = 'min'
    optimizer.fixed_strategy = 'probs'
    return optimizer


class SampledSearchContractTests(unittest.TestCase):
    def test_invalid_probability_rows_fail_before_creating_solver(self):
        optimizer = make_diving_optimizer()
        for probabilities in ([[0., 0.]], [[-1., 2.]], [[np.nan, 1.]], [[1., 0.], [0., 1.]]):
            with self.subTest(probabilities=probabilities), patch.object(diving.gp, 'Env') as environment:
                with self.assertRaises(ValueError):
                    optimizer.apply_sampled_strategy('/unused', probabilities, [0], sample_times=1)
                environment.assert_not_called()

    def test_reported_time_includes_sequential_samples_in_each_worker(self):
        optimizer = make_diving_optimizer()
        clock = [0.]
        def make_model(*args, **kwargs):
            model = MagicMock(ModelSense=gp.GRB.MINIMIZE, SolCount=1, ObjVal=0.)
            model.getVars.return_value = [SimpleNamespace(VType=gp.GRB.BINARY, LB=0., UB=1.)]
            def optimize(callback):
                clock[0] += 2.
            model.optimize.side_effect = optimize
            return model
        with patch.dict(diving.os.environ, {'SLURM_CPUS_PER_TASK': '1'}), \
                patch.object(diving.gp, 'Env'), patch.object(diving.gp, 'read', side_effect=make_model), \
                patch.object(diving.time, 'time', side_effect=lambda: clock[0]):
            best, elapsed, _ = optimizer.apply_sampled_strategy('/unused', [[0.5, 0.5]], [0],
                                                               sample_times=3, fixed_ratio=0)
        self.assertEqual(elapsed, 6.)
        self.assertIsNotNone(best)

    def test_failed_search_disposes_all_models_before_their_environments(self):
        for workers, reverse_results in ((1, False), (2, False), (2, True)):
            with self.subTest(workers=workers, reverse_results=reverse_results):
                optimizer = make_diving_optimizer()
                environments, models, events = [], [], []
                def make_environment(**kwargs):
                    environment = MagicMock()
                    environment.index = len(environments)
                    environment.dispose.side_effect = lambda: events.append(('environment', environment.index))
                    environments.append(environment)
                    return environment
                def read_model(path, env=None):
                    model = MagicMock(ModelSense=gp.GRB.MINIMIZE, SolCount=1, ObjVal=0.)
                    model.getVars.return_value = [SimpleNamespace(VType=gp.GRB.BINARY, LB=0., UB=1.)]
                    should_fail = len(models) == 1 if workers == 1 else env.index == 0
                    model.dispose.side_effect = lambda: events.append(('model', id(model)))
                    if should_fail:
                        model.optimize.side_effect = RuntimeError('worker solve failed')
                    models.append((env, model))
                    return model
                def ordered_results(futures):
                    return reversed(futures) if reverse_results else iter(futures)
                with patch.dict(diving.os.environ, {'SLURM_CPUS_PER_TASK': str(workers)}), \
                        patch.object(diving.gp, 'Env', side_effect=make_environment), \
                        patch.object(diving.gp, 'read', side_effect=read_model), \
                        patch.object(diving.concurrent.futures, 'as_completed', side_effect=ordered_results):
                    with self.assertRaisesRegex(RuntimeError, 'worker solve failed'):
                        optimizer.apply_sampled_strategy('/unused', [[0.5, 0.5]], [0],
                                                         sample_times=2, fixed_ratio=0)
                self.assertEqual(len(models), 2)
                for environment, model in models:
                    model.dispose.assert_called_once_with()
                    self.assertLess(events.index(('model', id(model))),
                                    events.index(('environment', environment.index)))
                for environment in environments:
                    environment.dispose.assert_called_once_with()


class RealSampledSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.environment = gp.Env(params={'OutputFlag': 0, 'Threads': 1})
        except gp.GurobiError as error:
            raise unittest.SkipTest(f'Gurobi environment unavailable: {error}')

    @classmethod
    def tearDownClass(cls):
        cls.environment.dispose()

    def test_sample_cannot_relax_original_fixed_variable_domain(self):
        with gp.Model(env=self.environment) as original, TemporaryDirectory() as directory:
            variable = original.addVar(vtype=gp.GRB.BINARY, lb=1, ub=1)
            original.setObjective(variable)
            original.update()
            path = str(Path(directory) / 'fixed.mps')
            original.write(path)
            optimizer = make_diving_optimizer()
            with self.assertRaisesRegex(ValueError, 'original variable domain'):
                optimizer.apply_sampled_strategy(path, [[1., 0.]], [0], sample_times=1)

    def test_actual_maximization_sense_overrides_default_minimum(self):
        with gp.Model(env=self.environment) as original, TemporaryDirectory() as directory:
            variable = original.addVar(vtype=gp.GRB.BINARY)
            original.setObjective(variable, gp.GRB.MAXIMIZE)
            original.update()
            path = str(Path(directory) / 'maximize.mps')
            original.write(path)
            for workers in (1, 2):
                with self.subTest(workers=workers):
                    optimizer = make_diving_optimizer()
                    with patch.dict(diving.os.environ, {'SLURM_CPUS_PER_TASK': str(workers)}), \
                            patch.object(torch, 'multinomial', return_value=torch.tensor([[0, 1]])):
                        best, _, best_index = optimizer.apply_sampled_strategy(
                            path, [[0.5, 0.5]], [0], sample_times=2)
                    self.assertIsNotNone(best)
                    try:
                        self.assertEqual(best.ObjVal, 1.)
                        self.assertEqual(best_index, 1)
                    finally:
                        environment = best._fmip_environment
                        best.dispose()
                        environment.dispose()


if __name__ == '__main__':
    unittest.main()
