"""Second-pass regressions, including tiny real Gurobi solves when licensed."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import threading
from unittest.mock import MagicMock, patch

import gurobipy as gp
import numpy as np
import pandas as pd
import torch

from downstream import gurobi_apollo as apollo
from downstream import gurobi_neuraldiving as diving
from downstream import gurobi_predsearch as predsearch
from fmip.utils import gp_utils
from fmip.utils.mvb import MVB
from fmip.utils.solver_validation import binary_probabilities, validate_trust_region
from test_training_inference import make_model


class DownstreamContractTests(unittest.TestCase):
    def test_binary_guard_accepts_zero_mask_column_and_rejects_general_domains(self):
        variables = [SimpleNamespace(VType='B', LB=0, UB=1)]
        result = binary_probabilities([[0.2, 0.8, 0.]], variables)
        np.testing.assert_allclose(result, [[0.2, 0.8]])
        with self.assertRaisesRegex(ValueError, 'outside classes'):
            binary_probabilities([[0.2, 0.5, 0.3]], variables)
        variables[0] = SimpleNamespace(VType='I', LB=0, UB=2)
        with self.assertRaisesRegex(ValueError, 'integer domains'):
            binary_probabilities([[0.2, 0.8]], variables)

    def test_trust_region_rejects_overlapping_fixed_sets(self):
        with self.assertRaisesRegex(ValueError, 'not overlap'):
            validate_trust_region(0.8, 0.4, 0.2)

    def test_legacy_checkpoint_restores_architecture_without_dataset(self):
        original = make_model(discrete=False, classes=4)
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'legacy.ckpt'
            torch.save({'state_dict': original.state_dict()}, checkpoint)
            restored = gp_utils.load_policy_checkpoint(checkpoint, sense='max', device='cpu')
        self.assertEqual(restored.args.n_layers, 1)
        self.assertEqual(restored.max_integer_num, 4)
        self.assertFalse(restored.only_discrete)
        self.assertEqual(restored.args.sense, 'max')
        for key, value in original.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key], value)

    def test_checkpoint_preserves_nondefault_flow_configuration(self):
        original = make_model(discrete=True, mode='mask', classes=3)
        original.args.flow_schedule = 'cosine'
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'metadata.ckpt'
            torch.save({'state_dict': original.state_dict(),
                        'hyper_parameters': {'param_args': vars(original.args)}}, checkpoint)
            restored = gp_utils.load_policy_checkpoint(checkpoint, device='cpu')
        self.assertEqual(restored.DFM.mode, 'mask')
        self.assertEqual(restored.args.flow_schedule, 'cosine')
        self.assertTrue(restored.only_discrete)

    def test_fmipump_fails_before_reading_model_when_feature_absent(self):
        with patch.object(gp_utils, 'MIPmodel') as reader:
            with self.assertRaisesRegex(NotImplementedError, 'not implemented'):
                gp_utils.pred_FMIPump('/unused', SimpleNamespace())
        reader.assert_not_called()

    def test_prediction_uses_policy_device(self):
        components = dict.fromkeys(['x_cons', 'x_Cvars', 'x_Ivars', 'C2cons_edge_attr',
                                   'I2cons_edge_attr', 'C2cons_edge_index', 'I2cons_edge_index'])
        graph = MagicMock()
        graph.to.return_value = graph
        model = SimpleNamespace(device=torch.device('cpu'), inference=MagicMock(return_value={}))
        with patch.object(gp_utils, 'toHeteroData', return_value=graph):
            gp_utils.pred_by_model(SimpleNamespace(generGraphCom=lambda: components), model)
        graph.to.assert_called_once_with(torch.device('cpu'))

    def test_apollo_does_not_launch_solver_after_inference_exhausts_budget(self):
        clock = [0.]
        optimizer = apollo.ApolloOptimizer.__new__(apollo.ApolloOptimizer)
        optimizer.problem_root = '/unused'
        optimizer.timelimit = 1.
        model = MagicMock(ModelSense=gp.GRB.MINIMIZE)
        model.getVars.return_value = []
        def predict(_):
            clock[0] += 2.
            return {'Ivars': np.empty((0, 2))}
        optimizer.get_prediction = predict
        optimizer.apply_predsearch_strategy = MagicMock()
        with patch.object(apollo.gp, 'read', return_value=model), patch.object(apollo, 'MIPmodel'), \
                patch.object(apollo.time, 'time', side_effect=lambda: clock[0]):
            best, elapsed = optimizer.apply_apollo_strategy('x.mps', [0.1, 0.1, 0.1, 3])
        self.assertIsNone(best)
        self.assertEqual(elapsed, 2.)
        optimizer.apply_predsearch_strategy.assert_not_called()

    def test_apollo_zero_budget_skips_solver(self):
        optimizer = apollo.ApolloOptimizer.__new__(apollo.ApolloOptimizer)
        optimizer.timelimit = 0.
        model = MagicMock()
        self.assertEqual(optimizer.apply_predsearch_strategy(model, None, 0, 0, 0), (None, None))
        model.optimize.assert_not_called()

    def test_apollo_history_time_is_cumulative_across_solves(self):
        apollo.time_points.clear()
        apollo.iter_points.clear()
        apollo.obj_values.clear()
        model = SimpleNamespace(_history_time_offset=10., cbGet=lambda field: {
            gp.GRB.Callback.RUNTIME: 0.5, gp.GRB.Callback.MIP_OBJBST: 2.,
            gp.GRB.Callback.MIP_ITRCNT: 3.}[field])
        try:
            apollo.save_history(model, gp.GRB.Callback.MIP)
            self.assertEqual(apollo.time_points, [10.5])
        finally:
            apollo.time_points.clear()
            apollo.iter_points.clear()
            apollo.obj_values.clear()

    def test_predsearch_saves_results_and_creates_parent(self):
        optimizer = predsearch.PredsearchOptimizer.__new__(predsearch.PredsearchOptimizer)
        optimizer.result_dict = {'params': {'instance': {'predsearch_obj': 2.}}}
        optimizer.solving_history_dict = {'params': {'instance': {'time': [1.]}}}
        with TemporaryDirectory() as directory:
            prefix = Path(directory) / 'nested' / 'results'
            optimizer.save_all_results(str(prefix))
            result = pd.read_csv(str(prefix) + '_params.csv', index_col=0)
            self.assertEqual(result.loc['instance', 'predsearch_obj'], 2.)
            self.assertTrue(Path(str(prefix) + '_params_history.pkl').exists())

    def test_mvb_low_success_probability_does_not_force_zero_ones(self):
        _, upper = MVB._getHoeffdingBound(np.empty(0), np.array([0.9, 0.9]), 0.4, 0.9)
        self.assertGreaterEqual(upper, 1.8)


class RealGurobiDownstreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.env = gp.Env(params={'OutputFlag': 0, 'Threads': 1})
        except gp.GurobiError as error:
            raise unittest.SkipTest(f'Gurobi environment unavailable: {error}')

    @classmethod
    def tearDownClass(cls):
        cls.env.dispose()

    def model(self):
        model = gp.Model(env=self.env)
        model.addVar(lb=0, ub=10, name='continuous')
        first = model.addVar(vtype=gp.GRB.BINARY, name='first')
        second = model.addVar(vtype=gp.GRB.BINARY, name='second')
        model.addConstr(first + second <= 1)
        model.setObjective(2 * first + 3 * second)
        model.update()
        self.addCleanup(model.dispose)
        return model

    def test_actual_predsearch_budget_constrains_binary_variables(self):
        model = self.model()
        optimizer = predsearch.PredsearchOptimizer.__new__(predsearch.PredsearchOptimizer)
        optimizer.problem_root = '/unused'
        optimizer.set_threads = 1
        optimizer.timelimit = 2
        optimizer.get_prediction = lambda: {'Ivars': np.array([[0.9, 0.1], [0.1, 0.9]])}
        with patch.object(predsearch.gp, 'read', return_value=model), patch.object(predsearch, 'MIPmodel'):
            solved, _ = optimizer.apply_predsearch_strategy('unused', [0., 0.5, 0.])
        self.assertIs(solved, model)
        self.assertAlmostEqual(model.getVarByName('second').X, 1.)
        self.assertAlmostEqual(model.ObjVal, 3.)

    def test_actual_neural_diving_probability_strategy_fixes_most_confident(self):
        model = self.model()
        optimizer = diving.NeuralDivingOptimizer.__new__(diving.NeuralDivingOptimizer)
        optimizer.set_threads = 1
        optimizer.timelimit = 2
        optimizer.sense = 'min'
        optimizer.fixed_strategy = 'probs'
        with patch.object(diving.gp, 'read', return_value=model), \
                patch.object(torch, 'multinomial', return_value=torch.tensor([[0], [1]])):
            best, _, _ = optimizer.apply_sampled_strategy('unused', [[0.99, 0.01], [0.4, 0.6]],
                                                          [1, 2], sample_times=1, fixed_ratio=0.5)
        self.assertIs(best, model)
        self.addCleanup(best._fmip_environment.dispose)
        self.assertEqual((model.getVarByName('first').LB, model.getVarByName('first').UB), (0., 0.))
        self.assertEqual((model.getVarByName('second').LB, model.getVarByName('second').UB), (0., 1.))

    def test_mvb_warm_start_uses_registered_subset(self):
        original = self.model()
        optimizer = MVB(1, 3)
        optimizer.registerModel(original)
        self.addCleanup(optimizer._mvbmodel.dispose)
        optimizer.registerVars([1, 2])
        optimizer.setParam(0., 1., [1., 0.7], [0.9], [0.9])
        result = optimizer.getMultiVarBranch(warm=True, Xpred=np.array([[0.1, 0.9], [0.9, 0.1]]))
        result.update()
        self.assertEqual(result.getVarByName('first').Start, 1.)
        self.assertEqual(result.getVarByName('second').Start, 0.)
        self.assertEqual(result.getVarByName('continuous').Start, gp.GRB.UNDEFINED)

    def test_parallel_neural_diving_uses_distinct_main_thread_environments(self):
        original = self.model()
        optimizer = diving.NeuralDivingOptimizer.__new__(diving.NeuralDivingOptimizer)
        optimizer.set_threads, optimizer.timelimit = 1, 2
        optimizer.sense, optimizer.fixed_strategy = 'min', 'probs'
        real_env, real_read = gp.Env, gp.read
        created_on, used_environments = [], []
        def make_env(**kwargs):
            created_on.append(threading.get_ident())
            return real_env(params={'Threads': 1, 'OutputFlag': 0})
        def read(path, env=None):
            used_environments.append(id(env))
            return real_read(path, env=env)
        with TemporaryDirectory() as directory:
            path = str(Path(directory) / 'instance.mps')
            original.write(path)
            with patch.dict(diving.os.environ, {'SLURM_CPUS_PER_TASK': '2'}), \
                    patch.object(diving.gp, 'Env', side_effect=make_env), \
                    patch.object(diving.gp, 'read', side_effect=read):
                best, _, _ = optimizer.apply_sampled_strategy(path, [[0.5, 0.5], [0.5, 0.5]],
                                                              [1, 2], sample_times=2, fixed_ratio=0)
        self.assertIsNotNone(best)
        try:
            self.assertEqual(len(set(used_environments)), 2)
            self.assertEqual(created_on, [threading.get_ident()] * 2)
            self.assertAlmostEqual(best.ObjVal, 0.)
        finally:
            environment = best._fmip_environment
            best.dispose()
            environment.dispose()

    def test_mvb_complementary_subproblem_uses_subset_indices(self):
        original = self.model()
        optimizer = MVB(1, 3)
        optimizer.registerModel(original)
        self.addCleanup(optimizer._mvbmodel.dispose)
        optimizer.registerVars([1, 2])
        generated = optimizer.generate_subproblem(original, [1], [], 1, 0, lowCut=False)
        generated.optimize()
        self.assertEqual(generated.SolCount, 1)
        self.assertAlmostEqual(generated.getVarByName('second').X, 1.)
        self.assertAlmostEqual(generated.getVarByName('continuous').X, 0.)

    def test_actual_predsearch_and_mvb_reject_nonbinary_integer_domain(self):
        model = gp.Model(env=self.env)
        self.addCleanup(model.dispose)
        model.addVar(vtype=gp.GRB.INTEGER, lb=0, ub=2)
        model.update()
        optimizer = predsearch.PredsearchOptimizer.__new__(predsearch.PredsearchOptimizer)
        optimizer.problem_root = '/unused'
        optimizer.get_prediction = lambda: {'Ivars': np.array([[0.2, 0.3, 0.5]])}
        with patch.object(predsearch.gp, 'read', return_value=model), patch.object(predsearch, 'MIPmodel'):
            with self.assertRaisesRegex(ValueError, 'integer domains'):
                optimizer.apply_predsearch_strategy('unused', [0.5, 0, 0])
        mvb = MVB(0, 1)
        mvb.registerModel(model)
        self.addCleanup(mvb._mvbmodel.dispose)
        with self.assertRaisesRegex(ValueError, 'integer domains'):
            mvb.registerVars([0])


if __name__ == '__main__':
    unittest.main()
