"""Second-pass data contracts, small graphs, and licensed tiny LP integration."""

import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gurobipy as gp
import numpy as np
import torch
from torch import nn
from torch_geometric.data import Batch

from fmip.dataset import GraphMILPDataset, SplitSchedule, toHeteroData
from fmip.models.gnn import MILPGCN
from fmip.models.nn import normalization, node_output
from fmip.utils import lp_utils
from fmip.utils.milp_reader import MIPmodel
from fmip.utils.milp_utils import cal_objective_batch, cal_objective_graph, duplicate_hetero_data, feasibility
from test_data_network import make_graph


class DatasetContractTests(unittest.TestCase):
    def test_cached_near_integer_labels_are_rounded_not_truncated(self):
        graph = make_graph()
        graph['Ivars'].x[:, -1] = torch.tensor([0.999999, 0.000001])
        holder = SimpleNamespace(_data=graph)
        GraphMILPDataset.calculate_max_integer_(holder)
        torch.testing.assert_close(graph.extract_Ivars_label().long(), torch.tensor([1, 0]))

    def test_noninteger_or_nan_labels_fail_explicitly(self):
        for invalid in (0.4, float('nan')):
            graph = make_graph()
            graph['Ivars'].x[0, -1] = invalid
            with self.assertRaisesRegex(ValueError, 'integral'):
                GraphMILPDataset.calculate_max_integer_(SimpleNamespace(_data=graph))

    def test_cache_identity_includes_source_selection_and_label_mode(self):
        keys = [GraphMILPDataset.cache_directory(root, '/cache', 'train', files, labeled)
                for root, files, labeled in
                [('/one', ['a.lp'], True), ('/two', ['a.lp'], True),
                 ('/one', ['b.lp'], True), ('/one', ['a.lp'], False)]]
        self.assertEqual(len(set(keys)), 4)

    def test_cached_batch_rejects_a_graph_without_integers(self):
        graph = make_graph()
        holder = SimpleNamespace(_data=graph, slices={'Ivars': {'x': torch.tensor([0, 0, 2])}})
        with self.assertRaisesRegex(ValueError, 'Every FMIP'):
            GraphMILPDataset.calculate_max_integer_(holder)

    def make_files(self, root, count):
        (root / 'problem').mkdir()
        (root / 'solution').mkdir()
        extensions = ('.lp', '.mps', '.mps.gz')
        for i in range(count):
            (root / 'problem' / f'p{i}{extensions[i % 3]}').touch()
            (root / 'solution' / f'p{i}_sol.pkl').touch()
        (root / 'problem' / 'README.txt').touch()

    def test_mixed_extensions_split_into_disjoint_nonempty_sets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_files(root, 3)
            splits = SplitSchedule(str(root), str(root / 'cache')).get_split()
            self.assertTrue(all(len(values) == 1 for values in splits.values()))
            self.assertEqual({name for values in splits.values() for name in values},
                             {'p0.lp', 'p1.mps', 'p2.mps.gz'})

    def test_too_small_split_fails_without_leaking_instances(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_files(root, 2)
            with self.assertRaisesRegex(ValueError, 'At least 3'):
                SplitSchedule(str(root), str(root / 'cache'))

    def test_bad_instance_is_not_silently_dropped(self):
        holder = SimpleNamespace(problem_root='/unused', solution_root=None,
                                 file_name_list=['bad.lp'])
        with patch('fmip.dataset.MIPmodel', side_effect=ValueError('invalid model')):
            with self.assertRaisesRegex(ValueError, 'bad.lp'):
                GraphMILPDataset.process(holder)


class SmallGraphAndObjectiveTests(unittest.TestCase):
    def test_duplicate_batch_preserves_partitions_metadata_and_scalar_mode(self):
        first, second = make_graph(), make_graph(continuous=False)
        first['name'], second['name'] = 'first.lp', 'second.mps'
        first.model_sense, second.model_sense = 1, -1
        first.objective_constant, second.objective_constant = 3., -4.
        original = Batch.from_data_list([first, second])
        original.only_discrete = True
        duplicate = duplicate_hetero_data(original, 3)
        self.assertEqual(duplicate.num_graphs, 6)
        self.assertEqual(duplicate['name'], ['first.lp', 'second.mps'] * 3)
        torch.testing.assert_close(duplicate['Cvars'].ptr, torch.tensor([0, 1, 1, 2, 2, 3, 3]))
        torch.testing.assert_close(duplicate['Ivars'].ptr, torch.arange(0, 14, 2))
        self.assertIs(original.only_discrete, True)
        samples = duplicate.to_data_list()
        self.assertEqual([sample['Cvars'].x.shape[0] for sample in samples], [1, 0] * 3)
        for index, sample in enumerate(samples):
            expected = first if index % 2 == 0 else second
            torch.testing.assert_close(sample['Ivars', 'to', 'cons'].edge_index,
                                       expected['Ivars', 'to', 'cons'].edge_index)
            self.assertEqual(sample.model_sense.item(), expected.model_sense)
            self.assertEqual(sample.objective_constant.item(), expected.objective_constant)
            self.assertTrue(bool(sample.only_discrete))
        duplicate['Ivars'].x[0, 0] = 99
        self.assertNotEqual(original['Ivars'].x[0, 0].item(), 99)

    def test_duplicate_graph_rejects_zero_repetitions(self):
        with self.assertRaisesRegex(ValueError, 'positive integer'):
            duplicate_hetero_data(make_graph(), 0)

    def test_training_mode_checkpoint_supports_guidance_gradients(self):
        graph = make_graph()
        graph['Cvars'].x.requires_grad_(True)
        model = MILPGCN(6, 6, 1, 64, 1, 2).train()
        for layers in (model.per_layer_out_cons, model.per_layer_out_Ivars, model.per_layer_out_Cvars):
            for layer in layers:
                nn.init.normal_(layer[-1].weight, std=0.05)
        _, prediction = model(graph, torch.tensor([0.3]))
        gradient, = torch.autograd.grad(prediction.square().sum(), graph['Cvars'].x)
        self.assertEqual(gradient.shape, graph['Cvars'].x.shape)
        self.assertTrue(torch.isfinite(gradient).all())

    def test_single_node_head_supports_small_hidden_dimensions(self):
        for channels in (16, 32, 64, 96):
            with self.subTest(channels=channels):
                head = nn.Sequential(normalization(channels), nn.ReLU(), nn.Conv2d(channels, 2, 1))
                features = torch.randn(1, channels, requires_grad=True)
                result = node_output(head, features)
                result.square().sum().backward()
                self.assertEqual(result.shape, (1, 2))
                self.assertTrue(torch.isfinite(features.grad).all())

    def test_graph_without_constraints_has_finite_predictions(self):
        graph = make_graph()
        graph['cons'].x = torch.empty((0, 1))
        for kind in ('Ivars', 'Cvars'):
            graph[kind, 'to', 'cons'].edge_index = torch.empty((2, 0), dtype=torch.long)
            graph[kind, 'to', 'cons'].edge_attr = torch.empty((0, 1))
        model = MILPGCN(6, 6, 1, 32, 1, 2).eval()
        integers, continuous = model(graph, torch.tensor([0.5]))
        self.assertTrue(torch.isfinite(integers).all())
        self.assertTrue(torch.isfinite(continuous).all())

    def test_batched_objectives_include_continuous_nodes_and_constants(self):
        first, second = make_graph(), make_graph()
        first.objective_constant, second.objective_constant = 3., 7.
        batch = Batch.from_data_list([first, second])
        torch.testing.assert_close(cal_objective_graph(batch, reduction=False), torch.tensor([7., 11.]))
        self.assertEqual(cal_objective_graph(batch).item(), 18.)
        solutions_i = torch.stack([batch.extract_Ivars_label()] * 2)
        solutions_c = torch.stack([batch.extract_Cvars_label()] * 2)
        torch.testing.assert_close(cal_objective_batch(batch, solutions_i, solutions_c, reduction=False), torch.tensor([18., 18.]))
        self.assertEqual(cal_objective_batch(batch, solutions_i, solutions_c).item(), 36.)

    def test_batched_feasibility_does_not_drop_continuous_terms(self):
        graph = make_graph()
        batch = Batch.from_data_list([graph, graph.clone()])
        calculator = feasibility(False, True, False, False)
        value = calculator(batch, torch.zeros(4), torch.full((2,), 10.))
        self.assertGreater(value.item(), 0.)

    def test_sparse_column_selection_preserves_repeats_order_and_empty_shape(self):
        dense = torch.tensor([[1., 2., 3.], [4., 0., 6.]])
        sparse = dense.to_sparse()
        torch.testing.assert_close(lp_utils.select_columns(sparse, torch.tensor([2, 0, 2])).to_dense(), dense[:, [2, 0, 2]])
        self.assertEqual(lp_utils.select_columns(sparse, []).shape, (2, 0))

    def test_basis_masks_use_presence_flags_not_zero_bound_values(self):
        features = torch.tensor([[0., 0., 0., 1., 1., 0.], [0., 0., 0., 0., 0., 0.]])
        probabilities = torch.tensor([[0.2, 0.3, 0.5], [0.4, 0., 0.6]])
        expected = torch.tensor([[0.2, 0.3, 0.5], [0., 1., 0.]])
        torch.testing.assert_close(lp_utils.mask_and_divide_dominance(probabilities, features), expected)


class RealGurobiEncodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.environment = gp.Env(empty=True)
        cls.environment.setParam('OutputFlag', 0)
        cls.environment.start()

    @classmethod
    def tearDownClass(cls):
        cls.environment.dispose()

    def model(self):
        model = gp.Model(env=self.environment)
        model.Params.Threads = 1
        self.addCleanup(model.dispose)
        return model

    def reader(self, model):
        reader = MIPmodel.__new__(MIPmodel)
        reader.model = model
        return reader

    def test_real_mixed_encoding_preserves_objective_and_all_senses(self):
        model = self.model()
        c = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name='c')
        i = model.addVar(vtype=gp.GRB.BINARY, name='i')
        model.addConstr(c + 2 * i >= 1)
        model.addConstr(2 * c - i == 0)
        model.setObjective(3 * c + 5 * i + 7, gp.GRB.MAXIMIZE)
        model.update()
        components = self.reader(model).generGraphCom()
        graph = toHeteroData(**{key: components[key] for key in
            ('x_cons', 'x_Cvars', 'x_Ivars', 'C2cons_edge_attr', 'I2cons_edge_attr',
             'C2cons_edge_index', 'I2cons_edge_index', 'model_sense', 'objective_constant')})
        graph.input_Ivars_label_(torch.tensor([1.]))
        graph.input_Cvars_label_(torch.tensor([0.5]))
        self.assertEqual(graph.model_sense, -1)
        self.assertAlmostEqual(cal_objective_graph(graph).item(), 13.5)
        self.assertEqual(graph['cons'].x.shape, (3, 1))

    def test_solver_tolerance_label_rounding(self):
        model = self.model()
        model.addVar(vtype=gp.GRB.BINARY, name='x')
        model.update()
        with tempfile.TemporaryDirectory() as temporary:
            solution = Path(temporary) / 'solution.pkl'
            with solution.open('wb') as stream:
                pickle.dump({'label_info': {'x': 0.999999}}, stream)
            self.assertEqual(self.reader(model).generGraphCom(str(solution))['x_Ivars'][0, -1].item(), 1.)

    def test_non_linear_and_semi_continuous_models_are_rejected(self):
        quadratic = self.model()
        x = quadratic.addVar(vtype=gp.GRB.BINARY)
        quadratic.setObjective(x * x)
        quadratic.update()
        with self.assertRaisesRegex(ValueError, 'linear MILPs'):
            self.reader(quadratic).generGraphCom()
        semi = self.model()
        semi.addVar(vtype=gp.GRB.SEMICONT, lb=2, ub=5)
        semi.update()
        with self.assertRaisesRegex(ValueError, 'Unsupported variable type'):
            self.reader(semi).generGraphCom()

    def test_show_result_handles_infeasible_model(self):
        model = self.model()
        x = model.addVar(lb=0, ub=1)
        model.addConstr(x >= 2)
        result = self.reader(model).showResult()
        self.assertEqual(result['SolCount'], 0)
        self.assertIsNone(result['ObjVal'])

    def test_restored_lp_adapter_recovers_an_equality_basis_solution(self):
        model = self.model()
        x = model.addVar(lb=0, ub=10, name='x')
        model.addConstr(x == 2, name='fix')
        model.setObjective(x)
        model.update()
        with patch.object(lp_utils, 'MIPmodel', return_value=self.reader(model)):
            graph = lp_utils.create_graph_data('unused.lp', to_eq=False)
        np.testing.assert_allclose(lp_utils.recover_approx_solution(graph, [1]), [2.])


if __name__ == '__main__':
    unittest.main()
