"""Third audit regressions for standalone and batched MILP targets."""

import math
import unittest

import torch
from torch_geometric.data import Batch

from fmip.utils.milp_utils import (
    cal_feasibility_batch, cal_feasibility_graph,
    cal_objective_batch, cal_objective_graph, target_func, target_function_batch,
    gradient_function,
)
from test_data_network import make_graph


class StandaloneTargetTests(unittest.TestCase):
    def test_discrete_feasibility_accepts_none_continuous_values(self):
        graph = make_graph(continuous=False)
        graph.only_discrete = True
        integers = torch.tensor([3., 5.], requires_grad=True)
        violation = cal_feasibility_batch(graph, integers, None)
        torch.testing.assert_close(violation, torch.tensor(math.sqrt(2.)))
        violation.backward()
        torch.testing.assert_close(integers.grad, torch.full((2,), 1 / math.sqrt(2.)))

    def test_standalone_samples_need_no_name_or_batch_metadata(self):
        graph = make_graph()
        integers = torch.tensor([[0., 1.], [2., 1.]])
        continuous = torch.tensor([3.])
        actual = cal_feasibility_batch(graph, integers, continuous, reduction=False)
        torch.testing.assert_close(actual, torch.tensor([[1.], [3.]]))
        graph.input_Ivars_label_(integers[0])
        graph.input_Cvars_label_(continuous)
        torch.testing.assert_close(actual[0], cal_feasibility_graph(graph, reduction=False))

    def test_unnamed_graph_batch_keeps_constraint_partitions(self):
        graphs = [make_graph(continuous=False), make_graph(continuous=False)]
        for graph in graphs:
            graph.only_discrete = True
        batch = Batch.from_data_list(graphs)
        actual = cal_feasibility_batch(batch, torch.tensor([3., 5., 4., 4.]), None,
                                       reduction=False)
        torch.testing.assert_close(actual, torch.tensor([math.sqrt(2.), 2.]))

    def test_standalone_objective_and_guidance_target_keep_sample_axes(self):
        graph = make_graph()
        graph.objective_constant = 7.
        integers = torch.tensor([[0., 1.], [2., 1.]])
        continuous = torch.tensor([3.])
        values = cal_objective_batch(graph, integers, continuous, reduction=False)
        torch.testing.assert_close(values, torch.tensor([12., 14.]))
        graph.input_Ivars_label_(integers[0])
        graph.input_Cvars_label_(continuous)
        torch.testing.assert_close(cal_objective_graph(graph, reduction=False), values[:1])
        target = target_func(weight_c=1., weight_f=1., if_reduction=False,
                             if_only_graph=False, if_Ivars_batch=True, if_Cvars_batch=False)
        torch.testing.assert_close(target(graph, integers, continuous), torch.tensor([13., 17.]))

    def test_unreduced_target_keeps_single_graph_axis(self):
        graph = make_graph()
        graph.objective_constant = 7.
        integers = torch.tensor([[0., 1.], [2., 1.]])
        continuous = torch.tensor([3.])
        actual = target_function_batch(graph, integers, continuous, weight=1., reduction=False)
        torch.testing.assert_close(actual, torch.tensor([[13.], [17.]]))

    def test_target_separates_three_samples_and_two_graphs(self):
        graphs = [make_graph(continuous=False), make_graph(continuous=False)]
        for graph, constant in zip(graphs, (7., 11.)):
            graph.only_discrete = True
            graph.objective_constant = constant
        batch = Batch.from_data_list(graphs)
        integers = torch.tensor([[3., 5., 4., 4.], [0., 1., 2., 3.], [1., 2., 3., 4.]])
        actual = target_function_batch(batch, integers, None, weight=1., reduction=False)
        expected = torch.tensor([[20. + math.sqrt(2.), 25.], [9., 19.], [12., 23.]])
        torch.testing.assert_close(actual, expected)
        reduced = target_function_batch(batch, integers, None, weight=1.)
        torch.testing.assert_close(reduced, torch.tensor(105. + math.sqrt(7.)))

    def test_repeated_gradient_calls_do_not_accumulate_or_change_features(self):
        graph = make_graph()
        graph.input_Cvars_label_(torch.tensor([3.]))
        before = graph.clone()
        for _ in range(2):
            with torch.no_grad():
                integers, continuous = gradient_function(graph)
            torch.testing.assert_close(integers, torch.tensor([11., 2.]))
            torch.testing.assert_close(continuous, torch.tensor([11.]))
            for kind in ('Ivars', 'Cvars'):
                self.assertFalse(graph[kind].x.requires_grad)
                self.assertIsNone(graph[kind].x.grad)
                torch.testing.assert_close(graph[kind].x, before[kind].x)

    def test_gradient_keeps_existing_feature_gradients_and_handles_discrete_graph(self):
        graph = make_graph(continuous=False)
        graph.only_discrete = True
        graph['Ivars'].x.requires_grad_()
        graph['Ivars'].x.grad = torch.full_like(graph['Ivars'].x, 7.)
        integers, continuous = gradient_function(graph)
        torch.testing.assert_close(integers, torch.tensor([1., 2.]))
        self.assertEqual(continuous.shape, (0,))
        torch.testing.assert_close(graph['Ivars'].x.grad, torch.full_like(graph['Ivars'].x, 7.))


if __name__ == '__main__':
    unittest.main()
