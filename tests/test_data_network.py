"""Regressions for graph bounds and independent graph batching."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch_geometric.data import Batch

from fmip.dataset import GraphMILP, GraphMILPDataset
from fmip.models.gnn import MILPGCN
from fmip.models.gnn_baseline import MILPGCNNonRes
from fmip.models.layers import LinearEncoder
from fmip.models.nn import node_output, normalization


def make_graph(offset=0., continuous=True):
    graph = GraphMILP()
    graph['Ivars'].x = torch.tensor([
        [0., 1., 1., 1., 1., 0.],
        [0., 1., 2., 1., 1., 1.],
    ])
    graph['Ivars'].x[:, 2] += offset
    graph['Cvars'].x = (torch.tensor([[0., 4., 1. + offset, 1., 1., 2.]])
                          if continuous else torch.empty((0, 6)))
    graph['cons'].x = torch.tensor([[2. + offset], [4. + offset]])
    graph['Ivars', 'to', 'cons'].edge_index = torch.tensor([[0, 1], [0, 1]])
    graph['Ivars', 'to', 'cons'].edge_attr = torch.ones((2, 1))
    graph['Cvars', 'to', 'cons'].edge_index = (
        torch.tensor([[0, 0], [0, 1]]) if continuous else torch.empty((2, 0), dtype=torch.long))
    graph['Cvars', 'to', 'cons'].edge_attr = (
        torch.ones((2, 1)) if continuous else torch.empty((0, 1)))
    return graph


class GraphProjectionTests(unittest.TestCase):
    def test_inplace_projection_keeps_variable_types_separate(self):
        graph = make_graph()
        graph['Ivars'].x[:, -1] = torch.tensor([-2., 3.])
        graph['Cvars'].x[:, -1] = 6.
        graph.projection_()
        torch.testing.assert_close(graph.extract_Ivars_label(), torch.tensor([0., 1.]))
        torch.testing.assert_close(graph.extract_Cvars_label(), torch.tensor([4.]))

    def test_missing_bounds_do_not_clamp_to_placeholder_zero(self):
        graph = GraphMILP()
        graph['Cvars'].x = torch.tensor([
            [0., 0., 1., 1., 0., 0.],  # nonnegative, no upper bound
            [0., 0., 1., 0., 0., 0.],  # free variable
        ])
        values = torch.tensor([7., -3.])
        torch.testing.assert_close(graph.projection_Cvars(values), values)

    def test_integer_projection_is_a_valid_distribution(self):
        graph = make_graph()
        graph['Ivars'].x[0, :2] = torch.tensor([1., 1.])
        probabilities = torch.tensor([[0.9, 0.1, 0.], [0., 0., 1.]])
        projected = graph.projection_Ivars(probabilities)
        torch.testing.assert_close(projected, torch.tensor([[0., 1., 0.], [0.5, 0.5, 0.]]))
        torch.testing.assert_close(projected.sum(-1), torch.ones(2))

    def test_integer_projection_rejects_unrepresented_domain(self):
        graph = make_graph()
        graph['Ivars'].x[0, :2] = torch.tensor([3., 4.])
        with self.assertRaisesRegex(ValueError, 'no class'):
            graph.projection_Ivars(torch.full((2, 2), 0.5))

    def test_integer_projection_respects_absent_upper_bound(self):
        graph = make_graph()
        graph['Ivars'].x[:, 1] = 0.
        graph['Ivars'].x[:, 4] = 0.
        probabilities = torch.tensor([[0.2, 0.3, 0.5], [0.5, 0.4, 0.1]])
        torch.testing.assert_close(graph.projection_Ivars(probabilities), probabilities)

    def test_class_count_uses_allowed_values_not_only_solution(self):
        graph = make_graph()
        graph['Ivars'].x[:, -1] = 0.
        holder = SimpleNamespace(_data=graph)
        GraphMILPDataset.calculate_max_integer_(holder)
        self.assertEqual(holder.max_integer_num, 2)
        graph['Ivars'].x[0, 1] = 4.
        GraphMILPDataset.calculate_max_integer_(holder)
        self.assertEqual(holder.max_integer_num, 5)

    def test_trusted_cache_roundtrip_uses_corrected_encoding_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / 'cache'
            processed = Path(GraphMILPDataset.cache_directory(str(root), str(cache), prefix='test'))
            processed.mkdir(parents=True)
            # Legacy caches must not take precedence over the corrected format.
            (processed / 'data.pt').write_text('legacy cache')
            torch.save(GraphMILPDataset.collate([make_graph(), make_graph(offset=1.)]),
                       processed / 'data_v3.pt')
            dataset = GraphMILPDataset(str(root), str(cache), prefix='test')
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset.max_integer_num, 2)
            torch.testing.assert_close(dataset[1]['Ivars'].x, make_graph(offset=1.)['Ivars'].x)
            self.assertEqual((processed / 'data.pt').read_text(), 'legacy cache')


class NodeHeadTests(unittest.TestCase):
    def test_linear_encoder_can_run(self):
        encoder = LinearEncoder(3, 8, layer=2)
        self.assertEqual(encoder(torch.ones((4, 3))).shape, (4, 8))

    def test_head_batching_matches_separate_graphs_and_has_gradients(self):
        torch.manual_seed(0)
        head = nn.Sequential(normalization(64), nn.ReLU(), nn.Conv2d(64, 2, 1))
        first = torch.randn(3, 64)
        second = 20. + 5. * torch.randn(2, 64)
        combined = torch.cat([first, second]).requires_grad_()
        expected = torch.cat([node_output(head, first), node_output(head, second)])
        actual = node_output(head, combined, torch.tensor([0, 0, 0, 1, 1]))
        torch.testing.assert_close(actual, expected)
        actual.square().sum().backward()
        self.assertTrue(torch.isfinite(combined.grad).all())
        self.assertTrue(torch.isfinite(head[-1].weight.grad).all())

    def test_empty_modality_has_correct_output_shape(self):
        head = nn.Sequential(normalization(64), nn.ReLU(), nn.Conv2d(64, 2, 1))
        self.assertEqual(node_output(head, torch.empty((0, 64))).shape, (0, 2))

    def test_full_models_preserve_graph_predictions_in_batch(self):
        for model_type in (MILPGCN, MILPGCNNonRes):
            for discrete in (False, True):
                with self.subTest(model=model_type.__name__, discrete=discrete):
                    torch.manual_seed(7)
                    model = model_type(6, 6, 1, 64, 1, 2, only_discrete=discrete).eval()
                    # Residual branches start at zero. Activate them to also
                    # exercise message passing rather than only embeddings.
                    for collection in (model.per_layer_out_cons,
                                       model.per_layer_out_Ivars,
                                       model.per_layer_out_Cvars):
                        for layer in collection:
                            nn.init.normal_(layer[-1].weight, std=0.05)
                    first = make_graph(continuous=not discrete)
                    second = make_graph(offset=20., continuous=not discrete)
                    batch = Batch.from_data_list([first, second])
                    with torch.no_grad():
                        expected_i, expected_c = model(first, torch.tensor([0.25]))
                        actual_i, actual_c = model(batch, torch.tensor([0.25]))
                    torch.testing.assert_close(actual_i[:2], expected_i, rtol=1e-5, atol=1e-6)
                    if not discrete:
                        torch.testing.assert_close(actual_c[:1], expected_c, rtol=1e-5, atol=1e-6)

    def test_mixed_model_accepts_empty_continuous_modality(self):
        model = MILPGCN(6, 6, 1, 64, 1, 2).eval()
        with torch.no_grad():
            integers, continuous = model(make_graph(continuous=False), torch.tensor([0.5]))
        self.assertEqual(integers.shape, (2, 2))
        self.assertEqual(continuous.shape, (0,))


if __name__ == '__main__':
    unittest.main()
