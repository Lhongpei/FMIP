"""Numerical invariants, effective optimizer updates, and prediction publication."""

from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from fmip.utils.lr_schedulers import get_schedule_fn
from fmip.utils.milp_utils import cal_objective_graph
from test_data_network import make_graph
from test_training_inference import make_model, make_batch


class ObjectiveOffsetTests(unittest.TestCase):
    def test_guidance_ranking_is_invariant_to_large_objective_offsets(self):
        model = make_model(discrete=True)
        graph = make_graph(continuous=False)
        graph.only_discrete = True
        samples = torch.tensor([[0., 1.], [1., 0.]])
        for sense in (1, -1):
            graph.model_sense = sense
            graph.objective_constant = 0.
            expected = model._guidance_target(model.Ivars_batch_target, graph, samples, None)
            torch.testing.assert_close(expected, sense * torch.tensor([0.004, 0.002]))
            for offset in (1e12, -1e12):
                with self.subTest(sense=sense, offset=offset):
                    graph.objective_constant = offset
                    actual = model._guidance_target(model.Ivars_batch_target, graph, samples, None)
                    torch.testing.assert_close(actual, expected)

    def test_continuous_guidance_and_reported_objectives_keep_distinct_contracts(self):
        model = make_model()
        graph = make_graph()
        integers = torch.tensor([0., 1.])
        continuous = torch.tensor([1.], requires_grad=True)
        graph.objective_constant = 0.
        expected = model._guidance_target(model.Cvars_batch_target, graph, integers, continuous)
        expected_grad, = torch.autograd.grad(expected, continuous)
        graph.objective_constant = 1e12
        actual = model._guidance_target(model.Cvars_batch_target, graph, integers, continuous)
        actual_grad, = torch.autograd.grad(actual, continuous)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_grad, expected_grad)
        graph.objective_constant = 7.
        self.assertEqual(cal_objective_graph(graph).item(), 11.)

    def test_seeded_guided_inference_is_unchanged_by_an_objective_offset(self):
        model = make_model().eval()
        graph = make_graph()
        graph.objective_constant = 0.
        torch.manual_seed(57)
        expected = model.inference(graph, guidance=True)
        graph.objective_constant = 1e12
        torch.manual_seed(57)
        actual = model.inference(graph, guidance=True)
        np.testing.assert_allclose(actual['Ivars'], expected['Ivars'])
        np.testing.assert_allclose(actual['Cvars'], expected['Cvars'])


class OptimizerUpdateTests(unittest.TestCase):
    def test_single_step_one_cycle_performs_a_real_parameter_update(self):
        parameter = torch.nn.Parameter(torch.tensor([1.]))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scheduler = get_schedule_fn('one-cycle', 1)(optimizer)
        parameter.square().sum().backward()
        optimizer.step()
        scheduler.step()
        torch.testing.assert_close(parameter.detach(), torch.tensor([0.8]))
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.)


class PredictionPublicationTests(unittest.TestCase):
    def test_nonfinite_model_predictions_are_not_replaced_with_uniform_probabilities(self):
        model = make_model(discrete=True).eval()
        invalid_logits = torch.full((2, 2), float('nan'))
        with patch.object(model, 'pred_clean_sample', return_value=(invalid_logits, None)):
            with self.assertRaisesRegex(ValueError, 'finite and nonnegative'):
                model.inference(make_graph(continuous=False))

    def test_failed_prediction_write_preserves_the_previous_complete_file(self):
        for discrete in (False, True):
            with self.subTest(discrete=discrete), TemporaryDirectory() as directory:
                model = make_model(discrete)
                graph = make_batch(continuous=not discrete)
                destination = Path(directory) / 'toy_pred.pkl'
                previous = {'Ivars': np.array([[0.2, 0.8], [0.3, 0.7]])}
                original = pickle.dumps(previous)
                destination.write_bytes(original)

                def fail_during_serialization(value, stream):
                    stream.write(b'partial prediction')
                    raise OSError('simulated interrupted output')

                with patch('fmip.pl_gmip_model.pickle.dump', side_effect=fail_during_serialization):
                    with self.assertRaisesRegex(OSError, 'interrupted output'):
                        model.save_labels(torch.full((2, 2), 0.5), torch.tensor([2.]), graph, directory)
                self.assertEqual(destination.read_bytes(), original)
                self.assertEqual(list(Path(directory).iterdir()), [destination])


if __name__ == '__main__':
    unittest.main()
