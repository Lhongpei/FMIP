"""Small graph regressions for the real Lightning model and samplers."""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from pytorch_lightning import Trainer
from torch_geometric.data import Batch

from fmip.pl_gmip_model import GMIPModel
from fmip.pl_meta_model import COMetaModel
from fmip.train import arg_parser
from test_data_network import make_graph


def make_model(discrete=False, mode="uniform", classes=2):
    args = arg_parser([])
    args.max_integer_num = classes
    args.only_discrete = discrete
    args.discrete_flow_mode = mode
    args.n_layers = 1
    args.hidden_dim = 64
    args.inference_steps = 3
    args.inference_schedule = "linear"
    args.num_workers = 0
    args.batch_size = 8
    model = GMIPModel(args, load_datasets=False)
    model.log = lambda *args, **kwargs: None
    model.R = 4
    model.N_iter = 1
    return model


def make_batch(continuous=True):
    graph = make_graph(continuous=continuous)
    graph['name'] = 'toy.mps.gz'
    return Batch.from_data_list([graph])


class TrainingInferenceTests(unittest.TestCase):
    def test_training_noise_preserves_times_and_node_alignment(self):
        model = make_model()
        times = np.array([0.2, 0.5, 0.8])
        counts = torch.tensor([2, 0, 1])
        labels = torch.tensor([1., 2., 3.])
        expected_times = torch.tensor([0.2, 0.2, 0.8])
        with patch.object(model.DFM, 'sample', return_value=labels) as sampler:
            _, discrete_times = model.discrete_training_noise(labels, counts, times)
        torch.testing.assert_close(sampler.call_args.args[1], expected_times)
        torch.testing.assert_close(discrete_times, expected_times)
        with patch('torch.randn_like', return_value=torch.zeros_like(labels)):
            sample, continuous_times, velocity = model.continuous_training_noise(labels, counts, times)
        torch.testing.assert_close(continuous_times, expected_times)
        torch.testing.assert_close(sample, expected_times * labels)
        torch.testing.assert_close(velocity, labels)

    def test_training_backward_for_mixed_discrete_and_empty_continuous(self):
        for discrete, continuous in ((False, True), (True, False), (False, False)):
            with self.subTest(discrete=discrete, continuous=continuous):
                model = make_model(discrete=discrete)
                batch = make_batch(continuous=continuous)
                loss = model.training_step(batch, 0)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_real_inference_and_guidance_return_valid_probabilities(self):
        for discrete in (False, True):
            for mode in ('uniform', 'mask'):
                for guided in (False, True):
                    with self.subTest(discrete=discrete, mode=mode, guided=guided):
                        model = make_model(discrete, mode, 3 if mode == 'mask' else 2).eval()
                        result = model.inference(make_graph(continuous=not discrete), guidance=guided)
                        probabilities = torch.tensor(result['Ivars'])
                        self.assertTrue(torch.isfinite(probabilities).all())
                        self.assertTrue((probabilities >= 0).all())
                        torch.testing.assert_close(probabilities.sum(-1), torch.ones(2))
                        if mode == 'mask':
                            torch.testing.assert_close(probabilities[:, -1], torch.zeros(2))
                        if not discrete:
                            self.assertEqual(result['Cvars'].shape, (1,))
                            self.assertTrue(np.isfinite(result['Cvars']).all())

    def test_mask_is_excluded_even_for_unbounded_integer_variables(self):
        model = make_model(True, 'mask', 3)
        graph = make_graph(continuous=False)
        graph['Ivars'].x[:, 4] = 0
        probabilities = model.project_integer_probabilities(graph, torch.tensor([[0., 0., 1.], [0.2, 0.3, 0.5]]))
        torch.testing.assert_close(probabilities, torch.tensor([[0.5, 0.5, 0.], [0.4, 0.6, 0.]]))

    def test_multiclass_parallel_samples_keep_class_dimension(self):
        model = make_model(True, classes=4).eval()
        graph = make_graph(continuous=False)
        graph['Ivars'].x[:, 1] = 3
        result = model.inference(graph, batchsize=2)
        self.assertEqual(result['Ivars'].shape, (2, 2, 4))

    def test_parallel_sampling_honors_guidance(self):
        model = make_model(True).eval()
        prediction = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
        with patch.object(model, 'guided_inference_core', return_value=(prediction, None)) as guided:
            result = model.inference(make_graph(continuous=False), guidance=True, batchsize=2)
        self.assertEqual(guided.call_count, 2)
        self.assertEqual(result['Ivars'].shape, (2, 2, 2))

    def test_probability_metrics_do_not_apply_second_softmax(self):
        model = make_model(True)
        prediction = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
        metrics = model.cal_metrics(make_batch(False), prediction, torch.tensor([0, 1]))
        self.assertAlmostEqual(metrics['ce_loss'], -(math.log(0.8) + math.log(0.7)) / 2, places=6)

    def test_feasibility_uses_final_continuous_prediction(self):
        model = make_model()
        batch = make_batch()
        prediction = torch.tensor([[1., 0.], [0., 1.]])
        with patch.object(model, 'inference_core', return_value=(prediction, torch.tensor([10.]))):
            metrics = model.inference_batch_multimodal(batch, 0)
        torch.testing.assert_close(batch.extract_Cvars_label(), torch.tensor([10.]))
        self.assertGreater(metrics['feasibility'], 0)

    def test_parallel_evaluation_supports_pure_discrete_graphs(self):
        model = make_model(True).eval()
        metrics = model.inference_parallel(make_batch(False), 0)
        self.assertEqual(metrics['Cvar_mse'], 0)
        self.assertLessEqual(metrics['metric'], 0)

    def test_loader_with_zero_workers_keeps_small_training_dataset(self):
        model = make_model()
        model.train_dataset = [make_graph()]
        self.assertEqual(len(list(model.train_dataloader())), 1)
        model.validation_dataset = [make_graph()]
        model.args.validation_examples = 10
        self.assertEqual(len(list(model.val_dataloader())), 1)

    def test_classes_cover_all_splits_and_reserve_mask(self):
        args = arg_parser([])
        args.discrete_flow_mode = 'mask'
        def load_datasets(model):
            model.train_dataset = SimpleNamespace(max_integer_num=5)
            model.validation_dataset = SimpleNamespace(max_integer_num=3)
            model.test_dataset = SimpleNamespace(max_integer_num=2)
        with patch.object(COMetaModel, 'load_datasets', load_datasets):
            model = COMetaModel(args)
        self.assertEqual(model.max_integer_num, 6)

    def test_optimizer_schedule_uses_lightning_step_estimate(self):
        model = make_model()
        model._trainer = SimpleNamespace(estimated_stepping_batches=7)
        self.assertEqual(model.get_total_num_training_steps(), 7)

    def test_lightning_can_run_optimizer_steps(self):
        model = make_model()
        model.train_dataset = [make_graph()]
        model.validation_dataset = [make_graph()]
        trainer = Trainer(accelerator='cpu', devices=1, max_steps=2, max_epochs=2,
                          logger=False, enable_checkpointing=False,
                          enable_progress_bar=False, enable_model_summary=False,
                          num_sanity_val_steps=0, limit_val_batches=0)
        trainer.fit(model)
        self.assertEqual(trainer.global_step, 2)

    def test_nonlinear_terminal_sample_respects_bounds(self):
        model = make_model()
        model.CFM = type(model.CFM)('cosine')
        graph = make_graph()
        graph.input_Cvars_label_(torch.zeros(1))
        with patch.object(model, 'pred_clean_sample', return_value=(torch.zeros(2, 2), torch.tensor([4.]))):
            _, continuous = model.mixed_denoise_step(graph, torch.tensor([0.]), torch.tensor([1.]))
        self.assertTrue((continuous <= 4).all())

    def test_focal_training_supports_more_than_two_classes(self):
        args = arg_parser([])
        args.max_integer_num = 4
        args.only_discrete = True
        args.use_focal_loss = True
        args.n_layers = 1
        model = GMIPModel(args, load_datasets=False)
        model.log = lambda *args, **kwargs: None
        graph = make_graph(continuous=False)
        graph['Ivars'].x[:, 1] = 3
        graph['Ivars'].x[:, -1] = 3
        loss = model.training_step(Batch.from_data_list([graph]), 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()


if __name__ == '__main__':
    unittest.main()
