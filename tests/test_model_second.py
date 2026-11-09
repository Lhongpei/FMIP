"""Integration regressions found in the second independent model audit."""
import math
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gurobipy as gp
import numpy as np
import torch
from pytorch_lightning import Trainer
from torch_geometric.data import Batch

from fmip.pl_gmip_model import GMIPModel
from fmip.train import arg_parser
from fmip.utils.milp_utils import cal_feasibility_graph
from test_data_network import make_graph
from test_training_inference import make_model, make_batch


class ModelSecondTests(unittest.TestCase):
    def test_fp16_subnormal_allowed_mass_still_normalizes(self):
        graph = make_graph(continuous=False)
        graph['Ivars'].x[:, 0] = 1
        probabilities = torch.tensor([[1., 1e-6], [1., 1e-6]], dtype=torch.float16)
        projected = graph.projection_Ivars(probabilities)
        torch.testing.assert_close(projected, torch.tensor([[0., 1.], [0., 1.]], dtype=torch.float16))

    def test_inference_preserves_callers_ground_truth(self):
        model = make_model().eval()
        graph = make_graph()
        original = graph.clone()
        model.inference(graph, guidance=True)
        torch.testing.assert_close(graph['Ivars'].x, original['Ivars'].x)
        torch.testing.assert_close(graph['Cvars'].x, original['Cvars'].x)

    def test_parallel_inference_accepts_existing_batch(self):
        model = make_model(True).eval()
        result = model.inference(make_batch(False), batchsize=3)
        self.assertEqual(result['Ivars'].shape, (3, 2, 2))

    def test_pure_parallel_evaluation_supports_discrete_and_empty_cvars(self):
        for discrete in (True, False):
            model = make_model(discrete).eval()
            model.R = 2
            result = model.pure_inference_parallel(make_batch(False), 0)
            self.assertTrue(all(np.isfinite(value) for value in result.values()))

    def test_zero_weight_target_batch_has_zero_finite_loss(self):
        args = arg_parser(['--only_discrete', '--max_integer_num', '2', '--ce_weight', '1', '0'])
        model = GMIPModel(args, load_datasets=False)
        logits = torch.zeros((2, 2), requires_grad=True)
        loss = model.discrete_training_loss(logits, torch.ones(2, dtype=torch.long))
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_out_of_bounds_continuous_prediction_keeps_learning_gradient(self):
        model = make_model()
        prediction = torch.nn.Parameter(torch.tensor([-2.]))
        with patch.object(model, 'pred_clean_sample', return_value=(torch.zeros(2, 2), prediction)):
            loss = model.training_step(make_batch(), 0)
        loss.backward()
        self.assertAlmostEqual(prediction.grad.item(), -8.)

    def test_single_sample_guidance_is_finite(self):
        model = make_model().eval()
        model.R = 1
        result = model.inference(make_graph(), guidance=True)
        self.assertTrue(np.isfinite(result['Ivars']).all())
        self.assertTrue(np.isfinite(result['Cvars']).all())

    def test_guidance_inside_inference_mode_can_compute_gradients(self):
        model = make_model().eval()
        with torch.inference_mode():
            result = model.inference(make_graph(), guidance=True)
        self.assertTrue(np.isfinite(result['Cvars']).all())

    def test_guidance_on_default_training_mode_model(self):
        model = make_model()
        self.assertTrue(model.training)
        result = model.inference(make_graph(), guidance=True)
        self.assertTrue(np.isfinite(result['Cvars']).all())

    def test_guidance_with_empty_continuous_nodes(self):
        model = make_model().eval()
        result = model.inference(make_graph(continuous=False), guidance=True)
        self.assertEqual(result['Cvars'].shape, (0,))

    def test_binary_mask_metrics_and_single_class_labels_are_finite(self):
        model = make_model(True, 'mask', 3)
        probabilities = torch.tensor([[0.8, 0.2, 0.], [0.3, 0.7, 0.]])
        for labels, ap in ((torch.zeros(2), 0.), (torch.ones(2), 1.)):
            metrics = model.cal_metrics(make_batch(False), probabilities, labels)
            self.assertTrue(all(math.isfinite(float(value)) for value in metrics.values()))
            self.assertEqual(metrics['pr_auc'], ap)

    def test_guidance_uses_problem_objective_sense(self):
        model = make_model(True)
        graph = make_graph(continuous=False)
        graph.model_sense = -1
        target = model.Ivars_batch_target
        with patch.object(target, 'objective_cal', return_value=torch.tensor([4., 1.])), \
             patch.object(target, 'feasibility_cal', return_value=torch.tensor([0., 0.])):
            value = model._guidance_target(target, graph, torch.zeros((2, 2)), None)
        self.assertLess(value[0], value[1])

    def test_guidance_option_is_applied_separately_to_graphs(self):
        model = make_model(True)
        model.args.use_guidance = True
        batch = Batch.from_data_list([make_graph(continuous=False), make_graph(continuous=False)])
        with patch.object(model, 'inference_parallel', return_value={'metric': 1.}) as guided:
            result = model.test_step(batch, 0)
        self.assertEqual(guided.call_count, 2)
        self.assertEqual(result['metric'], 1.)

    def test_guidance_cli_counts_and_mask_weights_are_honored(self):
        args = arg_parser(['--num_discrete_sample', '1', '--num_continuous_guide_step', '3',
                           '--discrete_flow_mode', 'mask', '--max_integer_num', '3',
                           '--ce_weight', '0.3', '0.7'])
        model = GMIPModel(args, load_datasets=False)
        self.assertEqual((model.R, model.N_iter), (1, 3))
        torch.testing.assert_close(model.discrete_loss.weight, torch.tensor([0.3, 0.7, 0.]))

    def test_prediction_save_names_cover_supported_extensions(self):
        model = make_model(True)
        graphs = []
        for name in ('a.lp', 'b.mps', 'c.mps.gz', 'd.proto.lp'):
            graph = make_graph(continuous=False)
            graph['name'] = name
            graphs.append(graph)
        batch = Batch.from_data_list(graphs)
        with tempfile.TemporaryDirectory() as directory:
            model.save_labels(torch.full((8, 2), 0.5), None, batch, directory)
            self.assertEqual({p.name for p in Path(directory).iterdir()},
                             {'a_pred.pkl', 'b_pred.pkl', 'c_pred.pkl', 'd_pred.pkl'})

    def test_real_lightning_guided_validation_uses_normal_tensors(self):
        model = make_model()
        del model.log  # Use real Lightning logging, not the unit-test stub.
        graph = make_graph()
        graph['name'] = 'validation.lp'
        model.validation_dataset = [graph]
        model.args.inference_type = 'parallel'
        trainer = Trainer(accelerator='cpu', devices=1, logger=False,
                          enable_checkpointing=False, enable_progress_bar=False,
                          enable_model_summary=False)  # default inference_mode=True
        result = trainer.validate(model, verbose=False)
        self.assertTrue(math.isfinite(result[0]['val/metric']))


class RealDatasetPipelineTests(unittest.TestCase):
    def test_solver_file_dataset_train_validate_checkpoint_reload(self):
        try:
            environment = gp.Env(params={'OutputFlag': 0, 'Threads': 1})
        except gp.GurobiError as error:
            self.skipTest(f'Gurobi license unavailable: {error}')
        with environment, tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            problems = root / 'toy' / 'problem'
            solutions = root / 'toy' / 'solution'
            problems.mkdir(parents=True)
            solutions.mkdir(parents=True)
            for index in range(6):
                with gp.Model(env=environment) as solver:
                    y = solver.addVar(lb=0, ub=4, name='y')
                    b = solver.addVar(vtype=gp.GRB.BINARY, name='b')
                    z = solver.addVar(lb=0, ub=4, name='z')
                    solver.addConstr(y + b >= 1)
                    solver.addConstr(z == y)
                    solver.addConstr(y + b <= 3)
                    solver.setObjective(b + y + 0.1 * z + 2, gp.GRB.MINIMIZE)
                    solver.optimize()
                    self.assertEqual(solver.Status, gp.GRB.OPTIMAL)
                    solver.write(str(problems / f'case{index}.lp'))
                    with (solutions / f'case{index}_sol.pkl').open('wb') as stream:
                        pickle.dump({'label_info': {v.VarName: v.X for v in solver.getVars()}}, stream)
            args = arg_parser(['--dataset_root', directory, '--dataset_name', 'toy',
                               '--dataset_cache', str(root / 'cache'), '--num_workers', '0',
                               '--batch_size', '2', '--hidden_dim', '32', '--n_layers', '1',
                               '--inference_steps', '2', '--num_epochs', '2'])
            model = GMIPModel(args)
            example = Batch.from_data_list([model.train_dataset[0]])
            self.assertAlmostEqual(cal_feasibility_graph(example).item(), 0.)
            trainer = Trainer(accelerator='cpu', devices=1, max_steps=2, max_epochs=2,
                              logger=False, enable_checkpointing=False,
                              enable_progress_bar=False, enable_model_summary=False,
                              num_sanity_val_steps=0)
            trainer.fit(model)
            self.assertEqual(trainer.global_step, 2)
            self.assertTrue(math.isfinite(trainer.validate(model, verbose=False)[0]['val/metric']))
            checkpoint = root / 'model.ckpt'
            trainer.save_checkpoint(checkpoint)
            restored = GMIPModel.load_from_checkpoint(checkpoint, map_location='cpu')
            self.assertFalse(hasattr(restored, 'train_dataset'))
            self.assertEqual(restored.args.n_layers, 1)
            self.assertEqual(restored.args.hidden_dim, 32)
            self.assertEqual(restored.max_integer_num, 2)
            for key, expected in model.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key], expected)
            prediction = restored.eval().inference(model.test_dataset[0])
            self.assertEqual(prediction['Ivars'].shape, (1, 2))
            self.assertTrue(np.isfinite(prediction['Cvars']).all())


if __name__ == '__main__':
    unittest.main()
