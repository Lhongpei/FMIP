"""Small GPU integration checks; run only in a GPU Slurm allocation."""
import unittest

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.precision import MixedPrecision

from test_data_network import make_graph
from test_training_inference import make_model, make_batch


@unittest.skipUnless(torch.cuda.is_available(), 'requires a GPU Slurm allocation')
class GPUIntegrationTests(unittest.TestCase):
    def test_amp_guidance_ignores_large_objective_offsets(self):
        model = make_model().cuda().eval()
        graph = make_graph().cuda()
        graph.objective_constant = 0.
        torch.manual_seed(57)
        with torch.autocast('cuda', dtype=torch.float16):
            expected = model.inference(graph, guidance=True)
        graph.objective_constant = 1e12
        torch.manual_seed(57)
        with torch.autocast('cuda', dtype=torch.float16):
            actual = model.inference(graph, guidance=True)
        np.testing.assert_allclose(actual['Ivars'], expected['Ivars'], atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(actual['Cvars'], expected['Cvars'], atol=1e-6, rtol=1e-5)

    def test_sparse_constraint_arithmetic_stays_float32_under_amp(self):
        from fmip.utils.milp_utils import constraint_matmul
        matrix = torch.diag(torch.tensor([2., 3.], device='cuda')).to_sparse()
        values = torch.ones(2, device='cuda', requires_grad=True)
        with torch.autocast('cuda', dtype=torch.float16):
            result = constraint_matmul(matrix, values)
            loss = result.square().sum()
        self.assertEqual(result.dtype, torch.float32)
        loss.backward()
        torch.testing.assert_close(values.grad, torch.tensor([8., 18.], device='cuda'))

    def test_cuda_training_and_all_flow_paths(self):
        torch.manual_seed(23)
        for discrete in (False, True):
            for mode in ('uniform', 'mask'):
                for schedule in ('linear', 'cosine', 'linearVP'):
                    with self.subTest(discrete=discrete, mode=mode, schedule=schedule):
                        model = make_model(discrete, mode, 3 if mode == 'mask' else 2).cuda()
                        model.CFM = type(model.CFM)(schedule)
                        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                        loss = model.training_step(make_batch(not discrete).cuda(), 0)
                        self.assertTrue(torch.isfinite(loss))
                        loss.backward()
                        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                        optimizer.step()
                        result = model.eval().inference(make_graph(continuous=not discrete).cuda(), guidance=True)
                        self.assertTrue(np.isfinite(result['Ivars']).all())
                        np.testing.assert_allclose(result['Ivars'].sum(-1), 1., atol=1e-5)
                        if not discrete:
                            self.assertTrue(np.isfinite(result['Cvars']).all())

    def test_lightning_mixed_precision_train_and_guided_validate(self):
        model = make_model()
        del model.log
        original_weights = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        model.train_dataset = [make_graph() for _ in range(2)]
        graph = make_graph()
        graph['name'] = 'gpu-validation.lp'
        model.validation_dataset = [graph]
        model.args.inference_type = 'parallel'
        # A short smoke test must make real updates rather than consume its
        # entire step budget calibrating the default large AMP loss scale.
        precision = MixedPrecision('16-mixed', 'cuda', scaler=torch.amp.GradScaler('cuda', init_scale=1.))
        trainer = Trainer(accelerator='gpu', devices=1, plugins=precision,
                          max_steps=2, max_epochs=2, logger=False,
                          enable_checkpointing=False, enable_progress_bar=False,
                          enable_model_summary=False, num_sanity_val_steps=0)
        trainer.fit(model)
        self.assertEqual(trainer.global_step, 2)
        self.assertTrue(any(not torch.equal(parameter.detach().cpu(), original_weights[name])
                            for name, parameter in model.named_parameters()))
        result = trainer.validate(model, verbose=False)
        self.assertTrue(np.isfinite(result[0]['val/metric']))
        model.args.inference_type = 'batch'
        model.inference_steps = 100
        model.inference_scheduler = type(model.inference_scheduler)(100, 'cosine')
        result = trainer.validate(model, verbose=False)
        self.assertTrue(np.isfinite(result[0]['val/metric']))
        model.cuda().eval()  # Lightning validation teardown moves the model to CPU.
        with torch.autocast('cuda', dtype=torch.float16):
            prediction = model.inference(make_graph().cuda())
        self.assertEqual(prediction['Ivars'].dtype, np.float32)

    def test_cuda_batched_graph_outputs_match_independent_graphs(self):
        from torch_geometric.data import Batch
        model = make_model().cuda().eval()
        first = make_graph().cuda()
        second = make_graph(offset=20.).cuda()
        with torch.no_grad():
            expected = model(first, torch.tensor([0.25], device='cuda'))
            actual = model(Batch.from_data_list([first, second]), torch.tensor([0.25], device='cuda'))
        torch.testing.assert_close(actual[0][:2], expected[0], rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(actual[1][:1], expected[1], rtol=1e-4, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
