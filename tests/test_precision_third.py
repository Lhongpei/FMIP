"""CPU mixed precision is the CLI fallback when --fp16 has no CUDA device."""

import unittest

import numpy as np
import torch
from pytorch_lightning import Trainer

from test_data_network import make_graph
from test_training_inference import make_model
from fmip.utils.flow_scheduler import DiscreteFlow


class CPUMixedPrecisionTests(unittest.TestCase):
    def test_low_precision_probabilities_keep_fractional_terminal_time(self):
        for dtype in (torch.float16, torch.bfloat16):
            probabilities = torch.tensor([[0.25, 0.75]], dtype=dtype)
            actual = DiscreteFlow().next_state_probs(probabilities, torch.tensor([0]), 0.9999, 1.)
            self.assertEqual(actual.dtype, torch.float32)
            torch.testing.assert_close(actual, probabilities.float())

    def test_bfloat16_inference_preserves_terminal_probabilities(self):
        for discrete in (True, False):
            model = make_model(discrete).eval()
            model.inference_steps = 100
            model.inference_scheduler = type(model.inference_scheduler)(100, 'cosine')
            with self.subTest(discrete=discrete):
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    result = model.inference(make_graph(continuous=not discrete))
                self.assertEqual(result['Ivars'].dtype, np.float32)
                self.assertTrue(np.isfinite(result['Ivars']).all())
                np.testing.assert_allclose(result['Ivars'].sum(-1), 1., atol=1e-5)

    def test_real_cpu_bfloat16_training_and_validation(self):
        model = make_model()
        del model.log
        original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        graph = make_graph()
        graph['name'] = 'cpu-amp.lp'
        model.train_dataset = [graph]
        model.validation_dataset = [graph]
        trainer = Trainer(accelerator='cpu', devices=1, precision='bf16-mixed',
                          max_steps=2, max_epochs=2, logger=False, enable_checkpointing=False,
                          enable_progress_bar=False, enable_model_summary=False,
                          num_sanity_val_steps=0)
        trainer.fit(model)
        self.assertTrue(any(not torch.equal(parameter.detach(), original[name])
                            for name, parameter in model.named_parameters()))
        self.assertTrue(np.isfinite(trainer.validate(model, verbose=False)[0]['val/metric']))


if __name__ == '__main__':
    unittest.main()
