"""Second audit: numerical boundaries, independent batches, and CLI contracts."""

from contextlib import redirect_stderr
import io
import math
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from fmip import train
from fmip.utils.flow_scheduler import ContinuousFlow, DiscreteFlow, InferenceScheduler
from fmip.utils.inference_schedulers import (
    CondOTScheduler, PolynomialConvexScheduler, VPScheduler,
    LinearVPScheduler, CosineScheduler,
)
from fmip.utils.lr_schedulers import get_one_cycle, get_schedule_fn
from fmip.utils.utils import at, at_onehot, generate_unique_samples, find_respect_solution


class NumericalSchedulerTests(unittest.TestCase):
    def test_inverse_snr_endpoints_and_large_values(self):
        snr = torch.tensor([0., 1., 1e20, float('inf')], dtype=torch.float64)
        for scheduler in (CondOTScheduler(), PolynomialConvexScheduler(2),
                          LinearVPScheduler(), CosineScheduler()):
            with self.subTest(scheduler=type(scheduler).__name__):
                times = scheduler.snr_inverse(snr)
                self.assertTrue(torch.isfinite(times).all())
                self.assertEqual(times[0].item(), 0.)
                self.assertEqual(times[-1].item(), 1.)
                self.assertTrue((times[1:] >= times[:-1]).all())

    def test_vp_inverse_handles_constant_beta_and_near_terminal_time(self):
        times = torch.tensor([0., 0.25, 0.9, 1 - 1e-10, 1.], dtype=torch.float64)
        for bounds in ((0.1, 20.), (0.5, 0.5)):
            scheduler = VPScheduler(*bounds)
            output = scheduler(times)
            snr = output.alpha_t / output.sigma_t
            torch.testing.assert_close(scheduler.snr_inverse(snr), times, rtol=1e-8, atol=1e-12)
            self.assertEqual(scheduler.snr_inverse(torch.tensor([float('inf')])).item(), 1.)

    def test_cosine_endpoint_has_zero_noise_in_float32(self):
        schedule = CosineScheduler()
        output = schedule(torch.tensor([1.]))
        torch.testing.assert_close(output.sigma_t, torch.zeros(1))
        torch.testing.assert_close(schedule.snr_inverse(output.alpha_t / output.sigma_t), torch.ones(1))

    def test_inverse_snr_rejects_invalid_domain(self):
        for scheduler in (CondOTScheduler(), VPScheduler(), LinearVPScheduler(), CosineScheduler()):
            for invalid in (-1., float('nan')):
                with self.subTest(scheduler=type(scheduler).__name__, value=invalid):
                    with self.assertRaises(ValueError):
                        scheduler.snr_inverse(torch.tensor([invalid]))
        for bounds in ((0., 1.), (2., 1.), (1., float('inf'))):
            with self.assertRaises(ValueError):
                VPScheduler(*bounds)

    def test_inference_schedule_rejects_zero_steps_and_bad_indices(self):
        for steps in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                InferenceScheduler(steps)
        with self.assertRaises(ValueError):
            InferenceScheduler(3, 'missing')
        for index in (-1, 3):
            with self.assertRaises(ValueError):
                InferenceScheduler(3)(index)

    def test_inference_intervals_preserve_state_dtype(self):
        prediction = torch.tensor([[0.2, 0.8]], dtype=torch.float32)
        start = torch.tensor([0.25], dtype=torch.float64)
        end = torch.tensor([0.5], dtype=torch.float64)
        probabilities = DiscreteFlow().next_state_probs(prediction, torch.tensor([0]), start, end)
        self.assertEqual(probabilities.dtype, prediction.dtype)
        sample = ContinuousFlow().denoise(torch.tensor([1.]), torch.tensor([0.]), start, end)
        self.assertEqual(sample.dtype, torch.float32)

    def test_zero_length_terminal_step_is_finite_identity(self):
        current = torch.tensor([1, 0])
        prediction = torch.tensor([[0.2, 0.8], [0.6, 0.4]])
        probabilities = DiscreteFlow().next_state_probs(prediction, current, 1., 1.)
        torch.testing.assert_close(probabilities, torch.nn.functional.one_hot(current, 2).float())
        values = torch.tensor([2., -1.])
        for scheduler in ('linear', 'cosine', 'linearVP'):
            torch.testing.assert_close(ContinuousFlow(scheduler).denoise(values, values, 1., 1.), values)

    def test_flow_rejects_invalid_intervals(self):
        for start, end in ((-0.1, 0.2), (0.5, 0.4), (0., 1.1), (float('nan'), 1.)):
            with self.subTest(start=start, end=end):
                with self.assertRaises(ValueError):
                    DiscreteFlow().next_state_probs(torch.tensor([[0.5, 0.5]]), torch.tensor([0]), start, end)
                with self.assertRaises(ValueError):
                    ContinuousFlow().denoise(torch.ones(1), torch.zeros(1), start, end)

    def test_one_cycle_never_produces_negative_learning_rate(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scheduler = get_one_cycle(optimizer, 2)
        for _ in range(5):
            optimizer.step()
            scheduler.step()
            self.assertGreaterEqual(optimizer.param_groups[0]['lr'], 0)
        with self.assertRaises(ValueError):
            get_schedule_fn('cosine-decay', 0)


class UtilityBatchTests(unittest.TestCase):
    def test_onehot_coefficients_match_integer_lookup_without_batch_mixing(self):
        matrices = torch.tensor([[[1., 2.], [3., 4.]], [[10., 20.], [30., 40.]]])
        times = torch.tensor([0, 1])
        labels = torch.tensor([[0, 1, 0], [1, 0, 1]])
        onehot = torch.nn.functional.one_hot(labels, 2).float()
        torch.testing.assert_close(at_onehot(matrices, times, onehot), at(matrices, times, labels))
        torch.testing.assert_close(at_onehot(matrices, times, onehot[:, :1]), at(matrices, times, labels[:, :1]))

    def test_unique_sampling_returns_at_most_requested_count(self):
        draws = [torch.tensor([[0, 0], [0, 0]]), torch.tensor([[0, 1], [1, 0]])]
        with patch('torch.multinomial', side_effect=draws):
            samples = generate_unique_samples(np.full((2, 2), 0.5), 2, max_attempts=1)
        np.testing.assert_array_equal(samples, np.array([[0, 0], [0, 1]]))
        self.assertEqual(generate_unique_samples(np.ones((3, 2)), 0).shape, (0, 3))

    def test_solution_names_replace_only_final_extension(self):
        self.assertEqual(find_respect_solution('/solutions', 'foo.lp.backup.lp'), '/solutions/foo.lp.backup_sol.pkl')
        self.assertEqual(find_respect_solution('/solutions', 'upper.MPS.GZ'), '/solutions/upper_sol.pkl')
        self.assertEqual(find_respect_solution('/solutions', 'noext'), '/solutions/noext_sol.pkl')


class CliExecutionTests(unittest.TestCase):
    def test_cli_rejects_invalid_counts_and_modes(self):
        for options in (['--inference_steps', '0'], ['--num_workers', '-1'],
                        ['--flow_schedule', 'unknown'], ['--batch_size', '0'],
                        ['--max_integer_num', '0'], ['--use_activation_checkpoint'],
                        ['--num_discrete_guide_step', '2']):
            with self.subTest(options=options), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    train.arg_parser(options)

    def test_no_action_or_invalid_resume_fails_before_dataset_loading(self):
        for options in ([], ['--do_train', '--resume_weight_only'], ['--do_train', '--do_valid_only']):
            with patch.object(train, 'GMIPModel') as model:
                with self.assertRaises(ValueError):
                    train.main(train.arg_parser(options))
                model.assert_not_called()

    def test_short_training_uses_available_checkpoint_and_guidance_gradients(self):
        args = train.arg_parser(['--do_train', '--do_test', '--num_epochs', '1', '--use_guidance'])
        trainer = MagicMock()
        checkpoint = SimpleNamespace(best_model_path='', last_model_path='last.ckpt')
        logger = SimpleNamespace(save_dir='/tmp/test', name='run', version='1', _id='1')
        with patch.object(train, 'GMIPModel') as model_class, \
             patch.object(train, 'Trainer', return_value=trainer) as trainer_class, \
             patch.object(train, 'WandbLogger', return_value=logger), \
             patch.object(train, 'ModelCheckpoint', return_value=checkpoint), \
             patch.object(train, 'rank_zero_info'):
            train.main(args)
        trainer.test.assert_called_once_with(model_class.return_value, ckpt_path='last.ckpt')
        self.assertEqual(trainer_class.call_args.kwargs['check_val_every_n_epoch'], 1)
        self.assertFalse(trainer_class.call_args.kwargs['inference_mode'])

    def test_weight_only_resume_explicitly_loads_datasets_once(self):
        args = train.arg_parser(['--do_train', '--resume_weight_only', '--ckpt_path', 'weights.ckpt'])
        trainer = MagicMock()
        logger = SimpleNamespace(save_dir='/tmp/test', name='run', version='1', _id='1')
        with patch.object(train, 'GMIPModel') as model_class, \
             patch.object(train, 'Trainer', return_value=trainer), \
             patch.object(train, 'WandbLogger', return_value=logger), \
             patch.object(train, 'ModelCheckpoint'), patch.object(train, 'rank_zero_info'), \
             patch.object(torch, 'load', return_value={}), \
             patch('fmip.utils.gp_utils.checkpoint_model_args', side_effect=lambda _, runtime_args: runtime_args):
            train.main(args)
        model_class.assert_not_called()
        model_class.load_from_checkpoint.assert_called_once_with('weights.ckpt', param_args=args, load_datasets=True)
        trainer.fit.assert_called_once_with(model_class.load_from_checkpoint.return_value)

    def test_distributed_strategy_allows_variable_types_to_be_unused(self):
        args = train.arg_parser(['--do_train'])
        logger = SimpleNamespace(save_dir='/tmp/test', name='run', version='1', _id='1')
        with patch.object(train, 'GMIPModel'), patch.object(train, 'Trainer') as trainer_class, \
             patch.object(train, 'WandbLogger', return_value=logger), \
             patch.object(train, 'ModelCheckpoint'), patch.object(train, 'rank_zero_info'), \
             patch.object(train, 'DDPStrategy') as strategy, \
             patch.object(torch.cuda, 'device_count', return_value=2), \
             patch.object(torch.cuda, 'is_available', return_value=True):
            train.main(args)
        strategy.assert_called_once_with(find_unused_parameters=True)
        self.assertIs(trainer_class.call_args.kwargs['strategy'], strategy.return_value)


if __name__ == '__main__':
    unittest.main()
