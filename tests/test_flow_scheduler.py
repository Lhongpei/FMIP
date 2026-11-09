import math
import unittest
from unittest.mock import patch

import torch

from fmip.utils.flow_scheduler import ContinuousFlow, DiscreteFlow


class FlowSchedulerTests(unittest.TestCase):
    def test_continuous_sample_preserves_fractional_times(self):
        clean = torch.tensor([2.0, 4.0, 6.0, 8.0])
        noise = torch.tensor([-2.0, -4.0, -6.0, -8.0])
        with patch("torch.randn_like", return_value=noise):
            sample, velocity = ContinuousFlow().sample(clean, [0.0, 0.25, 0.5, 1.0])
        torch.testing.assert_close(sample, torch.tensor([-2.0, -2.0, 0.0, 8.0]))
        torch.testing.assert_close(velocity, clean - noise)

    def test_continuous_denoise_matches_each_training_path(self):
        clean = torch.tensor([2.0, -3.0], dtype=torch.float64)
        noise = torch.tensor([-1.0, 4.0], dtype=torch.float64)
        time = 0.3
        step = 0.05
        epsilon = 1e-6
        paths = {
            "linear": lambda t: t * clean + (1 - t) * noise,
            "cosine": lambda t: math.sin(math.pi * t / 2) * clean
            + math.cos(math.pi * t / 2) * noise,
            "linearVP": lambda t: t * clean + math.sqrt(1 - t**2) * noise,
        }
        for scheduler, path in paths.items():
            with self.subTest(scheduler=scheduler):
                current = path(time)
                velocity = (path(time + epsilon) - path(time - epsilon)) / (2 * epsilon)
                actual = ContinuousFlow(scheduler).denoise(clean, current, time, time + step)
                torch.testing.assert_close(actual, current + step * velocity)

    def test_uniform_terminal_distribution_is_clean_prediction(self):
        flow = DiscreteFlow(max_integer_num=3)
        prediction = torch.tensor([[0.1, 0.2, 0.7], [0.4, 0.5, 0.1]])
        current = torch.tensor([0, 2])
        torch.testing.assert_close(flow.next_state(prediction, current, 0.4, 1.0), prediction)

    def test_uniform_stochastic_step_has_valid_probabilities(self):
        flow = DiscreteFlow(max_integer_num=3)
        flow.noise = 10
        prediction = torch.tensor([[0.0, 0.5, 0.5]])
        probabilities = flow.next_state_probs(prediction, torch.tensor([0]), 0.0, 0.5)
        self.assertTrue(torch.all(probabilities >= 0))
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(1))
        self.assertEqual(flow.next_state(prediction, torch.tensor([0]), 0.0, 0.5).shape, (1,))

    def test_mask_transition_unmasks_current_mask_tokens(self):
        flow = DiscreteFlow(max_integer_num=3, mode="mask")
        prediction = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        current = torch.tensor([2, 0])
        probabilities = flow.next_state_probs(prediction, current, 0.0, 0.5)
        torch.testing.assert_close(probabilities, torch.tensor([[0.5, 0.0, 0.5], [1.0, 0.0, 0.0]]))
        terminal = flow.next_state(prediction, current, 0.5, 1.0)
        torch.testing.assert_close(terminal, torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
        torch.testing.assert_close(current, torch.tensor([2, 0]))

    def test_mask_rates_preserve_unmasked_tokens_without_noise(self):
        flow = DiscreteFlow(max_integer_num=3, mode="mask")
        predictions = torch.tensor([[0, 1], [1, 1]])
        rates = flow.rate_matrix(predictions, torch.tensor([2, 0]), 0.5, 0.75)
        expected = torch.tensor([[[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                                 [[0.0, 2.0, 0.0], [0.0, 0.0, 0.0]]])
        torch.testing.assert_close(rates, expected)

    def test_mask_noise_remasks_current_clean_tokens(self):
        flow = DiscreteFlow(max_integer_num=3, mode="mask")
        flow.noise = 2
        prediction = torch.tensor([[0.0, 1.0, 0.0]])
        probabilities = flow.next_state_probs(prediction, torch.tensor([0]), 0.0, 0.25)
        torch.testing.assert_close(probabilities, torch.tensor([[0.5, 0.0, 0.5]]))
        rates = flow.rate_matrix(torch.tensor([[1]]), torch.tensor([0]), 0.0, 0.25)
        torch.testing.assert_close(rates, torch.tensor([[[0.0, 0.0, 2.0]]]))


if __name__ == "__main__":
    unittest.main()
