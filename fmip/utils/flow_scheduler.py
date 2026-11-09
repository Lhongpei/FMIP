"""Schedulers for Flow Matching Models"""

import math
import numpy as np
import torch


def _time_interval(t_start, t_end, reference):
    """Use the state dtype/device and reject intervals outside the flow domain."""
    start = torch.as_tensor(t_start, device=reference.device, dtype=reference.dtype)
    end = torch.as_tensor(t_end, device=reference.device, dtype=reference.dtype)
    if start.numel() != 1 or end.numel() != 1:
        raise ValueError("Inference requires scalar start and end times")
    if not (torch.isfinite(start).all() and torch.isfinite(end).all()
            and (start >= 0).all() and (end <= 1).all() and (end >= start).all()):
        raise ValueError("Inference times must satisfy 0 <= t_start <= t_end <= 1")
    return start, end


# from flow_scheduler import *
class DiscreteFlow(object):
    """Discrete Flow process with linear beta scheduling

    Args:
        max_integer_num (int): Number of classes
        mode (str): Noise mode in ["uniform", "mask"]
            notice that "mask" assume real class number is max_integer_num - 1
    """

    def __init__(self, max_integer_num=2, mode="uniform"):
        self.mode_options = ["uniform", "mask"]
        if mode not in self.mode_options:
            raise ValueError(f"Unsupported discrete flow mode: {mode}")
        if not isinstance(max_integer_num, int) or max_integer_num < (2 if mode == "mask" else 1):
            raise ValueError("The class count must be positive and mask mode needs a separate mask class")
        self.max_integer_num = max_integer_num
        self.mode = mode
        self.noise = 0

    def x_0_sample(self, shape, device=None) -> torch.Tensor:
        return (
            torch.randint(0, self.max_integer_num, shape, device=device)
            if self.mode == "uniform"
            else torch.full(shape, self.max_integer_num - 1, device=device)
        )

        # if mode ==
        ## noise * dt * D is the average number of dimensions that get re-masked each timestep

    def sample(self, x1, t) -> torch.Tensor:
        t = torch.as_tensor(t, device=x1.device, dtype=torch.float32)
        xt = x1.clone().long()
        num_nodes = x1.shape[0]
        t = t.view(-1)
        corrupt_mask = torch.rand(num_nodes, device=x1.device) < 1 - t
        if self.mode == "uniform":
            uniform_noise = torch.randint(
                0, self.max_integer_num, (num_nodes,), device=x1.device
            )
            value = uniform_noise[corrupt_mask]
        else:
            value = self.max_integer_num - 1
        xt[corrupt_mask] = value
        return xt

    # def obtain_rate_matrix(self, x1_probs: torch.Tensor, t_start, t_end) -> torch.Tensor:
    #     dt = t_end - t_start
    #     t = t_start
    #     device = x1_probs.device
    #     if type(t) != torch.Tensor:
    #         t = torch.tensor(t, device=device)
    #     if self.mode == "uniform":
    
    def next_state(
        self, x1_probs: torch.Tensor, xt: torch.Tensor, t_start, t_end
        ) -> torch.Tensor:
        """
        https://github.com/andrew-cr/discrete_flow_models

        x1_probs(softmaxed): torch.Tensor, shape (node_size, max_integer_num)
        xt: torch.Tensor, shape (node_size,)
        t: float, time step

        """
        step_probs = self.next_state_probs(x1_probs, xt, t_start, t_end)
        if t_end < 1.0:
            return torch.multinomial(step_probs, 1).squeeze(-1)
        return step_probs
    
    def next_state_probs(self, x1_probs: torch.Tensor, xt: torch.Tensor, t_start, t_end
        ) -> torch.Tensor:
        """
        https://github.com/andrew-cr/discrete_flow_models

        x1_probs(softmaxed): torch.Tensor, shape (node_size, max_integer_num)
        xt: torch.Tensor, shape (node_size,)
        t: float, time step

        """
        # Low-precision probabilities must not round a fractional time to one.
        if x1_probs.dtype in (torch.float16, torch.bfloat16):
            x1_probs = x1_probs.float()
        t, end = _time_interval(t_start, t_end, x1_probs)
        dt = end - t
        if dt == 0:
            return torch.nn.functional.one_hot(xt, self.max_integer_num).to(x1_probs)
        if self.noise < 0:
            raise ValueError("Discrete flow noise must be nonnegative")
        N = self.noise if end < 1.0 else 0
        if self.mode == "uniform":
            x1_probs_at_xt = torch.gather(x1_probs, -1, xt[:, None])
            step_probs = (
                dt * ((1 + N + N * (self.max_integer_num - 1) * t) / (1 - t)) * x1_probs
                + dt * N * x1_probs_at_xt
            ).clamp(max=1.0)

            # Calculate the on-diagnoal step probabilities
            # 1) Zero out the diagonal entries
            step_probs.scatter_(-1, xt[:, None], 0)
            # 2) Calculate the diagonal entries
            step_probs.scatter_(
                -1, xt[:, None], 1 - step_probs.sum(-1, keepdim=True)
            )
        else:
            mask_index = self.max_integer_num - 1
            current_probs = torch.nn.functional.one_hot(xt, self.max_integer_num).to(x1_probs)
            mask_probs = torch.zeros_like(x1_probs)
            mask_probs[:, mask_index] = 1
            unmask_probability = (dt * (1 + N * t) / (1 - t)).clamp(0, 1)
            remask_probability = (dt * N).clamp(0, 1)
            step_probs = torch.where(
                (xt == mask_index)[:, None],
                unmask_probability * x1_probs + (1 - unmask_probability) * current_probs,
                (1 - remask_probability) * current_probs + remask_probability * mask_probs,
            )
        # A large stochastic step can exhaust the probability of staying put.
        # Keep the sampling weights nonnegative and the returned rows normalized.
        step_probs = step_probs.clamp(min=0.0)
        return step_probs / step_probs.sum(dim=-1, keepdim=True)
    
    def rate_matrix(self, x1_pred: torch.Tensor, xt: torch.Tensor, t_start, t_end) -> torch.Tensor: #This function is suitable for batched x1_pred
        reference = torch.empty((), device=x1_pred.device, dtype=torch.float32)
        t, end = _time_interval(t_start, t_end, reference)
        if self.noise < 0:
            raise ValueError("Discrete flow noise must be nonnegative")
        if t >= 1:
            return torch.zeros((*x1_pred.shape, self.max_integer_num), device=x1_pred.device)
        N = self.noise if end < 1.0 else 0
        x1_eq_xt = x1_pred == xt
        x1_onehot = torch.nn.functional.one_hot(x1_pred, self.max_integer_num).float()
        if self.mode == "mask":
            mask_index = self.max_integer_num - 1
            is_masked = (xt == mask_index)[..., None]
            mask_onehot = torch.nn.functional.one_hot(
                torch.full_like(xt, mask_index), self.max_integer_num
            ).float()
            return (
                (1 + N * t) / (1 - t) * x1_onehot * is_masked
                + N * mask_onehot * ~is_masked
            )
        
        rate_m = N * x1_eq_xt[..., None] + (1 + N + N * (self.max_integer_num - 1) * t)/ (1 - t) * x1_onehot
        return rate_m
        
        
        

class ContinuousFlow(object):
    def __init__(self, scheduler="linear"):
        self.scheduler = PathScheduler(scheduler)

    def x_0_sample(self, shape, device=None, dtype=torch.float32) -> torch.Tensor:
        return torch.randn(shape, device=device, dtype=dtype)

    def sample(self, x1, t) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.as_tensor(t, device=x1.device, dtype=x1.dtype)
        t = t.view(-1)
        dict_sche = self.scheduler(t)
        alpha_t, sigma_t = dict_sche["alpha_t"], dict_sche["sigma_t"]
        d_alpha_t, d_sigma_t = dict_sche["d_alpha_t"], dict_sche["d_sigma_t"]
        noise = torch.randn_like(x1)
        xt = alpha_t * x1 + sigma_t * noise
        dxt = d_alpha_t * x1 + d_sigma_t * noise
        return xt, dxt

    def denoise(
        self, v_pred: torch.Tensor, xt: torch.Tensor, t_start, t_end
    ) -> torch.Tensor:
        # The network predicts the clean endpoint, not the path velocity.
        # Convert that prediction using the same path used for training.
        t, end = _time_interval(t_start, t_end, xt)
        if end == t:
            return xt.clone()
        schedule = self.scheduler(t)
        noise_est = (xt - schedule["alpha_t"] * v_pred) / schedule["sigma_t"]
        v_est = schedule["d_alpha_t"] * v_pred + schedule["d_sigma_t"] * noise_est
        return xt + v_est * (end - t)


class CosineScheduler:
    def __init__(self):
        self.pi = math.pi

    def __call__(self, t: torch.Tensor) -> dict:
        alpha_t = torch.sin(self.pi / 2 * t)
        sigma_t = torch.where(t == 1, torch.zeros_like(t), torch.cos(self.pi / 2 * t))
        d_alpha_t = (self.pi / 2) * sigma_t
        d_sigma_t = -(self.pi / 2) * torch.sin(self.pi / 2 * t)
        return {
            "alpha_t": alpha_t,
            "sigma_t": sigma_t,
            "d_alpha_t": d_alpha_t,
            "d_sigma_t": d_sigma_t,
        }


class Linearcheduler:
    def __call__(self, t: torch.Tensor) -> dict:
        return {
            "alpha_t": t,
            "sigma_t": 1 - t,
            "d_alpha_t": torch.ones_like(t),
            "d_sigma_t": -torch.ones_like(t),
        }


class LinearVPScheduler:
    def __call__(self, t: torch.Tensor) -> dict:
        return {
            "alpha_t": t,
            "sigma_t": (1 - t**2) ** 0.5,
            "d_alpha_t": torch.ones_like(t),
            "d_sigma_t": -t / (1 - t**2) ** 0.5,
        }


class PathScheduler:
    def __init__(self, scheduler="linear"):
        self.scheduler = scheduler
        if scheduler == "linear":
            self.scheduler = Linearcheduler()
        elif scheduler == "cosine":
            self.scheduler = CosineScheduler()
        elif scheduler == "linearVP":
            self.scheduler = LinearVPScheduler()
        else:
            raise ValueError(f"Unsupported flow scheduler: {scheduler}")

    def __call__(self, t) -> dict:
        return self.scheduler(t)


class InferenceScheduler:
    def __init__(self, inference_steps, scheduler="linear"):
        if not isinstance(inference_steps, int) or inference_steps <= 0:
            raise ValueError("inference_steps must be a positive integer")
        if scheduler not in ("linear", "cosine"):
            raise ValueError(f"Unsupported scheduler: {scheduler}")
        self.inference_steps = inference_steps
        self.scheduler = scheduler

    def __call__(self, i):
        if not isinstance(i, int) or not 0 <= i < self.inference_steps:
            raise ValueError("Inference step index is outside the schedule")
        if self.scheduler == "linear":
            t_start = 0 + float(i) / self.inference_steps
            t_end = 0 + float(i + 1) / self.inference_steps
            t_end = np.clip(t_end, 0, 1)
            return t_start, t_end
        
        elif self.scheduler == "cosine":
            t_start = 0 + np.sin(float(i) / self.inference_steps * np.pi / 2)
            t_end = 0 + np.sin(float(i + 1) / self.inference_steps * np.pi / 2)
            t_end = np.clip(t_end, 0, 1)
            return t_start, t_end

        else:
            raise ValueError(f"Unsupported scheduler: {self.scheduler}")
