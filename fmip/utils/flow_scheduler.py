"""Schedulers for Flow Matching Models"""

import math
import numpy as np
import torch


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
        self.max_integer_num = max_integer_num
        self.mode = mode
        self.noise = 0

    def x_0_sample(self, shape) -> torch.Tensor:
        return (
            torch.randint(0, self.max_integer_num, shape)
            if self.mode == "uniform"
            else torch.full(shape, self.max_integer_num - 1)
        )

        # if mode ==
        ## noise * dt * D is the average number of dimensions that get re-masked each timestep

    def sample(self, x1, t) -> torch.Tensor:
        if type(t) != torch.Tensor:
            t = torch.tensor(t, device=x1.device)
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
        dt = t_end - t_start
        t = t_start
        device = x1_probs.device
        if type(t) != torch.Tensor:
            t = torch.tensor(t, device=device)
        if self.mode == "uniform":
            N = self.noise if t + dt < 1.0 else 0
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
            ).clamp(min=0.0)
            if t + dt < 1.0:
                xt = torch.multinomial(step_probs, 1).squeeze(-1)
            else:
                return step_probs
        
        elif self.mode == "mask":
            # raise NotImplementedError("Mask mode is not implemented yet")
            num_samples = x1_probs.shape[0]
            x1 = torch.multinomial(x1_probs, 1).squeeze(-1)
            will_unmask = torch.rand(num_samples, device=device) < (
                dt * (1 + self.noise * t) / (1 - t)
            )
            will_unmask = will_unmask & (x1 == self.max_integer_num - 1)
            will_mask = torch.rand(num_samples, device=device) < dt * self.noise
            will_mask = will_mask & (x1 != self.max_integer_num - 1)
            xt[will_unmask] = x1[will_unmask]

            if t + dt < 1.0:
                xt[will_mask] = self.max_integer_num - 1

        return xt
    
    def next_state_probs(self, x1_probs: torch.Tensor, xt: torch.Tensor, t_start, t_end
        ) -> torch.Tensor:
        """
        https://github.com/andrew-cr/discrete_flow_models

        x1_probs(softmaxed): torch.Tensor, shape (node_size, max_integer_num)
        xt: torch.Tensor, shape (node_size,)
        t: float, time step

        """
        dt = t_end - t_start
        t = t_start
        device = x1_probs.device
        if type(t) != torch.Tensor:
            t = torch.tensor(t, device=device)
        if self.mode == "uniform":
            N = self.noise if t + dt < 1.0 else 0
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
            ).clamp(min=0.0)
        return step_probs
    
    def rate_matrix(self, x1_pred: torch.Tensor, xt: torch.Tensor, t_start, t_end) -> torch.Tensor: #This function is suitable for batched x1_pred
        t = t_start
        device = x1_pred.device
        if type(t) != torch.Tensor:
            t = torch.tensor(t, device=device)
        N = self.noise if t_end < 1.0 else 0
        x1_eq_xt = x1_pred == xt
        x1_onehot = torch.nn.functional.one_hot(x1_pred, self.max_integer_num).float()
        
        rate_m = N * x1_eq_xt[..., None] + (1 + N + N * (self.max_integer_num - 1) * t)/ (1 - t) * x1_onehot
        return rate_m
        
        
        

class ContinuousFlow(object):
    def __init__(self, scheduler="linear"):
        self.scheduler = PathScheduler(scheduler)

    def x_0_sample(self, shape) -> torch.Tensor:
        return torch.randn(shape)

    def sample(self, x1, t) -> tuple[torch.Tensor, torch.Tensor]:
        if type(t) != torch.Tensor:
            t = torch.tensor(t)
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
        v_est = (v_pred - xt) / (1 - t_start)
        return xt + v_est * (t_end - t_start)


class CosineScheduler:
    def __init__(self):
        self.pi = math.pi

    def __call__(self, t: torch.Tensor) -> dict:
        alpha_t = torch.sin(self.pi / 2 * t)
        sigma_t = torch.cos(self.pi / 2 * t)
        d_alpha_t = (self.pi / 2) * torch.cos(self.pi / 2 * t)
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

    def __call__(self, t) -> dict:
        return self.scheduler(t)


class InferenceScheduler:
    def __init__(self, inference_steps, scheduler="linear"):
        self.inference_steps = inference_steps
        self.scheduler = scheduler

    def __call__(self, i):
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
