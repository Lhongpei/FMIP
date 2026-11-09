# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing import Union
import math

import torch

from torch import Tensor


def _validate_snr(snr: Tensor) -> Tensor:
    if torch.isnan(snr).any() or (snr < 0).any():
        raise ValueError("SNR must be nonnegative and not NaN")
    return snr


@dataclass
class SchedulerOutput:
    r"""Represents a sample of a conditional-flow generated probability path.

    Attributes:
        alpha_t (Tensor): :math:`\alpha_t`, shape (...).
        sigma_t (Tensor): :math:`\sigma_t`, shape (...).
        d_alpha_t (Tensor): :math:`\frac{\partial}{\partial t}\alpha_t`, shape (...).
        d_sigma_t (Tensor): :math:`\frac{\partial}{\partial t}\sigma_t`, shape (...).

    """

    alpha_t: Tensor = field(metadata={"help": "alpha_t"})
    sigma_t: Tensor = field(metadata={"help": "sigma_t"})
    d_alpha_t: Tensor = field(metadata={"help": "Derivative of alpha_t."})
    d_sigma_t: Tensor = field(metadata={"help": "Derivative of sigma_t."})


class Scheduler(ABC):
    """Base Scheduler class."""

    @abstractmethod
    def __call__(self, t: Tensor) -> SchedulerOutput:
        r"""
        Args:
            t (Tensor): times in [0,1], shape (...).

        Returns:
            SchedulerOutput: :math:`\alpha_t,\sigma_t,\frac{\partial}{\partial t}\alpha_t,\frac{\partial}{\partial t}\sigma_t`
        """
        ...

    @abstractmethod
    def snr_inverse(self, snr: Tensor) -> Tensor:
        r"""
        Computes :math:`t` from the signal-to-noise ratio :math:`\frac{\alpha_t}{\sigma_t}`.

        Args:
            snr (Tensor): The signal-to-noise, shape (...)

        Returns:
            Tensor: t, shape (...)
        """
        ...


class ConvexScheduler(Scheduler):
    @abstractmethod
    def __call__(self, t: Tensor) -> SchedulerOutput:
        r"""Scheduler for convex paths.

        Args:
            t (Tensor): times in [0,1], shape (...).

        Returns:
            SchedulerOutput: :math:`\alpha_t,\sigma_t,\frac{\partial}{\partial t}\alpha_t,\frac{\partial}{\partial t}\sigma_t`
        """
        ...

    @abstractmethod
    def kappa_inverse(self, kappa: Tensor) -> Tensor:
        r"""
        Computes :math:`t` from :math:`\kappa_t`.

        Args:
            kappa (Tensor): :math:`\kappa`, shape (...)

        Returns:
            Tensor: t, shape (...)
        """
        ...

    def snr_inverse(self, snr: Tensor) -> Tensor:
        r"""
        Computes :math:`t` from the signal-to-noise ratio :math:`\frac{\alpha_t}{\sigma_t}`.

        Args:
            snr (Tensor): The signal-to-noise, shape (...)

        Returns:
            Tensor: t, shape (...)
        """
        snr = _validate_snr(snr)
        kappa_t = torch.where(torch.isinf(snr), torch.ones_like(snr), snr / (1.0 + snr))

        return self.kappa_inverse(kappa=kappa_t)


class CondOTScheduler(ConvexScheduler):
    """CondOT Scheduler."""

    def __call__(self, t: Tensor) -> SchedulerOutput:
        return SchedulerOutput(
            alpha_t=t,
            sigma_t=1 - t,
            d_alpha_t=torch.ones_like(t),
            d_sigma_t=-torch.ones_like(t),
        )

    def kappa_inverse(self, kappa: Tensor) -> Tensor:
        return kappa


class PolynomialConvexScheduler(ConvexScheduler):
    """Polynomial Scheduler."""

    def __init__(self, n: Union[float, int]) -> None:
        assert isinstance(
            n, (float, int)
        ), f"`n` must be a float or int. Got {type(n)=}."
        assert n > 0, f"`n` must be positive. Got {n=}."

        self.n = n

    def __call__(self, t: Tensor) -> SchedulerOutput:
        return SchedulerOutput(
            alpha_t=t**self.n,
            sigma_t=1 - t**self.n,
            d_alpha_t=self.n * (t ** (self.n - 1)),
            d_sigma_t=-self.n * (t ** (self.n - 1)),
        )

    def kappa_inverse(self, kappa: Tensor) -> Tensor:
        return torch.pow(kappa, 1.0 / self.n)


class VPScheduler(Scheduler):
    """Variance Preserving Scheduler."""

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0) -> None:
        if not (math.isfinite(beta_min) and math.isfinite(beta_max)
                and 0 < beta_min <= beta_max):
            raise ValueError("VP beta values must satisfy 0 < beta_min <= beta_max")
        self.beta_min = beta_min
        self.beta_max = beta_max
        super().__init__()

    def __call__(self, t: Tensor) -> SchedulerOutput:
        b = self.beta_min
        B = self.beta_max
        T = 0.5 * (1 - t) ** 2 * (B - b) + (1 - t) * b
        dT = -(1 - t) * (B - b) - b

        return SchedulerOutput(
            alpha_t=torch.exp(-0.5 * T),
            sigma_t=torch.sqrt(-torch.expm1(-T)),
            d_alpha_t=-0.5 * dT * torch.exp(-0.5 * T),
            d_sigma_t=0.5 * dT * torch.exp(-T) / torch.sqrt(-torch.expm1(-T)),
        )

    def snr_inverse(self, snr: Tensor) -> Tensor:
        snr = _validate_snr(snr)
        # log(1 + 1 / snr**2), without overflow from squaring large SNR.
        T = torch.logaddexp(torch.zeros_like(snr), -2 * torch.log(snr))
        b = self.beta_min
        B = self.beta_max
        if B == b:
            return 1 - T / b
        # Rationalization avoids cancellation near t=1.
        delta = 2 * T / (b + torch.sqrt(b**2 + 2 * (B - b) * T))
        delta = torch.where(torch.isinf(T), T, delta)
        t = 1 - delta
        return t


class LinearVPScheduler(Scheduler):
    """Linear Variance Preserving Scheduler."""

    def __call__(self, t: Tensor) -> SchedulerOutput:
        return SchedulerOutput(
            alpha_t=t,
            sigma_t=(1 - t**2) ** 0.5,
            d_alpha_t=torch.ones_like(t),
            d_sigma_t=-t / (1 - t**2) ** 0.5,
        )

    def snr_inverse(self, snr: Tensor) -> Tensor:
        snr = _validate_snr(snr)
        return torch.where(torch.isinf(snr), torch.ones_like(snr),
                           snr / torch.hypot(snr, torch.ones_like(snr)))


class CosineScheduler(Scheduler):
    """Cosine Scheduler."""

    def __call__(self, t: Tensor) -> SchedulerOutput:
        pi = torch.pi
        sigma = torch.where(t == 1, torch.zeros_like(t), torch.cos(pi / 2 * t))
        return SchedulerOutput(
            alpha_t=torch.sin(pi / 2 * t),
            sigma_t=sigma,
            d_alpha_t=pi / 2 * sigma,
            d_sigma_t=-pi / 2 * torch.sin(pi / 2 * t),
        )

    def snr_inverse(self, snr: Tensor) -> Tensor:
        return 2.0 * torch.atan(_validate_snr(snr)) / torch.pi
