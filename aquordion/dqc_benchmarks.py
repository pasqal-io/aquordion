from __future__ import annotations

from typing import Callable

import pyqtorch as pyq
import torch
from torch import Tensor
from torch.nn import ParameterDict

from .vqe_benchmarks import native_expectation_pyq

VARIABLES = ("x", "y")
N_VARIABLES = len(VARIABLES)
X_POS, Y_POS = [i for i in range(N_VARIABLES)]
BATCH_SIZE = 250


def dqc_pyq_adam(
    circuit: pyq.QuantumCircuit,
    observable: pyq.Observable,
    inputs_embedded: ParameterDict,
    LR: float = 1e-2,
    N_epochs: int = 30,
) -> Callable:
    """Taken from https://pasqal-io.github.io/pyqtorch/latest/pde/"""

    class DomainSampling(torch.nn.Module):

        def __init__(self) -> None:
            super().__init__()

        def exp_fn(self, inputs: Tensor) -> Tensor:
            return native_expectation_pyq(
                circuit,
                observable,
                {
                    **inputs_embedded,
                    **{VARIABLES[X_POS]: inputs[:, X_POS], VARIABLES[Y_POS]: inputs[:, Y_POS]},
                },
            )

        def sample(self, requires_grad: bool = False) -> Tensor:
            return torch.rand((BATCH_SIZE, N_VARIABLES), requires_grad=requires_grad)

        def left_boundary(self) -> Tensor:  # u(0,y)=0
            sample = self.sample()
            sample[:, X_POS] = 0.0
            return self.exp_fn(sample).pow(2).mean()

        def right_boundary(self) -> Tensor:  # u(L,y)=0
            sample = self.sample()
            sample[:, X_POS] = 1.0
            return self.exp_fn(sample).pow(2).mean()

        def top_boundary(self) -> Tensor:  # u(x,H)=0
            sample = self.sample()
            sample[:, Y_POS] = 1.0
            return self.exp_fn(sample).pow(2).mean()

        def bottom_boundary(self) -> Tensor:  # u(x,0)=f(x)
            sample = self.sample()
            sample[:, Y_POS] = 0.0
            return (self.exp_fn(sample) - torch.sin(torch.pi * sample[:, 0])).pow(2).mean()

        def interior(self) -> Tensor:  # uxx+uyy=0
            sample = self.sample(requires_grad=True)
            exp_eval = self.exp_fn(sample)
            dfdxy = torch.autograd.grad(
                exp_eval,
                sample,
                torch.ones_like(exp_eval),
                create_graph=True,
            )[0]
            dfdxxdyy = torch.autograd.grad(
                dfdxy,
                sample,
                torch.ones_like(dfdxy),
            )[0]

            return (dfdxxdyy[:, X_POS] + dfdxxdyy[:, Y_POS]).pow(2).mean()

    d_sampler = DomainSampling()

    def opt_pyq() -> None:
        optimizer = torch.optim.Adam(inputs_embedded.values(), lr=LR, foreach=False)

        def loss_fn() -> Tensor:
            return (
                d_sampler.left_boundary()
                + d_sampler.right_boundary()
                + d_sampler.top_boundary()
                + d_sampler.bottom_boundary()
                + d_sampler.interior()
            )

        for _ in range(N_epochs):
            optimizer.zero_grad()
            loss = loss_fn()
            loss.backward()
            optimizer.step()

    return opt_pyq
