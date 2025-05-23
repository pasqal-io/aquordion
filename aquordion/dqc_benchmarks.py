from __future__ import annotations

from typing import Callable

import horqrux
import jax
import jax.numpy as jnp
import optax
import pyqtorch as pyq
import torch
from jax import Array
from torch import Tensor
from torch.nn import ParameterDict

from .vqe_benchmarks import native_expectation_horqrux, native_expectation_pyq

VARIABLES = ("x", "y")
N_VARIABLES = len(VARIABLES)
X_POS, Y_POS = [i for i in range(N_VARIABLES)]
BATCH_SIZE = 250


class DomainSampling(torch.nn.Module):
    """Class for solving with pyqtorch a Laplace equation using a parametrized quantum circuit (PQC) and an observable.
    When passing as inputs points from a domain of definition of the equation,
    the output expectations are considered outputs of a function.
    As the model (PQC + observable) can be differentiated, we can learn how well it
    fits the system of equations forming the Laplace.

    As inputs, we sample from a square domain and compute the loss as a sum of
    interior points contributions and the boundary conditions.

    """

    def __init__(
        self,
        circuit: pyq.QuantumCircuit,
        observable: pyq.Observable,
        inputs_embedded: ParameterDict,
        n_shots: int = 0,
        diff_mode: pyq.DiffMode = pyq.DiffMode.AD,
    ) -> None:
        super().__init__()
        self.circuit = circuit
        self.observable = observable
        self.inputs_embedded = inputs_embedded
        self.n_shots = n_shots
        self.diff_mode = diff_mode

    def exp_fn(self, inputs: Tensor) -> Tensor:
        return native_expectation_pyq(
            self.circuit,
            self.observable,
            inputs={
                **self.inputs_embedded,
                **{VARIABLES[X_POS]: inputs[:, X_POS], VARIABLES[Y_POS]: inputs[:, Y_POS]},
            },
            diff_mode=self.diff_mode,
            n_shots=self.n_shots,
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


def dqc_pyq_adam(
    circuit: pyq.QuantumCircuit,
    observable: pyq.Observable,
    inputs_embedded: ParameterDict,
    LR: float = 1e-2,
    n_shots: int = 0,
    N_epochs: int = 30,
    diff_mode: pyq.DiffMode = pyq.DiffMode.AD,
) -> Callable:
    """Taken from https://pasqal-io.github.io/pyqtorch/latest/pde/"""

    d_sampler = DomainSampling(circuit, observable, inputs_embedded, n_shots, diff_mode)

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


def dqc_horqrux_adam(
    circuit: horqrux.QuantumCircuit,
    observable: horqrux.Observable,
    inputs: Array,
    LR: float = 1e-2,
    N_epochs: int = 30,
    n_shots: int = 0,
    diff_mode: horqrux.DiffMode = horqrux.DiffMode.AD,
) -> Callable:
    """Taken from https://pasqal-io.github.io/horqrux/latest/dqc/"""

    def opt_horqux() -> None:
        optimizer = optax.adam(learning_rate=LR)

        def optimize_step(param_vals: Array, opt_state: Array, grads: dict[str, Array]) -> tuple:
            updates, opt_state = optimizer.update(grads, opt_state, param_vals)
            param_vals = optax.apply_updates(param_vals, updates)
            return param_vals, opt_state

        def loss_fn(param_vals: Array, key: jax.random.PRNGKey) -> Array:
            """The loss function is the sum of all expectation value for the observable components."""
            values = dict(zip(circuit.vparams, param_vals))

            def pde_loss(x: Array, y: Array) -> Array:
                x = x.reshape(-1, 1)
                y = y.reshape(-1, 1)

                # boundary conditions loss terms are calculated below
                left = (jnp.zeros_like(y), y)  # u(0,y)=0
                right = (jnp.ones_like(y), y)  # u(L,y)=0
                top = (x, jnp.ones_like(x))  # u(x,H)=0
                bottom = (x, jnp.zeros_like(x))  # u(x,0)=f(x)
                terms = jnp.dstack(list(map(jnp.hstack, [left, right, top, bottom]))).squeeze(0)
                exp_fn = lambda xy: native_expectation_horqrux(
                    circuit,
                    observable,
                    values | {"x": xy[0], "y": xy[1]},
                    n_shots=n_shots,
                    diff_mode=diff_mode,
                )
                loss_left, loss_right, loss_top, loss_bottom = jax.vmap(exp_fn, in_axes=(1,))(terms)
                loss_bottom -= jnp.sin(jnp.pi * x)

                hessian = jax.hessian(exp_fn)(
                    jnp.concatenate(
                        [
                            x.reshape(
                                1,
                            ),
                            y.reshape(
                                1,
                            ),
                        ]
                    )
                )
                loss_interior = hessian[:, X_POS] + hessian[:, Y_POS]
                return jnp.sum(
                    jnp.concatenate(
                        list(
                            map(
                                lambda term: jnp.power(term, 2).reshape(-1, 1),  # type: ignore[no-any-return]
                                [loss_left, loss_right, loss_top, loss_bottom, loss_interior],
                            )
                        )
                    )
                )

            return jnp.mean(
                jax.vmap(pde_loss, in_axes=(0, 0))(
                    *jax.random.uniform(key, (N_VARIABLES, BATCH_SIZE))
                )
            )

        def train_step(i: int, param_vals_opt_state: tuple) -> tuple:
            param_vals, opt_state = param_vals_opt_state
            _, grads = jax.value_and_grad(loss_fn)(param_vals, jax.random.PRNGKey(i))
            return optimize_step(param_vals, opt_state, grads)

        param_vals = inputs.clone()
        opt_state = optimizer.init(param_vals)
        param_vals, opt_state = jax.lax.fori_loop(0, N_epochs, train_step, (param_vals, opt_state))

    return opt_horqux
