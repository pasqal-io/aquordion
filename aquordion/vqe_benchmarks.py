from __future__ import annotations

from typing import Callable

import horqrux
import jax
import optax
import pyqtorch as pyq
import torch
from jax import Array
from torch import Tensor
from torch.nn import ParameterDict


def native_expectation_pyq(
    circuit: pyq.QuantumCircuit,
    observable: pyq.Observable,
    inputs: dict[str, Tensor],
    diff_mode: pyq.DiffMode = pyq.DiffMode.AD,
    n_shots: int = 0,
) -> Tensor:
    """Expectation with native PyQTorch."""
    return pyq.expectation(
        circuit,
        state=pyq.zero_state(max(circuit.n_qubits, len(observable.qubit_support))),
        observable=observable,
        values=inputs,
        diff_mode=diff_mode,
        n_shots=n_shots if n_shots > 0 else None,
    )


def native_expectation_horqrux(
    circuit: horqrux.QuantumCircuit,
    observable: horqrux.Observable,
    inputs: dict[str, Array],
    diff_mode: horqrux.DiffMode = horqrux.DiffMode.AD,
    n_shots: int = 0,
) -> Array:
    """Expectation with native Horqrux."""
    return horqrux.expectation(
        state=horqrux.zero_state(max(circuit.n_qubits, len(observable.qubit_support))),
        circuit=circuit,
        observables=[observable],
        values=inputs,
        diff_mode=diff_mode,
        n_shots=n_shots,
    )


def vqe_pyq_adam(
    circuit: pyq.QuantumCircuit,
    observable: pyq.Observable,
    inputs_embedded: ParameterDict,
    diff_mode: pyq.DiffMode = pyq.DiffMode.AD,
    n_shots: int = 0,
    LR: float = 1e-2,
    N_epochs: int = 30,
) -> Callable:

    def opt_pyq() -> None:
        optimizer = torch.optim.Adam(inputs_embedded.values(), lr=LR, foreach=False)

        def loss_fn(vals: ParameterDict) -> Tensor:
            return native_expectation_pyq(
                circuit, observable, vals, diff_mode=diff_mode, n_shots=n_shots
            )

        for _ in range(N_epochs):
            optimizer.zero_grad()
            loss = loss_fn(inputs_embedded)
            loss.backward()
            optimizer.step()

    return opt_pyq


def vqe_horqrux_adam(
    circuit: horqrux.QuantumCircuit,
    observable: horqrux.Observable,
    inputs: Array,
    diff_mode: horqrux.DiffMode = horqrux.DiffMode.AD,
    n_shots: int = 0,
    LR: float = 1e-2,
    N_epochs: int = 30,
) -> Callable:

    def opt_horqux() -> None:
        optimizer = optax.adam(learning_rate=LR)

        def optimize_step(param_vals: Array, opt_state: Array, grads: dict[str, Array]) -> tuple:
            updates, opt_state = optimizer.update(grads, opt_state, param_vals)
            param_vals = optax.apply_updates(param_vals, updates)
            return param_vals, opt_state

        def loss_fn(param_vals: Array) -> Array:
            """The loss function is the sum of all expectation value for the observable components."""
            values = dict(zip(circuit.vparams, param_vals))
            return jax.numpy.sum(
                native_expectation_horqrux(
                    circuit, observable, values, diff_mode=diff_mode, n_shots=n_shots
                )
            )

        def train_step(i: int, param_vals_opt_state: tuple) -> tuple:
            param_vals, opt_state = param_vals_opt_state
            _, grads = jax.value_and_grad(loss_fn)(param_vals)
            return optimize_step(param_vals, opt_state, grads)

        param_vals = inputs.clone()
        opt_state = optimizer.init(param_vals)
        param_vals, opt_state = jax.lax.fori_loop(0, N_epochs, train_step, (param_vals, opt_state))

    return opt_horqux
