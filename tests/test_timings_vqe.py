from __future__ import annotations

from typing import Callable

import horqrux
import jax
import optax
import pytest
import torch
from jax import Array
from qadence import AbstractBlock
from torch import Tensor
from torch.nn import ParameterDict

from aquordion.benchmarks import (
    bknd_horqrux,
    bknd_pyqtorch,
    native_expectation_horqrux,
    native_expectation_pyq,
)

N_epochs = 20
LR = 0.01


def test_vqe_pyq(
    benchmark: pytest.Fixture,
    benchmark_vqe_ansatz: tuple[Callable, int, int],
    h2_hamiltonian: AbstractBlock,
) -> None:
    fn_circuit, n_qubits, n_layers = benchmark_vqe_ansatz
    circuit, params = fn_circuit(n_qubits, n_layers)
    torch.manual_seed(0)
    # avoid multiple conversion
    (circ, obs, embed_fn, params_conv) = bknd_pyqtorch.convert(circuit, h2_hamiltonian)
    values = {p: torch.rand(1, requires_grad=True) for p in params}

    circ = circ.native
    obs = obs[0].native

    def opt_pyq() -> None:
        inputs_embedded = ParameterDict({p: v for p, v in embed_fn(params_conv, values).items()})
        optimizer = torch.optim.Adam(inputs_embedded.values(), lr=LR, foreach=False)

        def loss_fn(vals: ParameterDict) -> Tensor:
            return native_expectation_pyq(circ, obs, vals)

        for _ in range(N_epochs):
            optimizer.zero_grad()
            loss = loss_fn(inputs_embedded)
            loss.backward()
            optimizer.step()

    benchmark.pedantic(opt_pyq, rounds=5)


def test_vqe_horqrux(
    benchmark: pytest.Fixture,
    benchmark_vqe_ansatz: tuple[Callable, int, int],
    h2_hamiltonian: AbstractBlock,
) -> None:
    fn_circuit, n_qubits, n_layers = benchmark_vqe_ansatz
    circuit, _ = fn_circuit(n_qubits, n_layers)

    # avoid multiple conversion
    (circ, obs, _, _) = bknd_horqrux.convert(circuit, h2_hamiltonian)

    ansatz = horqrux.QuantumCircuit(circ.native.n_qubits, list(iter(circ.native)))

    observable = horqrux.Observable(list(iter(obs[0].native)))

    def opt_horqux() -> None:
        key = jax.random.PRNGKey(42)
        init_param_vals = jax.random.uniform(key, shape=(ansatz.n_vparams,))
        optimizer = optax.adam(learning_rate=LR)

        def optimize_step(param_vals: Array, opt_state: Array, grads: dict[str, Array]) -> tuple:
            updates, opt_state = optimizer.update(grads, opt_state, param_vals)
            param_vals = optax.apply_updates(param_vals, updates)
            return param_vals, opt_state

        def loss_fn(param_vals: Array) -> Array:
            """The loss function is the sum of all expectation value for the observable components."""
            values = dict(zip(ansatz.vparams, param_vals))
            return jax.numpy.sum(native_expectation_horqrux(ansatz, observable, values))

        def train_step(i: int, param_vals_opt_state: tuple) -> tuple:
            param_vals, opt_state = param_vals_opt_state
            _, grads = jax.value_and_grad(loss_fn)(param_vals)
            return optimize_step(param_vals, opt_state, grads)

        param_vals = init_param_vals.clone()
        opt_state = optimizer.init(param_vals)
        param_vals, opt_state = jax.lax.fori_loop(0, N_epochs, train_step, (param_vals, opt_state))

    benchmark.pedantic(opt_horqux, rounds=5)
