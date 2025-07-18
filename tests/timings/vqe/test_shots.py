from __future__ import annotations

from typing import Callable

import horqrux
import jax
import pyqtorch
import pytest
import torch
from qadence import AbstractBlock
from torch.nn import ParameterDict

from aquordion.api_benchmarks import (
    bknd_horqrux,
    bknd_pyqtorch,
)
from aquordion.vqe_benchmarks import vqe_horqrux_adam, vqe_pyq_adam

N_epochs_shots = 10
n_shots = 100


@pytest.mark.timeout(10)
def test_vqe_pyq(
    benchmark: pytest.Fixture,
    benchmark_vqe_ansatz: tuple[Callable, int, int],
    h2_hamiltonian: AbstractBlock,
) -> None:
    fn_circuit, n_qubits, n_layers = benchmark_vqe_ansatz
    circuit, params = fn_circuit(n_qubits, n_layers)
    torch.manual_seed(0)
    (circ, obs, embed_fn, params_conv) = bknd_pyqtorch.convert(circuit, h2_hamiltonian)
    values = {p: torch.rand(1, requires_grad=True) for p in params}

    circ = circ.native
    obs = obs[0].native
    inputs_embedded = ParameterDict({p: v for p, v in embed_fn(params_conv, values).items()})

    opt_pyq = vqe_pyq_adam(
        circ,
        obs,
        inputs_embedded,
        diff_mode=pyqtorch.DiffMode.GPSR,
        n_shots=n_shots,
        N_epochs=N_epochs_shots,
    )
    benchmark.pedantic(opt_pyq, rounds=5)


@pytest.mark.timeout(10)
def test_vqe_horqrux(
    benchmark: pytest.Fixture,
    benchmark_vqe_ansatz: tuple[Callable, int, int],
    h2_hamiltonian: AbstractBlock,
) -> None:
    fn_circuit, n_qubits, n_layers = benchmark_vqe_ansatz
    circuit, _ = fn_circuit(n_qubits, n_layers)

    (circ, obs, _, _) = bknd_horqrux.convert(circuit, h2_hamiltonian)

    ansatz = horqrux.QuantumCircuit(circ.native.n_qubits, list(iter(circ.native)))

    observable = horqrux.Observable(list(iter(obs[0].native)))

    key = jax.random.PRNGKey(42)
    init_param_vals = jax.random.uniform(key, shape=(ansatz.n_vparams,))

    opt_horqux = vqe_horqrux_adam(
        ansatz,
        observable,
        init_param_vals,
        diff_mode=horqrux.DiffMode.GPSR,
        n_shots=n_shots,
        N_epochs=N_epochs_shots,
    )

    benchmark.pedantic(opt_horqux, rounds=5)
