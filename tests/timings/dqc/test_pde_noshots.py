from __future__ import annotations

from typing import Callable

import pyqtorch as pyq
import pytest
import torch
from qadence import QuantumCircuit, Z, hamiltonian_factory
from torch.nn import ParameterDict

from aquordion.api_benchmarks import (
    bknd_pyqtorch,
)
from aquordion.dqc_benchmarks import dqc_pyq_adam


def test_dqc_pyq(
    benchmark: pytest.Fixture,
    benchmark_dqc_ansatz: tuple[Callable, int, int],
) -> None:
    fn_circuit, n_qubits, n_layers = benchmark_dqc_ansatz
    ansatz = fn_circuit(n_qubits, n_layers)
    feature_map = [pyq.RX(i, "x") for i in range(n_qubits // 2)] + [
        pyq.RX(i, "y") for i in range(n_qubits // 2, n_qubits)
    ]

    circuit = QuantumCircuit(n_qubits, ansatz)
    total_magnetization = hamiltonian_factory(n_qubits, detuning=Z)
    values = {p: torch.rand(1, requires_grad=True) for p in circuit.unique_parameters}
    # avoid multiple conversion
    (circ, obs, embed_fn, params_conv) = bknd_pyqtorch.convert(circuit, total_magnetization)
    circ = pyq.QuantumCircuit(n_qubits, feature_map + circ.native.operations)
    obs = obs[0].native
    inputs_embedded = ParameterDict({p: v for p, v in embed_fn(params_conv, values).items()})
    opt_pyq = dqc_pyq_adam(circ, obs, inputs_embedded, N_epochs=10)
    benchmark.pedantic(opt_pyq, rounds=5)
