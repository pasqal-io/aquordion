from __future__ import annotations

from typing import Callable

import strategies as st
import torch
from hypothesis import given, settings
from qadence import AbstractBlock, QuantumCircuit
from qadence.backends.jax_utils import jarr_to_tensor
from qadence.states import equivalent_state

from aquordion.benchmark_api import expectation_horqrux, expectation_pyq, run_horqrux, run_pyq
from aquordion.utils import values_to_jnp

ATOL_32 = 1e-07  # 32 bit precision


def test_run_circuits(fn_circuit: Callable, random_circuit_integers: tuple[int, int]) -> None:
    n_qubits, n_layers = random_circuit_integers
    circuit, params = fn_circuit(n_qubits, n_layers)
    inputs = {p: torch.rand(1) for p in params}

    wf_pyqtorch = run_pyq(circuit, inputs)
    wf_horqrux = jarr_to_tensor(run_horqrux(circuit, values_to_jnp(inputs)))
    assert equivalent_state(wf_pyqtorch, wf_horqrux, atol=ATOL_32)


@given(st.ansatz(), st.observables())
@settings(deadline=None)
def test_expectation_circuits(
    ansatz: tuple[QuantumCircuit, list[str]], observable: AbstractBlock
) -> None:

    circuit, params = ansatz
    if observable.n_qubits > circuit.n_qubits:
        circuit = QuantumCircuit(observable.n_qubits, circuit.block)
    inputs = {p: torch.rand(1) for p in params}

    pyqtorch_expectation = expectation_pyq(circuit, observable, inputs)
    horqrux_expectation = jarr_to_tensor(
        expectation_horqrux(circuit, observable, values_to_jnp(inputs)), pyqtorch_expectation.dtype
    )
    assert torch.allclose(pyqtorch_expectation, horqrux_expectation, atol=ATOL_32)
