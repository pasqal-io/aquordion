from __future__ import annotations

import pytest
import strategies as st

from typing import Callable
import random 
import torch

from hypothesis import given, settings

from qadence import AbstractBlock, QuantumCircuit
from qadence.ml_tools.utils import rand_featureparameters
from qadence.backends.jax_utils import jarr_to_tensor
from qadence.states import equivalent_state

from aquordion.benchmarks_circuits import circuit_A, circuit_B, circuit_C
from aquordion.benchmark_api import run_pyq, run_horqrux, expectation_pyq, expectation_horqrux
from aquordion.utils import values_to_jnp

ATOL_32 = 1e-07  # 32 bit precision

@pytest.mark.parametrize("fn_circuit", [circuit_A, circuit_B, circuit_C])
def test_run_circuits(fn_circuit: Callable) -> None:
    n_qubits = random.randint(1, 5)
    n_layers = random.randint(1, 5)
    circuit, params = fn_circuit(n_qubits, n_layers)
    inputs = {p: torch.rand(1) for p in params}

    wf_pyqtorch = run_pyq(circuit, inputs)
    wf_horqrux = jarr_to_tensor(run_horqrux(circuit, values_to_jnp(inputs)))
    assert equivalent_state(wf_pyqtorch, wf_horqrux, atol=ATOL_32)

@pytest.mark.parametrize("fn_circuit", [circuit_A, circuit_B, circuit_C])
@given(st.observables())
@settings(deadline=None)
def test_expectation_circuits(fn_circuit: Callable, observable: AbstractBlock) -> None:
    n_qubits = random.randint(1, 5)
    n_layers = random.randint(1, 5)
    circuit, params = fn_circuit(n_qubits, n_layers)
    if observable.n_qubits > circuit.n_qubits:
        circuit = QuantumCircuit(observable.n_qubits, circuit.block)
    inputs = {p: torch.rand(1) for p in params}

    pyqtorch_expectation = expectation_pyq(circuit, observable, inputs)
    horqrux_expectation = jarr_to_tensor(expectation_horqrux(circuit, observable, values_to_jnp(inputs)), pyqtorch_expectation.dtype)
    assert torch.allclose(pyqtorch_expectation, horqrux_expectation, atol=ATOL_32)
