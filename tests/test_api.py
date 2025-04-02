from __future__ import annotations

import strategies as st
import torch
from hypothesis import given, settings
from qadence import AbstractBlock, QuantumCircuit
from qadence.backends.jax_utils import jarr_to_tensor
from qadence.ml_tools.utils import rand_featureparameters
from qadence.states import equivalent_state

from aquordion.benchmarks import expectation_horqrux, expectation_pyq, run_horqrux, run_pyq
from aquordion.utils import values_to_jnp

ATOL_32 = 1e-07  # 32 bit precision


@given(st.restricted_circuits())
@settings(deadline=None)
def test_run_for_random_circuit(circuit: QuantumCircuit) -> None:
    inputs = rand_featureparameters(circuit, 1)
    wf_pyqtorch = run_pyq(circuit, inputs)
    wf_horqrux = jarr_to_tensor(run_horqrux(circuit, values_to_jnp(inputs)))
    assert equivalent_state(wf_pyqtorch, wf_horqrux, atol=ATOL_32)


@given(st.restricted_circuits(), st.observables())
@settings(deadline=None)
def test_expectation_for_random_circuit(circuit: QuantumCircuit, observable: AbstractBlock) -> None:

    if observable.n_qubits > circuit.n_qubits:
        circuit = QuantumCircuit(observable.n_qubits, circuit.block)
    inputs = rand_featureparameters(circuit, 1)
    pyqtorch_expectation = expectation_pyq(circuit, observable, inputs)
    horqrux_expectation = jarr_to_tensor(
        expectation_horqrux(circuit, observable, values_to_jnp(inputs)), pyqtorch_expectation.dtype
    )
    assert torch.allclose(pyqtorch_expectation, horqrux_expectation, atol=ATOL_32)
