from __future__ import annotations

import torch
import pytest
import tests.strategies as st
from hypothesis import given, settings
from tests.strategies import BACKENDS

from aquordion import QuantumCircuit
from aquordion.backends.api import backend_factory
from aquordion.backends.jax_utils import jarr_to_tensor, tensor_to_jnp
from aquordion.types import BackendName
from aquordion.utils_parameters import rand_featureparameters

from tests.api.tols import ATOL_DICT


@given(st.restricted_circuits(), st.observables())
@settings(deadline=None)
@pytest.mark.parametrize("backend", BACKENDS)
def test_expectation_for_random_circuit(backend: BackendName, circuit: QuantumCircuit, observable: AbstractBlock) -> None:
    if observable.n_qubits > circuit.n_qubits:
        circuit = QuantumCircuit(observable.n_qubits, circuit.block)
    
    cfg = {"_use_gate_params": True}
    inputs = rand_featureparameters(circuit, 1)

    # pyqtorch
    bknd_pyqtorch = backend_factory(backend=BackendName.PYQTORCH, configuration=cfg)
    (circ_pyqtorch, obs_pyqtorch, embed_pyqtorch, params_pyqtorch) = bknd_pyqtorch.convert(circuit, observable)
    pyqtorch_expectation = bknd_pyqtorch.expectation(circ_pyqtorch, obs_pyqtorch, embed_pyqtorch(params_pyqtorch, inputs))

    bknd = backend_factory(backend=backend, configuration=cfg)
    (circ, obs, embed, params) = bknd.convert(circuit, observable)
    if inputs and backend == BackendName.HORQRUX:
        inputs = {k: tensor_to_jnp(v) for k, v in inputs.items()}
    expectation = bknd.expectation(circ, obs, embed(params, inputs))
    if backend == BackendName.HORQRUX:
        expectation = jarr_to_tensor(expectation, dtype=torch.double)

    assert torch.allclose(pyqtorch_expectation, expectation, atol=ATOL_DICT[backend])
