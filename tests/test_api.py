from __future__ import annotations

import strategies as st
import torch
from hypothesis import given, settings
from qadence import AbstractBlock, QuantumCircuit
from qadence.backends.api import backend_factory
from qadence.backends.jax_utils import jarr_to_tensor, tensor_to_jnp
from qadence.ml_tools.utils import rand_featureparameters
from qadence.states import equivalent_state
from qadence.types import BackendName

ATOL_32 = 1e-07  # 32 bit precision
ATOL_DICT = {
    BackendName.PYQTORCH: ATOL_32,
    BackendName.HORQRUX: ATOL_32,
}


@given(st.restricted_circuits())
@settings(deadline=None)
def test_run_for_random_circuit(circuit: QuantumCircuit) -> None:
    cfg = {"_use_gate_params": True}
    inputs = rand_featureparameters(circuit, 1)

    # pyqtorch
    bknd_pyqtorch = backend_factory(backend=BackendName.PYQTORCH, configuration=cfg)
    (circ_pyqtorch, _, embed_pyqtorch, params_pyqtorch) = bknd_pyqtorch.convert(circuit)
    wf_pyqtorch = bknd_pyqtorch.run(circ_pyqtorch, embed_pyqtorch(params_pyqtorch, inputs))

    # horqrux
    backend = BackendName.HORQRUX
    bknd = backend_factory(backend=backend, configuration=cfg)
    (circ, _, embed, params) = bknd.convert(circuit)
    if inputs and backend == BackendName.HORQRUX:
        inputs = {k: tensor_to_jnp(v) for k, v in inputs.items()}
    wf = bknd.run(circ, embed(params, inputs))
    if backend == BackendName.HORQRUX:
        wf = jarr_to_tensor(wf)

    assert equivalent_state(wf_pyqtorch, wf, atol=ATOL_DICT[backend])


@given(st.restricted_circuits(), st.observables())
@settings(deadline=None)
def test_expectation_for_random_circuit(circuit: QuantumCircuit, observable: AbstractBlock) -> None:
    if observable.n_qubits > circuit.n_qubits:
        circuit = QuantumCircuit(observable.n_qubits, circuit.block)

    cfg = {"_use_gate_params": True}
    inputs = rand_featureparameters(circuit, 1)

    # pyqtorch
    bknd_pyqtorch = backend_factory(backend=BackendName.PYQTORCH, configuration=cfg)
    (circ_pyqtorch, obs_pyqtorch, embed_pyqtorch, params_pyqtorch) = bknd_pyqtorch.convert(
        circuit, observable
    )
    pyqtorch_expectation = bknd_pyqtorch.expectation(
        circ_pyqtorch, obs_pyqtorch, embed_pyqtorch(params_pyqtorch, inputs)
    )

    # horqrux
    backend = BackendName.HORQRUX
    bknd = backend_factory(backend=backend, configuration=cfg)
    (circ, obs, embed, params) = bknd.convert(circuit, observable)
    if inputs and backend == BackendName.HORQRUX:
        inputs = {k: tensor_to_jnp(v) for k, v in inputs.items()}
    expectation = bknd.expectation(circ, obs, embed(params, inputs))
    if backend == BackendName.HORQRUX:
        expectation = jarr_to_tensor(expectation, pyqtorch_expectation.dtype)

    assert torch.allclose(pyqtorch_expectation, expectation, atol=ATOL_DICT[backend])
