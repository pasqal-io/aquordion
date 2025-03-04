from __future__ import annotations

import pytest
import strategies as st
from hypothesis import given, settings
from strategies import BACKENDS

from aquordion import QuantumCircuit
from aquordion.backends.api import backend_factory
from aquordion.backends.jax_utils import jarr_to_tensor, tensor_to_jnp
from aquordion.states import equivalent_state
from aquordion.types import BackendName
from aquordion.utils_parameters import rand_featureparameters

ATOL_32 = 1e-07  # 32 bit precision
ATOL_DICT = {
    BackendName.PYQTORCH: ATOL_32,
    BackendName.HORQRUX: ATOL_32,
}


@given(st.restricted_circuits())
@settings(deadline=None)
@pytest.mark.parametrize("backend", BACKENDS)
def test_run_for_random_circuit(backend: BackendName, circuit: QuantumCircuit) -> None:
    cfg = {"_use_gate_params": True}
    inputs = rand_featureparameters(circuit, 1)

    # pyqtorch
    bknd_pyqtorch = backend_factory(backend=BackendName.PYQTORCH, configuration=cfg)
    (circ_pyqtorch, _, embed_pyqtorch, params_pyqtorch) = bknd_pyqtorch.convert(circuit)
    wf_pyqtorch = bknd_pyqtorch.run(circ_pyqtorch, embed_pyqtorch(params_pyqtorch, inputs))

    bknd = backend_factory(backend=backend, configuration=cfg)
    (circ, _, embed, params) = bknd.convert(circuit)
    if inputs and backend == BackendName.HORQRUX:
        inputs = {k: tensor_to_jnp(v) for k, v in inputs.items()}
    wf = bknd.run(circ, embed(params, inputs))
    if backend == BackendName.HORQRUX:
        wf = jarr_to_tensor(wf)

    assert equivalent_state(wf_pyqtorch, wf, atol=ATOL_DICT[backend])
