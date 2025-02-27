from __future__ import annotations

import pytest
import torch
import strategies as st
from strategies import BACKENDS
from hypothesis import given, settings

from qadence import QuantumCircuit
from qadence.types import BackendName
from qadence.backends.api import backend_factory
from qadence.ml_tools.utils import rand_featureparameters
from qadence.backends.jax_utils import jarr_to_tensor, tensor_to_jnp
from qadence.states import equivalent_state

@given(st.digital_circuits())
@settings(deadline=None)
def test_run_methods(circuit: QuantumCircuit) -> None:

    wfs = []
    for b in BACKENDS:
        bknd = backend_factory(backend=b, diff_mode=None)
        (conv_circ, _, embed, params) = bknd.convert(circuit)
        inputs = rand_featureparameters(circuit, 1)
        if inputs and b == BackendName.HORQRUX:
            inputs = {k: tensor_to_jnp(v) for k, v in inputs.items()}
        wf = bknd.run(conv_circ, embed(params, inputs))
        if b == BackendName.HORQRUX:
            wf = jarr_to_tensor(wf)
        wfs += [wf]
    
    for wf in wfs[1:]:
        assert equivalent_state(wf, wfs[0])
    

    
