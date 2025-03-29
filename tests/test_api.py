from __future__ import annotations

import random
from typing import Callable

import pyqtorch as pyq
import pytest
import torch

from aquordion.circuits import circuit_A, circuit_B, circuit_C
from aquordion.conversion import jarr_to_tensor, pyq_to_horqrux, tensor_to_jnp, values_to_jax


@pytest.mark.parametrize("fn_circuit", [circuit_A, circuit_B, circuit_C])
def test_run_circuits(fn_circuit: Callable) -> None:
    n_qubits = random.randint(1, 5)
    n_layers = random.randint(1, 5)

    circ, params = fn_circuit(n_qubits, n_layers)
    values = {p: torch.rand(1) for p in params}

    state = pyq.random_state(n_qubits)
    pyq_output = circ(state, values)

    jax_circ = pyq_to_horqrux(circ)
    horqrux_output = jarr_to_tensor(jax_circ(tensor_to_jnp(state), values_to_jax(values)))

    assert torch.allclose(horqrux_output, pyq_output)
