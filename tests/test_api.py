from __future__ import annotations

import random
from typing import Callable

import horqrux as hx
import pyqtorch as pyq
import pytest
import strategies as st
import torch
from hypothesis import given, settings
from pyqtorch import QuantumCircuit

from aquordion.circuits import circuit_A, circuit_B, circuit_C
from aquordion.conversion import jarr_to_tensor, pyqcircuit_to_horqrux, tensor_to_jnp, values_to_jax

list_circuits_fn = [circuit_A, circuit_B, circuit_C]


@pytest.mark.parametrize("fn_circuit", list_circuits_fn)
def test_run_circuits(fn_circuit: Callable) -> None:
    n_qubits = random.randint(1, 5)
    n_layers = random.randint(1, 5)

    circ, params = fn_circuit(n_qubits, n_layers)
    values = {p: torch.rand(1) for p in params}

    state = pyq.random_state(n_qubits)
    pyq_output = circ(state, values)

    jax_circ = pyqcircuit_to_horqrux(circ)
    horqrux_output = jarr_to_tensor(jax_circ(tensor_to_jnp(state), values_to_jax(values)))

    assert torch.allclose(horqrux_output, pyq_output)


@pytest.mark.parametrize("fn_circuit", list_circuits_fn)
def test_expectation_circuits(fn_circuit: Callable) -> None:
    n_qubits = random.randint(1, 5)
    n_layers = random.randint(1, 5)

    circ, params = fn_circuit(n_qubits, n_layers)
    values = {p: torch.rand(1) for p in params}

    state = pyq.random_state(n_qubits)
    obs_op = pyq.Z(0)
    observable = pyq.Observable([obs_op])
    pyq_output = pyq.expectation(circ, state, values, observable)

    jax_circ = pyqcircuit_to_horqrux(circ)
    jax_obs = [hx.Observable([hx.Z(0)])]
    horqrux_output = jarr_to_tensor(
        hx.expectation(tensor_to_jnp(state), jax_circ, jax_obs, values_to_jax(values)),
        torch.float64,
    )

    assert torch.allclose(horqrux_output, pyq_output)


@given(st.digital_circuits())
@settings(deadline=None)
def test_run_strategies(circ_params: tuple[QuantumCircuit, tuple[str, ...]]) -> None:
    circ, params = circ_params
    values = {p: torch.rand(1) for p in params}

    state = pyq.random_state(circ.n_qubits)
    pyq_output = circ(state, values)
    jax_circ = pyqcircuit_to_horqrux(circ)
    horqrux_output = jarr_to_tensor(jax_circ(tensor_to_jnp(state), values_to_jax(values)))

    assert torch.allclose(horqrux_output, pyq_output)


@given(st.digital_circuits())
@settings(deadline=None)
def test_expectation_strategies(circ_params: tuple[QuantumCircuit, tuple[str, ...]]) -> None:
    circ, params = circ_params
    values = {p: torch.rand(1) for p in params}
    state = pyq.random_state(circ.n_qubits)

    obs_op = pyq.Z(0)
    observable = pyq.Observable([obs_op])
    pyq_output = pyq.expectation(circ, state, values, observable)

    jax_circ = pyqcircuit_to_horqrux(circ)
    jax_obs = [hx.Observable([hx.Z(0)])]
    horqrux_output = jarr_to_tensor(
        hx.expectation(tensor_to_jnp(state), jax_circ, jax_obs, values_to_jax(values)),
        torch.float64,
    )

    assert torch.allclose(horqrux_output, pyq_output)
