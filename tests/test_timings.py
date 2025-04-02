from __future__ import annotations

from typing import Callable

import pytest
import torch

from aquordion.benchmarks import run_horqrux, run_pyq
from aquordion.utils import values_to_jnp


@pytest.mark.parametrize(
    "n_qubits",
    [
        2,
        5,
    ],
)
@pytest.mark.parametrize(
    "n_layers",
    [
        2,
        5,
    ],
)
def test_run_pyq(
    benchmark: pytest.Fixture, fn_circuit: Callable, n_qubits: int, n_layers: int
) -> None:

    circuit, params = fn_circuit(n_qubits, n_layers)
    torch.manual_seed(0)
    inputs = {p: torch.rand(1) for p in params}

    benchmark(run_pyq, circuit, inputs)


@pytest.mark.parametrize(
    "n_qubits",
    [
        2,
        5,
    ],
)
@pytest.mark.parametrize(
    "n_layers",
    [
        2,
        5,
    ],
)
def test_run_horqrux(
    benchmark: pytest.Fixture, fn_circuit: Callable, n_qubits: int, n_layers: int
) -> None:

    circuit, params = fn_circuit(n_qubits, n_layers)
    torch.manual_seed(0)
    inputs = {p: torch.rand(1) for p in params}

    jnp_inputs = values_to_jnp(inputs)
    benchmark(run_horqrux, circuit, jnp_inputs)
