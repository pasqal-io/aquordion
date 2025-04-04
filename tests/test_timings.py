from __future__ import annotations

from typing import Callable

import pytest
import torch
from qadence import Z

from aquordion.benchmarks import expectation_horqrux, expectation_pyq, run_horqrux, run_pyq
from aquordion.utils import values_to_jnp


def test_run_pyq(benchmark: pytest.Fixture, benchmark_circuit: tuple[Callable, int, int]) -> None:
    print(benchmark_circuit)
    fn_circuit, n_qubits, n_layers = benchmark_circuit
    circuit, params = fn_circuit(n_qubits, n_layers)
    torch.manual_seed(0)
    inputs = {p: torch.rand(1) for p in params}

    benchmark.pedantic(run_pyq, args=(circuit, inputs), rounds=10)


def test_run_horqrux(
    benchmark: pytest.Fixture, benchmark_circuit: tuple[Callable, int, int]
) -> None:
    fn_circuit, n_qubits, n_layers = benchmark_circuit
    circuit, params = fn_circuit(n_qubits, n_layers)
    torch.manual_seed(0)
    inputs = {p: torch.rand(1) for p in params}

    jnp_inputs = values_to_jnp(inputs)
    benchmark.pedantic(run_horqrux, args=(circuit, jnp_inputs), rounds=10)


def test_expectation_pyq(
    benchmark: pytest.Fixture, benchmark_circuit: tuple[Callable, int, int]
) -> None:
    fn_circuit, n_qubits, n_layers = benchmark_circuit
    circuit, params = fn_circuit(n_qubits, n_layers)
    torch.manual_seed(0)
    inputs = {p: torch.rand(1) for p in params}
    observable = Z(0)

    benchmark.pedantic(expectation_pyq, args=(circuit, observable, inputs), rounds=10)


def test_expectation_horqrux(
    benchmark: pytest.Fixture, benchmark_circuit: tuple[Callable, int, int]
) -> None:
    fn_circuit, n_qubits, n_layers = benchmark_circuit
    circuit, params = fn_circuit(n_qubits, n_layers)
    torch.manual_seed(0)
    inputs = {p: torch.rand(1) for p in params}
    jnp_inputs = values_to_jnp(inputs)
    observable = Z(0)

    benchmark.pedantic(expectation_horqrux, args=(circuit, observable, jnp_inputs), rounds=10)
