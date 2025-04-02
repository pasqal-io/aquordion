from __future__ import annotations

import random
from typing import Callable

import pytest
from qadence import CNOT, MCZ, RX, RY, RZ, H, QuantumCircuit, chain


def circuit_A(n_qubits: int, n_layers: int = 1) -> tuple[QuantumCircuit, list[str]]:
    """A simple circuit structure without entangling layers coming from https://arxiv.org/pdf/2009.02823.pdf"""
    ops = []
    params = []
    for j in range(n_layers):
        for i in range(n_qubits):
            ops += [RX(i, f"theta_{j}_{i}_0"), RZ(i, f"theta_{j}_{i}_1")]
            params += [f"theta_{j}_{i}_0", f"theta_{j}_{i}_1"]

    circ = QuantumCircuit(n_qubits, chain(*ops))
    return circ, params


def circuit_B(n_qubits: int, n_layers: int = 1) -> tuple[QuantumCircuit, list[str]]:
    """A circuit structure with CZ entangling layers coming from https://arxiv.org/pdf/2009.02823.pdf"""
    ops = []
    params = []
    entangler = [MCZ((i,), i + 1) for i in range(n_qubits - 1)]
    H_transform = [H(i) for i in range(n_qubits)]

    for j in range(n_layers):
        ops += H_transform + entangler
        for i in range(n_qubits):
            ops += [RX(i, f"theta_{j}_{i}")]
            params += [f"theta_{j}_{i}"]

    circ = QuantumCircuit(n_qubits, chain(*ops))
    return circ, params


def circuit_C(n_qubits: int, n_layers: int = 1) -> tuple[QuantumCircuit, list[str]]:
    """A circuit structure with CNOT entangling layers coming from https://arxiv.org/pdf/2009.02823.pdf"""
    ops = []
    params = []
    entangler = [CNOT(i, i + 1) for i in range(n_qubits - 1)]

    for j in range(n_layers):
        for i in range(n_qubits):
            ops += [RY(i, f"theta_{j}_{i}_0"), RZ(i, f"theta_{j}_{i}_1")]
            params += [f"theta_{j}_{i}_0", f"theta_{j}_{i}_1"]
        ops += entangler

    for i in range(n_qubits):
        ops += [RY(i, f"theta_{n_layers}_{i}_0"), RZ(i, f"theta_{n_layers}_{i}_1")]
        params += [f"theta_{n_layers}_{i}_0", f"theta_{n_layers}_{i}_1"]

    circ = QuantumCircuit(n_qubits, chain(*ops))
    return circ, params


@pytest.fixture
def random_circuit_integers() -> tuple[int, int]:
    n_qubits = random.randint(1, 5)
    n_layers = random.randint(1, 5)
    return n_qubits, n_layers


@pytest.fixture(
    params=[
        circuit_A,
        circuit_B,
        circuit_C,
    ],
    ids=["circuit_A", "circuit_B", "circuit_C"],
)
def fn_circuit(request) -> Callable:  # type: ignore[no-untyped-def]
    return request.param  # type: ignore[no-any-return]
