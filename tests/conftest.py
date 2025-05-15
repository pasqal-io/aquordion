from __future__ import annotations

import itertools
import random
from typing import Any, Callable

import pytest
from qadence import CNOT, MCZ, RX, RY, RZ, AbstractBlock, H, I, QuantumCircuit, X, Y, Z, chain, hea

N_qubits_list = [
    2,
    5,
    10,
    15,
]

N_layers_list = [2, 5]

N_qubits_list_vqe = [
    4,
    10,
]


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
def fn_circuit(request: pytest.Fixture) -> Callable:
    return request.param  # type: ignore[no-any-return]


ids_benchmarks: list = list(
    itertools.product(["circuit_A", "circuit_B", "circuit_C"], N_qubits_list, N_layers_list)
)
ids_benchmarks = [f"{id[0]} n:{id[1]} D:{id[2]}" for id in ids_benchmarks]


ids_vqe_benchmarks: list = list(
    itertools.product(["circuit_A", "circuit_B", "circuit_C"], N_qubits_list_vqe, N_layers_list)
)
ids_vqe_benchmarks = [f"{id[0]} n:{id[1]} D:{id[2]}" for id in ids_vqe_benchmarks]


@pytest.fixture(
    params=list(
        itertools.product(
            [
                circuit_A,
                circuit_B,
                circuit_C,
            ],
            N_qubits_list_vqe,
            N_layers_list,
        )
    ),
    ids=ids_vqe_benchmarks,
)
def benchmark_vqe_ansatz(
    request: pytest.Fixture,
) -> Any:
    return request.param


@pytest.fixture(
    params=list(
        itertools.product(
            [hea],
            N_qubits_list_vqe,
            N_layers_list,
        ),
    ),
    ids=[f"HEA n:{id[0]} D:{id[1]}" for id in itertools.product(N_qubits_list_vqe, N_layers_list)],
)
def benchmark_dqc_ansatz(
    request: pytest.Fixture,
) -> Any:
    return request.param


@pytest.fixture(
    params=list(
        itertools.product(
            [
                circuit_A,
                circuit_B,
                circuit_C,
            ],
            N_qubits_list,
            N_layers_list,
        )
    ),
    ids=ids_benchmarks,
)
def benchmark_circuit(
    request: pytest.Fixture,
) -> Any:
    return request.param


@pytest.fixture
def h2_hamiltonian() -> AbstractBlock:
    return (
        -0.09963387941370971 * I(0)
        + 0.17110545123720233 * Z(0)
        + 0.17110545123720225 * Z(1)
        + 0.16859349595532533 * Z(0) * Z(1)
        + 0.04533062254573469 * Y(0) * X(1) * X(2) * Y(3)
        - 0.04533062254573469 * Y(0) * Y(1) * X(2) * X(3)
        - 0.04533062254573469 * X(0) * X(1) * Y(2) * Y(3)
        + 0.04533062254573469 * X(0) * Y(1) * Y(2) * X(3)
        - 0.22250914236600539 * Z(2)
        + 0.12051027989546245 * Z(0) * Z(2)
        - 0.22250914236600539 * Z(3)
        + 0.16584090244119712 * Z(0) * Z(3)
        + 0.16584090244119712 * Z(1) * Z(2)
        + 0.12051027989546245 * Z(1) * Z(3)
        + 0.1743207725924201 * Z(2) * Z(3)
    )
