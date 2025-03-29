from __future__ import annotations

from pyqtorch import CNOT, CZ, RX, RY, RZ, H, QuantumCircuit


def circuit_A(n_qubits: int, n_layers: int = 1) -> tuple[QuantumCircuit, list[str]]:
    ops = []
    params = []
    for j in range(n_layers):
        for i in range(n_qubits):
            ops += [RX(i, f"theta_{j}_{i}_0"), RZ(i, f"theta_{j}_{i}_1")]
            params += [f"theta_{j}_{i}_0",f"theta_{j}_{i}_1" ]

    circ = QuantumCircuit(n_qubits, ops)
    return circ, params


def circuit_B(n_qubits: int, n_layers: int = 1) -> tuple[QuantumCircuit, list[str]]:
    ops = []
    params = []
    entangler = [CZ(i, i + 1) for i in range(n_qubits - 1)]
    H_transform = [H(i) for i in range(n_qubits)]

    for j in range(n_layers):
        ops += H_transform + entangler
        for i in range(n_qubits):
            ops += [RX(i, f"theta_{j}_{i}")]
            params += [f"theta_{j}_{i}"]

    circ = QuantumCircuit(n_qubits, ops)
    return circ, params


def circuit_C(n_qubits: int, n_layers: int = 1) -> tuple[QuantumCircuit, list[str]]:
    ops = []
    params = []
    entangler = [CNOT(i, i + 1) for i in range(n_qubits - 1)]

    for j in range(n_layers):
        for i in range(n_qubits):
            ops += [RY(i, f"theta_{j}_{i}_0"), RZ(i, f"theta_{j}_{i}_1")]
            params += [f"theta_{j}_{i}_0",f"theta_{j}_{i}_1" ]
        ops += entangler

    for i in range(n_qubits):
        ops += [RY(i, f"theta_{n_layers}_{i}_0"), RZ(i, f"theta_{n_layers}_{i}_1")]
        params += [f"theta_{n_layers}_{i}_0",f"theta_{n_layers}_{i}_1" ]

    circ = QuantumCircuit(n_qubits, ops)
    return circ, params