from __future__ import annotations

from pyqtorch import QuantumCircuit


def extract_parameters(circuit: QuantumCircuit) -> tuple[str, ...]:
    """Extract string parameter names out of a circuit.

    Args:
        circuit (QuantumCircuit): Circuit.

    Returns:
        tuple[str, ...]: Tuple of string parameter names.
    """
    parameters: tuple = tuple()
    for op in circuit.flatten():
        if hasattr(op, "param_name") and isinstance(op.param_name, str):
            parameters += (op.param_name,)
    return parameters
