from __future__ import annotations

from aquordion.blocks.utils import chain
from aquordion.circuit import QuantumCircuit
from aquordion.operations import I


def fill_identities(circ: QuantumCircuit) -> QuantumCircuit:
    empty_wires = set(range(circ.n_qubits)) - set(circ.block.qubit_support)
    if len(empty_wires) > 0:
        ids = chain(I(i) for i in empty_wires)
        return QuantumCircuit(circ.n_qubits, chain(circ.block, ids))
    return circ
