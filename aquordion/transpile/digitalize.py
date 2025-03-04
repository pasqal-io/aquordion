from __future__ import annotations

from functools import singledispatch
from typing import overload

from aquordion.blocks import AbstractBlock, CompositeBlock
from aquordion.blocks.utils import _construct
from aquordion.circuit import QuantumCircuit
from aquordion.operations import HamEvo, U
from aquordion.types import LTSOrder


@overload
def digitalize(
    circuit: QuantumCircuit, approximation: LTSOrder = LTSOrder.BASIC
) -> QuantumCircuit: ...


@overload
def digitalize(block: AbstractBlock, approximation: LTSOrder = LTSOrder.BASIC) -> AbstractBlock: ...


@singledispatch
def digitalize(
    circ_or_block: AbstractBlock | QuantumCircuit, approximation: LTSOrder
) -> AbstractBlock | QuantumCircuit:
    raise NotImplementedError(f"digitalize is not implemented for {type(circ_or_block)}")


@digitalize.register  # type: ignore[attr-defined]
def _(block: AbstractBlock, approximation: LTSOrder = LTSOrder.BASIC) -> AbstractBlock:
    if isinstance(block, CompositeBlock):
        return _construct(type(block), tuple(digitalize(b, approximation) for b in block.blocks))
    elif isinstance(block, HamEvo):
        return block.digital_decomposition(approximation=approximation)
    elif isinstance(block, U):
        return block.digital_decomposition()
    else:
        return block


@digitalize.register  # type: ignore[attr-defined]
def _(circuit: QuantumCircuit, approximation: LTSOrder = LTSOrder.BASIC) -> QuantumCircuit:
    return QuantumCircuit(circuit.n_qubits, digitalize(circuit.block, approximation))
