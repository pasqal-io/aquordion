from __future__ import annotations

from functools import singledispatch
from typing import Any

from torch import Tensor, rand

from aquordion.blocks import AbstractBlock, parameters
from aquordion.circuit import QuantumCircuit
from aquordion.parameters import Parameter, stringify

__all__ = []  # type: ignore


@singledispatch
def rand_featureparameters(x: QuantumCircuit | AbstractBlock, *args: Any) -> dict[str, Tensor]:
    raise NotImplementedError(f"Unable to generate random featureparameters for object {type(x)}.")


@rand_featureparameters.register
def _(block: AbstractBlock, batch_size: int = 1) -> dict[str, Tensor]:
    non_number_params = [p for p in parameters(block) if not p.is_number]
    feat_params: list[Parameter] = [p for p in non_number_params if not p.trainable]
    return {stringify(p): rand(batch_size, requires_grad=False) for p in feat_params}


@rand_featureparameters.register
def _(circuit: QuantumCircuit, batch_size: int = 1) -> dict[str, Tensor]:
    return rand_featureparameters(circuit.block, batch_size)
