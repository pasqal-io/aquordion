from __future__ import annotations

from collections import Counter

from jax import Array
from qadence import AbstractBlock, QuantumCircuit
from qadence.backends.api import backend_factory
from qadence.types import BackendName
from torch import Tensor

cfg = {"_use_gate_params": True}
bknd_pyqtorch = backend_factory(backend=BackendName.PYQTORCH, configuration=cfg)
bknd_horqrux = backend_factory(backend=BackendName.HORQRUX, configuration=cfg)


def run_pyq(circuit: QuantumCircuit, inputs: dict[str, Tensor]) -> Tensor:
    """Run with PyQTorch"""
    (circ, _, embed_fn, params) = bknd_pyqtorch.convert(circuit)
    return bknd_pyqtorch.run(circ, embed_fn(params, inputs))


def run_horqrux(circuit: QuantumCircuit, inputs: dict[str, Array]) -> Array:
    """Run with Horqrux"""
    (circ, _, embed_fn, params) = bknd_horqrux.convert(circuit)
    return bknd_horqrux.run(circ, embed_fn(params, inputs))


def sample_pyq(circuit: QuantumCircuit, inputs: dict[str, Tensor], n_shots: int = 500) -> Counter:
    """Sample with PyQTorch"""
    (circ, _, embed_fn, params) = bknd_pyqtorch.convert(circuit)
    samples: Counter = bknd_pyqtorch.sample(circ, embed_fn(params, inputs), n_shots=n_shots)
    return samples


def sample_horqrux(
    circuit: QuantumCircuit, inputs: dict[str, Array], n_shots: int = 500
) -> Counter:
    """Sample with Horqrux"""
    (circ, _, embed_fn, params) = bknd_horqrux.convert(circuit)
    samples: Counter = bknd_horqrux.sample(circ, embed_fn(params, inputs), n_shots=n_shots)
    return samples


def expectation_pyq(
    circuit: QuantumCircuit, observable: AbstractBlock, inputs: dict[str, Tensor]
) -> Tensor:
    """Expectation with PyQTorch."""
    (circ, obs, embed_fn, params) = bknd_pyqtorch.convert(circuit, observable)
    return bknd_pyqtorch.expectation(circ, obs, embed_fn(params, inputs))


def expectation_horqrux(
    circuit: QuantumCircuit, observable: AbstractBlock, inputs: dict[str, Array]
) -> Array:
    """Expectation with Horqrux"""
    (circ, obs, embed_fn, params) = bknd_horqrux.convert(circuit, observable)
    return bknd_horqrux.expectation(circ, obs, embed_fn(params, inputs))
