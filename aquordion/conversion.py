from __future__ import annotations

from typing import Any

import horqrux
import jax.numpy as jnp
import pyqtorch as pyq
from jax import Array, device_get
from torch import Tensor, cdouble, from_numpy

pyq_to_horqrux_types_match = {
    pyq.RX: horqrux.RX,
    pyq.RY: horqrux.RY,
    pyq.RZ: horqrux.RZ,
    pyq.H: horqrux.H,
    pyq.Z: horqrux.Z,
    pyq.X: horqrux.X,
    pyq.Y: horqrux.Y,
    pyq.CZ: horqrux.Z,
    pyq.CNOT: horqrux.X,
    pyq.CY: horqrux.Y,
    pyq.I: horqrux.I,
    pyq.H: horqrux.H,
    pyq.S: horqrux.S,
    pyq.T: horqrux.T,
}


def pyqcircuit_to_horqrux(qc: pyq.QuantumCircuit) -> horqrux.QuantumCircuit:
    horqrux_ops = list()
    for op in qc.flatten():
        call_op = pyq_to_horqrux_types_match[type(op)]
        if isinstance(op, pyq.primitives.Parametric):
            param = (
                op.param_name if isinstance(op.param_name, str) else tensor_to_jnp(op.param_name)
            )
            horqrux_ops.append(call_op(target=op.target, control=op.control, param=param))
        else:
            horqrux_ops.append(call_op(target=op.target, control=op.control))
    return horqrux.QuantumCircuit(qc.n_qubits, horqrux_ops)


def jarr_to_tensor(arr: Array, dtype: Any = cdouble) -> Tensor:
    return from_numpy(device_get(arr)).to(dtype=dtype)


def tensor_to_jnp(tensor: Tensor, dtype: Any = jnp.complex128) -> Array:
    return (
        jnp.array(tensor.numpy(), dtype=dtype)
        if not tensor.requires_grad
        else jnp.array(tensor.detach().numpy(), dtype=dtype)
    )


def values_to_jax(param_values: dict[str, Tensor]) -> dict[str, Array]:
    return {key: jnp.array(value.detach().numpy()) for key, value in param_values.items()}
