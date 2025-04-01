from __future__ import annotations

from torch import Tensor
from jax import Array
from qadence.backends.jax_utils import tensor_to_jnp

def values_to_jnp(values: dict[str, Tensor]) -> dict[str, Array]:
    if values:
        values = {k: tensor_to_jnp(v) for k, v in values.items()}
    return values