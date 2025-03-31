from __future__ import annotations

from enum import Enum
from typing import Union
import numpy as np

TNumber = Union[int, float, complex, np.int64, np.float64]


class StrEnum(str, Enum):
    def __str__(self) -> str:
        """Used when dumping enum fields in a schema."""
        ret: str = self.value
        return ret

    @classmethod
    def list(cls) -> list[str]:
        return list(map(lambda c: c.value, cls))  # type: ignore


class _BackendName(StrEnum):
    """The available backends for running circuits."""

    PYQTORCH = "pyqtorch"
    """The Pyqtorch backend."""
    HORQRUX = "horqrux"
    """The horqrux backend."""

class ParameterType(StrEnum):
    """Parameter types available in qadence."""

    VARIATIONAL = "Variational"
    """VariationalParameters are trainable."""
    FIXED = "Fixed"
    """Fixed/ constant parameters are neither trainable nor act as input."""

