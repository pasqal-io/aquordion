from __future__ import annotations
from enum import Enum

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