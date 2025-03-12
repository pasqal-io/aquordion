from __future__ import annotations
from aquordion.types import BackendName

ATOL_32 = 1e-07  # 32 bit precision
ATOL_DICT = {
    BackendName.PYQTORCH: ATOL_32,
    BackendName.HORQRUX: ATOL_32,
}