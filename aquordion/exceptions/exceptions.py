from __future__ import annotations


class AquordionException(Exception):
    pass


class NotSupportedError(AquordionException):
    pass


class NotPauliBlockError(AquordionException):
    pass
