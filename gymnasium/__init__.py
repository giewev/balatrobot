"""Minimal stub of the :mod:`gymnasium` package used for unit testing.

This project only relies on :mod:`gymnasium.spaces` for type annotations and
shape declarations inside the lightweight pseudo environment code.  Installing
the full Gymnasium dependency is unnecessary for the unit tests, so we provide
just enough of the API for the modules to import successfully.
"""

from dataclasses import dataclass


@dataclass
class _Box:
    low: float
    high: float
    shape: tuple
    dtype: object | None = None


class _Dict(dict):
    def __init__(self, mapping):
        super().__init__(mapping)


class _SpacesModule:
    Box = _Box
    Dict = _Dict


spaces = _SpacesModule()

__all__ = ["spaces"]
