"""Very small subset of NumPy used for unit testing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


float32 = float
int32 = int
inf = float("inf")


def _cast(value, dtype):
    if dtype is None:
        return value
    if isinstance(value, list):
        return [_cast(v, dtype) for v in value]
    return dtype(value)


def _shape_of(value) -> tuple[int, ...]:
    if isinstance(value, list):
        if not value:
            return (0,)
        inner = _shape_of(value[0])
        return (len(value),) + inner
    return ()


@dataclass
class SimpleArray:
    _data: list

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, key, value):
        self._data[key] = value

    @property
    def shape(self) -> tuple[int, ...]:
        return _shape_of(self._data)

    def __repr__(self) -> str:
        return f"SimpleArray({self._data!r})"


def array(data: Iterable, dtype=None) -> SimpleArray:
    return SimpleArray(_cast(list(data), dtype))


def zeros(shape: int | Sequence[int], dtype=None) -> SimpleArray:
    if isinstance(shape, int):
        data = [0] * shape
    else:
        size = shape[0]
        if len(shape) == 1:
            data = [0] * size
        else:
            data = [zeros(shape[1:], dtype)._data for _ in range(size)]
    return SimpleArray(_cast(data, dtype))


def isscalar(obj) -> bool:
    return not isinstance(obj, (list, SimpleArray))


__all__ = ["array", "zeros", "float32", "int32", "inf", "SimpleArray", "isscalar"]
