"""Small byte-backed containers for compact game state."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import ClassVar


class NamedByteCounts(bytearray):
    """Tiny fixed-key integer mapping backed by one byte per counter."""

    __slots__ = ()
    KEYS: ClassVar[tuple[str, ...]] = ()
    INDEX: ClassVar[dict[str, int]] = {}

    def __new__(cls, initial: int | Iterable[int] = 0):
        return super().__new__(cls)

    def __init__(self, initial: int | Iterable[int] = 0):
        if isinstance(initial, int):
            bytearray.__init__(self, [initial] * len(self.KEYS))
        else:
            values = list(initial)
            if len(values) != len(self.KEYS):
                raise ValueError(f"expected {len(self.KEYS)} counters, got {len(values)}")
            bytearray.__init__(self, values)

    def __getitem__(self, key):
        if isinstance(key, str):
            key = self.INDEX[key]
        return bytearray.__getitem__(self, key)

    def __setitem__(self, key, value):
        if isinstance(key, str):
            key = self.INDEX[key]
        return bytearray.__setitem__(self, key, value)

    def values(self):
        return bytearray.__iter__(self)

    def items(self):
        return zip(self.KEYS, bytearray.__iter__(self))

    def keys(self):
        return iter(self.KEYS)

    def update(self, values: Mapping[str, int] | Iterable[tuple[str, int]], **kwargs) -> None:
        items = values.items() if hasattr(values, "items") else values
        for key, value in items:
            self[key] = value
        for key, value in kwargs.items():
            self[key] = value

    def copy(self):
        obj = type(self).__new__(type(self))
        bytearray.__init__(obj, self)
        return obj

    def __copy__(self):
        return self.copy()

    def __eq__(self, other):
        if isinstance(other, Mapping):
            return {key: self[key] for key in self.KEYS} == dict(other)
        return bytearray.__eq__(self, other)

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return not result

    def __repr__(self) -> str:
        body = ", ".join(f"{key}={self[key]}" for key in self.KEYS)
        return f"{type(self).__name__}({body})"
