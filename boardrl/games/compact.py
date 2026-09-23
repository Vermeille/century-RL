"""Small byte-backed containers for compact game state.

The game engines still expose rich values (Card objects, suit strings, named
counters) while the mutable state stores only byte-sized IDs/counts.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, ClassVar


class IndexedByteArray(bytearray):
    """A bytearray whose public elements are values from a constant table.

    Subclasses define ``VALUES``. Internally each element is the index into that
    tuple, so copies are compact C-level byte-buffer copies while game code can
    continue to work with the rich values.
    """

    __slots__ = ()
    VALUES: ClassVar[Sequence[Any]] = ()
    ID_BY_VALUE: ClassVar[dict[Any, int]] = {}

    def __new__(cls, values: Iterable[Any] = ()):
        return super().__new__(cls)

    def __init__(self, values: Iterable[Any] = ()):
        bytearray.__init__(self, (self._encode(value) for value in values))

    @classmethod
    def _encode(cls, value: Any) -> int:
        try:
            return cls.ID_BY_VALUE[value]
        except KeyError as exc:
            raise ValueError(f"{value!r} is not a valid {cls.__name__} value") from exc

    @classmethod
    def from_ids(cls, ids: bytes | bytearray | memoryview) -> "IndexedByteArray":
        obj = cls.__new__(cls)
        bytearray.__init__(obj, ids)
        return obj

    def ids(self) -> memoryview:
        """Read-only-by-convention zero-copy view of the stored byte IDs."""
        return memoryview(self)

    def __iter__(self) -> Iterator[Any]:
        values = self.VALUES
        return (values[index] for index in bytearray.__iter__(self))

    def __getitem__(self, index):
        value = bytearray.__getitem__(self, index)
        if isinstance(index, slice):
            return type(self).from_ids(value)
        return self.VALUES[value]

    def __setitem__(self, index, value):
        if isinstance(index, slice):
            return bytearray.__setitem__(self, index, (self._encode(v) for v in value))
        return bytearray.__setitem__(self, index, self._encode(value))

    def __contains__(self, value: object) -> bool:
        try:
            encoded = self._encode(value)
        except (KeyError, TypeError, ValueError):
            return False
        return bytearray.__contains__(self, encoded)

    def append(self, value: Any) -> None:
        bytearray.append(self, self._encode(value))

    def extend(self, values: Iterable[Any]) -> None:
        bytearray.extend(self, (self._encode(value) for value in values))

    def insert(self, index: int, value: Any) -> None:
        bytearray.insert(self, index, self._encode(value))

    def remove(self, value: Any) -> None:
        bytearray.remove(self, self._encode(value))

    def pop(self, index: int = -1):
        return self.VALUES[bytearray.pop(self, index)]

    def count(self, value: Any) -> int:
        return bytearray.count(self, self._encode(value))

    def index(self, value: Any, *args) -> int:
        return bytearray.index(self, self._encode(value), *args)

    def copy(self):
        # Do not route the raw IDs back through the rich-value constructor.
        # ``bytearray.__init__`` consumes the source buffer directly in C.
        obj = type(self).__new__(type(self))
        bytearray.__init__(obj, self)
        return obj

    def __copy__(self):
        return self.copy()

    def __eq__(self, other):
        if isinstance(other, (list, tuple)):
            return list(self) == list(other)
        return bytearray.__eq__(self, other)

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return not result

    def __repr__(self) -> str:
        return f"{type(self).__name__}({list(self)!r})"


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