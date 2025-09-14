from typing import (
    TypeVar, Generic, Callable, Iterator, Iterable, Any, Optional, Union,
    Dict, List, Tuple, Set, Type
)

T = TypeVar('T')
U = TypeVar('U')
K = TypeVar('K')
V = TypeVar('V')

Predicate = Callable[[T], bool]
Selector = Callable[[T], U]
KeySelector = Callable[[T], K]
Comparer = Callable[[T, T], int]
Accumulator = Callable[[U, T], U]


class TreeNode(Generic[T]):
    """represents a node in tree traversal with path and depth context"""

    def __init__(self, value: T, path: List[T], depth: int):
        self.value = value
        self.path = path  # no copy for performance
        self.depth = depth

    def __repr__(self) -> str:
        return f"TreeNode(value={self.value}, depth={self.depth}, path_length={len(self.path)})"


class ParseResult(Generic[T]):
    """encapsulates results of parsing operations"""

    def __init__(self, successes: List[T], failures: List[str]):
        self.successes = successes
        self.failures = failures

    @property
    def has_failures(self) -> bool: return len(self.failures) > 0

    @property
    def success_count(self) -> int: return len(self.successes)

    @property
    def failure_count(self) -> int: return len(self.failures)

    def __repr__(self) -> str:
        return f"ParseResult(successes={self.success_count}, failures={self.failure_count})"


class NestedGroup(Generic[K, T]):
    """represents a group with hierarchical child groups"""

    def __init__(self, key: K, items: List[T], children: List['NestedGroup[K, T]']):
        self.key = key
        self.items = items
        self.children = children

    @property
    def is_leaf(self) -> bool: return not self.children

    def __repr__(self) -> str:
        return f"NestedGroup(key={self.key}, items={len(self.items)}, children={len(self.children)})"


class TreeItem(Generic[T, K]):
    """represents an item in a constructed tree structure"""

    def __init__(self, value: T, children: List['TreeItem[T, K]']):
        self.value = value
        self.children = children

    @property
    def is_leaf(self) -> bool: return not self.children

    def __repr__(self) -> str:
        return f"TreeItem(value={self.value}, children={len(self.children)})"


class MemoizedEnumerable(Generic[T]):
    """
    advanced memoized enumerable with lazy evaluation and partial caching
    supports streaming and partial materialization
    """

    def __init__(self, data_func: Callable[[], Iterator[T]]):
        self._source_func = data_func
        self._cache = []
        self._source_iterator = None
        self._is_fully_enumerated = False

    def _get_iterator(self):
        """get or create the source iterator"""
        if self._source_iterator is None:
            self._source_iterator = iter(self._source_func())
        return self._source_iterator

    def _materialize_to_index(self, target_index: int):
        """materialize the cache up to (and including) the target index"""
        if self._is_fully_enumerated:
            return

        iterator = self._get_iterator()
        while len(self._cache) <= target_index and not self._is_fully_enumerated:
            try:
                item = next(iterator)
                self._cache.append(item)
            except StopIteration:
                self._is_fully_enumerated = True
                break

    def _materialize_all(self):
        """fully materialize the enumerable into cache"""
        if self._is_fully_enumerated:
            return

        iterator = self._get_iterator()
        try:
            while True:
                item = next(iterator)
                self._cache.append(item)
        except StopIteration:
            self._is_fully_enumerated = True

    def __iter__(self) -> Iterator[T]:
        # yield from cache first
        for item in self._cache:
            yield item

        if not self._is_fully_enumerated:
            iterator = self._get_iterator()
            try:
                while True:
                    item = next(iterator)
                    self._cache.append(item)
                    yield item
            except StopIteration:
                self._is_fully_enumerated = True

    def __getitem__(self, index):
        """support indexing by materializing up to the requested index"""
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self[i] for i in range(start, stop, step)]

        if index < 0:
            self._materialize_all()
            if abs(index) > len(self._cache):
                raise IndexError("index out of range")
            return self._cache[index]

        self._materialize_to_index(index)

        if index < len(self._cache):
            return self._cache[index]
        else:
            raise IndexError("index out of range")

    def __len__(self):
        """get the length by fully materializing if necessary"""
        self._materialize_all()
        return len(self._cache)