from __future__ import annotations
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from bisect import bisect_left, bisect_right
from .types import *

# --- core functionality ---
from .extensions.core import _CoreOperations

# --- accessors ---
from .extensions.set import SetAccessor
from .extensions.join import JoinAccessor
from .extensions.grouping import GroupingAccessor
from .extensions.stats import StatsAccessor
from .extensions.combinatorics import CombinatoricsAccessor
from .extensions.utility import UtilityAccessor
from .extensions.terminal import TerminalAccessor
from .extensions.tree import TreeAccessor

# --- abstract base class ---

class IEnumerable(ABC, Generic[T]):
    @abstractmethod
    def _get_data(self) -> List[T]:
        """get the underlying data as a list"""
        pass

# --- base enumerable implementation ---

class _BaseEnumerable(IEnumerable[T]):
    def __init__(self, data_func: Callable[[], List[T]]):
        """init with a function that returns data when called"""
        self._data_func = data_func
        self._cached_result: Optional[List[T]] = None
        self._is_cached = False

    def _get_data(self) -> List[T]:
        """get the current data, caching the result"""
        if not self._is_cached:
            self._cached_result = self._data_func()
            self._is_cached = True
        return self._cached_result

    def _try_numpy_optimization(self, data: List[T], operation: str, func: Optional[Callable] = None) -> Optional[List[T]]:
        """try to optimize operations using numpy when possible."""
        try:
            if data and all(isinstance(x, (int, float, complex)) for x in data):
                arr = np.array(data)
                if operation == 'where':
                    mask = np.vectorize(func)(arr)
                    return arr[mask].tolist()
                elif operation == 'select':
                    return np.vectorize(func)(arr).tolist()
                elif operation == 'distinct':
                    return np.unique(arr).tolist()
            return None
        except (TypeError, ValueError, AttributeError): # catch specific errors
            return None

    def __iter__(self) -> Iterator[T]:
        return iter(self._get_data())

    def __len__(self) -> int:
        return self.to.count()

# --- main enumerable class ---

class Enumerable(
    _BaseEnumerable[T],
    _CoreOperations[T]
):
    """a powerful, linq-inspired enumerable class for python iterables."""
    def __init__(self, data_func: Callable[[], List[T]]):
        super().__init__(data_func)
        # --- initialize accessors ---
        self.set = SetAccessor(self)
        self.join = JoinAccessor(self)
        self.group = GroupingAccessor(self)
        self.stats = StatsAccessor(self)
        self.comb = CombinatoricsAccessor(self)
        self.util = UtilityAccessor(self)
        self.to = TerminalAccessor(self)
        self.tree = TreeAccessor(self)

# --- ordered enumerable class ---

class OrderedEnumerable(Enumerable[T]):
    """represents a sorted sequence, allowing for subsequent orderings."""

    def __init__(self, data_func: Callable[[], List[T]], sort_keys: List[Tuple[Callable, bool]]):
        super().__init__(data_func)
        self._original_data_func = data_func
        self._sort_keys = sort_keys
        # reset cache flags after parent init
        self._cached_result: Optional[List[T]] = None
        self._is_cached = False

    def _get_data(self) -> List[T]:
        """overrides base to apply all sorts at once using stable sort."""
        if not self._is_cached:
            data = self._original_data_func()
            # python's sort is stable, so we sort from the last key to the first
            for key_selector, is_descending in reversed(self._sort_keys):
                data = sorted(data, key=key_selector, reverse=is_descending)
            self._cached_result = data
            self._is_cached = True
        return self._cached_result

    def _get_full_key_selector(self) -> Callable[[T], Tuple]:
        """creates a single selector that returns a tuple of all sort keys."""
        return lambda item: tuple(key_selector(item) for key_selector, _ in self._sort_keys)

    def then_by(self, key_selector: KeySelector[T, K]) -> 'OrderedEnumerable[T]':
        """secondary sort ascending"""
        new_keys = self._sort_keys + [(key_selector, False)]
        return OrderedEnumerable(self._original_data_func, new_keys)

    def then_by_descending(self, key_selector: KeySelector[T, K]) -> 'OrderedEnumerable[T]':
        """secondary sort descending"""
        new_keys = self._sort_keys + [(key_selector, True)]
        return OrderedEnumerable(self._original_data_func, new_keys)

    def find_by_key(self, *key_prefix: Any) -> 'Enumerable[T]':
        """
        efficiently finds all items matching a key prefix using binary search (o(log n)).

        example:
            users.order_by(lambda u: u.state).then_by(lambda u: u.city)
                 .find_by_key("ca", "los angeles")
        """

        def find_data():
            if not self._sort_keys:
                raise TypeError("find_by_key requires at least one order_by call.")
            if len(key_prefix) > len(self._sort_keys):
                raise ValueError("more search keys provided than sort levels exist.")

            # check for descending sorts, which bisect cannot handle natively
            if any(is_desc for _, is_desc in self._sort_keys[:len(key_prefix)]):
                raise NotImplementedError("find_by_key does not support descending-sorted keys.")

            sorted_data = self._get_data()

            # create a selector that only extracts the part of the key we are searching for
            prefix_len = len(key_prefix)
            prefix_selector = lambda item: self._get_full_key_selector()(item)[:prefix_len]

            # use the `key` argument for direct, efficient binary search
            start_index = bisect_left(sorted_data, key_prefix, key=prefix_selector)
            end_index = bisect_right(sorted_data, key_prefix, key=prefix_selector)

            return sorted_data[start_index:end_index]

        return Enumerable(find_data)

    def between_keys(self, lower_bound: Union[Any, Tuple], upper_bound: Union[Any, Tuple]) -> 'Enumerable[T]':
        """
        efficiently gets a slice of items where the sort key(s) are between the bounds.

        example:
            products.order_by(lambda p: p.price).between_keys(10.00, 49.99)
        """

        def between_data():
            if not self._sort_keys:
                raise TypeError("between_keys requires at least one order_by call.")

            lower = lower_bound if isinstance(lower_bound, tuple) else (lower_bound,)
            upper = upper_bound if isinstance(upper_bound, tuple) else (upper_bound,)

            if len(lower) != len(upper):
                raise ValueError("lower and upper bound tuples must have the same length.")

            bound_len = len(lower)
            if any(is_desc for _, is_desc in self._sort_keys[:bound_len]):
                raise NotImplementedError("between_keys does not support descending-sorted keys.")

            sorted_data = self._get_data()
            bound_selector = lambda item: self._get_full_key_selector()(item)[:bound_len]

            start_index = bisect_left(sorted_data, lower, key=bound_selector)
            end_index = bisect_right(sorted_data, upper, key=bound_selector)

            return sorted_data[start_index:end_index]

        return Enumerable(between_data)

    def merge_with(self, other: 'OrderedEnumerable[T]') -> 'Enumerable[T]':
        """
        efficiently merges this sorted sequence with another compatible sorted sequence (o(n + m)).
        raises an error if the sort keys and directions are not identical.
        """

        def merge_data():
            if [sk[1] for sk in self._sort_keys] != [sk[1] for sk in other._sort_keys]:
                raise TypeError("cannot merge enumerables with different sort keys or directions.")

            list_a = self._get_data()
            list_b = other._get_data()

            # build a single comparison function based on the shared sort definition
            key_selector = self._get_full_key_selector()
            desc_flags = [is_desc for _, is_desc in self._sort_keys]

            def compare_items(item1, item2):
                key1, key2 = key_selector(item1), key_selector(item2)
                for i, (k1, k2) in enumerate(zip(key1, key2)):
                    is_desc = desc_flags[i]
                    if k1 < k2: return 1 if is_desc else -1
                    if k1 > k2: return -1 if is_desc else 1
                return 0

            # classic two-pointer merge algorithm using the comparator
            result = []
            i, j = 0, 0
            while i < len(list_a) and j < len(list_b):
                if compare_items(list_a[i], list_b[j]) <= 0:
                    result.append(list_a[i])
                    i += 1
                else:
                    result.append(list_b[j])
                    j += 1

            result.extend(list_a[i:])
            result.extend(list_b[j:])
            return result

        return Enumerable(merge_data)

    def lag_in_order(self, periods: int = 1, fill_value: Optional[T] = None) -> 'Enumerable[Optional[T]]':
        """
        shifts elements by periods based on the sorted order.
        different from the standard lag(), which uses original sequence order.
        """

        def lag_sorted_data():
            sorted_data = self._get_data()
            if periods <= 0: return sorted_data
            return ([fill_value] * periods) + sorted_data[:-periods]

        return Enumerable(lag_sorted_data)

    def lead_in_order(self, periods: int = 1, fill_value: Optional[T] = None) -> 'Enumerable[Optional[T]]':
        """
        shifts elements forward by periods based on the sorted order.
        different from the standard lead(), which uses original sequence order.
        """

        def lead_sorted_data():
            sorted_data = self._get_data()
            if periods <= 0: return sorted_data
            return sorted_data[periods:] + ([fill_value] * periods)

        return Enumerable(lead_sorted_data)