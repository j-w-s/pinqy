"""
'    __________.___ _______   ________ _____.___.
'    \______   \   |\      \  \_____  \\__  |   |
'     |     ___/   |/   |   \  /  / \  \/   |   |
'     |    |   |   /    |    \/   \_/.  \____   |
'     |____|   |___\____|__  /\_____\ \_/ ______|
'                          \/        \__>/
"""

import numpy as np
import pandas as pd
from typing import (
    TypeVar, Generic, Callable, Iterator, Iterable, Any, Optional, Union,
    Dict, List, Tuple, Set, Type
)
from functools import reduce
from collections import defaultdict
from itertools import (
    chain, accumulate, dropwhile, takewhile,
    permutations as itertools_permutations,
    combinations as itertools_combinations, zip_longest
)
from itertools import batched
from abc import ABC, abstractmethod
from bisect import bisect_right
import math
import random
from bisect import bisect_left, bisect_right
from itertools import product as itertools_product
from itertools import combinations_with_replacement as itertools_combinations_with_replacement
from itertools import groupby as itertools_groupby
from collections import deque

# --- type definitions ---

T = TypeVar('T')
U = TypeVar('U')
K = TypeVar('K')
V = TypeVar('V')

Predicate = Callable[[T], bool]
Selector = Callable[[T], U]
KeySelector = Callable[[T], K]
Comparer = Callable[[T, T], int]
Accumulator = Callable[[U, T], U]

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
        return self.count()

# --- operation mixins for organization and extensibility ---

# --- core operations ---

class _CoreOperationsMixin(Generic[T]):
    def where(self: '_BaseEnumerable[T]', predicate: Predicate[T]) -> 'Enumerable[T]':
        """filter elements based on a predicate"""
        def filter_data():
            data = self._get_data()
            optimized = self._try_numpy_optimization(data, 'where', predicate)
            if optimized is not None: return optimized
            return [x for x in data if predicate(x)]
        return Enumerable(filter_data)

    def select(self: '_BaseEnumerable[T]', selector: Selector[T, U]) -> 'Enumerable[U]':
        """project each element to a new form"""
        def map_data():
            data = self._get_data()
            optimized = self._try_numpy_optimization(data, 'select', selector)
            if optimized is not None: return optimized
            return [selector(x) for x in data]
        return Enumerable(map_data)

    def select_many(self: '_BaseEnumerable[T]', selector: Selector[T, Iterable[U]]) -> 'Enumerable[U]':
        """project and flatten sequences"""
        def flat_map_data():
            # a nested list comprehension is often more performant and is functionally equivalent
            return [item for sublist in self.select(selector) for item in sublist]
        return Enumerable(flat_map_data)

    def order_by(self: '_BaseEnumerable[T]', key_selector: Callable[[T], K]) -> 'OrderedEnumerable[T]':
        """sort elements by a key"""
        return OrderedEnumerable(self._data_func, [(key_selector, False)])

    def order_by_descending(self: '_BaseEnumerable[T]', key_selector: Callable[[T], K]) -> 'OrderedEnumerable[T]':
        """sort elements by a key in descending order"""
        return OrderedEnumerable(self._data_func, [(key_selector, True)])

    def take(self: '_BaseEnumerable[T]', count: int) -> 'Enumerable[T]':
        """take the first 'count' elements"""
        return Enumerable(lambda: self._get_data()[:count])

    def skip(self: '_BaseEnumerable[T]', count: int) -> 'Enumerable[T]':
        """skip the first 'count' elements"""
        return Enumerable(lambda: self._get_data()[count:])

    def take_while(self: '_BaseEnumerable[T]', predicate: Predicate[T]) -> 'Enumerable[T]':
        """take elements while predicate is true"""
        # itertools.takewhile is the most efficient implementation for this
        return Enumerable(lambda: list(takewhile(predicate, self._get_data())))

    def skip_while(self: '_BaseEnumerable[T]', predicate: Predicate[T]) -> 'Enumerable[T]':
        """skip elements while predicate is true"""
        # itertools.dropwhile is the most efficient implementation for this
        return Enumerable(lambda: list(dropwhile(predicate, self._get_data())))

    def select_with_index(self: '_BaseEnumerable[T]', selector: Callable[[T, int], U]) -> 'Enumerable[U]':
        """project each element to a new form, using the element's index"""
        def map_with_index_data():
            return [selector(item, index) for index, item in enumerate(self._get_data())]
        return Enumerable(map_with_index_data)

    def reverse(self: '_BaseEnumerable[T]') -> 'Enumerable[T]':
        """inverts the order of the elements in a sequence"""
        def reverse_data():
            return list(reversed(self._get_data()))
        return Enumerable(reverse_data)

    def append(self: '_BaseEnumerable[T]', element: T) -> 'Enumerable[T]':
        """appends a value to the end of the sequence"""
        def append_data():
            # itertools.chain is the most memory-efficient way to combine iterables
            return list(chain(self._get_data(), [element]))
        return Enumerable(append_data)

    def prepend(self: '_BaseEnumerable[T]', element: T) -> 'Enumerable[T]':
        """adds a value to the beginning of the sequence"""
        def prepend_data():
            # chain is also optimal for prepending
            return list(chain([element], self._get_data()))
        return Enumerable(prepend_data)

    def default_if_empty(self: '_BaseEnumerable[T]', default_value: T) -> 'Enumerable[T]':
        """returns the elements of a sequence, or a default value in a singleton collection if the sequence is empty"""
        def default_data():
            data = self._get_data()
            return data if data else [default_value]
        return Enumerable(default_data)

    def of_type(self: '_BaseEnumerable[T]', type_filter: Type[U]) -> 'Enumerable[U]':
        """filters the elements of a sequence based on a specified type"""
        # this is syntactic sugar over where() but improves readability and intent
        # the type hint Type[U] ensures the user passes a class/type, not an instance
        return self.where(lambda item: isinstance(item, type_filter))

    def batched(self: '_BaseEnumerable[T]', size: int) -> 'Enumerable[Tuple[T, ...]]':
        """
        batches elements of a sequence into tuples of a specified size. requires python 3.12+.
        the last batch may be smaller than the requested size.
        """
        if size <= 0:
            raise ValueError("batch size must be positive")
        def batched_data():
            # convert the resulting tuples to lists to maintain the list-of-lists convention of chunk()
            return [list(batch) for batch in batched(self._get_data(), size)]
        return Enumerable(batched_data)

    def for_each(self: '_BaseEnumerable[T]', action: Callable[[T], Any]) -> 'Enumerable[T]':
        """
        performs the specified action on each element of a sequence for side-effects.
        this is a terminal operation in spirit but returns the original enumerable to allow chaining.
        """
        def for_each_data():
            data = self._get_data()
            for item in data:
                action(item)
            # returns the original data to the next link in the chain
            return data
        return Enumerable(for_each_data)

    def as_ordered(self: '_BaseEnumerable[T]') -> 'OrderedEnumerable[T]':
        """
        treats the current sequence as already ordered, allowing 'then_by' to be called.
        this does not perform a sort. use it only when the source is pre-sorted.
        """
        # create an orderedenumerable with a no-op sort
        # the key is constant, so python's stable sort preserves the original order
        return OrderedEnumerable(self._data_func, [(lambda x: None, False)])

# --- set operations ---
class _SetOperationsMixin(Generic[T]):
    """
    provides core and advanced set-theoretic operations.
    this includes standard union, intersection, and difference,
    as well as multiset (bag) operations and similarity metrics.
    """

    def distinct(self: '_BaseEnumerable[T]', key_selector: Optional[KeySelector[T, K]] = None) -> 'Enumerable[T]':
        """return distinct elements. preserves order of first appearance."""
        def distinct_data():
            data = self._get_data()
            if key_selector is None:
                # python 3.7+ dicts are ordered, making dict.fromkeys a highly efficient, order-preserving unique filter.
                optimized = self._try_numpy_optimization(data, 'distinct')
                if optimized is not None: return optimized
                return list(dict.fromkeys(data))
            else:
                seen = set()
                # the walrus operator (:=) in a list comprehension is concise and efficient
                # 'and not seen.add(key)' is a trick to perform the side-effect within the expression
                return [item for item in data if (key := key_selector(item)) not in seen and not seen.add(key)]
        return Enumerable(distinct_data)

    def union(self: '_BaseEnumerable[T]', other: Iterable[T]) -> 'Enumerable[T]':
        """return the order-preserving union of two sequences (distinct elements)."""
        # chain is a fast iterator that avoids creating an intermediate concatenated list
        # dict.fromkeys handles the order-preserving unique filtering.
        return Enumerable(lambda: list(dict.fromkeys(chain(self._get_data(), other))))

    def intersect(self: '_BaseEnumerable[T]', other: Iterable[T]) -> 'Enumerable[T]':
        """return the order-preserving intersection of two sequences."""
        def intersect_data():
            # building a set from the second iterable provides o(1) average time complexity for lookups
            other_set = set(other)
            # this preserves the order from the first (self) sequence
            return [x for x in self._get_data() if x in other_set]
        return Enumerable(intersect_data)

    def except_(self: '_BaseEnumerable[T]', other: Iterable[T]) -> 'Enumerable[T]':
        """return elements from the first sequence not in the second (set difference)."""
        def except_data():
            other_set = set(other)
            return [x for x in self._get_data() if x not in other_set]
        return Enumerable(except_data)

    def symmetric_difference(self: '_BaseEnumerable[T]', other: Iterable[T]) -> 'Enumerable[T]':
        """return elements that are in one sequence or the other, but not both."""
        def symmetric_difference_data():
            self_data = self._get_data()
            other_data = list(other)
            # sets provide the fastest way to compute the symmetric difference
            self_set = set(self_data)
            other_set = set(other_data)
            diff_set = self_set.symmetric_difference(other_set)
            # iterate through the original sequences to preserve order of first appearance
            seen = set()
            return [item for item in chain(self_data, other_data)
                    if item in diff_set and item not in seen and not seen.add(item)]
        return Enumerable(symmetric_difference_data)

    def concat(self: '_BaseEnumerable[T]', other: Iterable[T]) -> 'Enumerable[T]':
        """concatenate with another sequence, preserving all elements and order."""
        return Enumerable(lambda: self._get_data() + list(other))

    # --- multiset operations ---

    def multiset_intersect(self: '_BaseEnumerable[T]', other: Iterable[T]) -> 'Enumerable[T]':
        """
        returns the intersection of two multisets (bags), respecting element counts.
        ex: [1, 1, 2, 3] & [1, 2, 2] -> [1, 2]
        """
        from collections import Counter
        def multiset_intersect_data():
            # counter is a highly optimized dict subclass for counting hashable objects
            self_counts = Counter(self._get_data())
            other_counts = Counter(other)
            # the '&' operator on counters computes their multiset intersection (min(c1[x], c2[x]))
            intersection_counts = self_counts & other_counts
            # .elements() is an efficient iterator over the items repeating as many times as their count
            return list(intersection_counts.elements())
        return Enumerable(multiset_intersect_data)

    def except_by_count(self: '_BaseEnumerable[T]', other: Iterable[T]) -> 'Enumerable[T]':
        """
        returns the difference of two multisets (bags), respecting element counts.
        ex: [1, 1, 2, 3] - [1, 2, 2] -> [1, 3]
        """
        from collections import Counter
        def except_by_count_data():
            self_counts = Counter(self._get_data())
            other_counts = Counter(other)
            # the '-' operator on counters computes their multiset difference (c1[x] - c2[x]), dropping non-positive
            diff_counts = self_counts - other_counts
            return list(diff_counts.elements())
        return Enumerable(except_by_count_data)

    # --- boolean set checks ---

    def is_subset_of(self: '_BaseEnumerable[T]', other: Iterable[T]) -> bool:
        """determines whether this sequence is a subset of another."""
        # set operations are the fastest way to perform these checks
        return set(self._get_data()).issubset(set(other))

    def is_superset_of(self: '_BaseEnumerable[T]', other: Iterable[T]) -> bool:
        """determines whether this sequence is a superset of another."""
        return set(self._get_data()).issuperset(set(other))

    def is_proper_subset_of(self: '_BaseEnumerable[T]', other: Iterable[T]) -> bool:
        """determines whether this sequence is a proper subset of another (subset, but not equal)."""
        # the '<' operator on sets is the most efficient check for a proper subset
        return set(self._get_data()) < set(other)

    def is_proper_superset_of(self: '_BaseEnumerable[T]', other: Iterable[T]) -> bool:
        """determines whether this sequence is a proper superset of another (superset, but not equal)."""
        # the '>' operator on sets is the most efficient check for a proper superset
        return set(self._get_data()) > set(other)

    def is_disjoint_with(self: '_BaseEnumerable[T]', other: Iterable[T]) -> bool:
        """determines whether this sequence has no elements in common with another."""
        return set(self._get_data()).isdisjoint(set(other))

    # --- similarity metrics ---

    def jaccard_similarity(self: '_BaseEnumerable[T]', other: Iterable[T]) -> float:
        """
        calculates the jaccard similarity coefficient between the two sequences.
        defined as the size of the intersection divided by the size of the union.
        returns a float between 0.0 and 1.0.
        """
        self_set = set(self._get_data())
        other_set = set(other)

        intersection_size = len(self_set.intersection(other_set))
        union_size = len(self_set.union(other_set))

        # handle the edge case of two empty sets, which are perfectly similar
        if union_size == 0:
            return 1.0

        return intersection_size / union_size

# --- join operations ---

class _JoinOperationsMixin(Generic[T]):
    def join(self: '_BaseEnumerable[T]', inner: Iterable[U], outer_key_selector: KeySelector[T, K],
             inner_key_selector: KeySelector[U, K],
             result_selector: Callable[[T, U], V]) -> 'Enumerable[V]':
        """inner join two sequences based on matching keys"""
        def join_data():
            inner_lookup = defaultdict(list)
            for inner_item in inner:
                inner_lookup[inner_key_selector(inner_item)].append(inner_item)
            result = []
            for outer_item in self._get_data():
                outer_key = outer_key_selector(outer_item)
                if outer_key in inner_lookup:
                    for inner_item in inner_lookup[outer_key]:
                        result.append(result_selector(outer_item, inner_item))
            return result
        return Enumerable(join_data)

    def left_join(self: '_BaseEnumerable[T]', inner: Iterable[U], outer_key_selector: KeySelector[T, K],
                  inner_key_selector: KeySelector[U, K],
                  result_selector: Callable[[T, Optional[U]], V],
                  default_inner: Optional[U] = None) -> 'Enumerable[V]':
        """left outer join - includes all outer elements even without matches"""
        def left_join_data():
            inner_lookup = defaultdict(list)
            for inner_item in inner:
                inner_lookup[inner_key_selector(inner_item)].append(inner_item)
            result = []
            for outer_item in self._get_data():
                outer_key = outer_key_selector(outer_item)
                matched_inners = inner_lookup.get(outer_key)
                if matched_inners:
                    for inner_item in matched_inners:
                        result.append(result_selector(outer_item, inner_item))
                else:
                    result.append(result_selector(outer_item, default_inner))
            return result
        return Enumerable(left_join_data)

    def right_join(self: '_BaseEnumerable[T]', inner: Iterable[U], outer_key_selector: KeySelector[T, K],
                   inner_key_selector: KeySelector[U, K],
                   result_selector: Callable[[Optional[T], U], V],
                   default_outer: Optional[T] = None) -> 'Enumerable[V]':
        """right outer join - includes all inner elements even without matches"""
        def right_join_data():
            outer_lookup = defaultdict(list)
            for outer_item in self._get_data():
                outer_lookup[outer_key_selector(outer_item)].append(outer_item)
            result = []
            inner_list = list(inner)
            for inner_item in inner_list:
                inner_key = inner_key_selector(inner_item)
                if inner_key in outer_lookup:
                    for outer_item in outer_lookup[inner_key]:
                        result.append(result_selector(outer_item, inner_item))
                else:
                    result.append(result_selector(default_outer, inner_item))
            return result
        return Enumerable(right_join_data)

    def full_outer_join(self: '_BaseEnumerable[T]', inner: Iterable[U], outer_key_selector: KeySelector[T, K],
                        inner_key_selector: KeySelector[U, K],
                        result_selector: Callable[[Optional[T], Optional[U]], V],
                        default_outer: Optional[T] = None,
                        default_inner: Optional[U] = None) -> 'Enumerable[V]':
        """full outer join - includes all elements from both sequences"""
        def full_join_data():
            outer_lookup = defaultdict(list)
            for outer_item in self._get_data():
                outer_lookup[outer_key_selector(outer_item)].append(outer_item)
            inner_lookup = defaultdict(list)
            for inner_item in inner:
                inner_lookup[inner_key_selector(inner_item)].append(inner_item)
            all_keys = set(outer_lookup.keys()) | set(inner_lookup.keys())
            result = []
            for key in all_keys:
                outer_items = outer_lookup.get(key, [default_outer])
                inner_items = inner_lookup.get(key, [default_inner])
                for outer_item in outer_items:
                    for inner_item in inner_items:
                        result.append(result_selector(outer_item, inner_item))
            return result
        return Enumerable(full_join_data)

    def group_join(self: '_BaseEnumerable[T]', inner: Iterable[U], outer_key_selector: KeySelector[T, K],
                   inner_key_selector: KeySelector[U, K],
                   result_selector: Callable[[T, List[U]], V]) -> 'Enumerable[V]':
        """group join - groups inner elements by outer key"""
        def group_join_data():
            inner_lookup = defaultdict(list)
            for inner_item in inner:
                inner_lookup[inner_key_selector(inner_item)].append(inner_item)
            return [result_selector(o, inner_lookup.get(outer_key_selector(o), [])) for o in self._get_data()]
        return Enumerable(group_join_data)

    def cross_join(self: '_BaseEnumerable[T]', inner: Iterable[U]) -> 'Enumerable[Tuple[T, U]]':
        """cartesian product of two sequences"""
        def cross_join_data():
            inner_data = list(inner)
            return [(outer_item, inner_item) for outer_item in self._get_data() for inner_item in inner_data]
        return Enumerable(cross_join_data)

    def zip_with(self: '_BaseEnumerable[T]', other: Iterable[U], result_selector: Callable[[T, U], V]) -> 'Enumerable[V]':
        """zip two sequences with custom result selector"""
        return Enumerable(lambda: [result_selector(t, u) for t, u in zip(self._get_data(), other)])

    def zip_longest_with(self: '_BaseEnumerable[T]', other: Iterable[U],
                         result_selector: Callable[[Optional[T], Optional[U]], V],
                         default_self: Optional[T] = None, default_other: Optional[U] = None) -> 'Enumerable[V]':
        """zip sequences padding shorter with defaults"""
        def manual_zip_longest():
            self_data, other_data = self._get_data(), list(other)
            max_len = max(len(self_data), len(other_data))
            result = []
            for i in range(max_len):
                s_item = self_data[i] if i < len(self_data) else default_self
                o_item = other_data[i] if i < len(other_data) else default_other
                result.append(result_selector(s_item, o_item))
            return result
        return Enumerable(manual_zip_longest)

# --- grouping/windowing operations ---

class _GroupingAndWindowingMixin(Generic[T]):
    def group_by(self: '_BaseEnumerable[T]', key_selector: KeySelector[T, K]) -> Dict[K, List[T]]:
        """group elements by a key"""
        groups = defaultdict(list)
        for item in self._get_data():
            groups[key_selector(item)].append(item)
        return dict(groups)

    def group_by_multiple(self: '_BaseEnumerable[T]', *key_selectors: KeySelector[T, Any]) -> Dict[Tuple, List[T]]:
        """group by multiple keys returning composite key tuples"""
        groups = defaultdict(list)
        for item in self._get_data():
            groups[tuple(selector(item) for selector in key_selectors)].append(item)
        return dict(groups)

    def group_by_with_aggregate(self: '_BaseEnumerable[T]', key_selector: KeySelector[T, K],
                                element_selector: Selector[T, U],
                                result_selector: Callable[[K, List[U]], V]) -> Dict[K, V]:
        """group by key then transform each group"""
        groups = defaultdict(list)
        for item in self._get_data():
            groups[key_selector(item)].append(element_selector(item))
        return {key: result_selector(key, elements) for key, elements in groups.items()}

    def partition(self: '_BaseEnumerable[T]', predicate: Predicate[T]) -> Tuple[List[T], List[T]]:
        """partition elements based on predicate"""
        true_items, false_items = [], []
        for item in self._get_data():
            (true_items if predicate(item) else false_items).append(item)
        return true_items, false_items

    def chunk(self: '_BaseEnumerable[T]', size: int) -> 'Enumerable[List[T]]':
        """split into chunks of specified size"""
        return Enumerable(lambda: [self._get_data()[i:i + size] for i in range(0, len(self._get_data()), size)])

    def window(self: '_BaseEnumerable[T]', size: int) -> 'Enumerable[List[T]]':
        """create sliding windows of specified size"""
        def window_data():
            data = self._get_data()
            if len(data) < size: return []
            return [data[i:i + size] for i in range(len(data) - size + 1)]
        return Enumerable(window_data)

    def pairwise(self: '_BaseEnumerable[T]') -> 'Enumerable[Tuple[T, T]]':
        """return consecutive pairs"""
        def pairwise_data():
            data = self._get_data()
            if len(data) < 2: return []
            return list(zip(data, data[1:]))
        return Enumerable(pairwise_data)

    def batch_by(self: '_BaseEnumerable[T]', key_selector: KeySelector[T, K]) -> 'Enumerable[List[T]]':
        """batch consecutive elements with same key"""
        def batch_data():
            data = self._get_data()
            if not data: return []
            result = []
            it = iter(data)
            first_item = next(it)
            current_batch, current_key = [first_item], key_selector(first_item)
            for item in it:
                item_key = key_selector(item)
                if item_key == current_key:
                    current_batch.append(item)
                else:
                    result.append(current_batch)
                    current_batch, current_key = [item], item_key
            result.append(current_batch)
            return result
        return Enumerable(batch_data)

# --- stats/math operations ---

class _StatisticalOperationsMixin(Generic[T]):
    def _get_values(self: '_BaseEnumerable[T]', selector: Optional[Selector[T, Union[int, float]]] = None) -> List[Union[int, float]]:
        """helper to extract numeric values for statistical operations."""
        if selector: return self.select(selector).to_list()
        data = self.to_list()
        if data and not all(isinstance(x, (int, float)) for x in data):
            raise TypeError("sequence contains non-numeric types for statistical operation.")
        return data

    def _calculate_stats_welford(self: '_BaseEnumerable[T]', selector: Optional[Selector[T, Union[int, float]]]) -> Tuple[int, float, float]:
        """calculates count, mean, and variance in a single pass. returns (count, mean, variance)."""
        count, mean, m2 = 0, 0.0, 0.0
        for x in self._get_values(selector):
            count += 1
            delta = x - mean
            mean += delta / count
            delta2 = x - mean
            m2 += delta * delta2
        # using population variance, as is common with numpy/pandas std dev default (ddof=0)
        variance = m2 / count if count > 0 else 0.0
        return count, mean, variance

    def sum(self: '_BaseEnumerable[T]', selector: Optional[Selector[T, Union[int, float]]] = None) -> Union[int, float]:
        """calc sum"""
        values = self._get_values(selector)
        try: return np.sum(values)
        except (TypeError, ValueError): return sum(values)

    def average(self: '_BaseEnumerable[T]', selector: Optional[Selector[T, Union[int, float]]] = None) -> float:
        """calc average"""
        count, mean, _ = self._calculate_stats_welford(selector)
        if count == 0: raise ValueError("cannot calculate average of empty sequence")
        return mean

    def std_dev(self: '_BaseEnumerable[T]', selector: Optional[Selector[T, Union[int, float]]] = None) -> float:
        """calculate standard deviation"""
        count, _, variance = self._calculate_stats_welford(selector)
        if count == 0: raise ValueError("cannot calculate standard deviation of empty sequence")
        return math.sqrt(variance)

    def min(self: '_BaseEnumerable[T]', selector: Optional[Selector[T, Any]] = None) -> T:
        """find minimum"""
        data = self._get_data()
        if not data: raise ValueError("cannot find minimum of empty sequence")
        return min(data, key=selector) if selector else min(data)

    def max(self: '_BaseEnumerable[T]', selector: Optional[Selector[T, Any]] = None) -> T:
        """find maximum"""
        data = self._get_data()
        if not data: raise ValueError("cannot find maximum of empty sequence")
        return max(data, key=selector) if selector else max(data)

    def median(self: '_BaseEnumerable[T]', selector: Optional[Selector[T, Union[int, float]]] = None) -> float:
        """calculate median value"""
        sorted_values = sorted(self._get_values(selector))
        n = len(sorted_values)
        if n == 0: raise ValueError("cannot calculate median of empty sequence")
        mid = n // 2
        return (sorted_values[mid] + sorted_values[mid - 1]) / 2 if n % 2 == 0 else sorted_values[mid]

    def percentile(self: '_BaseEnumerable[T]', q: float, selector: Optional[Selector[T, Union[int, float]]] = None) -> float:
        """calculate percentile (0 <= q <= 100)"""
        sorted_vals = sorted(self._get_values(selector))
        if not sorted_vals: raise ValueError("cannot calculate percentile of empty sequence")
        if not 0 <= q <= 100: raise ValueError("percentile must be between 0 and 100")
        k = (len(sorted_vals) - 1) * (q / 100.0)
        f, c = math.floor(k), math.ceil(k)
        if f == c: return sorted_vals[int(k)]
        return sorted_vals[int(f)] * (c - k) + sorted_vals[int(c)] * (k - f)

    def mode(self: '_BaseEnumerable[T]', selector: Optional[Selector[T, K]] = None) -> K:
        """find most frequent element"""
        data = self.select(selector).to_list() if selector else self.to_list()
        if not data: raise ValueError("cannot find mode of empty sequence")
        return max(set(data), key=data.count)

# --- ts stats operations ---

class _TimeSeriesAndAdvancedStatsMixin(Generic[T]):
    def rolling_window(self: '_BaseEnumerable[T]', window_size: int, aggregator: Callable[[List[T]], U]) -> 'Enumerable[U]':
        """apply aggregator to rolling windows"""
        return self.window(window_size).select(aggregator)

    def rolling_sum(self: '_BaseEnumerable[T]', window_size: int,
                    selector: Optional[Selector[T, Union[int, float]]] = None) -> 'Enumerable[Union[int, float]]':
        """rolling sum using existing operations"""
        return self.rolling_window(window_size, lambda w: from_iterable(w).sum(selector))

    def rolling_average(self: '_BaseEnumerable[T]', window_size: int,
                        selector: Optional[Selector[T, Union[int, float]]] = None) -> 'Enumerable[float]':
        """rolling average using existing operations"""
        return self.rolling_window(window_size, lambda w: from_iterable(w).average(selector))

    def lag(self: '_BaseEnumerable[T]', periods: int = 1, fill_value: Optional[T] = None) -> 'Enumerable[Optional[T]]':
        """shift elements by periods"""
        def lag_data():
            data = self._get_data()
            if periods <= 0: return data
            return ([fill_value] * periods) + data[:-periods]
        return Enumerable(lag_data)

    def lead(self: '_BaseEnumerable[T]', periods: int = 1, fill_value: Optional[T] = None) -> 'Enumerable[Optional[T]]':
        """shift elements forward"""
        def lead_data():
            data = self._get_data()
            if periods <= 0: return data
            return data[periods:] + ([fill_value] * periods)
        return Enumerable(lead_data)

    def diff(self: '_BaseEnumerable[T]', periods: int = 1) -> 'Enumerable[T]':
        """difference with previous elements"""
        lagged = self.lag(periods, fill_value=None)
        # zip automatically stops at the shorter iterable, correctly handling the length change
        return self.zip_with(lagged, lambda current, prev: current - prev if prev is not None and current is not None else None).skip(periods)

    def scan(self: '_BaseEnumerable[T]', accumulator: Accumulator[U, T], seed: U) -> 'Enumerable[U]':
        """produce intermediate accumulation values, including seed"""
        return Enumerable(lambda: list(accumulate(self._get_data(), accumulator, initial=seed)))

    def cumulative_sum(self: '_BaseEnumerable[T]', selector: Optional[Selector[T, Union[int, float]]] = None) -> 'Enumerable[Union[int, float]]':
        """cumulative sum"""
        values = self._get_values(selector)
        return Enumerable(lambda: list(accumulate(values, lambda acc, x: acc + x)))

    def cumulative_product(self: '_BaseEnumerable[T]', selector: Optional[Selector[T, Union[int, float]]] = None) -> 'Enumerable[Union[int, float]]':
        """cumulative product"""
        values = self._get_values(selector)
        return Enumerable(lambda: list(accumulate(values, lambda acc, x: acc * x)))

    def cumulative_max(self: '_BaseEnumerable[T]', selector: Optional[Selector[T, K]] = None) -> 'Enumerable[K]':
        """cumulative maximum"""
        values = self.select(selector).to_list() if selector else self.to_list()
        return Enumerable(lambda: list(accumulate(values, max)))

    def cumulative_min(self: '_BaseEnumerable[T]', selector: Optional[Selector[T, K]] = None) -> 'Enumerable[K]':
        """cumulative minimum"""
        values = self.select(selector).to_list() if selector else self.to_list()
        return Enumerable(lambda: list(accumulate(values, min)))

    def rank(self: '_BaseEnumerable[T]', selector: Optional[Selector[T, K]] = None, ascending: bool = True) -> 'Enumerable[int]':
        """rank elements (1-based) using an efficient o(n log n) algorithm."""
        def rank_data():
            data = self._get_data()
            values = [(selector(item) if selector else item) for item in data]
            indexed = sorted(enumerate(values), key=lambda p: p[1], reverse=not ascending)
            ranks = [0] * len(data)
            for rank, (original_index, _) in enumerate(indexed):
                ranks[original_index] = rank + 1
            return ranks
        return Enumerable(rank_data)

    def dense_rank(self: '_BaseEnumerable[T]', selector: Optional[Selector[T, K]] = None, ascending: bool = True) -> 'Enumerable[int]':
        """dense rank elements (1-based) using an efficient o(n log n) algorithm."""
        def dense_rank_data():
            values = self.select(selector) if selector else self
            unique_sorted = values.distinct().order_by(lambda x: x, descending=not ascending).to_list()
            rank_map = {val: i + 1 for i, val in enumerate(unique_sorted)}
            return [rank_map[selector(item) if selector else item] for item in self._get_data()]
        return Enumerable(dense_rank_data)

    def quantile_cut(self: '_BaseEnumerable[T]', q: int, selector: Optional[Selector[T, Union[int, float]]] = None) -> 'Enumerable[int]':
        """cut into q quantile bins using an efficient algorithm."""
        if q <= 0: raise ValueError("number of quantiles (q) must be positive")
        def cut_data():
            values_en = from_iterable(self._get_values(selector))
            quantiles = [values_en.percentile(i * 100 / q) for i in range(1, q)]
            return [bisect_right(quantiles, val) for val in values_en]
        return Enumerable(cut_data)

    def normalize(self: '_BaseEnumerable[T]', selector: Optional[Selector[T, float]] = None) -> 'Enumerable[float]':
        """min-max normalization"""
        values = from_iterable(self._get_values(selector))
        min_val, max_val = values.min(), values.max()
        if max_val == min_val: return values.select(lambda _: 0.0)
        return values.select(lambda x: (x - min_val) / (max_val - min_val))

    def standardize(self: '_BaseEnumerable[T]', selector: Optional[Selector[T, float]] = None) -> 'Enumerable[float]':
        """z-score standardization"""
        values = from_iterable(self._get_values(selector))
        mean_val, std_val = values.average(), values.std_dev()
        if std_val == 0: return values.select(lambda _: 0.0)
        return values.select(lambda x: (x - mean_val) / std_val)

    def outliers_iqr(self: '_BaseEnumerable[T]', selector: Optional[Selector[T, float]] = None, factor: float = 1.5) -> 'Enumerable[T]':
        """detect outliers using iqr method"""
        values = from_iterable(self._get_values(selector))
        q1, q3 = values.percentile(25), values.percentile(75)
        iqr = q3 - q1
        lower_bound, upper_bound = q1 - factor * iqr, q3 + factor * iqr
        sel = selector if selector else lambda item: item
        return self.where(lambda item: not (lower_bound <= sel(item) <= upper_bound))

# --- combinatorics ---

class _CombinatorialAndAdvancedMixin(Generic[T]):
    def binomial_coefficient(self: '_BaseEnumerable[T]', r: int) -> int:
        """binomial coefficient n choose r, using python's optimized math.comb"""
        n = self.count()
        return math.comb(n, r)

    def permutations(self: '_BaseEnumerable[T]', r: Optional[int] = None) -> 'Enumerable[Tuple[T, ...]]':
        """generate permutations using the efficient itertools implementation."""
        r = r or self.count()
        return from_iterable(itertools_permutations(self.to_list(), r))

    def combinations(self: '_BaseEnumerable[T]', r: int) -> 'Enumerable[Tuple[T, ...]]':
        """generate combinations using the efficient itertools implementation."""
        return from_iterable(itertools_combinations(self.to_list(), r))

    def combinations_with_replacement(self: '_BaseEnumerable[T]', r: int) -> 'Enumerable[Tuple[T, ...]]':
        """
        generate combinations with replacement using the efficient itertools implementation.
        elements are treated as if they were replaced after each pick.
        """
        return from_iterable(itertools_combinations_with_replacement(self.to_list(), r))

    def power_set(self: '_BaseEnumerable[T]') -> 'Enumerable[Tuple[T, ...]]':
        """
        generates the power set (the set of all subsets) of the sequence.
        this is implemented functionally by chaining combinations of every possible length (from 0 to n).
        """

        def power_set_data():
            data = self.to_list()
            # chain combinations of length 0, 1, 2, ..., n
            return chain.from_iterable(itertools_combinations(data, r) for r in range(len(data) + 1))

        return Enumerable(lambda: list(power_set_data()))

    def cartesian_product(self: '_BaseEnumerable[T]', *others: Iterable[Any]) -> 'Enumerable[Tuple[Any, ...]]':
        """
        computes the cartesian product with other iterables using the efficient itertools.product.
        returns an enumerable of tuples.
        """

        def product_data():
            # gathers all iterables, including self, and unpacks them into itertools.product
            all_iterables = [self._get_data()] + [list(o) for o in others]
            return list(itertools_product(*all_iterables))

        return Enumerable(product_data)

    def run_length_encode(self: '_BaseEnumerable[T]') -> 'Enumerable[Tuple[T, int]]':
        """
        performs run-length encoding on the sequence. consecutive identical elements
        are grouped into (element, count) tuples.
        """

        def rle_data():
            data = self._get_data()
            # itertools.groupby is perfect for this, grouping consecutive identical items
            return [(key, len(list(group))) for key, group in itertools_groupby(data)]

        return Enumerable(rle_data)

    def flatten(self: '_BaseEnumerable[T]', depth: int = 1) -> 'Enumerable[Any]':
        """flatten nested sequences to a specified depth"""
        if depth <= 0: return self
        current = self
        for _ in range(depth):
            current = current.select_many(
                lambda x: x if isinstance(x, Iterable) and not isinstance(x, (str, bytes)) else [x])
        return current

    def transpose(self: '_BaseEnumerable[T]') -> 'Enumerable[List[Any]]':
        """transpose a matrix-like structure (list of lists) using the efficient itertools implementation."""
        return Enumerable(lambda: [list(col) for col in zip_longest(*self._get_data())])

    def unzip(self: '_BaseEnumerable[T]') -> Tuple['Enumerable[Any]', ...]:
        """
        the inverse of zip. transforms an enumerable of tuples/lists into a tuple of enumerables.
        e.g., [(a, 1), (b, 2)] -> ( (a, b), (1, 2) )
        """
        data = self.to_list()
        if not data:
            return tuple()
        # the pythonic way to unzip is with zip(*)
        unzipped_cols = zip(*data)
        return tuple(from_iterable(col) for col in unzipped_cols)

    def intersperse(self: '_BaseEnumerable[T]', separator: T) -> 'Enumerable[T]':
        """intersperse separator between elements"""
        data = self.to_list()
        if len(data) < 2: return from_iterable(data)
        it = iter(data)
        interspersed = chain.from_iterable(zip(it, repeat(separator, len(data) - 1)))
        return from_iterable(interspersed).concat(list(it))

    def sample(self: '_BaseEnumerable[T]', n: int, replace: bool = False,
               random_state: Optional[int] = None) -> 'Enumerable[T]':
        """random sampling"""

        def sample_data():
            if random_state: random.seed(random_state)
            data = self._get_data()
            if replace: return random.choices(data, k=n)
            return random.sample(data, min(n, len(data)))

        return Enumerable(sample_data)

    def stratified_sample(self: '_BaseEnumerable[T]', key_selector: KeySelector[T, K],
                          samples_per_group: int) -> 'Enumerable[T]':
        """stratified sampling"""

        def stratified_data():
            groups = self.group_by(key_selector)
            result = []
            for group_items in groups.values():
                result.extend(from_iterable(group_items).sample(min(samples_per_group, len(group_items))).to_list())
            return result

        return Enumerable(stratified_data)

    def bootstrap_sample(self: '_BaseEnumerable[T]', n_samples: int = 1000,
                         sample_size: Optional[int] = None) -> 'Enumerable[Enumerable[T]]':
        """bootstrap sampling"""
        size = sample_size or self.count()
        return from_range(0, n_samples).select(lambda _: self.sample(size, replace=True))

    def memoize(self: '_BaseEnumerable[T]') -> 'Enumerable[T]':
        """
        evaluates the enumerable chain and caches the result. subsequent operations on the
        returned enumerable will start from this cached state, preventing re-computation
        of the preceding chain.
        """
        # calling _get_data() triggers the full computation and caching
        cached_data = self._get_data()
        # return a new enumerable that starts with the already-computed data
        return Enumerable(lambda: cached_data)

    def pipe(self: '_BaseEnumerable[T]', func: Callable[..., U], *args, **kwargs) -> U:
        """
        pipes the enumerable object into an external function. enables custom, chainable operations.
        example: .pipe(my_custom_plot_function, title='my data')
        """
        return func(self, *args, **kwargs)

    def side_effect(self: '_BaseEnumerable[T]', action: Callable[[T], Any]) -> 'Enumerable[T]':
        """
        performs a side-effect action for each element in the sequence without modifying it.
        primarily used for debugging, e.g., .side_effect(print).
        """

        def side_effect_data():
            data = self._get_data()
            for item in data:
                action(item)
            return data

        return Enumerable(side_effect_data)

    def topological_sort(self: '_BaseEnumerable[T]',
                         dependency_selector: Callable[[T], Iterable[T]]) -> 'Enumerable[T]':
        """
        performs a topological sort on the sequence, treated as nodes in a directed acyclic graph (dag).
        the dependency_selector function must return an iterable of nodes that a given node depends on.
        raises a valueerror if a cycle is detected.
        """

        def sort_data():
            data = self.to_list()
            data_set = set(data)

            # kahn's algorithm for topological sorting
            in_degree = {u: 0 for u in data}
            # adjacency list: u -> v means u must come before v
            adj = {u: [] for u in data}

            for u in data:
                for v in dependency_selector(u):
                    if v not in data_set:
                        # ignoring dependencies outside the current enumerable set
                        continue
                    adj[v].append(u)
                    in_degree[u] += 1

            # queue for nodes with no incoming edges
            queue = deque([u for u in data if in_degree[u] == 0])
            sorted_list = []

            while queue:
                u = queue.popleft()
                sorted_list.append(u)

                for v in adj[u]:
                    in_degree[v] -= 1
                    if in_degree[v] == 0:
                        queue.append(v)

            if len(sorted_list) != len(data):
                raise ValueError("a cycle was detected in the dependency graph.")

            return sorted_list

        return Enumerable(sort_data)

# --- terminal operations ---

class _TerminalOperationsMixin(Generic[T]):
    def to_list(self: '_BaseEnumerable[T]') -> List[T]:
        """convert to list"""
        return self._get_data()

    def to_array(self: '_BaseEnumerable[T]') -> np.ndarray:
        """convert to numpy array"""
        return np.array(self._get_data())

    def to_set(self: '_BaseEnumerable[T]') -> Set[T]:
        """convert to set"""
        return set(self._get_data())

    def to_dict(self: '_BaseEnumerable[T]', key_selector: KeySelector[T, K],
                value_selector: Optional[Selector[T, V]] = None) -> Dict[K, V]:
        """convert to dictionary"""
        val_sel = value_selector if value_selector else lambda item: item
        return {key_selector(item): val_sel(item) for item in self._get_data()}

    def to_pandas(self: '_BaseEnumerable[T]') -> pd.Series:
        """convert to pandas series"""
        return pd.Series(self._get_data())

    def to_df(self: '_BaseEnumerable[T]') -> pd.DataFrame:
        """convert to pandas dataframe"""
        return pd.DataFrame(self._get_data())

    def count(self: '_BaseEnumerable[T]', predicate: Optional[Predicate[T]] = None) -> int:
        """count elements"""
        if predicate is None: return len(self._get_data())
        return sum(1 for x in self._get_data() if predicate(x))

    def any(self: '_BaseEnumerable[T]', predicate: Optional[Predicate[T]] = None) -> bool:
        """check if any element satisfies condition"""
        data = self._get_data()
        if predicate is None: return len(data) > 0
        return any(predicate(x) for x in data)

    def all(self: '_BaseEnumerable[T]', predicate: Predicate[T]) -> bool:
        """check if all elements satisfy condition"""
        return all(predicate(x) for x in self._get_data())

    def first(self: '_BaseEnumerable[T]', predicate: Optional[Predicate[T]] = None) -> T:
        """get first element"""
        data = self._get_data()
        if predicate is None:
            if not data: raise ValueError("sequence contains no elements")
            return data[0]
        for item in data:
            if predicate(item): return item
        raise ValueError("no element satisfies the condition")

    def first_or_default(self: '_BaseEnumerable[T]', predicate: Optional[Predicate[T]] = None,
                         default: Optional[T] = None) -> Optional[T]:
        """get first element or default"""
        try: return self.first(predicate)
        except ValueError: return default

    def single(self: '_BaseEnumerable[T]', predicate: Optional[Predicate[T]] = None) -> T:
        """get single element, erroring if not exactly one"""
        data = [x for x in self._get_data() if predicate(x)] if predicate else self._get_data()
        if len(data) == 0: raise ValueError("sequence contains no matching elements")
        if len(data) > 1: raise ValueError("sequence contains more than one matching element")
        return data[0]

    def aggregate(self: '_BaseEnumerable[T]', accumulator: Accumulator[T, T], seed: Optional[T] = None) -> T:
        """applies accumulator function over sequence"""
        data = self._get_data()
        if not data and seed is None: raise ValueError("cannot aggregate empty sequence without seed")
        return reduce(accumulator, data, seed) if seed is not None else reduce(accumulator, data)

    def aggregate_with_selector(self: '_BaseEnumerable[T]', seed: U, accumulator: Accumulator[U, T],
                                result_selector: Selector[U, V]) -> V:
        """aggregate with seed and final transformation"""
        return result_selector(reduce(accumulator, self._get_data(), seed))

# --- main enumerable class ---

class Enumerable(
    _BaseEnumerable[T],
    _CoreOperationsMixin[T],
    _SetOperationsMixin[T],
    _JoinOperationsMixin[T],
    _GroupingAndWindowingMixin[T],
    _StatisticalOperationsMixin[T],
    _TimeSeriesAndAdvancedStatsMixin[T],
    _CombinatorialAndAdvancedMixin[T],
    _TerminalOperationsMixin[T]
):
    """a powerful, linq-inspired enumerable class for python iterables."""
    pass

# --- ordered enumerable class ---

class OrderedEnumerable(Enumerable[T]):
    """represents a sorted sequence, allowing for subsequent orderings."""

    def __init__(self, data_func: Callable[[], List[T]], sort_keys: List[Tuple[Callable, bool]]):
        self._original_data_func = data_func
        self._sort_keys = sort_keys
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

# --- factory functions ---

def from_iterable(data: Iterable[T]) -> Enumerable[T]:
    """create enumerable from iterable"""
    return Enumerable(lambda: list(data))

def from_range(start: int, count: int) -> Enumerable[int]:
    """create enumerable from range"""
    return Enumerable(lambda: list(range(start, start + count)))

def repeat(item: T, count: int) -> Enumerable[T]:
    """create enumerable with repeated item"""
    return Enumerable(lambda: [item] * count)

def empty() -> Enumerable[Any]:
    """create empty enumerable"""
    return Enumerable(lambda: [])

def generate(generator_func: Callable[[], T], count: int) -> Enumerable[T]:
    """generate sequence using a function"""
    return Enumerable(lambda: [generator_func() for _ in range(count)])

# --- aliases ---
pinqy = from_iterable
P = from_iterable