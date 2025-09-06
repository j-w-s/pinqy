from __future__ import annotations
import typing
from itertools import chain
from ..types import *

if typing.TYPE_CHECKING:
    from ..enumerable import Enumerable

class SetAccessor(Generic[T]):
    """
    provides core and advanced set-theoretic operations.
    this includes standard union, intersection, and difference,
    as well as multiset (bag) operations and similarity metrics.
    """
    def __init__(self, enumerable_instance: 'Enumerable[T]'):
        self._enumerable = enumerable_instance

    def distinct(self, key_selector: Optional[KeySelector[T, K]] = None) -> 'Enumerable[T]':
        """return distinct elements. preserves order of first appearance."""
        from ..enumerable import Enumerable
        def distinct_data():
            data = self._enumerable._get_data()
            if key_selector is None:
                # python 3.7+ dicts are ordered, making dict.fromkeys a highly efficient, order-preserving unique filter.
                optimized = self._enumerable._try_numpy_optimization(data, 'distinct')
                if optimized is not None: return optimized
                return list(dict.fromkeys(data))
            else:
                seen = set()
                # the walrus operator (:=) in a list comprehension is concise and efficient
                # 'and not seen.add(key)' is a trick to perform the side-effect within the expression
                return [item for item in data if (key := key_selector(item)) not in seen and not seen.add(key)]
        return Enumerable(distinct_data)

    def union(self, other: Iterable[T]) -> 'Enumerable[T]':
        """return the order-preserving union of two sequences (distinct elements)."""
        from ..enumerable import Enumerable
        # chain is a fast iterator that avoids creating an intermediate concatenated list
        # dict.fromkeys handles the order-preserving unique filtering.
        return Enumerable(lambda: list(dict.fromkeys(chain(self._enumerable._get_data(), other))))

    def intersect(self, other: Iterable[T]) -> 'Enumerable[T]':
        """return the order-preserving intersection of two sequences."""
        from ..enumerable import Enumerable
        def intersect_data():
            # building a set from the second iterable provides o(1) average time complexity for lookups
            other_set = set(other)
            # this preserves the order from the first (self) sequence
            return [x for x in self._enumerable._get_data() if x in other_set]
        return Enumerable(intersect_data)

    def except_(self, other: Iterable[T]) -> 'Enumerable[T]':
        """return elements from the first sequence not in the second (set difference)."""
        from ..enumerable import Enumerable
        def except_data():
            other_set = set(other)
            return [x for x in self._enumerable._get_data() if x not in other_set]
        return Enumerable(except_data)

    def symmetric_difference(self, other: Iterable[T]) -> 'Enumerable[T]':
        """return elements that are in one sequence or the other, but not both."""
        from ..enumerable import Enumerable
        def symmetric_difference_data():
            self_data = self._enumerable._get_data()
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

    def concat(self, other: Iterable[T]) -> 'Enumerable[T]':
        """concatenate with another sequence, preserving all elements and order."""
        from ..enumerable import Enumerable
        return Enumerable(lambda: self._enumerable._get_data() + list(other))

    # --- multiset operations ---

    def multiset_intersect(self, other: Iterable[T]) -> 'Enumerable[T]':
        """
        returns the intersection of two multisets (bags), respecting element counts.
        ex: [1, 1, 2, 3] & [1, 2, 2] -> [1, 2]
        """
        from collections import Counter
        from ..enumerable import Enumerable
        def multiset_intersect_data():
            # counter is a highly optimized dict subclass for counting hashable objects
            self_counts = Counter(self._enumerable._get_data())
            other_counts = Counter(other)
            # the '&' operator on counters computes their multiset intersection (min(c1[x], c2[x]))
            intersection_counts = self_counts & other_counts
            # .elements() is an efficient iterator over the items repeating as many times as their count
            return list(intersection_counts.elements())
        return Enumerable(multiset_intersect_data)

    def except_by_count(self, other: Iterable[T]) -> 'Enumerable[T]':
        """
        returns the difference of two multisets (bags), respecting element counts.
        ex: [1, 1, 2, 3] - [1, 2, 2] -> [1, 3]
        """
        from collections import Counter
        from ..enumerable import Enumerable
        def except_by_count_data():
            self_counts = Counter(self._enumerable._get_data())
            other_counts = Counter(other)
            # the '-' operator on counters computes their multiset difference (c1[x] - c2[x]), dropping non-positive
            diff_counts = self_counts - other_counts
            return list(diff_counts.elements())
        return Enumerable(except_by_count_data)

    # --- boolean set checks ---

    def is_subset_of(self, other: Iterable[T]) -> bool:
        """determines whether this sequence is a subset of another."""
        # set operations are the fastest way to perform these checks
        return set(self._enumerable._get_data()).issubset(set(other))

    def is_superset_of(self, other: Iterable[T]) -> bool:
        """determines whether this sequence is a superset of another."""
        return set(self._enumerable._get_data()).issuperset(set(other))

    def is_proper_subset_of(self, other: Iterable[T]) -> bool:
        """determines whether this sequence is a proper subset of another (subset, but not equal)."""
        # the '<' operator on sets is the most efficient check for a proper subset
        return set(self._enumerable._get_data()) < set(other)

    def is_proper_superset_of(self, other: Iterable[T]) -> bool:
        """determines whether this sequence is a proper superset of another (superset, but not equal)."""
        # the '>' operator on sets is the most efficient check for a proper superset
        return set(self._enumerable._get_data()) > set(other)

    def is_disjoint_with(self, other: Iterable[T]) -> bool:
        """determines whether this sequence has no elements in common with another."""
        return set(self._enumerable._get_data()).isdisjoint(set(other))

    # --- similarity metrics ---

    def jaccard_similarity(self, other: Iterable[T]) -> float:
        """
        calculates the jaccard similarity coefficient between the two sequences.
        defined as the size of the intersection divided by the size of the union.
        returns a float between 0.0 and 1.0.
        """
        self_set = set(self._enumerable._get_data())
        other_set = set(other)

        intersection_size = len(self_set.intersection(other_set))
        union_size = len(self_set.union(other_set))

        # handle the edge case of two empty sets, which are perfectly similar
        if union_size == 0:
            return 1.0

        return intersection_size / union_size