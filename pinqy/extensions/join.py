from __future__ import annotations
import typing
from collections import defaultdict
from ..types import *

if typing.TYPE_CHECKING:
    from ..enumerable import Enumerable

class JoinAccessor(Generic[T]):
    def __init__(self, enumerable_instance: 'Enumerable[T]'):
        self._enumerable = enumerable_instance

    def join(self, inner: Iterable[U], outer_key_selector: KeySelector[T, K],
             inner_key_selector: KeySelector[U, K],
             result_selector: Callable[[T, U], V]) -> 'Enumerable[V]':
        """inner join two sequences based on matching keys"""
        from ..enumerable import Enumerable
        def join_data():
            inner_lookup = defaultdict(list)
            for inner_item in inner:
                inner_lookup[inner_key_selector(inner_item)].append(inner_item)
            result = []
            for outer_item in self._enumerable._get_data():
                outer_key = outer_key_selector(outer_item)
                if outer_key in inner_lookup:
                    for inner_item in inner_lookup[outer_key]:
                        result.append(result_selector(outer_item, inner_item))
            return result
        return Enumerable(join_data)

    def left_join(self, inner: Iterable[U], outer_key_selector: KeySelector[T, K],
                  inner_key_selector: KeySelector[U, K],
                  result_selector: Callable[[T, Optional[U]], V],
                  default_inner: Optional[U] = None) -> 'Enumerable[V]':
        """left outer join - includes all outer elements even without matches"""
        from ..enumerable import Enumerable
        def left_join_data():
            inner_lookup = defaultdict(list)
            for inner_item in inner:
                inner_lookup[inner_key_selector(inner_item)].append(inner_item)
            result = []
            for outer_item in self._enumerable._get_data():
                outer_key = outer_key_selector(outer_item)
                matched_inners = inner_lookup.get(outer_key)
                if matched_inners:
                    for inner_item in matched_inners:
                        result.append(result_selector(outer_item, inner_item))
                else:
                    result.append(result_selector(outer_item, default_inner))
            return result
        return Enumerable(left_join_data)

    def right_join(self, inner: Iterable[U], outer_key_selector: KeySelector[T, K],
                   inner_key_selector: KeySelector[U, K],
                   result_selector: Callable[[Optional[T], U], V],
                   default_outer: Optional[T] = None) -> 'Enumerable[V]':
        """right outer join - includes all inner elements even without matches"""
        from ..enumerable import Enumerable
        def right_join_data():
            outer_lookup = defaultdict(list)
            for outer_item in self._enumerable._get_data():
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

    def full_outer_join(self, inner: Iterable[U], outer_key_selector: KeySelector[T, K],
                        inner_key_selector: KeySelector[U, K],
                        result_selector: Callable[[Optional[T], Optional[U]], V],
                        default_outer: Optional[T] = None,
                        default_inner: Optional[U] = None) -> 'Enumerable[V]':
        """full outer join - includes all elements from both sequences"""
        from ..enumerable import Enumerable
        def full_join_data():
            outer_lookup = defaultdict(list)
            for outer_item in self._enumerable._get_data():
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

    def group_join(self, inner: Iterable[U], outer_key_selector: KeySelector[T, K],
                   inner_key_selector: KeySelector[U, K],
                   result_selector: Callable[[T, List[U]], V]) -> 'Enumerable[V]':
        """group join - groups inner elements by outer key"""
        from ..enumerable import Enumerable
        def group_join_data():
            inner_lookup = defaultdict(list)
            for inner_item in inner:
                inner_lookup[inner_key_selector(inner_item)].append(inner_item)
            return [result_selector(o, inner_lookup.get(outer_key_selector(o), [])) for o in self._enumerable._get_data()]
        return Enumerable(group_join_data)

    def cross_join(self, inner: Iterable[U]) -> 'Enumerable[Tuple[T, U]]':
        """cartesian product of two sequences"""
        from ..enumerable import Enumerable
        def cross_join_data():
            inner_data = list(inner)
            return [(outer_item, inner_item) for outer_item in self._enumerable._get_data() for inner_item in inner_data]
        return Enumerable(cross_join_data)

    def zip_with(self, other: Iterable[U], result_selector: Callable[[T, U], V]) -> 'Enumerable[V]':
        """zip two sequences with custom result selector"""
        from ..enumerable import Enumerable
        return Enumerable(lambda: [result_selector(t, u) for t, u in zip(self._enumerable._get_data(), other)])

    def zip_longest_with(self, other: Iterable[U],
                         result_selector: Callable[[Optional[T], Optional[U]], V],
                         default_self: Optional[T] = None, default_other: Optional[U] = None) -> 'Enumerable[V]':
        """zip sequences padding shorter with defaults"""
        from ..enumerable import Enumerable
        def manual_zip_longest():
            self_data, other_data = self._enumerable._get_data(), list(other)
            max_len = max(len(self_data), len(other_data))
            result = []
            for i in range(max_len):
                s_item = self_data[i] if i < len(self_data) else default_self
                o_item = other_data[i] if i < len(other_data) else default_other
                result.append(result_selector(s_item, o_item))
            return result
        return Enumerable(manual_zip_longest)