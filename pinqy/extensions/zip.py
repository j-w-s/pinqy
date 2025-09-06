from __future__ import annotations
import typing
from itertools import zip_longest
from ..types import *

if typing.TYPE_CHECKING:
    from ..enumerable import Enumerable


class ZipAccessor(Generic[T]):
    def __init__(self, enumerable_instance: 'Enumerable[T]'):
        self._enumerable = enumerable_instance

    def zip_with(self, other: Iterable[U], result_selector: Callable[[T, U], V]) -> 'Enumerable[V]':
        """zip two sequences with custom result selector"""
        from ..enumerable import Enumerable
        return Enumerable(lambda: [result_selector(t, u) for t, u in zip(self._enumerable._get_data(), other)])

    def zip_longest_with(self, other: Iterable[U],
                         result_selector: Callable[[Optional[T], Optional[U]], V],
                         default_self: Optional[T] = None, default_other: Optional[U] = None) -> 'Enumerable[V]':
        """zip sequences padding shorter with defaults"""
        from ..enumerable import Enumerable
        def zip_longest_data():
            # use a sentinel object to distinguish from a fill value of none
            sentinel = object()
            self_data = self._enumerable._get_data()
            other_data = list(other)
            zipped = zip_longest(self_data, other_data, fillvalue=sentinel)

            result = []
            for t, u in zipped:
                s_item = t if t is not sentinel else default_self
                o_item = u if u is not sentinel else default_other
                result.append(result_selector(s_item, o_item))
            return result

        return Enumerable(zip_longest_data)