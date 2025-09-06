from __future__ import annotations
import typing
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