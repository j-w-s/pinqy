from __future__ import annotations
import typing
from collections import defaultdict
from itertools import batched
from ..types import *

if typing.TYPE_CHECKING:
    from ..enumerable import Enumerable

class GroupingAccessor(Generic[T]):
    def __init__(self, enumerable_instance: 'Enumerable[T]'):
        self._enumerable = enumerable_instance

    def group_by(self, key_selector: KeySelector[T, K]) -> Dict[K, List[T]]:
        """group elements by a key"""
        groups = defaultdict(list)
        for item in self._enumerable._get_data():
            groups[key_selector(item)].append(item)
        return dict(groups)

    def group_by_multiple(self, *key_selectors: KeySelector[T, Any]) -> Dict[Tuple, List[T]]:
        """group by multiple keys returning composite key tuples"""
        groups = defaultdict(list)
        for item in self._enumerable._get_data():
            groups[tuple(selector(item) for selector in key_selectors)].append(item)
        return dict(groups)

    def group_by_with_aggregate(self, key_selector: KeySelector[T, K],
                                element_selector: Selector[T, U],
                                result_selector: Callable[[K, List[U]], V]) -> Dict[K, V]:
        """group by key then transform each group"""
        groups = defaultdict(list)
        for item in self._enumerable._get_data():
            groups[key_selector(item)].append(element_selector(item))
        return {key: result_selector(key, elements) for key, elements in groups.items()}

    def group_by_nested(self, key_selector: KeySelector[T, K], sub_key_selector: KeySelector[T, V]) -> Dict[
        K, Dict[V, List[T]]]:
        """create nested groupings with primary and secondary keys"""
        from ..factories import from_iterable

        primary_groups = self.group_by(key_selector)
        nested_result = {}

        for primary_key, items in primary_groups.items():
            secondary_groups = from_iterable(items).group.group_by(sub_key_selector)
            nested_result[primary_key] = secondary_groups

        return nested_result

    def partition(self, predicate: Predicate[T]) -> Tuple[List[T], List[T]]:
        """partition elements based on predicate"""
        true_items, false_items = [], []
        for item in self._enumerable._get_data():
            (true_items if predicate(item) else false_items).append(item)
        return true_items, false_items

    def chunk(self, size: int) -> 'Enumerable[List[T]]':
        """split into chunks of specified size"""
        from ..enumerable import Enumerable
        return Enumerable(lambda: [self._enumerable._get_data()[i:i + size] for i in range(0, len(self._enumerable._get_data()), size)])

    def window(self, size: int) -> 'Enumerable[List[T]]':
        """create sliding windows of specified size"""
        from ..enumerable import Enumerable
        def window_data():
            data = self._enumerable._get_data()
            if len(data) < size: return []
            return [data[i:i + size] for i in range(len(data) - size + 1)]
        return Enumerable(window_data)

    def pairwise(self) -> 'Enumerable[Tuple[T, T]]':
        """return consecutive pairs"""
        from ..enumerable import Enumerable
        def pairwise_data():
            data = self._enumerable._get_data()
            if len(data) < 2: return []
            return list(zip(data, data[1:]))
        return Enumerable(pairwise_data)

    def batch_by(self, key_selector: KeySelector[T, K]) -> 'Enumerable[List[T]]':
        """batch consecutive elements with same key"""
        from ..enumerable import Enumerable
        def batch_data():
            data = self._enumerable._get_data()
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

    def batched(self, size: int) -> 'Enumerable[Tuple[T, ...]]':
        """
        batches elements of a sequence into tuples of a specified size. requires python 3.12+.
        the last batch may be smaller than the requested size.
        """
        from ..enumerable import Enumerable
        if size <= 0:
            raise ValueError("batch size must be positive")
        def batched_data():
            # convert the resulting tuples to lists to maintain the list-of-lists convention of chunk()
            return [list(batch) for batch in batched(self._enumerable._get_data(), size)]
        return Enumerable(batched_data)