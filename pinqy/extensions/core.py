from __future__ import annotations
import typing
from itertools import chain, takewhile, dropwhile, batched
from ..types import *

if typing.TYPE_CHECKING:
    from ..enumerable import OrderedEnumerable

class _CoreOperations(Generic[T]):
    def where(self: 'Enumerable[T]', predicate: Predicate[T]) -> 'Enumerable[T]':
        """filter elements based on a predicate"""
        from ..enumerable import Enumerable
        def filter_data():
            data = self._get_data()
            optimized = self._try_numpy_optimization(data, 'where', predicate)
            if optimized is not None: return optimized
            return [x for x in data if predicate(x)]
        # always return a base enumerable
        return Enumerable(filter_data)

    def select(self: 'Enumerable[T]', selector: Selector[T, U]) -> 'Enumerable[U]':
        """project each element to a new form"""
        from ..enumerable import Enumerable
        def map_data():
            data = self._get_data()
            optimized = self._try_numpy_optimization(data, 'select', selector)
            if optimized is not None: return optimized
            return [selector(x) for x in data]
        return Enumerable(map_data)

    def select_many(self: 'Enumerable[T]', selector: Selector[T, Iterable[U]]) -> 'Enumerable[U]':
        """project and flatten sequences"""
        from ..enumerable import Enumerable
        def flat_map_data():
            # a nested list comprehension is often more performant and is functionally equivalent
            return [item for sublist in self.select(selector) for item in sublist]
        return Enumerable(flat_map_data)

    def order_by(self: 'Enumerable[T]', key_selector: Callable[[T], K]) -> 'OrderedEnumerable[T]':
        """sort elements by a key"""
        from ..enumerable import OrderedEnumerable
        return OrderedEnumerable(self._data_func, [(key_selector, False)])

    def order_by_descending(self: 'Enumerable[T]', key_selector: Callable[[T], K]) -> 'OrderedEnumerable[T]':
        """sort elements by a key in descending order"""
        from ..enumerable import OrderedEnumerable
        return OrderedEnumerable(self._data_func, [(key_selector, True)])

    def take(self: 'Enumerable[T]', count: int) -> 'Enumerable[T]':
        """take the first 'count' elements"""
        from ..enumerable import Enumerable
        return Enumerable(lambda: self._get_data()[:count])

    def skip(self: 'Enumerable[T]', count: int) -> 'Enumerable[T]':
        """skip the first 'count' elements"""
        from ..enumerable import Enumerable
        return Enumerable(lambda: self._get_data()[count:])

    def take_while(self: 'Enumerable[T]', predicate: Predicate[T]) -> 'Enumerable[T]':
        """take elements while predicate is true"""
        from ..enumerable import Enumerable
        # itertools.takewhile is the most efficient implementation for this
        return Enumerable(lambda: list(takewhile(predicate, self._get_data())))

    def skip_while(self: 'Enumerable[T]', predicate: Predicate[T]) -> 'Enumerable[T]':
        """skip elements while predicate is true"""
        from ..enumerable import Enumerable
        # itertools.dropwhile is the most efficient implementation for this
        return Enumerable(lambda: list(dropwhile(predicate, self._get_data())))

    def select_with_index(self: 'Enumerable[T]', selector: Callable[[T, int], U]) -> 'Enumerable[U]':
        """project each element to a new form, using the element's index"""
        from ..enumerable import Enumerable
        def map_with_index_data():
            return [selector(item, index) for index, item in enumerate(self._get_data())]
        return Enumerable(map_with_index_data)

    def reverse(self: 'Enumerable[T]') -> 'Enumerable[T]':
        """inverts the order of the elements in a sequence"""
        from ..enumerable import Enumerable
        return Enumerable(lambda: list(reversed(self._get_data())))

    def append(self: 'Enumerable[T]', element: T) -> 'Enumerable[T]':
        """appends a value to the end of the sequence"""
        from ..enumerable import Enumerable
        def append_data():
            # itertools.chain is the most memory-efficient way to combine iterables
            return list(chain(self._get_data(), [element]))
        return Enumerable(append_data)

    def prepend(self: 'Enumerable[T]', element: T) -> 'Enumerable[T]':
        """adds a value to the beginning of the sequence"""
        from ..enumerable import Enumerable
        def prepend_data():
            # chain is also optimal for prepending
            return list(chain([element], self._get_data()))
        return Enumerable(prepend_data)

    def default_if_empty(self: 'Enumerable[T]', default_value: T) -> 'Enumerable[T]':
        """returns the elements of a sequence, or a default value in a singleton collection if the sequence is empty"""
        from ..enumerable import Enumerable
        def default_data():
            data = self._get_data()
            return data if data else [default_value]
        return Enumerable(default_data)

    def of_type(self: 'Enumerable[T]', type_filter: Type[U]) -> 'Enumerable[U]':
        """filters the elements of a sequence based on a specified type"""
        # this is syntactic sugar over where() but improves readability and intent
        # the type hint Type[U] ensures the user passes a class/type, not an instance
        return self.where(lambda item: isinstance(item, type_filter))

    def as_ordered(self: 'Enumerable[T]') -> 'OrderedEnumerable[T]':
        """
        treats the current sequence as already ordered, allowing 'then_by' to be called.
        this does not perform a sort. use it only when the source is pre-sorted.
        """
        from ..enumerable import OrderedEnumerable
        # create an orderedenumerable with a no-op sort
        # the key is constant, so python's stable sort preserves the original order
        return OrderedEnumerable(self._data_func, [(lambda x: None, False)])