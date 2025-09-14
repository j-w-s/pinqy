from __future__ import annotations
import typing
import random
from collections import deque
from itertools import groupby as itertools_groupby, zip_longest, chain
from ..types import *

if typing.TYPE_CHECKING:
    from ..enumerable import Enumerable


class UtilityAccessor(Generic[T]):
    def __init__(self, enumerable_instance: 'Enumerable[T]'):
        self._enumerable = enumerable_instance

    def for_each(self, action: Callable[[T], Any]) -> 'Enumerable[T]':
        """
        performs the specified action on each element of a sequence for side-effects.
        this is an EAGER operation that executes immediately.
        returns the original enumerable to allow chaining.
        """
        # this is an eager operation.
        for item in self._enumerable._get_data():
            action(item)
        # returns the original enumerable instance, not a new lazy one.
        return self._enumerable

    def run_length_encode(self) -> 'Enumerable[Tuple[T, int]]':
        """
        performs run-length encoding on the sequence. consecutive identical elements
        are grouped into (element, count) tuples.
        """
        from ..enumerable import Enumerable
        def rle_data():
            data = self._enumerable._get_data()
            # itertools.groupby is perfect for this, grouping consecutive identical items
            return [(key, len(list(group))) for key, group in itertools_groupby(data)]

        return Enumerable(rle_data)

    def flatten(self, depth: int = 1) -> 'Enumerable[Any]':
        """flatten nested sequences to a specified depth"""
        from ..enumerable import Enumerable

        def flatten_data():
            def flatten_recursive(items, current_depth):
                if current_depth <= 0:
                    return items

                result = []
                for item in items:
                    if isinstance(item, (list, tuple)) and not isinstance(item, (str, bytes)):
                        result.extend(flatten_recursive(item, current_depth - 1))
                    elif hasattr(item, '__iter__') and not isinstance(item, (str, bytes, dict)):
                        result.extend(flatten_recursive(list(item), current_depth - 1))
                    else:
                        result.append(item)
                return result

            return flatten_recursive(self._enumerable._get_data(), depth)

        return Enumerable(flatten_data)

    def transpose(self) -> 'Enumerable[List[Any]]':
        """transpose a matrix-like structure (list of lists) using the efficient itertools implementation."""
        from ..enumerable import Enumerable
        def transpose_data():
            data = self._enumerable._get_data()
            if not data:
                return []
            return [list(col) for col in zip_longest(*data)]

        return Enumerable(transpose_data)

    def unzip(self) -> Tuple['Enumerable[Any]', ...]:
        """
        the inverse of zip. transforms an enumerable of tuples/lists into a tuple of enumerables.
        e.g., [(a, 1), (b, 2)] -> ( (a, b), (1, 2) )
        """
        from ..factories import from_iterable
        data = self._enumerable.to.list()
        if not data:
            return tuple()
        # the pythonic way to unzip is with zip(*)
        unzipped_cols = zip(*data)
        return tuple(from_iterable(col) for col in unzipped_cols)

    def intersperse(self, separator: T) -> 'Enumerable[T]':
        """intersperse separator between elements"""
        from ..enumerable import Enumerable

        def intersperse_data():
            data = self._enumerable._get_data()
            if len(data) < 2:
                return data

            result = [data[0]]
            for item in data[1:]:
                result.append(separator)
                result.append(item)
            return result

        return Enumerable(intersperse_data)

    def sample(self, n: int, replace: bool = False,
               random_state: Optional[int] = None) -> 'Enumerable[T]':
        """random sampling"""
        from ..enumerable import Enumerable
        def sample_data():
            if random_state: random.seed(random_state)
            data = self._enumerable._get_data()
            if replace: return random.choices(data, k=n)
            return random.sample(data, min(n, len(data)))

        return Enumerable(sample_data)

    def stratified_sample(self, key_selector: KeySelector[T, K],
                          samples_per_group: int) -> 'Enumerable[T]':
        """stratified sampling"""
        from ..enumerable import Enumerable
        from ..factories import from_iterable
        def stratified_data():
            groups = self._enumerable.group.group_by(key_selector)
            result = []
            for group_items in groups.values():
                result.extend(
                    from_iterable(group_items).util.sample(min(samples_per_group, len(group_items))).to.list())
            return result

        return Enumerable(stratified_data)

    def bootstrap_sample(self, n_samples: int = 1000,
                         sample_size: Optional[int] = None) -> 'Enumerable[Enumerable[T]]':
        """bootstrap sampling"""
        from ..factories import from_range
        size = sample_size or self._enumerable.to.count()
        return from_range(0, n_samples).select(lambda _: self._enumerable.util.sample(size, replace=True))

    def memoize(self) -> 'Enumerable[T]':
        """
        returns a new enumerable that caches the result of this one upon first evaluation.
        this operation is LAZY. the cache is only populated when the new enumerable is
        iterated for the first time.
        """
        from ..enumerable import Enumerable
        # make the operation lazy.
        # pass the original enumerable's _get_data method as the data_func for the new one.
        # this means the original's data is only requested (and subsequently cached by itself)
        # when the *new* enumerable is first evaluated.
        return Enumerable(self._enumerable._get_data)

    def pipe(self, func: Callable[..., U], *args, **kwargs) -> U:
        """
        pipes the enumerable object into an external function. enables custom, chainable operations.
        example: .pipe(my_custom_plot_function, title='my data')
        """
        return func(self._enumerable, *args, **kwargs)

    def side_effect(self, action: Callable[[T], Any]) -> 'Enumerable[T]':
        """
        performs a side-effect action for each element as it passes through the sequence
        without modifying it. this operation is lazy and is primarily used for debugging
        pipelines without materializing the data.
        example: .where(...).side_effect(print).select(...)
        """
        from ..enumerable import Enumerable
        def lazy_side_effect_generator():
            # iterate through the source without calling _get_data() to remain lazy
            for item in self._enumerable:
                action(item)
                yield item

        # enumerable constructor expects a function that returns a list.
        # generator must be wrapped in a lambda that materializes it.
        return Enumerable(lambda: list(lazy_side_effect_generator()))

    def topological_sort(self, dependency_selector: Callable[[T], Iterable[T]]) -> 'Enumerable[T]':
        """
        performs a topological sort on the sequence, treated as nodes in a directed acyclic graph (dag).
        the dependency_selector function must return an iterable of nodes that a given node depends on.
        raises a valueerror if a cycle is detected.
        """
        from ..enumerable import Enumerable
        def sort_data():
            data = self._enumerable.to.list()

            # create a mapping from items to indices for hashable lookups
            item_to_index = {id(item): i for i, item in enumerate(data)}
            index_to_item = {i: item for i, item in enumerate(data)}

            in_degree = {i: 0 for i in range(len(data))}
            adj = {i: [] for i in range(len(data))}

            for i, item in enumerate(data):
                for dep in dependency_selector(item):
                    dep_index = None
                    for j, candidate in enumerate(data):
                        if (isinstance(dep, dict) and isinstance(candidate, dict) and
                            dep.get('id') == candidate.get('id')) or dep == candidate:
                            dep_index = j
                            break

                    if dep_index is not None:
                        adj[dep_index].append(i)
                        in_degree[i] += 1

            # kahn's algorithm
            queue = deque([i for i in range(len(data)) if in_degree[i] == 0])
            sorted_indices = []

            while queue:
                u = queue.popleft()
                sorted_indices.append(u)

                for v in adj[u]:
                    in_degree[v] -= 1
                    if in_degree[v] == 0:
                        queue.append(v)

            if len(sorted_indices) != len(data):
                raise ValueError("a cycle was detected in the dependency graph.")

            return [index_to_item[i] for i in sorted_indices]

        return Enumerable(sort_data)

    def flatten_deep(self, is_iterable_func: Optional[Callable[[Any], bool]] = None) -> 'Enumerable[Any]':
        """
        deeply flattens nested structures with custom iterable detection.
        similar to c#'s FlattenDeep but with python-specific optimizations.
        """
        from ..enumerable import Enumerable
        def is_iterable_default(item):
            return (isinstance(item, Iterable) and
                    not isinstance(item, (str, bytes, bytearray, dict)))

        is_iter_check = is_iterable_func or is_iterable_default

        def flatten_data():
            result = []
            stack = list(reversed(self._enumerable._get_data()))
            while stack:
                item = stack.pop()
                if is_iter_check(item):
                    stack.extend(reversed(list(item)))
                else:
                    result.append(item)
            return result

        return Enumerable(flatten_data)

    def pipe_through(self, *operations: Callable[['Enumerable[T]'], 'Enumerable[Any]']) -> 'Enumerable[Any]':
        """
        functional pipeline - applies operations in sequence.
        each operation receives the result of the previous.
        """
        result = self._enumerable
        for operation in operations:
            result = operation(result)
        return result

    def apply_if(self, condition: bool, operation: Callable[['Enumerable[T]'], 'Enumerable[T]']) -> 'Enumerable[T]':
        """conditionally apply operation based on boolean condition"""
        return operation(self._enumerable) if condition else self._enumerable

    def apply_when(self, predicate: Callable[['Enumerable[T]'], bool],
                   operation: Callable[['Enumerable[T]'], 'Enumerable[T]']) -> 'Enumerable[T]':
        """conditionally apply operation based on predicate evaluation"""
        return operation(self._enumerable) if predicate(self._enumerable) else self._enumerable

    def try_parse(self: 'UtilityAccessor[str]', parser: Callable[[str], Tuple[bool, T]]) -> 'ParseResult[T]':
        """
        attempts to parse string elements, collecting successes and failures.
        parser should return (success: bool, value: T) tuple.
        """
        successes, failures = [], []
        for item in self._enumerable._get_data():
            success, value = parser(item)
            if success:
                successes.append(value)
            else:
                failures.append(item)
        return ParseResult(successes, failures)

    def parse_or_default(self: 'UtilityAccessor[str]', parser: Callable[[str], Tuple[bool, T]],
                         default_value: T = None) -> 'Enumerable[T]':
        """parse with default fallback for failed attempts"""
        from ..enumerable import Enumerable
        def parse_with_default():
            return [parser(item)[1] if parser(item)[0] else default_value for item in self._enumerable._get_data()]

        return Enumerable(parse_with_default)

    def compose(self, *functions: Callable[['Enumerable[Any]'], 'Enumerable[Any]']) -> 'Enumerable[Any]':
        """
        functional composition - creates a new function by composing multiple functions.
        applies functions from left to right (unlike mathematical composition).
        """
        result = self._enumerable
        for func in functions:
            result = func(result)
        return result

    def apply_functions(self, functions: Iterable[Callable[[T], U]]) -> 'Enumerable[U]':
        """
        applies multiple functions to each element (cartesian product style).
        each item is transformed by each function, creating len(items) * len(functions) results.
        """
        from ..enumerable import Enumerable
        def apply_data():
            func_list = list(functions)
            return [func(item) for item in self._enumerable._get_data() for func in func_list]

        return Enumerable(apply_data)

    def lazy_where(self, contextual_predicate: Callable[[T, List[T]], bool]) -> 'Enumerable[T]':
        """
        contextual filtering where predicate has access to full materialized sequence.
        useful for operations requiring knowledge of the entire dataset.
        """
        from ..enumerable import Enumerable
        def contextual_filter():
            materialized = self._enumerable._get_data()
            return [item for item in materialized if contextual_predicate(item, materialized)]

        return Enumerable(contextual_filter)

    def unfold(self, seed_selector: Callable[[T], U],
               unfolder: Callable[[U], Optional[Tuple[V, U]]]) -> 'Enumerable[V]':
        """
        unfold operation - generates sequence from seeds using unfolder function.
        unfolder returns (next_value, next_seed) or None to terminate.
        """
        from ..enumerable import Enumerable
        def unfold_data():
            result = []
            for item in self._enumerable._get_data():
                current_seed = seed_selector(item)
                while True:
                    unfolded = unfolder(current_seed)
                    if unfolded is None: break
                    value, next_seed = unfolded
                    result.append(value)
                    current_seed = next_seed
            return result

        return Enumerable(unfold_data)

    def memoize_advanced(self) -> 'MemoizedEnumerable[T]':
        """
        advanced memoization with lazy evaluation and partial caching.
        more sophisticated than the basic memoize() method.
        """
        return MemoizedEnumerable(self._enumerable.__iter__)