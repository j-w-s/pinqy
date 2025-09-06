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
        this is a terminal operation in spirit but returns the original enumerable to allow chaining.
        """
        from ..enumerable import Enumerable
        def for_each_data():
            data = self._enumerable._get_data()
            for item in data:
                action(item)
            # returns the original data to the next link in the chain
            return data
        return Enumerable(for_each_data)

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
        if depth <= 0: return self._enumerable
        current = self._enumerable
        for _ in range(depth):
            current = current.select_many(
                lambda x: x if isinstance(x, Iterable) and not isinstance(x, (str, bytes)) else [x])
        return current

    def transpose(self) -> 'Enumerable[List[Any]]':
        """transpose a matrix-like structure (list of lists) using the efficient itertools implementation."""
        from ..enumerable import Enumerable
        return Enumerable(lambda: [list(col) for col in zip_longest(*self._enumerable._get_data())])

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
        from ..factories import from_iterable, repeat
        data = self._enumerable.to.list()
        if len(data) < 2: return from_iterable(data)
        it = iter(data)
        # a small correction: repeat is a factory, not a method
        interspersed = chain.from_iterable(zip(it, repeat(separator, len(data) - 1)._get_data()))
        return from_iterable(interspersed).set.concat(list(it))

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
                result.extend(from_iterable(group_items).util.sample(min(samples_per_group, len(group_items))).to.list())
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
        evaluates the enumerable chain and caches the result. subsequent operations on the
        returned enumerable will start from this cached state, preventing re-computation
        of the preceding chain.
        """
        from ..enumerable import Enumerable
        # calling _get_data() triggers the full computation and caching
        cached_data = self._enumerable._get_data()
        # return a new enumerable that starts with the already-computed data
        return Enumerable(lambda: cached_data)

    def pipe(self, func: Callable[..., U], *args, **kwargs) -> U:
        """
        pipes the enumerable object into an external function. enables custom, chainable operations.
        example: .pipe(my_custom_plot_function, title='my data')
        """
        return func(self._enumerable, *args, **kwargs)

    def side_effect(self, action: Callable[[T], Any]) -> 'Enumerable[T]':
        """
        performs a side-effect action for each element in the sequence without modifying it.
        primarily used for debugging, e.g., .side_effect(print).
        """
        from ..enumerable import Enumerable
        def side_effect_data():
            data = self._enumerable._get_data()
            for item in data:
                action(item)
            return data

        return Enumerable(side_effect_data)

    def topological_sort(self, dependency_selector: Callable[[T], Iterable[T]]) -> 'Enumerable[T]':
        """
        performs a topological sort on the sequence, treated as nodes in a directed acyclic graph (dag).
        the dependency_selector function must return an iterable of nodes that a given node depends on.
        raises a valueerror if a cycle is detected.
        """
        from ..enumerable import Enumerable
        def sort_data():
            data = self._enumerable.to.list()
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

    def flatten_deep(self, is_iterable_func: Optional[Callable[[Any], bool]] = None) -> 'Enumerable[Any]':
        """
        deeply flattens nested structures with custom iterable detection.
        similar to c#'s FlattenDeep but with python-specific optimizations.
        """
        from ..enumerable import Enumerable
        def is_iterable_default(item):
            return (isinstance(item, Iterable) and
                    not isinstance(item, (str, bytes, bytearray)))

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