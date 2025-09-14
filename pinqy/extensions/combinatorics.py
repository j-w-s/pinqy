import typing
import math
from itertools import (
    chain,
    permutations as itertools_permutations,
    combinations as itertools_combinations,
    product as itertools_product,
    combinations_with_replacement as itertools_combinations_with_replacement
)
from ..types import *

if typing.TYPE_CHECKING:
    from ..enumerable import Enumerable

class CombinatoricsAccessor(Generic[T]):
    def __init__(self, enumerable_instance: 'Enumerable[T]'):
        self._enumerable = enumerable_instance

    def binomial_coefficient(self, r: int) -> int:
        """binomial coefficient n choose r, using python's optimized math.comb"""
        n = self._enumerable.to.count()
        # math.comb raises valueerror for r < 0. return 0 for consistency.
        if r < 0:
            return 0
        return math.comb(n, r)

    def permutations(self, r: Optional[int] = None) -> 'Enumerable[Tuple[T, ...]]':
        """generate permutations using the efficient itertools implementation."""
        from ..factories import from_iterable
        r_val = self._enumerable.to.count() if r is None else r
        # itertools.permutations raises valueerror for r < 0. return empty enumerable.
        if r_val < 0:
            return from_iterable([])
        return from_iterable(itertools_permutations(self._enumerable.to.list(), r_val))

    def combinations(self, r: int) -> 'Enumerable[Tuple[T, ...]]':
        """generate combinations using the efficient itertools implementation."""
        from ..factories import from_iterable
        # itertools.combinations raises valueerror for r < 0. return empty enumerable.
        if r < 0:
            return from_iterable([])
        return from_iterable(itertools_combinations(self._enumerable.to.list(), r))

    def combinations_with_replacement(self, r: int) -> 'Enumerable[Tuple[T, ...]]':
        """
        generate combinations with replacement using the efficient itertools implementation.
        elements are treated as if they were replaced after each pick.
        """
        from ..factories import from_iterable
        # itertools.combinations_with_replacement raises valueerror for r < 0. return empty.
        if r < 0:
            return from_iterable([])
        return from_iterable(itertools_combinations_with_replacement(self._enumerable.to.list(), r))

    def power_set(self) -> 'Enumerable[Tuple[T, ...]]':
        """
        generates the power set (the set of all subsets) of the sequence.
        this is implemented functionally by chaining combinations of every possible length (from 0 to n).
        """
        from ..enumerable import Enumerable
        def power_set_data():
            data = self._enumerable.to.list()
            # chain combinations of length 0, 1, 2, ..., n
            return chain.from_iterable(itertools_combinations(data, r) for r in range(len(data) + 1))

        return Enumerable(lambda: list(power_set_data()))

    def cartesian_product(self, *others: Iterable[Any]) -> 'Enumerable[Tuple[Any, ...]]':
        """
        computes the cartesian product with other iterables using the efficient itertools.product.
        returns an enumerable of tuples.
        """
        from ..enumerable import Enumerable
        def product_data():
            # gathers all iterables, including self, and unpacks them into itertools.product
            all_iterables = [self._enumerable._get_data()] + [list(o) for o in others]
            return list(itertools_product(*all_iterables))

        return Enumerable(product_data)