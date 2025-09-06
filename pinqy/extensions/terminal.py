from __future__ import annotations
import typing
import numpy as np
import pandas as pd
from functools import reduce
from ..types import *

if typing.TYPE_CHECKING:
    from ..enumerable import Enumerable

class TerminalAccessor(Generic[T]):
    def __init__(self, enumerable_instance: 'Enumerable[T]'):
        self._enumerable = enumerable_instance

    def list(self) -> List[T]:
        """convert to list"""
        return self._enumerable._get_data()

    def array(self) -> np.ndarray:
        """convert to numpy array"""
        return np.array(self._enumerable._get_data())

    def set(self) -> Set[T]:
        """convert to set"""
        return set(self._enumerable._get_data())

    def dict(self, key_selector: KeySelector[T, K],
             value_selector: Optional[Selector[T, V]] = None) -> Dict[K, V]:
        """convert to dictionary"""
        val_sel = value_selector if value_selector else lambda item: item
        return {key_selector(item): val_sel(item) for item in self._enumerable._get_data()}

    def pandas(self) -> pd.Series:
        """convert to pandas series"""
        return pd.Series(self._enumerable._get_data())

    def df(self) -> pd.DataFrame:
        """convert to pandas dataframe"""
        return pd.DataFrame(self._enumerable._get_data())

    def count(self, predicate: Optional[Predicate[T]] = None) -> int:
        """count elements"""
        if predicate is None: return len(self._enumerable._get_data())
        return sum(1 for x in self._enumerable._get_data() if predicate(x))

    def any(self, predicate: Optional[Predicate[T]] = None) -> bool:
        """check if any element satisfies condition"""
        data = self._enumerable._get_data()
        if predicate is None: return len(data) > 0
        return any(predicate(x) for x in data)

    def all(self, predicate: Predicate[T]) -> bool:
        """check if all elements satisfy condition"""
        return all(predicate(x) for x in self._enumerable._get_data())

    def first(self, predicate: Optional[Predicate[T]] = None) -> T:
        """get first element"""
        data = self._enumerable._get_data()
        if predicate is None:
            if not data: raise ValueError("sequence contains no elements")
            return data[0]
        for item in data:
            if predicate(item): return item
        raise ValueError("no element satisfies the condition")

    def first_or_default(self, predicate: Optional[Predicate[T]] = None,
                         default: Optional[T] = None) -> Optional[T]:
        """get first element or default"""
        try: return self.first(predicate)
        except ValueError: return default

    def single(self, predicate: Optional[Predicate[T]] = None) -> T:
        """get single element, erroring if not exactly one"""
        data = [x for x in self._enumerable._get_data() if predicate(x)] if predicate else self._enumerable._get_data()
        if len(data) == 0: raise ValueError("sequence contains no matching elements")
        if len(data) > 1: raise ValueError("sequence contains more than one matching element")
        return data[0]

    def aggregate(self, accumulator: Accumulator[T, T], seed: Optional[T] = None) -> T:
        """applies accumulator function over sequence"""
        data = self._enumerable._get_data()
        if not data and seed is None: raise ValueError("cannot aggregate empty sequence without seed")
        return reduce(accumulator, data, seed) if seed is not None else reduce(accumulator, data)

    def aggregate_with_selector(self, seed: U, accumulator: Accumulator[U, T],
                                result_selector: Selector[U, V]) -> V:
        """aggregate with seed and final transformation"""
        return result_selector(reduce(accumulator, self._enumerable._get_data(), seed))