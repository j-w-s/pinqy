from __future__ import annotations
import typing
import math
import numpy as np
from itertools import accumulate
from bisect import bisect_right
from ..types import *

if typing.TYPE_CHECKING:
    from ..enumerable import Enumerable

class StatsAccessor(Generic[T]):
    def __init__(self, enumerable_instance: 'Enumerable[T]'):
        self._enumerable = enumerable_instance

    def _get_values(self, selector: Optional[Selector[T, Union[int, float]]] = None) -> List[Union[int, float]]:
        """helper to extract numeric values for statistical operations."""
        if selector: return self._enumerable.select(selector).to.list()
        data = self._enumerable.to.list()
        if data and not all(isinstance(x, (int, float)) for x in data):
            raise TypeError("sequence contains non-numeric types for statistical operation.")
        return data

    def _calculate_stats_welford(self, selector: Optional[Selector[T, Union[int, float]]]) -> Tuple[int, float, float]:
        """calculates count, mean, and variance in a single pass. returns (count, mean, variance)."""
        count, mean, m2 = 0, 0.0, 0.0
        for x in self._get_values(selector):
            count += 1
            delta = x - mean
            mean += delta / count
            delta2 = x - mean
            m2 += delta * delta2
        # using population variance, as is common with numpy/pandas std dev default (ddof=0)
        variance = m2 / count if count > 0 else 0.0
        return count, mean, variance

    def sum(self, selector: Optional[Selector[T, Union[int, float]]] = None) -> Union[int, float]:
        """calc sum"""
        values = self._get_values(selector)
        try:
            result = np.sum(values)
            return result.item() if hasattr(result, 'item') else result
        except (TypeError, ValueError):
            return sum(values)

    def average(self, selector: Optional[Selector[T, Union[int, float]]] = None) -> float:
        """calc average"""
        count, mean, _ = self._calculate_stats_welford(selector)
        if count == 0: raise ValueError("cannot calculate average of empty sequence")
        return mean

    def std_dev(self, selector: Optional[Selector[T, Union[int, float]]] = None) -> float:
        """calculate standard deviation"""
        count, _, variance = self._calculate_stats_welford(selector)
        if count == 0: raise ValueError("cannot calculate standard deviation of empty sequence")
        return math.sqrt(variance)

    def min(self, selector: Optional[Selector[T, Any]] = None) -> T:
        """find minimum"""
        data = self._enumerable._get_data()
        if not data: raise ValueError("cannot find minimum of empty sequence")
        return min(data, key=selector) if selector else min(data)

    def max(self, selector: Optional[Selector[T, Any]] = None) -> T:
        """find maximum"""
        data = self._enumerable._get_data()
        if not data: raise ValueError("cannot find maximum of empty sequence")
        return max(data, key=selector) if selector else max(data)

    def median(self, selector: Optional[Selector[T, Union[int, float]]] = None) -> float:
        """calculate median value"""
        sorted_values = sorted(self._get_values(selector))
        n = len(sorted_values)
        if n == 0: raise ValueError("cannot calculate median of empty sequence")
        mid = n // 2
        return (sorted_values[mid] + sorted_values[mid - 1]) / 2 if n % 2 == 0 else sorted_values[mid]

    def percentile(self, q: float, selector: Optional[Selector[T, Union[int, float]]] = None) -> float:
        """calculate percentile (0 <= q <= 100)"""
        sorted_vals = sorted(self._get_values(selector))
        if not sorted_vals: raise ValueError("cannot calculate percentile of empty sequence")
        if not 0 <= q <= 100: raise ValueError("percentile must be between 0 and 100")
        k = (len(sorted_vals) - 1) * (q / 100.0)
        f, c = math.floor(k), math.ceil(k)
        if f == c: return sorted_vals[int(k)]
        return sorted_vals[int(f)] * (c - k) + sorted_vals[int(c)] * (k - f)

    def mode(self, selector: Optional[Selector[T, K]] = None) -> K:
        """find most frequent element"""
        data = self._enumerable.select(selector).to.list() if selector else self._enumerable.to.list()
        if not data: raise ValueError("cannot find mode of empty sequence")
        return max(set(data), key=data.count)

    def rolling_window(self, window_size: int, aggregator: Callable[[List[T]], U]) -> 'Enumerable[U]':
        """apply aggregator to rolling windows"""
        return self._enumerable.group.window(window_size).select(aggregator)

    def rolling_sum(self, window_size: int,
                    selector: Optional[Selector[T, Union[int, float]]] = None) -> 'Enumerable[Union[int, float]]':
        """rolling sum using existing operations"""
        from ..factories import from_iterable
        return self.rolling_window(window_size, lambda w: from_iterable(w).stats.sum(selector))

    def rolling_average(self, window_size: int,
                        selector: Optional[Selector[T, Union[int, float]]] = None) -> 'Enumerable[float]':
        """rolling average using existing operations"""
        from ..factories import from_iterable
        return self.rolling_window(window_size, lambda w: from_iterable(w).stats.average(selector))

    def lag(self, periods: int = 1, fill_value: Optional[T] = None) -> 'Enumerable[Optional[T]]':
        """shift elements by periods"""
        from ..enumerable import Enumerable
        def lag_data():
            data = self._enumerable._get_data()
            if periods <= 0: return data
            return ([fill_value] * periods) + data[:-periods]
        return Enumerable(lag_data)

    def lead(self, periods: int = 1, fill_value: Optional[T] = None) -> 'Enumerable[Optional[T]]':
        """shift elements forward"""
        from ..enumerable import Enumerable
        def lead_data():
            data = self._enumerable._get_data()
            if periods <= 0: return data
            return data[periods:] + ([fill_value] * periods)
        return Enumerable(lead_data)

    def diff(self, periods: int = 1) -> 'Enumerable[T]':
        """difference with previous elements"""
        lagged = self.lag(periods, fill_value=None)
        # zip automatically stops at the shorter iterable, correctly handling the length change
        return self._enumerable.zip.zip_with(lagged, lambda current, prev: current - prev if prev is not None and current is not None else None).skip(periods)

    def scan(self, accumulator: Accumulator[U, T], seed: U) -> 'Enumerable[U]':
        """produce intermediate accumulation values, including seed"""
        from ..enumerable import Enumerable
        return Enumerable(lambda: list(accumulate(self._enumerable._get_data(), accumulator, initial=seed)))

    def cumulative_sum(self, selector: Optional[Selector[T, Union[int, float]]] = None) -> 'Enumerable[Union[int, float]]':
        """cumulative sum"""
        from ..enumerable import Enumerable
        values = self._get_values(selector)
        return Enumerable(lambda: list(accumulate(values, lambda acc, x: acc + x)))

    def cumulative_product(self, selector: Optional[Selector[T, Union[int, float]]] = None) -> 'Enumerable[Union[int, float]]':
        """cumulative product"""
        from ..enumerable import Enumerable
        values = self._get_values(selector)
        return Enumerable(lambda: list(accumulate(values, lambda acc, x: acc * x)))

    def cumulative_max(self, selector: Optional[Selector[T, K]] = None) -> 'Enumerable[K]':
        """cumulative maximum"""
        from ..enumerable import Enumerable
        values = self._enumerable.select(selector).to.list() if selector else self._enumerable.to.list()
        return Enumerable(lambda: list(accumulate(values, max)))

    def cumulative_min(self, selector: Optional[Selector[T, K]] = None) -> 'Enumerable[K]':
        """cumulative minimum"""
        from ..enumerable import Enumerable
        values = self._enumerable.select(selector).to.list() if selector else self._enumerable.to.list()
        return Enumerable(lambda: list(accumulate(values, min)))

    def rank(self, selector: Optional[Selector[T, K]] = None, ascending: bool = True) -> 'Enumerable[int]':
        """rank elements (1-based) using an efficient o(n log n) algorithm."""
        from ..enumerable import Enumerable
        def rank_data():
            data = self._enumerable._get_data()
            values = [(selector(item) if selector else item) for item in data]
            # use a stable sort on the original index to handle ties correctly
            indexed = sorted(enumerate(values), key=lambda p: p[1], reverse=not ascending)
            ranks = [0] * len(data)
            # assign rank, but if values are the same, assign the same rank
            for i in range(len(indexed)):
                original_index = indexed[i][0]
                # tie handling: if the current value is the same as the previous, use the same rank
                if i > 0 and indexed[i][1] == indexed[i-1][1]:
                    original_prev_index = indexed[i-1][0]
                    ranks[original_index] = ranks[original_prev_index]
                else:
                    ranks[original_index] = i + 1
            return ranks
        return Enumerable(rank_data)

    def dense_rank(self, selector: Optional[Selector[T, K]] = None, ascending: bool = True) -> 'Enumerable[int]':
        """dense rank elements (1-based) using an efficient o(n log n) algorithm."""
        from ..enumerable import Enumerable
        def dense_rank_data():
            values = self._enumerable.select(selector) if selector else self._enumerable
            distinct_values = values.set.distinct()

            # use order_by or order_by_descending conditionally
            if ascending:
                unique_sorted = distinct_values.order_by(lambda x: x).to.list()
            else:
                unique_sorted = distinct_values.order_by_descending(lambda x: x).to.list()

            rank_map = {val: i + 1 for i, val in enumerate(unique_sorted)}
            return [rank_map[selector(item) if selector else item] for item in self._enumerable._get_data()]
        return Enumerable(dense_rank_data)

    def quantile_cut(self, q: int, selector: Optional[Selector[T, Union[int, float]]] = None) -> 'Enumerable[int]':
        """cut into q quantile bins using an efficient algorithm."""
        from ..factories import from_iterable
        from ..enumerable import Enumerable
        if q <= 0: raise ValueError("number of quantiles (q) must be positive")
        def cut_data():
            values_en = from_iterable(self._get_values(selector))
            quantiles = [values_en.stats.percentile(i * 100 / q) for i in range(1, q)]
            return [bisect_right(quantiles, val) for val in values_en]
        return Enumerable(cut_data)

    def normalize(self, selector: Optional[Selector[T, float]] = None) -> 'Enumerable[float]':
        """min-max normalization"""
        from ..factories import from_iterable
        values = from_iterable(self._get_values(selector))
        min_val, max_val = values.stats.min(), values.stats.max()
        if max_val == min_val: return values.select(lambda _: 0.0)
        return values.select(lambda x: (x - min_val) / (max_val - min_val))

    def standardize(self, selector: Optional[Selector[T, float]] = None) -> 'Enumerable[float]':
        """z-score standardization"""
        from ..factories import from_iterable
        values = from_iterable(self._get_values(selector))
        mean_val, std_val = values.stats.average(), values.stats.std_dev()
        if std_val == 0: return values.select(lambda _: 0.0)
        return values.select(lambda x: (x - mean_val) / std_val)

    def outliers_iqr(self, selector: Optional[Selector[T, float]] = None, factor: float = 1.5) -> 'Enumerable[T]':
        """detect outliers using iqr method"""
        from ..factories import from_iterable
        values = from_iterable(self._get_values(selector))
        q1, q3 = values.stats.percentile(25), values.stats.percentile(75)
        iqr = q3 - q1
        lower_bound, upper_bound = q1 - factor * iqr, q3 + factor * iqr
        sel = selector if selector else lambda item: item
        return self._enumerable.where(lambda item: not (lower_bound <= sel(item) <= upper_bound))

    def lag_in_order(self, periods: int = 1, fill_value: Optional[T] = None) -> 'Enumerable[Optional[T]]':
        """
        shifts elements by periods, but requires the enumerable to be ordered.
        different from regular lag() which uses original sequence order.
        """
        from ..enumerable import OrderedEnumerable
        if not isinstance(self._enumerable, OrderedEnumerable):
            raise TypeError("lag_in_order can only be called on OrderedEnumerable instances. Use order_by() first.")

        return self._enumerable.lag_in_order(periods, fill_value)