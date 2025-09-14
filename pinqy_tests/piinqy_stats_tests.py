import time
import math
import numpy as np
import suite
from collections import namedtuple
from dgen import from_schema
from pinqy import P, from_iterable, from_range

test = suite.test
assert_that = suite.assert_that

sample_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
sample_floats = [1.5, 2.7, 3.1, 4.8, 5.2, 6.9, 7.3, 8.1, 9.6, 10.4]
outlier_data = [1, 2, 3, 4, 5, 100, 6, 7, 8, 9]

Person = namedtuple('Person', ['name', 'age', 'salary'])
people = [
    Person('alice', 25, 50000),
    Person('bob', 30, 60000),
    Person('charlie', 25, 55000),
    Person('diana', 35, 75000),
    Person('eve', 28, 62000)
]

time_series = [10, 12, 11, 15, 13, 18, 16, 20, 19, 22, 21, 25]


@test("sum without selector")
def test_sum_basic():
    result = P(sample_numbers).stats.sum()
    expected = sum(sample_numbers)
    assert_that(result == expected, f"basic sum failed: {result}")


@test("sum with selector")
def test_sum_with_selector():
    result = P(people).stats.sum(lambda p: p.salary)
    expected = sum(p.salary for p in people)
    assert_that(result == expected, f"sum with selector failed: {result}")


@test("average calculation")
def test_average():
    result = P(sample_numbers).stats.average()
    expected = sum(sample_numbers) / len(sample_numbers)
    assert_that(abs(result - expected) < 0.001, f"average failed: {result}")

    avg_age = P(people).stats.average(lambda p: p.age)
    expected_age = sum(p.age for p in people) / len(people)
    assert_that(abs(avg_age - expected_age) < 0.001, f"average age failed: {avg_age}")


@test("standard deviation calculation")
def test_std_dev():
    result = P(sample_numbers).stats.std_dev()

    # manual calculation using population std dev (n, not n-1)
    mean = sum(sample_numbers) / len(sample_numbers)
    variance = sum((x - mean) ** 2 for x in sample_numbers) / len(sample_numbers)
    expected = math.sqrt(variance)

    assert_that(abs(result - expected) < 0.001, f"std dev failed: {result} vs {expected}")


@test("min and max operations")
def test_min_max():
    min_val = P(sample_numbers).stats.min()
    max_val = P(sample_numbers).stats.max()

    assert_that(min_val == 1, f"min should be 1: {min_val}")
    assert_that(max_val == 10, f"max should be 10: {max_val}")

    youngest = P(people).stats.min(lambda p: p.age)
    oldest = P(people).stats.max(lambda p: p.age)

    assert_that(youngest.age == 25, f"youngest should be 25: {youngest.age}")
    assert_that(oldest.age == 35, f"oldest should be 35: {oldest.age}")


@test("median calculation")
def test_median():
    result = P(sample_numbers).stats.median()
    assert_that(result == 5.5, f"median should be 5.5: {result}")

    odd_numbers = [1, 3, 5, 7, 9]
    odd_median = P(odd_numbers).stats.median()
    assert_that(odd_median == 5, f"odd median should be 5: {odd_median}")

    salary_median = P(people).stats.median(lambda p: p.salary)
    expected_median = 60000  # middle value when sorted
    assert_that(salary_median == expected_median, f"salary median failed: {salary_median}")


@test("percentile calculations")
def test_percentile():
    p25 = P(sample_numbers).stats.percentile(25)
    p50 = P(sample_numbers).stats.percentile(50)
    p75 = P(sample_numbers).stats.percentile(75)
    p100 = P(sample_numbers).stats.percentile(100)

    assert_that(abs(p50 - 5.5) < 0.001, f"50th percentile should be ~5.5: {p50}")
    assert_that(p100 == 10, f"100th percentile should be 10: {p100}")
    assert_that(p25 < p50 < p75 < p100, f"percentiles should be ordered: {p25}, {p50}, {p75}, {p100}")


@test("mode calculation")
def test_mode():
    repeated_data = [1, 2, 2, 3, 3, 3, 4]
    mode = P(repeated_data).stats.mode()
    assert_that(mode == 3, f"mode should be 3: {mode}")

    age_mode = P(people).stats.mode(lambda p: p.age)
    assert_that(age_mode == 25, f"age mode should be 25: {age_mode}")


@test("rolling window operations")
def test_rolling_window():
    window_sums = P(time_series).stats.rolling_sum(3).to.list()

    # first window: 10 + 12 + 11 = 33
    assert_that(window_sums[0] == 33, f"first window sum should be 33: {window_sums[0]}")
    assert_that(len(window_sums) == len(time_series) - 2, f"rolling window count wrong: {len(window_sums)}")

    window_avgs = P(time_series).stats.rolling_average(3).to.list()
    assert_that(abs(window_avgs[0] - 11.0) < 0.001, f"first window avg should be ~11: {window_avgs[0]}")


@test("lag and lead operations")
def test_lag_lead():
    lagged = P(sample_numbers).stats.lag(2).to.list()
    expected_lag = [None, None, 1, 2, 3, 4, 5, 6, 7, 8]
    assert_that(lagged == expected_lag, f"lag failed: {lagged}")

    lagged_filled = P(sample_numbers).stats.lag(2, fill_value=0).to.list()
    expected_lag_filled = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert_that(lagged_filled == expected_lag_filled, f"lag with fill failed: {lagged_filled}")

    led = P(sample_numbers).stats.lead(2).to.list()
    expected_lead = [3, 4, 5, 6, 7, 8, 9, 10, None, None]
    assert_that(led == expected_lead, f"lead failed: {led}")


@test("diff operation")
def test_diff():
    diff_result = P(time_series).stats.diff(1).to.list()

    # skip the first element (no previous to diff against)
    expected_first = 12 - 10  # second minus first
    assert_that(diff_result[0] == expected_first, f"first diff should be 2: {diff_result[0]}")
    assert_that(len(diff_result) == len(time_series) - 1, f"diff length wrong: {len(diff_result)}")


@test("scan operation")
def test_scan():
    cumulative_sums = P([1, 2, 3, 4, 5]).stats.scan(lambda acc, x: acc + x, seed=0).to.list()
    expected = [0, 1, 3, 6, 10, 15]  # includes seed
    assert_that(cumulative_sums == expected, f"scan failed: {cumulative_sums}")


@test("cumulative operations")
def test_cumulative_ops():
    cum_sum = P([1, 2, 3, 4]).stats.cumulative_sum().to.list()
    assert_that(cum_sum == [1, 3, 6, 10], f"cumulative sum failed: {cum_sum}")

    cum_prod = P([1, 2, 3, 4]).stats.cumulative_product().to.list()
    assert_that(cum_prod == [1, 2, 6, 24], f"cumulative product failed: {cum_prod}")

    cum_max = P([3, 1, 4, 1, 5]).stats.cumulative_max().to.list()
    assert_that(cum_max == [3, 3, 4, 4, 5], f"cumulative max failed: {cum_max}")

    cum_min = P([3, 1, 4, 1, 5]).stats.cumulative_min().to.list()
    assert_that(cum_min == [3, 1, 1, 1, 1], f"cumulative min failed: {cum_min}")


@test("rank operations")
def test_rank():
    data = [10, 30, 20, 30, 40]
    ranks = P(data).stats.rank().to.list()

    # ties get the same rank, next rank is incremented by count of ties
    expected = [1, 3, 2, 3, 5]
    assert_that(ranks == expected, f"rank failed: {ranks}")

    desc_ranks = P(data).stats.rank(ascending=False).to.list()
    expected_desc = [5, 2, 4, 2, 1]
    assert_that(desc_ranks == expected_desc, f"descending rank failed: {desc_ranks}")


@test("dense rank operations")
def test_dense_rank():
    data = [10, 30, 20, 30, 40]
    dense_ranks = P(data).stats.dense_rank().to.list()

    # dense rank doesn't skip numbers after ties
    expected = [1, 3, 2, 3, 4]
    assert_that(dense_ranks == expected, f"dense rank failed: {dense_ranks}")

    salary_ranks = P(people).stats.dense_rank(lambda p: p.salary).to.list()
    assert_that(len(set(salary_ranks)) <= len(people), f"dense rank count wrong: {salary_ranks}")


@test("quantile cut operations")
def test_quantile_cut():
    quartiles = P(sample_numbers).stats.quantile_cut(4).to.list()

    # should have values 0, 1, 2, 3 representing quartiles
    unique_quartiles = set(quartiles)
    assert_that(unique_quartiles.issubset({0, 1, 2, 3}), f"quartile values wrong: {unique_quartiles}")
    assert_that(len(quartiles) == 10, f"quartile count wrong: {len(quartiles)}")


@test("normalize and standardize")
def test_normalize_standardize():
    normalized = P(sample_floats).stats.normalize().to.list()

    # min-max normalization should result in values between 0 and 1
    assert_that(min(normalized) == 0.0, f"min normalized should be 0: {min(normalized)}")
    assert_that(max(normalized) == 1.0, f"max normalized should be 1: {max(normalized)}")
    assert_that(all(0 <= x <= 1 for x in normalized), f"all values should be 0-1: {normalized}")

    standardized = P(sample_floats).stats.standardize().to.list()

    # z-score should have mean ~0 and std ~1
    std_mean = sum(standardized) / len(standardized)
    assert_that(abs(std_mean) < 0.001, f"standardized mean should be ~0: {std_mean}")


@test("outlier detection")
def test_outliers_iqr():
    outliers = P(outlier_data).stats.outliers_iqr().to.list()

    # 100 should be detected as outlier
    assert_that(100 in outliers, f"100 should be outlier: {outliers}")
    assert_that(len(outliers) >= 1, f"should find at least one outlier: {len(outliers)}")

    # test with custom factor
    strict_outliers = P(outlier_data).stats.outliers_iqr(factor=1.0).to.list()
    lenient_outliers = P(outlier_data).stats.outliers_iqr(factor=3.0).to.list()

    assert_that(len(strict_outliers) >= len(lenient_outliers),
                f"strict should find more outliers: {len(strict_outliers)} vs {len(lenient_outliers)}")


@test("empty sequence error handling")
def test_empty_sequences():
    empty = P([])

    try:
        empty.stats.average()
        assert_that(False, "empty average should raise error")
    except ValueError as e:
        assert_that("empty sequence" in str(e), f"unexpected error: {e}")

    try:
        empty.stats.min()
        assert_that(False, "empty min should raise error")
    except ValueError as e:
        assert_that("empty sequence" in str(e), f"unexpected error: {e}")

    try:
        empty.stats.median()
        assert_that(False, "empty median should raise error")
    except ValueError as e:
        assert_that("empty sequence" in str(e), f"unexpected error: {e}")


@test("non-numeric data error handling")
def test_non_numeric_errors():
    mixed_data = [1, 'hello', 3, 'world']

    try:
        P(mixed_data).stats.sum()
        assert_that(False, "mixed data sum should raise error")
    except TypeError as e:
        assert_that("non-numeric" in str(e), f"unexpected error: {e}")


@test("complex statistical chains")
def test_complex_chains():
    # analyze salary data with multiple operations
    salary_analysis = (P(people)
                       .select(lambda p: p.salary)
                       .stats.normalize()
                       .stats.rank()
                       .stats.cumulative_sum())

    result = salary_analysis.to.list()
    assert_that(len(result) == len(people), f"chain result length wrong: {len(result)}")

    # rolling analysis of time series
    rolling_analysis = (P(time_series)
                        .stats.rolling_average(3)
                        .stats.diff(1)
                        .to.list())

    assert_that(len(rolling_analysis) > 0, f"rolling analysis should produce results: {len(rolling_analysis)}")


@test("ordered enumerable integration")
def test_ordered_integration():
    # test lag_in_order vs regular lag
    ordered_people = P(people).order_by(lambda p: p.age)

    regular_lag = P(people).stats.lag(1).to.list()
    ordered_lag = ordered_people.stats.lag_in_order(1).to.list()

    # should be different due to ordering
    assert_that(regular_lag != ordered_lag, "regular and ordered lag should differ")
    assert_that(len(regular_lag) == len(ordered_lag),
                f"lag lengths should match: {len(regular_lag)} vs {len(ordered_lag)}")


@test("welford algorithm accuracy")
def test_welford_accuracy():
    # test internal welford calculation against manual
    large_numbers = from_range(1, 1000).to.list()

    # use internal method
    result = P(large_numbers).stats.average()
    manual_avg = sum(large_numbers) / len(large_numbers)

    assert_that(abs(result - manual_avg) < 0.001, f"welford average wrong: {result} vs {manual_avg}")

    std_result = P(large_numbers).stats.std_dev()
    manual_mean = sum(large_numbers) / len(large_numbers)
    manual_var = sum((x - manual_mean) ** 2 for x in large_numbers) / len(large_numbers)
    manual_std = math.sqrt(manual_var)

    assert_that(abs(std_result - manual_std) < 0.001, f"welford std dev wrong: {std_result} vs {manual_std}")


@test("performance with large datasets")
def test_performance():
    print()

    large_data = from_range(1, 100000).select(lambda x: x * 1.5).to.list()

    start = time.perf_counter()
    avg = P(large_data).stats.average()
    duration_avg = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    std = P(large_data).stats.std_dev()
    duration_std = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    ranks = P(large_data).stats.rank().to.list()
    duration_rank = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    normalized = P(large_data).stats.normalize().to.list()
    duration_norm = (time.perf_counter() - start) * 1000

    print(f"    {suite._c.grey}├─> average on 100k items: {duration_avg:.2f}ms{suite._c.reset}")
    print(f"    {suite._c.grey}├─> std_dev on 100k items: {duration_std:.2f}ms{suite._c.reset}")
    print(f"    {suite._c.grey}├─> rank on 100k items: {duration_rank:.2f}ms{suite._c.reset}")
    print(f"    {suite._c.grey}└─> normalize on 100k items: {duration_norm:.2f}ms{suite._c.reset}")

    assert_that(avg > 0, f"average should be positive: {avg}")
    assert_that(std > 0, f"std dev should be positive: {std}")
    assert_that(len(ranks) == 100000, f"rank count wrong: {len(ranks)}")
    assert_that(len(normalized) == 100000, f"normalized count wrong: {len(normalized)}")


@test("generated data statistical analysis")
def test_with_generated_data():
    schema = {
        'measurements': [{
            '_qen_items': {
                'value': ('pyfloat', {'min_value': 10, 'max_value': 100}),
                'category': {'_qen_provider': 'choice', 'from': ['A', 'B', 'C']},
                'timestamp': ('pyint', {'min_value': 1000, 'max_value': 2000})
            },
            '_qen_count': 100
        }]
    }

    data = from_schema(schema, seed=42).take(1).to.first()['measurements']

    # comprehensive statistical analysis
    values = P(data).select(lambda x: x['value'])

    avg = values.stats.average()
    median = values.stats.median()
    std = values.stats.std_dev()

    assert_that(10 <= avg <= 100, f"average should be in range: {avg}")
    assert_that(10 <= median <= 100, f"median should be in range: {median}")
    assert_that(std >= 0, f"std dev should be non-negative: {std}")

    # category analysis - fixed approach
    grouped_data = P(data).group.group_by(lambda x: x['category'])

    # convert the dictionary result to a list of statistics
    stats_list = []
    for category, items in grouped_data.items():
        stats_list.append({
            'category': category,
            'count': len(items),
            'avg_value': P(items).stats.average(lambda x: x['value'])
        })

    assert_that(len(stats_list) <= 3, f"should have max 3 categories: {len(stats_list)}")

    total_count = sum(s['count'] for s in stats_list)
    assert_that(total_count == 100, f"total count should be 100: {total_count}")


@test("statistical edge cases")
def test_edge_cases():
    # single value
    single = P([42])
    assert_that(single.stats.average() == 42, "single average should be value")
    assert_that(single.stats.std_dev() == 0, "single std dev should be 0")
    assert_that(single.stats.median() == 42, "single median should be value")

    # identical values
    identical = P([5, 5, 5, 5, 5])
    assert_that(identical.stats.std_dev() == 0, "identical std dev should be 0")
    assert_that(identical.stats.mode() == 5, "identical mode should be value")

    # two values
    two_vals = P([10, 20])
    assert_that(two_vals.stats.median() == 15, "two value median should be average")
    assert_that(two_vals.stats.percentile(50) == 15, "two value 50th percentile should be 15")


@test("numpy integration")
def test_numpy_integration():
    # test that numpy optimizations work when available
    numeric_data = from_range(1, 1000).to.list()

    result = P(numeric_data).stats.sum()
    expected = sum(numeric_data)

    assert_that(result == expected, f"numpy sum integration failed: {result} vs {expected}")

    # test with mixed data that should fall back to python
    mixed = [1, 2, 3, 'string', 5]
    try:
        P(mixed).stats.sum()
        assert_that(False, "mixed data should raise error")
    except TypeError:
        pass  # expected


@test("financial analysis simulation")
def test_financial_analysis():
    # simulate stock prices with returns analysis
    prices = [100, 102, 98, 105, 103, 110, 108, 115, 112, 120]

    returns = (P(prices)
               .stats.diff(1)
               .select(lambda x: x / 100 if x is not None else None)  # convert to returns
               .where(lambda x: x is not None))

    avg_return = returns.stats.average()
    return_volatility = returns.stats.std_dev()

    assert_that(avg_return > -1 and avg_return < 1, f"return should be reasonable: {avg_return}")
    assert_that(return_volatility >= 0, f"volatility should be non-negative: {return_volatility}")

    # cumulative returns
    cumulative = returns.stats.cumulative_sum().to.list()
    assert_that(len(cumulative) == len(returns.to.list()), f"cumulative length wrong: {len(cumulative)}")


@test("time series analysis patterns")
def test_time_series_patterns():
    # seasonal pattern simulation
    seasonal_data = []
    for i in range(48):  # 4 years of quarterly data
        base = 100
        trend = i * 0.5
        seasonal = 10 * math.sin(2 * math.pi * i / 4)  # quarterly pattern
        noise = (i % 7) - 3  # some noise
        seasonal_data.append(base + trend + seasonal + noise)

    # analyze with multiple window sizes
    ma_4 = P(seasonal_data).stats.rolling_average(4).to.list()  # annual moving average
    ma_12 = P(seasonal_data).stats.rolling_average(12).to.list()  # 3-year moving average

    assert_that(len(ma_4) == len(seasonal_data) - 3, f"4-period MA length wrong: {len(ma_4)}")
    assert_that(len(ma_12) == len(seasonal_data) - 11, f"12-period MA length wrong: {len(ma_12)}")

    # trend analysis
    first_half_avg = P(seasonal_data[:24]).stats.average()
    second_half_avg = P(seasonal_data[24:]).stats.average()

    assert_that(second_half_avg > first_half_avg, f"should show upward trend: {first_half_avg} vs {second_half_avg}")


if __name__ == "__main__":
    suite.run(title="pinqy stats operations test")