import sys
import os
import re
import random
import math
from typing import Tuple, Optional, List, Any
from collections import deque, defaultdict

from pinqy import p, MemoizedEnumerable
import dgen
from suite import test, assert_that, run


@test("for_each should execute action eagerly and return original enumerable")
def test_for_each():
    results = []

    def collect_action(x): results.append(x * 2)

    original = p([1, 2, 3, 4])
    returned = original.util.for_each(collect_action)

    assert_that(returned is original, "should return original enumerable")
    assert_that(results == [2, 4, 6, 8], "should execute action immediately")
    assert_that(returned.where(lambda x: x > 2).to.list() == [3, 4], "chaining should work")


@test("run_length_encode should group consecutive identical elements")
def test_run_length_encode():
    data = p("aaabbc")
    result = data.util.run_length_encode().to.list()
    expected = [('a', 3), ('b', 2), ('c', 1)]
    assert_that(result == expected, f"expected {expected}, got {result}")

    numbers = p([1, 1, 2, 2, 2, 3, 1])
    num_result = numbers.util.run_length_encode().to.list()
    expected_nums = [(1, 2), (2, 3), (3, 1), (1, 1)]
    assert_that(num_result == expected_nums, f"numbers: expected {expected_nums}, got {num_result}")


@test("flatten should flatten nested sequences to specified depth")
def test_flatten():
    nested = p([1, [2, [3, 4]], 5])

    depth_1 = nested.util.flatten(1).to.list()
    assert_that(depth_1 == [1, 2, [3, 4], 5], f"depth 1: got {depth_1}")

    depth_2 = nested.util.flatten(2).to.list()
    assert_that(depth_2 == [1, 2, 3, 4, 5], f"depth 2: got {depth_2}")

    depth_0 = nested.util.flatten(0).to.list()
    assert_that(depth_0 == [1, [2, [3, 4]], 5], f"depth 0: got {depth_0}")


@test("flatten_deep should recursively flatten all nested structures")
def test_flatten_deep():
    deeply_nested = p([1, [2, [3, [4, [5]]]], 6])
    result = deeply_nested.util.flatten_deep().to.list()
    assert_that(result == [1, 2, 3, 4, 5, 6], f"got {result}")

    mixed = p([1, "hello", [2, ["nested", 3]], [4]])
    mixed_result = mixed.util.flatten_deep().to.list()
    assert_that(mixed_result == [1, "hello", 2, "nested", 3, 4], f"mixed: got {mixed_result}")


@test("transpose should transpose matrix-like structures")
def test_transpose():
    matrix = p([[1, 2, 3], [4, 5, 6]])
    result = matrix.util.transpose().to.list()
    expected = [[1, 4], [2, 5], [3, 6]]
    assert_that(result == expected, f"expected {expected}, got {result}")

    empty_matrix = p([])
    empty_result = empty_matrix.util.transpose().to.list()
    assert_that(empty_result == [], "empty matrix should transpose to empty")


@test("unzip should transform enumerable of tuples to tuple of enumerables")
def test_unzip():
    paired = p([('a', 1), ('b', 2), ('c', 3)])
    letters, numbers = paired.util.unzip()

    assert_that(letters.to.list() == ['a', 'b', 'c'], f"letters: got {letters.to.list()}")
    assert_that(numbers.to.list() == [1, 2, 3], f"numbers: got {numbers.to.list()}")

    empty = p([])
    empty_result = empty.util.unzip()
    assert_that(empty_result == tuple(), "empty should return empty tuple")


@test("intersperse should place separator between elements")
def test_intersperse():
    data = p([1, 2, 3, 4])
    result = data.util.intersperse(0).to.list()
    expected = [1, 0, 2, 0, 3, 0, 4]
    assert_that(result == expected, f"expected {expected}, got {result}")

    single = p([42])
    single_result = single.util.intersperse(99).to.list()
    assert_that(single_result == [42], "single element should remain unchanged")

    empty = p([])
    empty_result = empty.util.intersperse(5).to.list()
    assert_that(empty_result == [], "empty should remain empty")


@test("sample should return random subset with reproducible results")
def test_sample():
    data = p(list(range(100)))

    sample_a = data.util.sample(10, random_state=42).to.list()
    sample_b = data.util.sample(10, random_state=42).to.list()
    assert_that(sample_a == sample_b, "same seed should produce same results")
    assert_that(len(sample_a) == 10, f"should return 10 items, got {len(sample_a)}")

    sample_with_replacement = data.util.sample(5, replace=True, random_state=123).to.list()
    assert_that(len(sample_with_replacement) == 5, "replacement sampling should work")

    small_data = p([1, 2, 3])
    large_sample = small_data.util.sample(10, random_state=456).to.list()
    assert_that(len(large_sample) == 3, "can't sample more than available without replacement")


@test("stratified_sample should ensure representation across groups")
def test_stratified_sample():
    schema = {
        'group': {'_qen_provider': 'choice', 'from': ['A', 'B', 'C']},
        'value': ('pyint', {'min_value': 1, 'max_value': 100})
    }
    generator = dgen.from_schema(schema, seed=42)
    data = generator.take(30).to.list()

    enumerable_data = p(data)
    stratified = enumerable_data.util.stratified_sample(
        lambda x: x['group'],
        samples_per_group=3
    ).to.list()

    groups = p(stratified).group.group_by(lambda x: x['group'])
    assert_that(len(groups) <= 3, "should have at most 3 groups")
    for group_items in groups.values():
        assert_that(len(group_items) <= 3, f"each group should have at most 3 items")


@test("bootstrap_sample should generate multiple samples with replacement")
def test_bootstrap_sample():
    data = p([1, 2, 3, 4, 5])

    bootstrap_samples = data.util.bootstrap_sample(n_samples=5, sample_size=3).to.list()
    assert_that(len(bootstrap_samples) == 5, f"should generate 5 samples, got {len(bootstrap_samples)}")

    for sample in bootstrap_samples:
        sample_list = sample.to.list()
        assert_that(len(sample_list) == 3, f"each sample should have 3 items, got {len(sample_list)}")
        for item in sample_list:
            assert_that(item in [1, 2, 3, 4, 5], f"all items should be from original data")


@test("memoize should cache results lazily")
def test_memoize():
    counter = [0]

    def count_generator():
        counter[0] += 1
        return [1, 2, 3]

    from pinqy.enumerable import Enumerable
    original = Enumerable(count_generator)
    memoized = original.util.memoize()

    assert_that(counter[0] == 0, "should not execute until first access")

    first_result = memoized.to.list()
    assert_that(counter[0] == 1, "should execute on first access")

    second_result = memoized.to.list()
    assert_that(counter[0] == 1, "should not execute again on second access")
    assert_that(first_result == second_result, "results should be identical")


@test("pipe should pass enumerable to external function")
def test_pipe():
    def custom_transform(enumerable, multiplier=1, add=0):
        return enumerable.select(lambda x: x * multiplier + add).to.list()

    data = p([1, 2, 3])
    result = data.util.pipe(custom_transform, multiplier=2, add=10)
    expected = [12, 14, 16]
    assert_that(result == expected, f"expected {expected}, got {result}")


@test("side_effect should perform lazy side effects without modification")
def test_side_effect():
    side_effects = []

    def log_effect(x): side_effects.append(f"processing {x}")

    data = p([1, 2, 3, 4])
    result = (data
              .where(lambda x: x % 2 == 0)
              .util.side_effect(log_effect)
              .select(lambda x: x * 2)
              .to.list())

    assert_that(result == [4, 8], f"transformation should work correctly: {result}")
    assert_that(side_effects == ["processing 2", "processing 4"], f"side effects: {side_effects}")


@test("topological_sort should handle dependency graphs correctly")
def test_topological_sort():
    tasks = [
        {'id': 'A', 'deps': []},
        {'id': 'B', 'deps': ['A']},
        {'id': 'C', 'deps': ['A']},
        {'id': 'D', 'deps': ['B', 'C']}
    ]

    def get_dependencies(task):
        deps = []
        for dep_id in task['deps']:
            for t in tasks:
                if t['id'] == dep_id:
                    deps.append(t)
        return deps

    sorted_tasks = p(tasks).util.topological_sort(get_dependencies).to.list()
    sorted_ids = [t['id'] for t in sorted_tasks]

    assert_that(sorted_ids.index('A') < sorted_ids.index('B'), "A should come before B")
    assert_that(sorted_ids.index('A') < sorted_ids.index('C'), "A should come before C")
    assert_that(sorted_ids.index('B') < sorted_ids.index('D'), "B should come before D")
    assert_that(sorted_ids.index('C') < sorted_ids.index('D'), "C should come before D")



@test("topological_sort should detect cycles")
def test_topological_sort_cycle_detection():
    cyclic_tasks = [
        {'id': 'X', 'deps': ['Y']},
        {'id': 'Y', 'deps': ['Z']},
        {'id': 'Z', 'deps': ['X']}
    ]

    def get_cyclic_deps(task):
        deps = []
        for dep_id in task['deps']:
            for t in cyclic_tasks:
                if t['id'] == dep_id:
                    deps.append(t)
        return deps

    try:
        p(cyclic_tasks).util.topological_sort(get_cyclic_deps).to.list()
        assert_that(False, "should have raised ValueError for cycle")
    except ValueError as e:
        assert_that("cycle" in str(e).lower(), f"error message should mention cycle: {e}")




@test("pipe_through should apply operations sequentially")
def test_pipe_through():
    def double(enum): return enum.select(lambda x: x * 2)

    def add_ten(enum): return enum.select(lambda x: x + 10)

    def filter_large(enum): return enum.where(lambda x: x > 15)

    result = (p([1, 2, 3, 4, 5])
              .util.pipe_through(double, add_ten, filter_large)
              .to.list())

    expected = [16, 18, 20]  # (1*2+10=12<15), (2*2+10=14<15), (3*2+10=16>=15), etc.
    assert_that(result == expected, f"expected {expected}, got {result}")


@test("apply_if should conditionally apply operations")
def test_apply_if():
    data = p([3, 1, 4, 1, 5])

    sorted_result = data.util.apply_if(True, lambda e: e.order_by(lambda x: x)).to.list()
    assert_that(sorted_result == [1, 1, 3, 4, 5], f"should sort when condition is True")

    unsorted_result = data.util.apply_if(False, lambda e: e.order_by(lambda x: x)).to.list()
    assert_that(unsorted_result == [3, 1, 4, 1, 5], f"should not sort when condition is False")


@test("apply_when should conditionally apply based on predicate")
def test_apply_when():
    short_data = p([1, 2, 3])
    long_data = p([1, 2, 3, 4, 5, 6])

    def has_more_than_4(enum): return enum.to.count() > 4

    def sort_desc(enum): return enum.order_by_descending(lambda x: x)

    short_result = short_data.util.apply_when(has_more_than_4, sort_desc).to.list()
    assert_that(short_result == [1, 2, 3], "short data should remain unchanged")

    long_result = long_data.util.apply_when(has_more_than_4, sort_desc).to.list()
    assert_that(long_result == [6, 5, 4, 3, 2, 1], "long data should be sorted descending")


@test("lazy_where should filter with access to full context")
def test_lazy_where():
    data = p([1, 2, 3, 4, 5])

    # filter elements greater than average
    above_avg = data.util.lazy_where(lambda item, all_items: item > sum(all_items) / len(all_items)).to.list()
    expected = [4, 5]  # avg is 3, so 4 and 5 are above
    assert_that(above_avg == expected, f"expected {expected}, got {above_avg}")

    # filter elements in top half by position
    top_half = data.util.lazy_where(lambda item, all_items: all_items.index(item) >= len(all_items) // 2).to.list()
    expected_top = [3, 4, 5]  # indices 2, 3, 4
    assert_that(top_half == expected_top, f"expected {expected_top}, got {top_half}")


@test("unfold should generate sequences from seeds")
def test_unfold():
    # fibonacci-like sequence from each starting number
    data = p([1, 2])

    def fib_unfolder(seed):
        if seed >= 100: return None
        return (seed, seed + 1)  # value, next_seed

    result = data.util.unfold(lambda x: x, fib_unfolder).to.list()
    expected = [1, 2, 3, 4, 5, 6]  # stops when reaching 100+ (which doesn't happen in this short sequence)

    # more accurately, each seed generates until >= 100
    # seed 1: generates 1, 2, 3, ..., 99
    # seed 2: generates 2, 3, 4, ..., 99
    # this would be a lot of numbers, so let's limit the unfolder

    def limited_unfolder(seed):
        if seed > 5: return None
        return (seed, seed + 1)

    limited_result = data.util.unfold(lambda x: x, limited_unfolder).to.list()
    # seed 1: 1, 2, 3, 4, 5
    # seed 2: 2, 3, 4, 5
    expected_limited = [1, 2, 3, 4, 5, 2, 3, 4, 5]
    assert_that(limited_result == expected_limited, f"expected {expected_limited}, got {limited_result}")


@test("try_parse should separate successful and failed parses")
def test_try_parse():
    def int_parser(s: str) -> Tuple[bool, Optional[int]]:
        try:
            return True, int(s)
        except ValueError:
            return False, None

    data = p(["123", "abc", "456", "def", "789"])
    result = data.util.try_parse(int_parser)

    assert_that(result.successes == [123, 456, 789], f"successes: {result.successes}")
    assert_that(result.failures == ["abc", "def"], f"failures: {result.failures}")
    assert_that(result.success_count == 3, f"success count: {result.success_count}")
    assert_that(result.failure_count == 2, f"failure count: {result.failure_count}")
    assert_that(result.has_failures == True, "should have failures")


@test("parse_or_default should use default for failed parses")
def test_parse_or_default():
    def float_parser(s: str) -> Tuple[bool, Optional[float]]:
        try:
            return True, float(s)
        except ValueError:
            return False, None

    data = p(["1.5", "invalid", "2.7", "also_invalid"])
    result = data.util.parse_or_default(float_parser, default_value=0.0).to.list()
    expected = [1.5, 0.0, 2.7, 0.0]
    assert_that(result == expected, f"expected {expected}, got {result}")


@test("compose should apply functions sequentially")
def test_compose():
    def multiply_by_2(enum): return enum.select(lambda x: x * 2)

    def add_1(enum): return enum.select(lambda x: x + 1)

    def filter_even(enum): return enum.where(lambda x: x % 2 == 0)

    result = (p([1, 2, 3, 4])
              .util.compose(multiply_by_2, add_1, filter_even)
              .to.list())

    # [1,2,3,4] -> [2,4,6,8] -> [3,5,7,9] -> [] (no even numbers)
    expected = []
    assert_that(result == expected, f"expected {expected}, got {result}")


@test("apply_functions should apply multiple functions to each element")
def test_apply_functions():
    functions = [lambda x: x * 2, lambda x: x + 10, lambda x: x ** 2]
    data = p([1, 2])

    result = data.util.apply_functions(functions).to.list()
    # for each element, apply each function
    # 1: [2, 11, 1], 2: [4, 12, 4]
    expected = [2, 11, 1, 4, 12, 4]
    assert_that(result == expected, f"expected {expected}, got {result}")


@test("memoize_advanced should provide sophisticated caching")
def test_memoize_advanced():
    counter = [0]

    def counting_generator():
        for i in range(3):
            counter[0] += 1
            yield i

    memo = MemoizedEnumerable(counting_generator)

    first_two = list(memo)[:2]
    assert_that(counter[0] == 3, f"should have called generator 3 times for full iteration, got {counter[0]}")

    all_items = list(memo)  # should use cache, no additional calls
    assert_that(counter[0] == 3, f"should still have called generator 3 times total, got {counter[0]}")
    assert_that(all_items == [0, 1, 2], f"should get all items: {all_items}")


@test("combinatorics permutations should generate all arrangements")
def test_permutations():
    data = p("abc")
    perms_2 = data.comb.permutations(2).to.list()
    expected = [('a', 'b'), ('a', 'c'), ('b', 'a'), ('b', 'c'), ('c', 'a'), ('c', 'b')]
    assert_that(len(perms_2) == 6, f"should have 6 permutations of length 2, got {len(perms_2)}")
    assert_that(set(perms_2) == set(expected), f"permutations should match expected")

    perms_all = data.comb.permutations().to.list()
    assert_that(len(perms_all) == 6, f"should have 6 total permutations, got {len(perms_all)}")


@test("combinatorics combinations should generate unique selections")
def test_combinations():
    data = p("abcd")
    combs_2 = data.comb.combinations(2).to.list()
    expected_len = 6  # C(4,2) = 6
    assert_that(len(combs_2) == expected_len, f"should have 6 combinations, got {len(combs_2)}")

    # check uniqueness and no repetition
    unique_combs = set(combs_2)
    assert_that(len(unique_combs) == len(combs_2), "all combinations should be unique")

    # check ordering (combinations should be in lexicographic order)
    assert_that(('a', 'b') in combs_2, "should contain ('a', 'b')")
    assert_that(('b', 'a') not in combs_2, "should not contain reverse ('b', 'a')")


@test("combinatorics combinations_with_replacement should allow repeated elements")
def test_combinations_with_replacement():
    data = p("ab")
    combs_wr_2 = data.comb.combinations_with_replacement(2).to.list()
    expected = [('a', 'a'), ('a', 'b'), ('b', 'b')]
    assert_that(len(combs_wr_2) == 3, f"should have 3 combinations with replacement, got {len(combs_wr_2)}")
    assert_that(set(combs_wr_2) == set(expected), f"expected {expected}, got {combs_wr_2}")


@test("combinatorics power_set should generate all subsets")
def test_power_set():
    data = p([1, 2])
    power_set = data.comb.power_set().to.list()
    expected = [(), (1,), (2,), (1, 2)]
    assert_that(len(power_set) == 4, f"should have 4 subsets, got {len(power_set)}")
    assert_that(set(power_set) == set(expected), f"expected {expected}, got {power_set}")


@test("combinatorics cartesian_product should generate all combinations with other sequences")
def test_cartesian_product():
    data1 = p([1, 2])
    data2 = ['a', 'b']
    data3 = ['x']

    cart_prod = data1.comb.cartesian_product(data2, data3).to.list()
    expected = [
        (1, 'a', 'x'), (1, 'b', 'x'),
        (2, 'a', 'x'), (2, 'b', 'x')
    ]
    assert_that(len(cart_prod) == 4, f"should have 4 products, got {len(cart_prod)}")
    assert_that(set(cart_prod) == set(expected), f"expected {expected}, got {cart_prod}")


@test("combinatorics binomial_coefficient should calculate n choose k")
def test_binomial_coefficient():
    data = p(range(5))  # [0, 1, 2, 3, 4], so n=5

    coeff_2 = data.comb.binomial_coefficient(2)
    expected_2 = 10  # C(5,2) = 10
    assert_that(coeff_2 == expected_2, f"C(5,2) should be {expected_2}, got {coeff_2}")

    coeff_0 = data.comb.binomial_coefficient(0)
    expected_0 = 1  # C(5,0) = 1
    assert_that(coeff_0 == expected_0, f"C(5,0) should be {expected_0}, got {coeff_0}")

    coeff_5 = data.comb.binomial_coefficient(5)
    expected_5 = 1  # C(5,5) = 1
    assert_that(coeff_5 == expected_5, f"C(5,5) should be {expected_5}, got {coeff_5}")


@test("combined operations should work together seamlessly")
def test_combined_operations():
    # generate test data with dgen
    schema = {
        'id': ('pyint', {'min_value': 1, 'max_value': 100}),
        'name': 'name',
        'category': {'_qen_provider': 'choice', 'from': ['A', 'B', 'C']},
        'score': ('pyfloat', {'min_value': 0.0, 'max_value': 100.0, 'right_digits': 2})
    }
    generator = dgen.from_schema(schema, seed=123)
    test_data = generator.take(20).to.list()

    # complex chained operation combining multiple util methods
    result = (p(test_data)
    .util.stratified_sample(lambda x: x['category'], 3)  # sample from each category
    .util.side_effect(lambda x: x.update({'processed': True}) or x)  # add flag
    .where(lambda x: x['score'] > 50.0)  # filter high scores
    .order_by(lambda x: x['score'])
    .util.pipe_through(
        lambda enum: enum.select(lambda x: (x['name'], x['score'])),  # extract name and score
        lambda enum: enum.util.intersperse(('---', 0.0))  # add separators
    ))

    final_result = result.to.list()

    # verify structure
    assert_that(len(final_result) > 0, "should have some results")

    # check that separators were added - only if there's more than one data item
    separators = [item for item in final_result if item[0] == '---']
    data_items = [item for item in final_result if item[0] != '---']

    if len(data_items) > 1:
        assert_that(len(separators) == len(data_items) - 1,
                    f"should have {len(data_items) - 1} separators for {len(data_items)} items, got {len(separators)}")
    else:
        assert_that(len(separators) == 0, "should have no separators for single item or empty result")



@test("complex combinatorics chains should handle realistic scenarios")
def test_complex_combinatorics_scenarios():
    # scenario: team formation from different skill groups
    developers = ['alice', 'bob', 'charlie']
    designers = ['diana', 'eve']
    managers = ['frank', 'grace']

    # all possible 2-person teams from developers
    dev_pairs = p(developers).comb.combinations(2).to.list()
    assert_that(len(dev_pairs) == 3, f"should have 3 dev pairs, got {len(dev_pairs)}")

    # cross-functional teams (1 dev, 1 designer, 1 manager)
    cross_functional = (p(developers)
                        .comb.cartesian_product(designers, managers)
                        .select(lambda team: {'dev': team[0], 'designer': team[1], 'manager': team[2]})
                        .to.list())

    expected_teams = 3 * 2 * 2  # 12 possible teams
    assert_that(len(cross_functional) == expected_teams,
                f"should have {expected_teams} teams, got {len(cross_functional)}")

    # check a specific team exists
    alice_diana_frank = any(
        team['dev'] == 'alice' and team['designer'] == 'diana' and team['manager'] == 'frank'
        for team in cross_functional
    )
    assert_that(alice_diana_frank, "should contain alice-diana-frank team")


@test("utility methods should handle edge cases gracefully")
def test_edge_cases():
    # empty sequences
    empty = p([])

    assert_that(empty.util.flatten().to.list() == [], "empty flatten should remain empty")
    assert_that(empty.util.run_length_encode().to.list() == [], "empty rle should remain empty")
    assert_that(empty.util.sample(5).to.list() == [], "empty sample should remain empty")
    assert_that(empty.util.intersperse('x').to.list() == [], "empty intersperse should remain empty")

    # single element
    single = p([42])
    assert_that(single.util.intersperse(0).to.list() == [42], "single element intersperse unchanged")
    assert_that(single.util.run_length_encode().to.list() == [(42, 1)], "single element rle")
    assert_that(single.util.flatten().to.list() == [42], "single element flatten unchanged")

    # type-mixed sequences
    mixed = p([1, 'hello', [2, 3], {'key': 'value'}])
    flattened_mixed = mixed.util.flatten().to.list()
    expected_mixed = [1, 'hello', 2, 3, {'key': 'value'}]
    assert_that(flattened_mixed == expected_mixed, f"mixed flatten: expected {expected_mixed}, got {flattened_mixed}")


@test("performance characteristics should be reasonable for medium datasets")
def test_performance_characteristics():
    import time

    # generate larger dataset
    large_schema = {
        'id': ('uuid4', {}),
        'value': ('pyint', {'min_value': 1, 'max_value': 1000}),
        'category': {'_qen_provider': 'choice', 'from': list('ABCDEFGHIJ')},
        'timestamp': ('date_time_this_year', {})
    }
    generator = dgen.from_schema(large_schema, seed=42)
    large_data = generator.take(1000).to.list()

    # test various operations for reasonable performance
    start_time = time.time()

    # complex chain with multiple util operations
    result = (p(large_data)
              .util.stratified_sample(lambda x: x['category'], 10)
              .where(lambda x: x['value'] > 500)
              .util.run_length_encode()
              .select(lambda x: x[0])  # get the items, not counts
              .util.flatten_deep()
              .order_by(lambda x: x['value'])
              .take(50)
              .to.list())

    end_time = time.time()
    execution_time = end_time - start_time

    assert_that(execution_time < 1.0, f"should complete in under 1 second, took {execution_time:.3f}s")
    assert_that(len(result) <= 50, f"should have at most 50 results, got {len(result)}")


@test("nested utility operations should maintain data integrity")
def test_nested_utility_operations():
    # create hierarchical test data
    nested_schema = {
        'department': {'_qen_provider': 'choice', 'from': ['eng', 'sales', 'hr']},
        'team': {'_qen_provider': 'choice', 'from': ['alpha', 'beta', 'gamma']},
        'employees': [{
            '_qen_items': {
                'name': 'name',
                'salary': ('pyint', {'min_value': 40000, 'max_value': 120000}),
                'skills': [{
                    '_qen_items': 'word',
                    '_qen_count': (2, 5)
                }]
            },
            '_qen_count': (3, 8)
        }]
    }
    generator = dgen.from_schema(nested_schema, seed=789)
    dept_data = generator.take(5).to.list()

    result = (p(dept_data)
              .select(lambda d: {
                  'dept': d['department'],
                  'employee_count': len(d['employees']),
                  'avg_salary': sum(emp['salary'] for emp in d['employees']) / len(d['employees']),
                  'employees': d['employees']
              })
              .util.side_effect(lambda x: None)  # no-op side effect
              .select_many(lambda d: [
                  {
                      'dept': d['dept'],
                      'employee': emp['name'],
                      'salary': emp['salary'],
                      'skills': emp['skills']
                  }
                  for emp in d['employees']
              ])
              .to.list())

    assert_that(len(result) > 0, "should have employee records")

    # verify structure
    for emp_record in result:
        assert_that('dept' in emp_record, "should have department")
        assert_that('employee' in emp_record, "should have employee name")
        assert_that('skills' in emp_record, "should have skills")
        assert_that('salary' in emp_record, "should have salary")


@test("utility methods should be chainable with other pinqy operations")
def test_chainability_with_core_operations():
    # test data: products with categories and ratings
    product_schema = {
        'name': ('word', {}),
        'category': {'_qen_provider': 'choice', 'from': ['electronics', 'books', 'clothing']},
        'rating': ('pyfloat', {'min_value': 1.0, 'max_value': 5.0, 'right_digits': 1}),
        'price': ('pyfloat', {'min_value': 10.0, 'max_value': 500.0, 'right_digits': 2}),
        'tags': [{
            '_qen_items': 'word',
            '_qen_count': (1, 4)
        }]
    }
    generator = dgen.from_schema(product_schema, seed=456)
    products = generator.take(25).to.list()

    # comprehensive chain mixing core operations with utility methods
    result = (p(products)
              .where(lambda p: p['rating'] >= 3.5)  # core: filter
              .util.stratified_sample(lambda p: p['category'], 3)  # util: sample
              .select(lambda p: {  # core: transform
        'name': p['name'],
        'category': p['category'],
        'price_rating_ratio': p['price'] / p['rating'],
        'tag_string': ' '.join(p['tags'])
    })
              .order_by(lambda p: p['price_rating_ratio'])  # core: sort
              .util.pipe_through(  # util: pipeline
        lambda enum: enum.take(10),  # core: limit
        lambda enum: enum.util.intersperse({'separator': True}),  # util: intersperse
        lambda enum: enum.where(lambda x: 'separator' not in x)  # core: filter separators back out
    )
              .group.group_by(lambda p: p['category'])  # accessor: group
              .items())  # dict items

    result_dict = dict(result)

    assert_that(len(result_dict) <= 3, "should have at most 3 categories")

    total_products = sum(len(products) for products in result_dict.values())
    assert_that(total_products <= 10, f"should have at most 10 total products after take(10)")


@test("error handling should be robust for malformed inputs")
def test_error_handling():
    # test with various problematic inputs
    problematic_data = p([None, 1, "string", [1, 2], {'key': 'value'}, 42.5])

    # operations that should handle mixed types gracefully
    try:
        # this should not crash, even with mixed types
        result = (problematic_data
                  .util.side_effect(lambda x: None)  # no-op
                  .where(lambda x: x is not None)
                  .to.list())

        assert_that(None not in result, "should have filtered out None values")
        assert_that(len(result) > 0, "should have some non-None values")
    except Exception as e:
        assert_that(False, f"should handle mixed types gracefully, but got: {e}")

    # test empty operations
    empty_result = (p([])
                    .util.run_length_encode()
                    .util.flatten()
                    .to.list())
    assert_that(empty_result == [], "empty chain should remain empty")


@test("memory efficiency should be maintained with lazy operations")
def test_lazy_evaluation_memory_efficiency():
    # create a large conceptual dataset but only process what's needed
    counter = [0]

    def expensive_generator():
        for i in range(10000):  # large range
            counter[0] += 1
            yield {'id': i, 'value': i * 2, 'category': f"cat_{i % 5}"}

    from pinqy.enumerable import Enumerable
    large_enum = Enumerable(lambda: list(expensive_generator()))

    # chain operations but only take first 5
    result = (large_enum
              .util.side_effect(lambda x: None)  # lazy side effect
              .where(lambda x: x['value'] > 10)
              .util.sample(3, random_state=42)  # this will force evaluation
              .to.list())

    # the expensive_generator should have been called because sample() needs the full data
    assert_that(counter[0] == 10000, f"generator should have been called 10000 times for sample, got {counter[0]}")
    assert_that(len(result) == 3, f"should have exactly 3 sampled items, got {len(result)}")


@test("string-specific utility operations should handle text processing")
def test_string_processing():
    text_data = p(["hello world", "pinqy library", "functional programming", "lazy evaluation"])

    # word extraction and processing
    words = (text_data
             .select_many(lambda text: text.split())  # core: flatten words
             .util.run_length_encode()  # util: count consecutive duplicates (there won't be any)
             .select(lambda x: x[0])  # extract words from (word, count) tuples
             .util.intersperse("--")  # util: add separators
             .where(lambda w: w != "--")  # core: remove separators again
             .set.distinct()  # accessor: unique words
             .order_by(lambda w: len(w))  # core: sort by length
             .to.list())

    expected_unique_words = ["hello", "world", "pinqy", "library", "functional", "programming", "lazy", "evaluation"]
    result_words = set(words)
    expected_set = set(expected_unique_words)

    assert_that(result_words == expected_set, f"expected words {expected_set}, got {result_words}")


if __name__ == "__main__":
    run(title="pinqy utility methods comprehensive test suite")