import time
import suite
from dgen import from_schema
# import new factory and support classes
from pinqy import from_iterable, from_range, repeat, empty, generate, Enumerable, create_tree_from_flat, ParseResult, P

# --- setup ---
test = suite.test
assert_that = suite.assert_that

# --- test data generation schemas ---

# schema for simple objects used in many tests
object_schema = {
    'id': ('pyint', {'min_value': 1, 'max_value': 100}),
    'name': 'word',
    'value': ('pyfloat', {'min_value': 0, 'max_value': 1000}),
    'category': {'_qen_provider': 'choice', 'from': ['a', 'b', 'c']}
}

# schema for users and posts to test joins
users_schema = {
    'user_id': ('pyint', {'min_value': 1, 'max_value': 5}),
    'name': 'name'
}

posts_schema = {
    'post_id': 'uuid4',
    'user_id': ('pyint', {'min_value': 1, 'max_value': 8}),  # includes users who don't exist for outer joins
    'content': 'sentence'
}


# --- factory function tests ---

@test("factory functions create enumerables correctly")
def test_factories():
    # from_iterable
    data = [1, 2, 3]
    en_list = from_iterable(data)
    assert_that(isinstance(en_list, Enumerable), "from_iterable should be an enumerable instance")
    assert_that(en_list.to.list() == data, "from_iterable.to.list() should return the original data")

    # from_range
    en_range = from_range(10, 5)
    expected_range = [10, 11, 12, 13, 14]
    assert_that(en_range.to.count() == 5, "from_range count should be 5")
    assert_that(en_range.to.list() == expected_range, "from_range data should match the expected range")

    # repeat
    en_repeat = repeat("a", 3)
    assert_that(en_repeat.to.list() == ["a", "a", "a"], "repeat data should be a list of repeated items")

    # empty
    en_empty = empty()
    assert_that(en_empty.to.count() == 0, "count of an empty enumerable should be 0")
    assert_that(en_empty.to.list() == [], "to.list() on empty should be an empty list")

    # generate
    en_gen = generate(lambda: 'x', 4)
    assert_that(en_gen.to.list() == ['x', 'x', 'x', 'x'],
                "generate should call the function the correct number of times")

    # p / P alias
    assert_that(P([1, 2, 3]).to.list() == [1, 2, 3], "P() alias should work like from_iterable")


# --- core operation tests ---

@test("core filtering and projection methods work correctly")
def test_core_filter_project():
    # where
    evens = from_range(1, 10).where(lambda x: x % 2 == 0)
    assert_that(evens.to.list() == [2, 4, 6, 8, 10], "where should only contain even numbers")

    # select
    names = from_schema(object_schema, seed=42).take(3).select(lambda x: x['name'])
    assert_that(all(isinstance(n, str) for n in names), "select: all projected items should be strings")
    assert_that(names.to.count() == 3, "select: count should remain the same after projection")

    # select_many
    schema = {'id': 'pyint', 'tags': [{'name': 'word', '_qen_count': 3}]}
    data = from_schema(schema, seed=1).take(2)
    all_tags = data.select_many(lambda x: x['tags'])
    assert_that(all_tags.to.count() == 6, "select_many: should be a flat list of all tags")
    assert_that(isinstance(all_tags.to.first(), dict), "select_many: items should be the tag dictionaries")

    # of_type
    mixed_data = from_iterable([1, "a", 2.0, 3, "b"])
    strings = mixed_data.of_type(str)
    assert_that(strings.to.list() == ["a", "b"], "of_type should filter by type")


@test("core partitioning and sequence methods work")
def test_core_partition_sequence():
    # skip and take
    page = from_range(0, 100).skip(10).take(5)
    assert_that(page.to.list() == [10, 11, 12, 13, 14], "skip/take should represent the second page of 5 items")

    # take_while
    tw_data = from_iterable([2, 4, 5, 8, 1]).take_while(lambda x: x % 2 == 0)
    assert_that(tw_data.to.list() == [2, 4], "take_while should stop at the first non-match")

    # skip_while
    sw_data = from_iterable([2, 4, 5, 8, 1]).skip_while(lambda x: x % 2 == 0)
    assert_that(sw_data.to.list() == [5, 8, 1], "skip_while should skip all initial matches")

    # reverse
    assert_that(from_range(1, 3).reverse().to.list() == [3, 2, 1], "reverse should be in reverse order")

    # append and prepend
    result = from_iterable([2, 3]).prepend(1).append(4)
    assert_that(result.to.list() == [1, 2, 3, 4], "append/prepend elements should be in the correct order")

    # default_if_empty
    assert_that(empty().default_if_empty(-1).to.list() == [-1],
                "default_if_empty should provide a value for an empty enumerable")
    assert_that(P([1, 2]).default_if_empty(-1).to.list() == [1, 2],
                "default_if_empty should not affect a non-empty enumerable")


# --- ordered enumerable tests ---

@test("order_by, then_by, and other ordered operations work correctly")
def test_ordered_enumerable():
    data = from_schema(object_schema, seed=10).take(100).set.distinct(key_selector=lambda x: x['id'])

    # order_by and then_by
    sorted_data = data.order_by(lambda x: x['category']).then_by_descending(lambda x: x['value']).to.list()
    assert_that(sorted_data[0]['category'] <= sorted_data[-1]['category'],
                "primary sort (category) should be ascending")
    group_a = [d for d in sorted_data if d['category'] == 'a']
    for i in range(len(group_a) - 1):
        assert_that(group_a[i]['value'] >= group_a[i + 1]['value'], "secondary sort (value) should be descending")

    # merge_with
    sorted1 = P([1, 3, 5, 7]).order_by(lambda x: x)
    sorted2 = P([2, 4, 6]).order_by(lambda x: x)
    merged = sorted1.merge_with(sorted2)
    assert_that(merged.to.list() == [1, 2, 3, 4, 5, 6, 7], "merge_with should correctly merge two sorted sequences")

    # lag_in_order and lead_in_order
    sorted_values = data.order_by(lambda x: x['value'])
    lagged = sorted_values.lag_in_order(1, fill_value={'value': -1}).select(lambda x: x['value']).to.list()
    lead = sorted_values.lead_in_order(1, fill_value={'value': -1}).select(lambda x: x['value']).to.list()
    original_values = sorted_values.select(lambda x: x['value']).to.list()

    assert_that(lagged[0] == -1, "lag_in_order should use fill value for first item")
    assert_that(lagged[1] == original_values[0], "lag_in_order should shift elements correctly")
    assert_that(lead[-1] == -1, "lead_in_order should use fill value for last item")
    assert_that(lead[0] == original_values[1], "lead_in_order should shift elements correctly")


# --- set operation tests ---

@test("set operations produce correct results")
def test_set_ops():
    a = from_iterable([1, 2, 2, 3])
    b = from_iterable([3, 4, 5, 5])

    # distinct
    assert_that(a.set.distinct().to.list() == [1, 2, 3], "distinct should only contain unique elements")

    # union
    assert_that(a.set.union(b).to.list() == [1, 2, 3, 4, 5], "union should be the distinct union of both lists")

    # intersect
    assert_that(a.set.intersect(b).to.list() == [3], "intersect should be the intersection of both lists")

    # except_
    assert_that(a.set.except_(b).to.list() == [1, 2], "except_ should be the elements in A but not in B")

    # concat
    assert_that(a.set.concat(b).to.list() == [1, 2, 2, 3, 3, 4, 5, 5],
                "concat should join two sequences without removing duplicates")

    # symmetric_difference
    assert_that(a.set.symmetric_difference(b).to.list() == [1, 2, 4, 5],
                "symmetric_difference should be elements in one but not both")


@test("boolean set operations return correct booleans")
def test_set_boolean_ops():
    a = P([1, 2])
    b = P([1, 2, 3])
    c = P([3, 4])

    assert_that(a.set.is_subset_of(b), "is_subset_of should be true")
    assert_that(not b.set.is_subset_of(a), "is_subset_of should be false")
    assert_that(b.set.is_superset_of(a), "is_superset_of should be true")
    assert_that(a.set.is_proper_subset_of(b), "is_proper_subset_of should be true")
    assert_that(not b.set.is_proper_subset_of(b), "is_proper_subset_of should be false")
    assert_that(a.set.is_disjoint_with(c), "is_disjoint_with should be true")
    assert_that(not a.set.is_disjoint_with(b), "is_disjoint_with should be false")


# --- join operation tests ---

@test("join operations correlate sequences correctly")
def test_joins():
    users = P([{'id': 1, 'name': 'A'}, {'id': 2, 'name': 'B'}, {'id': 3, 'name': 'C'}])  # C has no posts
    posts = P([{'pid': 101, 'uid': 1}, {'pid': 102, 'uid': 1}, {'pid': 103, 'uid': 2},
               {'pid': 104, 'uid': 4}])  # uid=4 has no user

    # inner join
    inner = users.join.join(posts, lambda u: u['id'], lambda p: p['uid'], lambda u, p: (u['name'], p['pid']))
    assert_that(inner.to.list() == [('A', 101), ('A', 102), ('B', 103)], "inner join is incorrect")

    # left join
    left = users.join.left_join(posts, lambda u: u['id'], lambda p: p['uid'],
                                lambda u, p: (u['name'], p['pid'] if p else None))
    assert_that(left.to.list() == [('A', 101), ('A', 102), ('B', 103), ('C', None)], "left_join is incorrect")

    # group_join
    group = users.join.group_join(posts, lambda u: u['id'], lambda p: p['uid'],
                                  lambda u, ps: (u['name'], ps.select(lambda p: p['pid']).to.list()))
    expected_group = [('A', [101, 102]), ('B', [103]), ('C', [])]
    assert_that(group.to.list() == expected_group, "group_join is incorrect")

    # cross_join
    cross = P([1, 2]).join.cross_join(['a', 'b'])
    assert_that(cross.to.list() == [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')], "cross_join is incorrect")


# --- zip ---
@test("zip operations combine sequences correctly")
def test_zip_ops():
    a = from_iterable([1, 2, 3])
    b = ['a', 'b']

    # zip_with
    result1 = a.zip.zip_with(b, lambda n, s: f"{n}-{s}")
    assert_that(result1.to.list() == ['1-a', '2-b'], "zip_with should stop at shorter sequence")

    # zip_longest_with
    result2 = a.zip.zip_longest_with(b, lambda n, s: f"{n}-{s}", default_self=0, default_other='z')
    assert_that(result2.to.list() == ['1-a', '2-b', '3-z'], "zip_longest_with should pad with defaults")


# --- grouping and windowing tests ---

@test("grouping and windowing methods structure data correctly")
def test_grouping_windowing():
    # group_by
    groups = from_schema(object_schema, seed=3).take(50).group.group_by(lambda x: x['category'])
    assert_that('a' in groups and 'b' in groups and 'c' in groups, "group_by should create a key for each category")
    assert_that(all(isinstance(v, list) for v in groups.values()), "group_by values should be lists")
    assert_that(all(item['category'] == 'a' for item in groups['a']), "all items in group 'a' must have category 'a'")

    # chunk
    chunks = from_range(1, 10).group.chunk(3)
    assert_that(chunks.to.count() == 4, "chunk: 10 items chunked by 3 should produce 4 chunks")
    assert_that(chunks.to.list()[-1] == [10], "chunk: the last chunk should contain the remainder")
    assert_that(chunks.to.list()[0] == [1, 2, 3], "chunk: the first chunk should be full")

    # window
    windows = from_range(1, 5).group.window(3)
    expected_windows = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    assert_that(windows.to.list() == expected_windows, "window should produce correct sliding windows")

    # pairwise
    pairs = from_range(1, 4).group.pairwise()
    assert_that(pairs.to.list() == [(1, 2), (2, 3), (3, 4)], "pairwise should produce overlapping pairs")

    # partition
    evens, odds = from_range(1, 10).group.partition(lambda x: x % 2 == 0)
    assert_that(evens == [2, 4, 6, 8, 10], "partition should separate true items")
    assert_that(odds == [1, 3, 5, 7, 9], "partition should separate false items")


@test("pivot creates a correct pivot table")
def test_pivot():
    sales_data = from_iterable([
        {'year': 2023, 'product': 'A', 'sales': 100}, {'year': 2023, 'product': 'B', 'sales': 150},
        {'year': 2024, 'product': 'A', 'sales': 120}, {'year': 2023, 'product': 'A', 'sales': 50},
        {'year': 2024, 'product': 'B', 'sales': 200},
    ])
    result = sales_data.group.pivot(
        row_selector=lambda r: r['year'],
        column_selector=lambda r: r['product'],
        aggregator=lambda group: group.stats.sum(lambda s: s['sales'])
    )
    expected = {2023: {'A': 150, 'B': 150}, 2024: {'A': 120, 'B': 200}}
    assert_that(result == expected, "pivot table structure or values are incorrect")
    assert_that(result[2023]['A'] == 150, "aggregated value for (2023, A) is wrong")


# --- statistical operation tests ---

@test("statistical aggregate methods compute correct values")
def test_stats_aggregates():
    data = from_iterable([10, 20, 30, 40, 50])
    assert_that(data.to.count() == 5, "count is incorrect")
    assert_that(data.stats.sum() == 150, "sum is incorrect")
    assert_that(data.stats.average() == 30, "average is incorrect")
    assert_that(data.stats.min() == 10, "min is incorrect")
    assert_that(data.stats.max() == 50, "max is incorrect")
    assert_that(data.stats.median() == 30, "median for odd-sized list is incorrect")
    assert_that(from_iterable([10, 20, 30, 40]).stats.median() == 25, "median for even-sized list is incorrect")
    assert_that(P([1, 2, 2, 3, 4, 7, 9]).stats.mode() == 2, "mode is incorrect")


@test("statistical methods raise errors on empty sequences")
def test_stats_errors():
    en = empty()
    try:
        en.stats.average()
        assert_that(False, "average on empty list should raise ValueError")
    except ValueError:
        pass
    try:
        en.stats.min()
        assert_that(False, "min on empty list should raise ValueError")
    except ValueError:
        pass


@test("cumulative and ranking statistical methods work")
def test_stats_cumulative_ranking():
    # cumulative
    cum_sum = P([1, 2, 3, 4]).stats.cumulative_sum().to.list()
    assert_that(cum_sum == [1, 3, 6, 10], "cumulative_sum is incorrect")

    # ranking
    rank_data = P([10, 30, 20, 30])
    ranks = rank_data.stats.rank(ascending=False).to.list()
    dense_ranks = rank_data.stats.dense_rank(ascending=False).to.list()
    assert_that(ranks == [4, 1, 3, 1], "rank should handle ties by skipping ranks")
    assert_that(dense_ranks == [3, 1, 2, 1], "dense_rank should handle ties without gaps")


# --- terminal operation tests ---

@test("terminal operations correctly finalize the query")
def test_terminals():
    data = from_schema(object_schema, seed=7).take(10)
    # any / all
    assert_that(data.to.any(lambda x: x['category'] == 'a'), "any should find matching element")
    assert_that(not data.to.all(lambda x: x['value'] > 500), "all should fail if one element doesn't match")
    # first / first_or_default
    assert_that(data.to.first(lambda x: x['category'] == 'a')['category'] == 'a', "first should return match")
    assert_that(data.to.first_or_default(lambda x: x['category'] == 'd', default="not found") == "not found",
                "first_or_default should return default")
    # single
    try:
        data.to.single()
        assert_that(False, "single on multi-element list should raise ValueError")
    except ValueError:
        pass
    assert_that(P([5]).to.single() == 5, "single on single-element list should return the element")
    # dict
    data_dict = data.set.distinct(lambda x: x['id']).to.dict(lambda x: x['id'], lambda x: x['name'])
    assert_that(isinstance(data_dict, dict), "to.dict should produce a dictionary")

    # aggregate
    total = P(range(1, 5)).to.aggregate(lambda acc, x: acc + x, seed=0)
    assert_that(total == 10, "aggregate should compute the correct reduction")


# --- combinatorics tests ---
@test("combinatorics methods generate correct sequences")
def test_combinatorics():
    data = P(['a', 'b', 'c'])
    # permutations
    perms = data.comb.permutations(2)
    assert_that(perms.to.count() == 6, "permutations count is incorrect")
    assert_that(P(perms.to.list()).to.set() == {('a', 'b'), ('a', 'c'), ('b', 'a'), ('b', 'c'), ('c', 'a'), ('c', 'b')},
                "permutations result is incorrect")

    # combinations
    combs = data.comb.combinations(2)
    assert_that(combs.to.count() == 3, "combinations count is incorrect")

    # power_set
    pset = P([1, 2]).comb.power_set()
    assert_that(pset.to.count() == 4, "power_set should have 2^n elements")
    assert_that(P(pset.to.list()).to.set() == {(), (1,), (2,), (1, 2)}, "power_set result is incorrect")


# --- utility and functional tests ---

@test("utility methods provide correct functional transformations")
def test_utils():
    # flatten_deep
    nested = from_iterable([1, [2, [3, 'a']], (4, 5)])
    assert_that(nested.util.flatten_deep().to.list() == [1, 2, 3, 'a', 4, 5],
                "flatten_deep should flatten all nested iterables")

    # pipe_through
    def filter_evens(en): return en.where(lambda x: x % 2 == 0)

    def square_all(en): return en.select(lambda x: x * x)

    result = from_range(1, 10).util.pipe_through(filter_evens, square_all).to.list()
    assert_that(result == [4, 16, 36, 64, 100], "pipe_through should apply both functions in sequence")

    # apply_if / apply_when
    data = from_range(1, 10)
    result1 = data.util.apply_if(True, lambda en: en.where(lambda x: x > 5)).to.list()
    assert_that(result1 == [6, 7, 8, 9, 10], "apply_if should be applied when condition is true")
    result2 = data.util.apply_if(False, lambda en: en.where(lambda x: x > 5)).to.list()
    assert_that(result2 == list(range(1, 11)), "apply_if should be skipped when condition is false")

    # side_effect vs for_each
    side_effect_list = []
    processed_data = data.util.side_effect(lambda x: side_effect_list.append(x)).select(lambda x: x * 2)
    assert_that(len(side_effect_list) == 0, "side_effect should be lazy and not execute yet")
    final_list = processed_data.to.list()
    assert_that(len(side_effect_list) == 10, "side_effect should execute upon materialization")
    assert_that(final_list == [2, 4, 6, 8, 10, 12, 14, 16, 18, 20], "side_effect should not alter the sequence")

    for_each_list = []
    data.util.for_each(lambda x: for_each_list.append(x))
    assert_that(len(for_each_list) == 10, "for_each should execute immediately")

    # memoize
    call_count = 0

    def gen_with_side_effect():
        nonlocal call_count
        for i in range(3):
            call_count += 1
            yield i

    memoized = from_iterable(gen_with_side_effect()).util.memoize()
    assert_that(call_count == 0, "memoize should be lazy")
    memoized.to.list()  # first materialization
    assert_that(call_count == 3, "memoize should call generator on first run")
    memoized.to.list()  # second materialization
    assert_that(call_count == 3, "memoize should used cached results on second run")


@test("try_parse separates successes and failures")
def test_try_parse():
    def parse_int(s):
        try:
            return True, int(s)
        except ValueError:
            return False, s

    result = P(['1', 'a', '2', 'b', '3']).util.try_parse(parse_int)
    assert_that(isinstance(result, ParseResult), "should return a ParseResult object")
    assert_that(result.successes == [1, 2, 3], "successes list is incorrect")
    assert_that(result.failures == ['a', 'b'], "failures list is incorrect")


# --- tree operation tests ---

@test("tree operations build and flatten hierarchies")
def test_tree_ops():
    flat_data = [
        {'id': 1, 'parent_id': None, 'name': 'root'}, {'id': 2, 'parent_id': 1, 'name': 'child_a'},
        {'id': 3, 'parent_id': 1, 'name': 'child_b'}, {'id': 4, 'parent_id': 2, 'name': 'grandchild_a1'},
    ]
    tree = from_iterable(flat_data).tree.build_tree(
        key_selector=lambda x: x['id'], parent_key_selector=lambda x: x['parent_id']
    ).to.list()
    assert_that(len(tree) == 1, "build_tree: should be one root node")
    root = tree[0]
    assert_that(len(root.children) == 2, "build_tree: root should have two children")
    child_a = from_iterable(root.children).to.first(lambda c: c.value['name'] == 'child_a')
    assert_that(len(child_a.children) == 1, "build_tree: child_a should have one child")

    # recursive_select
    tree_data = [{'id': 1, 'children': [{'id': 2, 'children': [{'id': 4}]}, {'id': 3}]}]
    flattened = from_iterable(tree_data).tree.recursive_select(
        child_selector=lambda node: node.get('children')
    ).select(lambda node: node['id']).to.list()
    assert_that(flattened == [1, 2, 4, 3], "recursive_select should perform a depth-first traversal")


# --- performance and scaling tests ---

@test("performance scaling test for common operations")
def test_performance_scaling():
    sizes = [1000, 10000, 100000, 1000000]

    def query_filter_select(data):
        return (data
                .where(lambda x: x['category'] == 'a' and x['value'] > 500)
                .select(lambda x: x['name'])
                .to.list())

    def query_sort_take(data):
        return (data
                .order_by_descending(lambda x: x['value'])
                .take(10)
                .to.list())

    def query_group_aggregate(data):
        return (data
        .group.group_by_with_aggregate(
            key_selector=lambda x: x['category'],
            element_selector=lambda x: x['value'],
            result_selector=lambda cat, vals: vals.stats.average()
        ))

    queries = {
        "Filter & Select (O(n))": query_filter_select,
        "Sort & Take (O(n log n))": query_sort_take,
        "Group & Aggregate (O(n))": query_group_aggregate
    }

    print(f"\n    {suite._c.warn}--- WARNING: Large-scale performance test starting ---{suite._c.reset}")
    print(f"    {suite._c.grey}This will be slow and consume significant memory (esp. at >1M records).{suite._c.reset}")

    query_col_width = 30
    size_headers = [f"{s:,}" for s in sizes]
    # calc column widths for alignment, adding padding for "ms" and spaces
    data_col_widths = [len(h) + 4 for h in size_headers]

    header_parts = [f"{'Query':<{query_col_width}}"]
    for i, h in enumerate(size_headers):
        header_parts.append(h.center(data_col_widths[i]))

    header = " | ".join(header_parts)
    print(f"\n    {suite._c.info}{header}{suite._c.reset}")
    print(f"    {suite._c.grey}{'-' * len(header)}{suite._c.reset}")

    for name, query_func in queries.items():
        result_parts = [f"{name:<{query_col_width}}"]
        for i, size in enumerate(sizes):
            print(f"    {suite._c.grey}└─> Preparing {size:,} records for '{name}'...{suite._c.reset}", end='\r')
            data = from_schema(object_schema, seed=42).take(size)

            start = time.perf_counter()
            query_func(data)
            duration = (time.perf_counter() - start) * 1000  # in ms

            result_str = f"{duration:.2f}ms".rjust(data_col_widths[i])
            result_parts.append(result_str)

        print(" | ".join(result_parts))

    print(f"    {suite._c.info}{'-' * len(header)}{suite._c.reset}")
    assert_that(True, "Performance test completed")


# --- run the suite ---
if __name__ == "__main__":
    suite.run(title="pinqy test")