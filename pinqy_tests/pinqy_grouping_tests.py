import suite
from dgen import from_schema
from pinqy import P, from_range, Enumerable, empty

# --- setup ---
test = suite.test
assert_that = suite.assert_that

# --- test data & schemas ---
object_schema = {
    'id': ('pyint', {'min_value': 1, 'max_value': 10}),
    'name': 'word',
    'category': {'_qen_provider': 'choice', 'from': ['a', 'b', 'c']},
    'value': ('pyfloat', {'min_value': 10, 'max_value': 100}),
    'is_active': {'_qen_provider': 'choice', 'from': [True, False]}
}
sales_schema = {
    'year': {'_qen_provider': 'choice', 'from': [2022, 2023]},
    'region': {'_qen_provider': 'choice', 'from': ['na', 'eu']},
    'product_id': ('pyint', {'min_value': 1, 'max_value': 3}),
    'sales': ('pyint', {'min_value': 100, 'max_value': 1000})
}


# --- group_by ---

@test("group_by correctly groups items by a key selector")
def test_group_by_core():
    data = from_schema(object_schema, seed=42).take(50)
    grouped = data.group.group_by(lambda x: x['category'])

    assert_that(isinstance(grouped, dict), "group_by should return a dictionary")
    assert_that(set(grouped.keys()) == {'a', 'b', 'c'}, "all categories should be present as keys")
    for cat, items in grouped.items():
        assert_that(isinstance(items, list), "each group should be a list")
        assert_that(all(item['category'] == cat for item in items),
                    f"all items in group '{cat}' must have that category")


@test("group_by handles different key types")
def test_group_by_key_types():
    # numeric key
    grouped_by_even_odd = from_range(0, 10).group.group_by(lambda x: x % 2)
    assert_that(set(grouped_by_even_odd.keys()) == {0, 1}, "keys should be 0 and 1 for even/odd")
    assert_that(grouped_by_even_odd[0] == [0, 2, 4, 6, 8], "group 0 should contain even numbers")
    assert_that(grouped_by_even_odd[1] == [1, 3, 5, 7, 9], "group 1 should contain odd numbers")

    # boolean key
    grouped_by_active = from_schema(object_schema, seed=1).take(20).group.group_by(lambda x: x['is_active'])
    assert_that(set(grouped_by_active.keys()) == {True, False}, "keys should be True and False")


@test("group_by handles edge cases")
def test_group_by_edges():
    # empty enumerable
    assert_that(empty().group.group_by(lambda x: x) == {}, "group_by on empty enumerable should be an empty dict")

    # all items map to one key
    grouped_single = P(['a', 'b', 'c']).group.group_by(lambda x: 'same_key')
    assert_that(list(grouped_single.keys()) == ['same_key'], "should have a single key")
    assert_that(grouped_single['same_key'] == ['a', 'b', 'c'], "all items should be in the single group")

    # each item maps to a unique key
    grouped_unique = from_range(0, 5).group.group_by(lambda x: x)
    assert_that(len(grouped_unique) == 5, "should have 5 unique groups")
    assert_that(grouped_unique[3] == [3], "each group should contain one item")


# --- group_by_multiple ---

@test("group_by_multiple correctly groups by a composite tuple key")
def test_group_by_multiple_core():
    data = from_schema(object_schema, seed=123).take(100)
    grouped = data.group.group_by_multiple(lambda x: x['category'], lambda x: x['is_active'])

    assert_that(isinstance(grouped, dict), "should return a dictionary")
    if grouped:
        first_key = next(iter(grouped.keys()))
        assert_that(isinstance(first_key, tuple) and len(first_key) == 2, "keys should be tuples of length 2")

        if ('a', True) in grouped:
            group_a_true = grouped[('a', True)]
            assert_that(all(item['category'] == 'a' and item['is_active'] for item in group_a_true),
                        "all items in group ('a', True) must match")


@test("group_by_multiple handles edge cases")
def test_group_by_multiple_edges():
    assert_that(empty().group.group_by_multiple(lambda x: x) == {},
                "group_by_multiple on empty should be an empty dict")


# --- group_by_with_aggregate ---

@test("group_by_with_aggregate correctly transforms groups")
def test_group_by_with_aggregate_core():
    data = from_schema(object_schema, seed=2).take(50)
    # scenario: get average value per category
    avg_values = data.group.group_by_with_aggregate(
        key_selector=lambda x: x['category'],
        element_selector=lambda x: x['value'],
        result_selector=lambda key, values: values.stats.average()
    )

    assert_that(set(avg_values.keys()) == {'a', 'b', 'c'}, "result should have category keys")
    assert_that(isinstance(avg_values['a'], float), "aggregated value should be a float average")

    # scenario: get names of active items per category
    active_names = data.group.group_by_with_aggregate(
        key_selector=lambda x: x['category'],
        element_selector=lambda x: x,
        result_selector=lambda key, items: items.where(lambda i: i['is_active']).select(lambda i: i['name']).to.list()
    )
    assert_that(isinstance(active_names['b'], list), "result should be a list of names")


@test("group_by_with_aggregate verifies enumerable is passed to selector")
def test_group_by_with_aggregate_enumerable_passing():
    def checker(key, values):
        assert_that(isinstance(values, Enumerable), "result_selector must receive an Enumerable instance")
        return values.to.count()

    from_range(0, 10).group.group_by_with_aggregate(
        key_selector=lambda x: x % 2,
        element_selector=lambda x: x,
        result_selector=checker
    )


# --- pivot ---

@test("pivot creates a correct pivot table-like structure")
def test_pivot_core():
    data = from_schema(sales_schema, seed=55).take(100)
    pivot_table = data.group.pivot(
        row_selector=lambda r: r['year'],
        column_selector=lambda r: r['region'],
        aggregator=lambda group: float(group.stats.sum(lambda s: s['sales']))
    )

    assert_that(set(pivot_table.keys()).issubset({2022, 2023}), "row keys must be a subset of possible years")
    for year, regions in pivot_table.items():
        assert_that(set(regions.keys()).issubset({'na', 'eu'}),
                    f"column keys for {year} must be a subset of possible regions")
        for region, total_sales in regions.items():
            assert_that(isinstance(total_sales, float), f"aggregated value for ({year}, {region}) must be a float")
            assert_that(total_sales >= 0, f"sales total should be non-negative: {total_sales}")


@test("pivot handles different aggregators")
def test_pivot_aggregators():
    data = from_schema(sales_schema, seed=56).take(100)
    pivot_table = data.group.pivot(
        row_selector=lambda r: r['year'],
        column_selector=lambda r: r['product_id'],
        aggregator=lambda group: group.to.count()
    )
    if 2022 in pivot_table:
        assert_that(set(pivot_table[2022].keys()).issubset({1, 2, 3}), "column keys should be product ids")
        if 1 in pivot_table[2022]:
            assert_that(isinstance(pivot_table[2022][1], int), "aggregated value should be an integer count")


@test("pivot handles empty enumerable")
def test_pivot_empty():
    assert_that(empty().group.pivot(lambda r: r, lambda c: c, lambda g: g.to.count()) == {},
                "pivot on empty enumerable should produce an empty dict")


# --- group_by_nested ---

@test("group_by_nested creates a two-level nested dictionary")
def test_group_by_nested_core():
    data = from_schema(sales_schema, seed=99).take(200)
    nested = data.group.group_by_nested(
        key_selector=lambda s: s['region'],
        sub_key_selector=lambda s: s['year']
    )
    assert_that(set(nested.keys()) == {'na', 'eu'}, "primary keys should be regions")
    assert_that(set(nested['na'].keys()) == {2022, 2023}, "secondary keys should be years")
    assert_that(isinstance(nested['na'][2022], list), "innermost value should be a list of items")
    assert_that(all(item['region'] == 'na' and item['year'] == 2022 for item in nested['na'][2022]),
                "items in nested group must match both keys")


# --- partition ---

@test("partition correctly splits a sequence into two lists")
def test_partition_core():
    evens, odds = from_range(0, 10).group.partition(lambda x: x % 2 == 0)
    assert_that(isinstance(evens, list) and isinstance(odds, list), "partition must return two lists")
    assert_that(evens == [0, 2, 4, 6, 8], "first list should contain items where predicate is true")
    assert_that(odds == [1, 3, 5, 7, 9], "second list should contain items where predicate is false")


@test("partition handles edge cases")
def test_partition_edges():
    all_true, none_false = from_range(0, 5).group.partition(lambda x: True)
    assert_that(all_true == [0, 1, 2, 3, 4], "all items should be in the true list")
    assert_that(none_false == [], "false list should be empty")

    none_true, all_false = from_range(0, 5).group.partition(lambda x: False)
    assert_that(none_true == [], "true list should be empty")
    assert_that(all_false == [0, 1, 2, 3, 4], "all items should be in the false list")

    empty_true, empty_false = empty().group.partition(lambda x: True)
    assert_that(empty_true == [] and empty_false == [], "partition on empty returns two empty lists")


# --- chunk & batched ---

@test("chunk splits a sequence into lists of a given size")
def test_chunk_core():
    chunks = from_range(0, 10).group.chunk(3).to.list()
    assert_that(len(chunks) == 4, "10 items chunked by 3 should produce 4 chunks")
    assert_that(chunks[0] == [0, 1, 2], "first chunk should be full")
    assert_that(chunks[3] == [9], "last chunk should contain the remainder")
    assert_that(isinstance(chunks[0], list), "chunks should be lists")


@test("chunk handles edge cases for size and length")
def test_chunk_edges():
    chunk = from_range(0, 5).group.chunk(10).to.list()
    assert_that(chunk == [[0, 1, 2, 3, 4]], "chunk size > length should produce one chunk")

    chunks = from_range(0, 9).group.chunk(3).to.list()
    assert_that(len(chunks) == 3, "perfectly divisible length should produce n/size chunks")
    assert_that(chunks[2] == [6, 7, 8], "last chunk of perfect division should be full")

    chunks = from_range(0, 3).group.chunk(1).to.list()
    assert_that(chunks == [[0], [1], [2]], "chunk size of 1 should wrap each item in a list")

    assert_that(empty().group.chunk(5).to.list() == [], "chunk on empty enumerable should be empty")


@test("chunk raises ValueError for invalid size")
def test_chunk_invalid_input():
    try:
        from_range(0, 10).group.chunk(0)
        assert_that(False, "chunk with size 0 should raise ValueError")
    except ValueError:
        pass

    try:
        from_range(0, 10).group.chunk(-1)
        assert_that(False, "chunk with negative size should raise ValueError")
    except ValueError:
        pass


@test("batched splits a sequence into tuples of a given size")
def test_batched_core():
    batches = from_range(0, 10).group.batched(3).to.list()
    assert_that(len(batches) == 4, "10 items batched by 3 should produce 4 batches")
    assert_that(batches[0] == (0, 1, 2), "first batch should be a full tuple")
    assert_that(batches[3] == (9,), "last batch should contain the remainder as a tuple")
    assert_that(isinstance(batches[0], tuple), "batches should be tuples")


# --- window ---

@test("window creates correct sliding windows")
def test_window_core():
    windows = from_range(1, 5).group.window(3).to.list()
    expected = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    assert_that(windows == expected, "window should produce correct sliding windows")
    assert_that(len(windows) == 3, "should produce (n - size + 1) windows")


@test("window handles edge cases for size and length")
def test_window_edges():
    assert_that(from_range(1, 3).group.window(4).to.list() == [], "window should be empty if size > length")

    assert_that(from_range(1, 3).group.window(3).to.list() == [[1, 2, 3]],
                "window should be one list if size == length")

    assert_that(from_range(1, 4).group.window(1).to.list() == [[1], [2], [3], [4]], "window size 1 should wrap each item")

    assert_that(empty().group.window(3).to.list() == [], "window on empty should be empty")


# --- pairwise ---

@test("pairwise creates correct overlapping pairs")
def test_pairwise_core():
    pairs = from_range(1, 5).group.pairwise().to.list()
    expected = [(1, 2), (2, 3), (3, 4), (4, 5)]
    assert_that(pairs == expected, "pairwise produces incorrect pairs")


@test("pairwise handles edge cases")
def test_pairwise_edges():
    assert_that(from_range(1, 1).group.pairwise().to.list() == [], "pairwise on list of 1 should be empty")
    assert_that(from_range(1, 2).group.pairwise().to.list() == [(1, 2)], "pairwise on list of 2 should be one pair")
    assert_that(empty().group.pairwise().to.list() == [], "pairwise on empty list should be empty")


# --- batch_by ---

@test("batch_by groups consecutive items with the same key")
def test_batch_by_core():
    data = P([1, 1, 2, 3, 3, 3, 2, 2, 1])
    batches = data.group.batch_by(lambda x: x).to.list()
    expected = [[1, 1], [2], [3, 3, 3], [2, 2], [1]]
    assert_that(batches == expected, "batch_by did not group consecutive items correctly")


@test("batch_by works with complex key selectors")
def test_batch_by_complex_key():
    data = from_range(1, 10)
    batches = data.group.batch_by(lambda x: x % 2 == 0).to.list()
    expected = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
    assert_that(batches == expected, "alternating keys should produce n batches of 1")
    assert_that(len(batches) == 10, "alternating keys should produce n batches")

    data2 = P([1, 3, 5, 2, 4, 8, 7, 9])
    batches2 = data2.group.batch_by(lambda x: x % 2 == 0).to.list()
    expected2 = [[1, 3, 5], [2, 4, 8], [7, 9]]
    assert_that(batches2 == expected2, "batch_by with key func should group consecutive similar keys")


@test("batch_by handles edge cases")
def test_batch_by_edges():
    assert_that(empty().group.batch_by(lambda x: x).to.list() == [], "batch_by on empty list should be empty")

    all_same = P([1, 1, 1, 1]).group.batch_by(lambda x: 1).to.list()
    assert_that(all_same == [[1, 1, 1, 1]], "batch_by on all same key should be one batch")

    all_diff = from_range(0, 4).group.batch_by(lambda x: x).to.list()
    assert_that(all_diff == [[0], [1], [2], [3]], "batch_by on all different keys should be n batches of 1")


# --- run the suite ---
if __name__ == "__main__":
    suite.run(title="pinqy grouping test")