import time
import suite
from dgen import from_schema
# import new factory and support classes
from pinqy import from_iterable, from_range, repeat, empty, Enumerable, create_tree_from_flat, ParseResult

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

@test("from_iterable creates an enumerable from a list")
def test_from_iterable():
    data = [1, 2, 3]
    en = from_iterable(data)
    assert_that(isinstance(en, Enumerable), "should be an enumerable instance")
    assert_that(en.to.list() == data, "to.list() should return the original data")


@test("from_range creates an enumerable of a specific range and count")
def test_from_range():
    en = from_range(10, 5)
    expected = [10, 11, 12, 13, 14]
    assert_that(en.to.count() == 5, "count should be 5")
    assert_that(en.to.list() == expected, "data should match the expected range")


@test("repeat creates an enumerable with a single repeated item")
def test_repeat():
    en = repeat("a", 3)
    assert_that(en.to.list() == ["a", "a", "a"], "data should be a list of repeated items")


@test("empty creates an enumerable with zero items")
def test_empty():
    en = empty()
    assert_that(en.to.count() == 0, "count of an empty enumerable should be 0")
    assert_that(en.to.list() == [], "to.list() on empty should be an empty list")


# --- core operation tests ---

@test("where filters elements based on a predicate")
def test_where():
    data = from_range(1, 10)
    evens = data.where(lambda x: x % 2 == 0)
    assert_that(evens.to.list() == [2, 4, 6, 8, 10], "should only contain even numbers")


@test("select projects each element to a new form")
def test_select():
    data = from_schema(object_schema, seed=42).take(3)
    names = data.select(lambda x: x['name'])
    assert_that(all(isinstance(n, str) for n in names), "all projected items should be strings")
    assert_that(names.to.count() == 3, "count should remain the same after projection")


@test("select_many projects and flattens a sequence")
def test_select_many():
    schema = {'id': 'pyint', 'tags': [{'name': 'word', '_qen_count': 3}]}
    data = from_schema(schema, seed=1).take(2)  # [ {id:1, tags:[{..},{..},{..}]}, {id:2, tags:[...]} ]
    all_tags = data.select_many(lambda x: x['tags'])
    assert_that(all_tags.to.count() == 6, "should be a flat list of all tags")
    assert_that(isinstance(all_tags.to.first(), dict), "items should be the tag dictionaries")


@test("order_by and then_by sort elements correctly")
def test_order_by_then_by():
    data = from_schema(object_schema, seed=10).take(100).set.distinct(key_selector=lambda x: x['id'])
    sorted_data = data.order_by(lambda x: x['category']).then_by_descending(lambda x: x['value']).to.list()

    # check primary sort
    assert_that(sorted_data[0]['category'] <= sorted_data[-1]['category'],
                "primary sort (category) should be ascending")
    # check secondary sort within a group
    group_a = [d for d in sorted_data if d['category'] == 'a']
    for i in range(len(group_a) - 1):
        assert_that(group_a[i]['value'] >= group_a[i + 1]['value'], "secondary sort (value) should be descending")


@test("skip and take correctly partition the sequence")
def test_skip_take():
    data = from_range(0, 100)
    page = data.skip(10).take(5)
    assert_that(page.to.list() == [10, 11, 12, 13, 14], "should represent the second page of 5 items")


@test("reverse inverts the order of a sequence")
def test_reverse():
    data = from_range(1, 3)
    assert_that(data.reverse().to.list() == [3, 2, 1], "should be in reverse order")


@test("append and prepend add elements correctly")
def test_append_prepend():
    data = from_iterable([2, 3])
    result = data.prepend(1).append(4)
    assert_that(result.to.list() == [1, 2, 3, 4], "elements should be in the correct order")


# --- set operation tests ---

@test("distinct returns unique elements")
def test_distinct():
    data = from_iterable([1, 2, 2, 3, 1, 3])
    assert_that(data.set.distinct().to.list() == [1, 2, 3],
                "should only contain unique elements in order of first appearance")


@test("union produces the distinct set of two sequences")
def test_union():
    a = from_iterable([1, 2, 3])
    b = from_iterable([3, 4, 5])
    assert_that(a.set.union(b).to.list() == [1, 2, 3, 4, 5], "should be the union of both lists")


@test("intersect produces the common elements of two sequences")
def test_intersect():
    a = from_iterable([1, 2, 3])
    b = from_iterable([3, 4, 5])
    assert_that(a.set.intersect(b).to.list() == [3], "should be the intersection of both lists")


@test("except_ produces the set difference of two sequences")
def test_except():
    a = from_iterable([1, 2, 3])
    b = from_iterable([3, 4, 5])
    assert_that(a.set.except_(b).to.list() == [1, 2], "should be the elements in A but not in B")


@test("concat joins two sequences without removing duplicates")
def test_concat():
    a = from_iterable([1, 2])
    b = from_iterable([2, 3])
    assert_that(a.set.concat(b).to.list() == [1, 2, 2, 3], "should be the simple concatenation of both lists")


# --- join operation tests ---

@test("join correctly performs an inner join")
def test_join():
    users = from_schema(users_schema, seed=1).take(5).set.distinct(lambda u: u['user_id'])
    posts = from_schema(posts_schema, seed=2).take(20)

    joined = users.join.join(
        posts,
        outer_key_selector=lambda u: u['user_id'],
        inner_key_selector=lambda p: p['user_id'],
        result_selector=lambda u, p: {'name': u['name'], 'content': p['content']}
    )

    assert_that(joined.to.any(), "join should produce results")
    assert_that('name' in joined.to.first() and 'content' in joined.to.first(),
                "result selector should format the output")


@test("left_join includes all elements from the left sequence")
def test_left_join():
    # user_id 5 will have no posts
    users = from_iterable([{'user_id': 1, 'name': 'A'}, {'user_id': 5, 'name': 'B'}])
    posts = from_iterable([{'post_id': 101, 'user_id': 1}])

    joined = users.join.left_join(
        posts,
        lambda u: u['user_id'],
        lambda p: p['user_id'],
        lambda u, p: {'name': u['name'], 'post_id': p['post_id'] if p else None}
    )

    results = joined.to.list()
    assert_that(len(results) == 2, "should be two results, one for each user")
    user_b_result = from_iterable(results).to.single(lambda r: r['name'] == 'B')
    assert_that(user_b_result['post_id'] is None, "user with no posts should have a null post_id")

# --- zip ---

@test("zip_with correctly combines sequences using the .zip accessor")
def test_zip_accessor():
    a = from_iterable([1, 2, 3])
    b = ['a', 'b', 'c']
    result = a.zip.zip_with(b, lambda n, s: f"{n}-{s}")
    assert_that(result.to.list() == ['1-a', '2-b', '3-c'], "zipped output is incorrect")

# --- grouping and windowing tests ---

@test("group_by correctly groups items by a key")
def test_group_by():
    data = from_schema(object_schema, seed=3).take(50)
    groups = data.group.group_by(lambda x: x['category'])

    assert_that('a' in groups and 'b' in groups and 'c' in groups, "should create a key for each category")
    assert_that(all(isinstance(v, list) for v in groups.values()), "values of the group should be lists")

    # verify that all items in a group have the correct category
    assert_that(all(item['category'] == 'a' for item in groups['a']), "all items in group 'a' must have category 'a'")


@test("chunk splits the sequence into fixed-size lists")
def test_chunk():
    data = from_range(1, 10)  # 10 items
    chunks = data.group.chunk(3)
    assert_that(chunks.to.count() == 4, "10 items chunked by 3 should produce 4 chunks")
    assert_that(chunks.to.list()[-1] == [10], "the last chunk should contain the remainder")
    assert_that(chunks.to.list()[0] == [1, 2, 3], "the first chunk should be full")


@test("window creates sliding windows of elements")
def test_window():
    data = from_range(1, 5)
    windows = data.group.window(3)
    expected = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    assert_that(windows.to.list() == expected, "should produce correct sliding windows")

@test("pivot creates a correct pivot table")
def test_pivot():
    sales_data = from_iterable([
        {'year': 2023, 'product': 'A', 'sales': 100},
        {'year': 2023, 'product': 'B', 'sales': 150},
        {'year': 2024, 'product': 'A', 'sales': 120},
        {'year': 2023, 'product': 'A', 'sales': 50},
        {'year': 2024, 'product': 'B', 'sales': 200},
    ])

    result = sales_data.group.pivot(
        row_selector=lambda r: r['year'],
        column_selector=lambda r: r['product'],
        aggregator=lambda group: group.stats.sum(lambda s: s['sales'])
    )

    expected = {
        2023: {'A': 150, 'B': 150},
        2024: {'A': 120, 'B': 200}
    }

    assert_that(result == expected, "pivot table structure or values are incorrect")
    assert_that(result[2023]['A'] == 150, "aggregated value for (2023, A) is wrong")

# --- statistical operation tests ---

@test("statistical methods compute correct values")
def test_stats():
    data = from_iterable([10, 20, 30, 40, 50])
    assert_that(data.to.count() == 5, "count is incorrect")
    assert_that(data.stats.sum() == 150, "sum is incorrect")
    assert_that(data.stats.average() == 30, "average is incorrect")
    assert_that(data.stats.min() == 10, "min is incorrect")
    assert_that(data.stats.max() == 50, "max is incorrect")
    assert_that(data.stats.median() == 30, "median for odd-sized list is incorrect")
    assert_that(from_iterable([10, 20, 30, 40]).stats.median() == 25, "median for even-sized list is incorrect")


@test("statistical methods raise errors on empty sequences")
def test_stats_errors():
    en = empty()
    try:
        en.stats.average()
        assert_that(False, "average on empty list should raise ValueError")
    except ValueError:
        pass  # expected
    try:
        en.stats.min()
        assert_that(False, "min on empty list should raise ValueError")
    except ValueError:
        pass  # expected


# --- terminal operation tests ---

@test("terminal operations correctly finalize the query")
def test_terminals():
    data = from_schema(object_schema, seed=7).take(10)

    # any
    assert_that(data.to.any(lambda x: x['category'] == 'a'), "any should find matching element")
    assert_that(not data.to.any(lambda x: x['category'] == 'd'), "any should not find non-existent element")

    # all
    assert_that(data.to.all(lambda x: x['value'] >= 0), "all should verify condition for all elements")
    assert_that(not data.to.all(lambda x: x['value'] > 500), "all should fail if one element doesn't match")

    # first / first_or_default
    first_a = data.to.first(lambda x: x['category'] == 'a')
    assert_that(first_a['category'] == 'a', "first should return the first matching element")
    assert_that(data.to.first_or_default(lambda x: x['category'] == 'd', default="not found") == "not found",
                "first_or_default should return default value")

    data_dict = data.set.distinct(lambda x: x['id']).to.dict(key_selector=lambda x: x['id'],
                                                             value_selector=lambda x: x['name'])
    assert_that(isinstance(data_dict, dict), "to_dict should produce a dictionary")
    assert_that(len(data_dict) > 0, "dictionary should not be empty")


# --- advanced functional & tree operation tests ---

@test("build_tree creates a correct hierarchical structure")
def test_build_tree():
    flat_data = [
        {'id': 1, 'parent_id': None, 'name': 'root'},
        {'id': 2, 'parent_id': 1, 'name': 'child_a'},
        {'id': 3, 'parent_id': 1, 'name': 'child_b'},
        {'id': 4, 'parent_id': 2, 'name': 'grandchild_a1'},
    ]
    tree = from_iterable(flat_data).tree.build_tree(
        key_selector=lambda x: x['id'],
        parent_key_selector=lambda x: x['parent_id']
    ).to.list()

    assert_that(len(tree) == 1, "should be one root node")
    root = tree[0]
    assert_that(root.value['name'] == 'root', "root value is incorrect")
    assert_that(len(root.children) == 2, "root should have two children")
    child_a = from_iterable(root.children).to.first(lambda c: c.value['name'] == 'child_a')
    assert_that(len(child_a.children) == 1, "child_a should have one child")
    assert_that(child_a.children[0].value['name'] == 'grandchild_a1', "grandchild value is incorrect")


@test("recursive_select flattens a tree structure")
def test_recursive_select():
    tree_data = [
        {'id': 1, 'children': [
            {'id': 2, 'children': [{'id': 4}]},
            {'id': 3}
        ]}
    ]
    flattened = from_iterable(tree_data).tree.recursive_select(
        child_selector=lambda node: node.get('children')
    ).select(lambda node: node['id']).to.list()

    assert_that(flattened == [1, 2, 4, 3], "should perform a depth-first traversal")


@test("flatten_deep recursively flattens nested lists")
def test_flatten_deep():
    nested = from_iterable([1, [2, [3, 'a']], (4, 5)])
    flat = nested.util.flatten_deep().to.list()
    assert_that(flat == [1, 2, 3, 'a', 4, 5], "should flatten all nested iterables")


@test("pipe_through applies a sequence of operations")
def test_pipe_through():
    def filter_evens(en):
        return en.where(lambda x: x % 2 == 0)

    def square_all(en):
        return en.select(lambda x: x * x)

    result = from_range(1, 10).util.pipe_through(filter_evens, square_all).to.list()
    assert_that(result == [4, 16, 36, 64, 100], "should apply both functions in sequence")


@test("apply_if and apply_when conditionally execute operations")
def test_apply_if_when():
    data = from_range(1, 10)
    # apply_if with true condition
    result1 = data.util.apply_if(True, lambda en: en.where(lambda x: x > 5)).to.list()
    assert_that(result1 == [6, 7, 8, 9, 10], "operation should be applied when condition is true")
    # apply_if with false condition
    result2 = data.util.apply_if(False, lambda en: en.where(lambda x: x > 5)).to.list()
    assert_that(result2 == list(range(1, 11)), "operation should be skipped when condition is false")
    # apply_when with true predicate
    result3 = data.util.apply_when(lambda en: en.to.any(lambda x: x == 10),
                                   lambda en: en.select(lambda x: x + 1)).to.list()
    assert_that(result3 == list(range(2, 12)), "operation should apply when predicate is true")
    # apply_when with false predicate
    result4 = data.util.apply_when(lambda en: en.to.any(lambda x: x == 11),
                                   lambda en: en.select(lambda x: x + 1)).to.list()
    assert_that(result4 == list(range(1, 11)), "operation should be skipped when predicate is false")


@test("try_parse separates successes and failures")
def test_try_parse():
    def parse_int(s):
        try:
            return True, int(s)
        except ValueError:
            return False, None

    data = from_iterable(['1', 'a', '2', 'b', '3'])
    result = data.util.try_parse(parse_int)

    assert_that(isinstance(result, ParseResult), "should return a ParseResult object")
    assert_that(result.successes == [1, 2, 3], "successes list is incorrect")
    assert_that(result.failures == ['a', 'b'], "failures list is incorrect")


# --- stress test with combined operations ---

@test("stress test with large data and chained operations")
def test_stress_pinqy():
    num_records = 10000
    print(f"\n    {suite._c.grey}generating {num_records} objects for pinqy stress test...{suite._c.reset}")

    data_stream = from_schema(object_schema, seed=99).take(num_records)

    print(f"    {suite._c.grey}running complex pinqy chain...{suite._c.reset}")
    start_time = time.perf_counter()

    # a complex chain of operations
    results = (data_stream
               .where(lambda x: x['category'] == 'a' and x['value'] > 100)
               .order_by_descending(lambda x: x['value'])
               .select(lambda x: {'id': x['id'], 'name': x['name'].upper()})
               .take(10)
               .to.list())

    duration = time.perf_counter() - start_time
    print(f"    {suite._c.grey}└─> pinqy chain finished in {duration:.3f} seconds.{suite._c.reset}")

    assert_that(len(results) <= 10, "final result count should be at most 10")
    if len(results) > 0:
        assert_that(results[0]['name'].isupper(), "select transformation was not applied")
    if len(results) > 1:
        assert_that(from_iterable(results).select(lambda x: x['id']).set.distinct().to.count() == len(results),
                    "ids should be unique")

# --- run the suite ---
if __name__ == "__main__":
    suite.run(title="pinqy test suite")