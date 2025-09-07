import os
import time
import suite
from dgen import from_schema
# import new factory and support classes
from pinqy import from_iterable, from_range, repeat, empty, Enumerable, create_tree_from_flat, ParseResult
import json
import uuid
from pathlib import Path
import suite
from dgen import from_json

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


# --- terminal operation edge cases ---

@test("first raises ValueError on empty or no-match sequences")
def test_first_errors():
    try:
        empty().to.first()
        assert_that(False, "first on empty list should raise ValueError")
    except ValueError:
        pass  # expected

    try:
        from_range(1, 10).to.first(lambda x: x > 10)
        assert_that(False, "first with no matching predicate should raise ValueError")
    except ValueError:
        pass  # expected


@test("single raises ValueError on empty, or multiple matches")
def test_single_errors():
    try:
        empty().to.single()
        assert_that(False, "single on empty list should raise ValueError")
    except ValueError:
        pass  # expected

    try:
        from_range(1, 10).to.single()
        assert_that(False, "single on list with multiple items should raise ValueError")
    except ValueError:
        pass  # expected

    try:
        from_range(1, 10).to.single(lambda x: x > 5)
        assert_that(False, "single with multiple matching items should raise ValueError")
    except ValueError:
        pass  # expected

    # test that it works for a single item
    result = from_range(1, 10).where(lambda x: x == 5).to.single()
    assert_that(result == 5, "single should return the one matching item")


# --- ordered enumerable edge cases ---

@test("find_by_key and between_keys work on sorted data")
def test_ordered_find():
    schema = {
        'state': {'_qen_provider': 'choice', 'from': ['ca', 'ny', 'tx']},
        'city': 'city',
        'pop': ('pyint', {'min_value': 1000, 'max_value': 10000})
    }
    data = from_schema(schema, seed=123).take(100)

    # order by state, then city
    ordered_data = data.order_by(lambda x: x['state']).then_by(lambda x: x['city'])

    # find by key
    ca_cities = ordered_data.find_by_key('ca').to.list()
    assert_that(from_iterable(ca_cities).to.any(), "should find cities in CA")
    assert_that(from_iterable(ca_cities).to.all(lambda x: x['state'] == 'ca'), "all found cities must be in CA")

    # between keys
    prices = from_iterable([10, 20, 30, 40, 50]).order_by(lambda x: x)
    in_range = prices.between_keys(20, 40).to.list()
    assert_that(in_range == [20, 30, 40], "between_keys should return inclusive range")


@test("find_by_key and between_keys work on descending sort")
def test_ordered_find_desc():
    # data is sorted [10, 9, 8, ..., 1]
    data = from_range(1, 10).order_by_descending(lambda x: x)

    # test find_by_key
    result_find = data.find_by_key(5).to.list()
    assert_that(result_find == [5], "find_by_key should find the correct element in a descending list")

    # test between_keys
    # note: the user provides bounds in a "natural" high-to-low order for descending sorts
    result_between = data.between_keys(7, 5).to.list()
    assert_that(result_between == [7, 6, 5], "between_keys should return the correct range from a descending list")

    # test that an empty result is returned for an invalid range
    result_empty = data.between_keys(4, 8).to.list()
    assert_that(result_empty == [], "between_keys on a descending sort should return empty for a low-to-high range")

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

    # test without including parents
    children_only = from_iterable(tree_data).tree.recursive_select(
        child_selector=lambda node: node.get('children'),
        include_parents=False
    ).select(lambda node: node['id']).to.list()
    assert_that(children_only == [2, 4, 3], "should not include the root/parent nodes")


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

# --- from_json tests ---

@test("from_json infers schema from a simple json string")
def test_from_json_simple_string():
    json_string = """
    {
        "id": 1,
        "is_active": true,
        "score": 95.5,
        "tags": ["alpha", "beta"]
    }
    """
    # create a generator from the json and take 10 items
    generator = from_json(json_string, seed=42)
    data = generator.take(10).to.list()

    assert_that(len(data) == 10, "should generate the requested number of items")
    first_item = data[0]
    assert_that(isinstance(first_item['id'], int), "id should be inferred as an integer")
    assert_that(isinstance(first_item['is_active'], bool), "is_active should be inferred as a boolean")
    assert_that(isinstance(first_item['score'], float), "score should be inferred as a float")
    assert_that(isinstance(first_item['tags'], list), "tags should be inferred as a list")
    assert_that(len(first_item['tags']) == 2, "inferred list should preserve original length")


@test("from_json correctly infers string heuristics")
def test_from_json_string_heuristics():
    json_string = """
    {
        "user_uuid": "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11",
        "email": "test.user@example.com",
        "user_name": "John Doe",
        "status": "pending",
        "description": "this is a multi-word sentence."
    }
    """
    # generate a single object to check its properties
    generator = from_json(json_string, seed=101)
    item = generator.take(1).to.first()

    # check uuid
    try:
        uuid.UUID(item['user_uuid'])
        is_valid_uuid = True
    except ValueError:
        is_valid_uuid = False
    assert_that(is_valid_uuid, "user_uuid should be a valid uuid4 string")

    # check other heuristics
    assert_that('@' in item['email'], "email should contain an '@' symbol")
    assert_that(' ' in item['user_name'], "user_name should be inferred as a multi-word name")
    assert_that(' ' not in item['status'], "status should be inferred as a single word")
    assert_that(len(item['description'].split()) > 1, "description should be inferred as a sentence")


@test("from_json loads from a file path")
def test_from_json_file():
    # this test creates and deletes its own file to remain self-contained
    file_name = "sample.json"
    json_content = [
        {"product_id": 100, "name": "widget", "in_stock": True},
        {"product_id": 200, "name": "gadget", "in_stock": False}
    ]

    try:
        # setup: create the json file
        with open(file_name, "w") as f:
            json.dump(json_content, f)

        # test: create generator from the file path
        generator = from_json(file_name, seed=1)
        data = generator.take(5).to.list()

        assert_that(len(data) == 5, "should generate 5 items")
        first_item = data[0]
        assert_that('product_id' in first_item and 'name' in first_item, "generated items should have the correct keys")
        assert_that(isinstance(first_item['product_id'], int), "product_id should be an int")
        assert_that(isinstance(first_item['in_stock'], bool), "in_stock should be a bool")

    finally:
        # teardown: clean up the file
        if os.path.exists(file_name):
            os.remove(file_name)


@test("from_json handles nested objects and lists of objects")
def test_from_json_nested():
    json_string = """
    {
        "order_id": "xyz-123",
        "customer": {
            "id": 5,
            "name": "alice"
        },
        "items": [
            { "sku": "a-01", "quantity": 2 },
            { "sku": "b-02", "quantity": 1 }
        ]
    }
    """
    generator = from_json(json_string, seed=99)
    data = generator.take(3).to.list()

    assert_that(len(data) == 3, "should generate 3 root objects")
    first_item = data[0]
    assert_that(isinstance(first_item['customer'], dict), "customer should be a nested dictionary")
    assert_that(isinstance(first_item['customer']['id'], int), "nested customer id should be an int")
    assert_that(isinstance(first_item['items'], list), "items should be a list")
    assert_that(len(first_item['items']) == 2, "nested list should have the same length as the template")
    assert_that(isinstance(first_item['items'][0]['quantity'], int), "quantity in list of objects should be an int")

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
    suite.run(title="dgen test suite")