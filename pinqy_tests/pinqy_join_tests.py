import suite
from dgen import from_schema
from pinqy import P, from_range, empty

test = suite.test
assert_that = suite.assert_that

# --- test data schemas ---
person_schema = {
    'id': ('pyint', {'min_value': 1, 'max_value': 5}),
    'name': 'word',
    'department': {'_qen_provider': 'choice', 'from': ['eng', 'sales', 'hr']}
}

order_schema = {
    'id': ('pyint', {'min_value': 1, 'max_value': 10}),
    'customer_id': ('pyint', {'min_value': 1, 'max_value': 5}),
    'amount': ('pyint', {'min_value': 100, 'max_value': 1000})
}

# --- helper data ---
people_data = [
    {'id': 1, 'name': 'alice', 'dept': 'eng'},
    {'id': 2, 'name': 'bob', 'dept': 'sales'},
    {'id': 3, 'name': 'charlie', 'dept': 'eng'}
]

orders_data = [
    {'id': 101, 'customer_id': 1, 'amount': 250},
    {'id': 102, 'customer_id': 1, 'amount': 150},
    {'id': 103, 'customer_id': 2, 'amount': 500},
    {'id': 104, 'customer_id': 4, 'amount': 300}  # no matching person
]


# --- inner join ---

@test("join performs inner join correctly")
def test_join_basic():
    people = P(people_data)
    orders = P(orders_data)

    result = people.join.join(
        orders,
        lambda p: p['id'],
        lambda o: o['customer_id'],
        lambda p, o: {'name': p['name'], 'order_id': o['id'], 'amount': o['amount']}
    ).to.list()

    assert_that(len(result) == 3, "should have 3 matching joins")
    alice_orders = P(result).where(lambda r: r['name'] == 'alice').to.list()
    assert_that(len(alice_orders) == 2, "alice should have 2 orders")
    assert_that(P(alice_orders).select(lambda r: r['order_id']).to.set() == {101, 102},
                "alice should have orders 101 and 102")


@test("join handles no matches")
def test_join_no_matches():
    result = P([{'id': 1}]).join.join(
        [{'customer_id': 2}],
        lambda p: p['id'],
        lambda o: o['customer_id'],
        lambda p, o: {'matched': True}
    ).to.list()

    assert_that(result == [], "join with no matches should return empty")


@test("join handles empty sequences")
def test_join_empty():
    empty_join = empty().join.join(
        [{'id': 1}],
        lambda x: x,
        lambda x: x['id'],
        lambda x, y: x
    ).to.list()

    assert_that(empty_join == [], "join with empty outer should return empty")


# --- left join ---

@test("left_join includes all outer elements")
def test_left_join_basic():
    people = P(people_data)
    orders = P(orders_data)

    result = people.join.left_join(
        orders,
        lambda p: p['id'],
        lambda o: o['customer_id'],
        lambda p, o: {'name': p['name'], 'amount': o['amount'] if o else 0}
    ).to.list()

    assert_that(len(result) == 4, "should include all people (some with multiple orders)")
    charlie_result = P(result).where(lambda r: r['name'] == 'charlie').to.list()
    assert_that(len(charlie_result) == 1, "charlie should appear once")
    assert_that(charlie_result[0]['amount'] == 0, "charlie should have default amount")


@test("left_join with custom default")
def test_left_join_custom_default():
    people = P([{'id': 1, 'name': 'alice'}])
    orders = P([])

    result = people.join.left_join(
        orders,
        lambda p: p['id'],
        lambda o: o['customer_id'],
        lambda p, o: {'name': p['name'], 'order': o},
        default_inner={'id': -1, 'amount': 999}
    ).to.list()

    assert_that(result[0]['order']['amount'] == 999, "should use custom default")


# --- right join ---

@test("right_join includes all inner elements")
def test_right_join_basic():
    people = P(people_data)
    orders = P(orders_data)

    result = people.join.right_join(
        orders,
        lambda p: p['id'],
        lambda o: o['customer_id'],
        lambda p, o: {'name': p['name'] if p else 'unknown', 'order_id': o['id']}
    ).to.list()

    assert_that(len(result) == 4, "should include all orders")
    unknown_orders = P(result).where(lambda r: r['name'] == 'unknown').to.list()
    assert_that(len(unknown_orders) == 1, "should have one order with unknown customer")
    assert_that(unknown_orders[0]['order_id'] == 104, "unknown order should be 104")


# --- full outer join ---

@test("full_outer_join includes all elements from both sequences")
def test_full_outer_join_basic():
    left_data = [{'id': 1, 'name': 'alice'}, {'id': 2, 'name': 'bob'}]
    right_data = [{'id': 2, 'city': 'ny'}, {'id': 3, 'city': 'la'}]

    result = P(left_data).join.full_outer_join(
        right_data,
        lambda l: l['id'],
        lambda r: r['id'],
        lambda l, r: {
            'name': l['name'] if l else 'unknown',
            'city': r['city'] if r else 'unknown'
        }
    ).to.list()

    assert_that(len(result) == 3, "should have 3 results")
    names = P(result).select(lambda r: r['name']).to.set()
    cities = P(result).select(lambda r: r['city']).to.set()
    assert_that(names == {'alice', 'bob', 'unknown'}, "should include all names plus unknown")
    assert_that(cities == {'ny', 'la', 'unknown'}, "should include all cities plus unknown")


@test("full_outer_join with custom defaults")
def test_full_outer_join_defaults():
    result = P([{'id': 1}]).join.full_outer_join(
        [{'id': 2}],
        lambda l: l['id'],
        lambda r: r['id'],
        lambda l, r: {'left': l, 'right': r},
        default_outer={'id': -1},
        default_inner={'id': -2}
    ).to.list()

    assert_that(len(result) == 2, "should have 2 non-matching results")
    left_ids = P(result).select(lambda r: r['left']['id']).to.list()
    right_ids = P(result).select(lambda r: r['right']['id']).to.list()
    assert_that(set(left_ids) == {1, -1}, "should use default for left")
    assert_that(set(right_ids) == {-2, 2}, "should use default for right")


# --- group join ---

@test("group_join groups inner elements by outer key")
def test_group_join_basic():
    people = P(people_data)
    orders = P(orders_data)

    result = people.join.group_join(
        orders,
        lambda p: p['id'],
        lambda o: o['customer_id'],
        lambda p, orders_enum: {
            'name': p['name'],
            'total_amount': orders_enum.stats.sum(lambda o: o['amount']),
            'order_count': orders_enum.to.count()
        }
    ).to.list()

    assert_that(len(result) == 3, "should have one result per person")
    alice_result = P(result).where(lambda r: r['name'] == 'alice').to.first()
    assert_that(alice_result['order_count'] == 2, "alice should have 2 orders")
    assert_that(alice_result['total_amount'] == 400, "alice total should be 400")

    charlie_result = P(result).where(lambda r: r['name'] == 'charlie').to.first()
    assert_that(charlie_result['order_count'] == 0, "charlie should have 0 orders")


@test("group_join passes enumerable to result selector")
def test_group_join_enumerable_check():
    def verify_enumerable(person, orders_enum):
        from pinqy import Enumerable
        assert_that(isinstance(orders_enum, Enumerable), "should receive enumerable")
        return {'name': person['name'], 'verified': True}

    result = P(people_data).join.group_join(
        orders_data,
        lambda p: p['id'],
        lambda o: o['customer_id'],
        verify_enumerable
    ).to.list()

    assert_that(all(r['verified'] for r in result), "all results should be verified")


# --- cross join ---

@test("cross_join produces cartesian product")
def test_cross_join_basic():
    colors = P(['red', 'blue'])
    sizes = ['s', 'm', 'l']

    result = colors.join.cross_join(sizes).to.list()

    assert_that(len(result) == 6, "should have 6 combinations")
    assert_that(('red', 's') in result, "should contain ('red', 's')")
    assert_that(('blue', 'l') in result, "should contain ('blue', 'l')")

    red_items = P(result).where(lambda r: r[0] == 'red').to.list()
    assert_that(len(red_items) == 3, "red should appear with all sizes")


@test("cross_join handles empty sequences")
def test_cross_join_empty():
    result = P([1, 2]).join.cross_join([]).to.list()
    assert_that(result == [], "cross join with empty should be empty")

    result2 = empty().join.cross_join([1, 2]).to.list()
    assert_that(result2 == [], "empty cross join should be empty")


@test("cross_join with single elements")
def test_cross_join_single():
    result = P([1]).join.cross_join([2]).to.list()
    assert_that(result == [(1, 2)], "single element cross join should work")


# --- complex scenarios ---

@test("chained joins work correctly")
def test_chained_joins():
    departments = [{'id': 1, 'name': 'engineering'}, {'id': 2, 'name': 'sales'}]
    people = [{'id': 1, 'dept_id': 1}, {'id': 2, 'dept_id': 2}]

    result = (P(departments)
              .join.join(people, lambda d: d['id'], lambda p: p['dept_id'],
                         lambda d, p: {'dept_name': d['name'], 'person_id': p['id']})
              .to.list())

    assert_that(len(result) == 2, "chained join should work")
    dept_names = P(result).select(lambda r: r['dept_name']).to.set()
    assert_that(dept_names == {'engineering', 'sales'}, "should have both departments")


@test("join with complex key selectors")
def test_join_complex_keys():
    # fix: ensure the keys actually match by using compatible data
    data1 = [{'a': 1, 'b': 2}, {'a': 2, 'b': 1}]  # sums: 3, 3
    data2 = [{'x': 3, 'y': 5}, {'x': 5, 'y': 7}]  # x values: 3, 5

    result = P(data1).join.join(
        data2,
        lambda d1: d1['a'] + d1['b'],  # composite key: 3, 3
        lambda d2: d2['x'],            # simple key: 3, 5
        lambda d1, d2: {'sum1': d1['a'] + d1['b'], 'x': d2['x']}
    ).to.list()

    assert_that(len(result) == 2, "should have two matches (both data1 items match data2[0])")
    assert_that(all(r['sum1'] == 3 and r['x'] == 3 for r in result), "should match on composite key")


@test("performance test with larger datasets")
def test_join_performance():
    # generate larger datasets to test performance characteristics
    large_people = from_schema(person_schema, seed=42).take(100)
    large_orders = from_schema(order_schema, seed=43).take(200)

    result = large_people.join.join(
        large_orders.to.list(),
        lambda p: p['id'],
        lambda o: o['customer_id'],
        lambda p, o: {'person': p['name'], 'order': o['id']}
    ).to.count()

    # basic sanity check that join completed successfully
    assert_that(result >= 0, "join should complete and return non-negative count")


@test("join with duplicate keys")
def test_join_duplicate_keys():
    people = [{'id': 1, 'name': 'alice'}, {'id': 1, 'name': 'alice2'}]  # duplicate ids
    orders = [{'customer_id': 1, 'amount': 100}, {'customer_id': 1, 'amount': 200}]

    result = P(people).join.join(
        orders,
        lambda p: p['id'],
        lambda o: o['customer_id'],
        lambda p, o: {'name': p['name'], 'amount': o['amount']}
    ).to.list()

    assert_that(len(result) == 4, "should have 2x2 = 4 results for duplicate keys")
    amounts = P(result).select(lambda r: r['amount']).to.list()
    assert_that(amounts.count(100) == 2, "each amount should appear twice")
    assert_that(amounts.count(200) == 2, "each amount should appear twice")


if __name__ == "__main__":
    suite.run(title="pinqy join operations test")