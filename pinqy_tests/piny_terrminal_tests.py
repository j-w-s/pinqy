import time
import numpy as np
import pandas as pd
import suite
from collections import namedtuple
from dgen import from_schema
from pinqy import P, from_iterable, from_range

test = suite.test
assert_that = suite.assert_that

Person = namedtuple('Person', ['name', 'age', 'city'])

sample_people = [
    Person('alice', 25, 'nyc'),
    Person('bob', 30, 'la'),
    Person('charlie', 25, 'nyc'),
    Person('diana', 35, 'chicago'),
    Person('eve', 28, 'la')
]

sample_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
mixed_data = [1, 'hello', 3.14, True, None, {'key': 'value'}, [1, 2, 3]]


@test("list conversion returns proper list")
def test_to_list():
    result = P(sample_numbers).to.list()
    assert_that(result == sample_numbers, f"list conversion failed: {result}")
    assert_that(isinstance(result, list), f"should return list type: {type(result)}")


@test("array conversion creates numpy array")
def test_to_array():
    result = P(sample_numbers).to.array()
    expected = np.array(sample_numbers)
    assert_that(np.array_equal(result, expected), f"array conversion failed: {result}")
    assert_that(isinstance(result, np.ndarray), f"should return ndarray: {type(result)}")


@test("set conversion removes duplicates")
def test_to_set():
    duplicates = [1, 2, 2, 3, 3, 3, 4]
    result = P(duplicates).to.set()
    expected = {1, 2, 3, 4}
    assert_that(result == expected, f"set conversion failed: {result}")
    assert_that(isinstance(result, set), f"should return set type: {type(result)}")


@test("dict conversion with key selector")
def test_to_dict_with_keys():
    result = P(sample_people).to.dict(lambda p: p.name)

    assert_that(len(result) == 5, f"dict should have 5 entries: {len(result)}")
    assert_that(result['alice'] == sample_people[0], f"alice mapping incorrect: {result['alice']}")
    assert_that(result['bob'] == sample_people[1], f"bob mapping incorrect: {result['bob']}")


@test("dict conversion with key and value selectors")
def test_to_dict_with_key_value():
    result = P(sample_people).to.dict(
        lambda p: p.name,
        lambda p: p.age
    )

    expected = {'alice': 25, 'bob': 30, 'charlie': 25, 'diana': 35, 'eve': 28}
    assert_that(result == expected, f"key-value dict conversion failed: {result}")


@test("pandas series conversion")
def test_to_pandas():
    result = P(sample_numbers).to.pandas()
    expected = pd.Series(sample_numbers)

    assert_that(isinstance(result, pd.Series), f"should return pandas series: {type(result)}")
    assert_that(result.equals(expected), f"series data incorrect: {result.tolist()}")


@test("dataframe conversion from objects")
def test_to_df():
    data = [
        {'name': 'alice', 'age': 25, 'city': 'nyc'},
        {'name': 'bob', 'age': 30, 'city': 'la'},
        {'name': 'charlie', 'age': 25, 'city': 'nyc'}
    ]

    result = P(data).to.df()
    expected = pd.DataFrame(data)

    assert_that(isinstance(result, pd.DataFrame), f"should return dataframe: {type(result)}")
    assert_that(result.equals(expected), f"dataframe conversion failed")
    assert_that(len(result) == 3, f"should have 3 rows: {len(result)}")


@test("count without predicate")
def test_count_basic():
    result = P(sample_numbers).to.count()
    assert_that(result == 10, f"basic count failed: {result}")


@test("count with predicate")
def test_count_with_predicate():
    result = P(sample_numbers).to.count(lambda x: x > 5)
    assert_that(result == 5, f"conditional count failed: {result}")

    even_count = P(sample_numbers).to.count(lambda x: x % 2 == 0)
    assert_that(even_count == 5, f"even count failed: {even_count}")


@test("any without predicate")
def test_any_basic():
    assert_that(P(sample_numbers).to.any(), "non-empty should return true")
    assert_that(not P([]).to.any(), "empty should return false")


@test("any with predicate")
def test_any_with_predicate():
    assert_that(P(sample_numbers).to.any(lambda x: x > 8), "should find numbers > 8")
    assert_that(not P(sample_numbers).to.any(lambda x: x > 20), "should not find numbers > 20")

    has_alice = P(sample_people).to.any(lambda p: p.name == 'alice')
    assert_that(has_alice, "should find alice")


@test("all with predicate")
def test_all_with_predicate():
    assert_that(P(sample_numbers).to.all(lambda x: x > 0), "all should be positive")
    assert_that(not P(sample_numbers).to.all(lambda x: x > 5), "not all should be > 5")

    all_adults = P(sample_people).to.all(lambda p: p.age >= 18)
    assert_that(all_adults, "all should be adults")


@test("first without predicate")
def test_first_basic():
    result = P(sample_numbers).to.first()
    assert_that(result == 1, f"first should be 1: {result}")

    try:
        P([]).to.first()
        assert_that(False, "empty sequence should raise error")
    except ValueError as e:
        assert_that("no elements" in str(e), f"unexpected error: {e}")


@test("first with predicate")
def test_first_with_predicate():
    result = P(sample_numbers).to.first(lambda x: x > 5)
    assert_that(result == 6, f"first > 5 should be 6: {result}")

    alice = P(sample_people).to.first(lambda p: p.name == 'alice')
    assert_that(alice.name == 'alice', f"should find alice: {alice}")

    try:
        P(sample_numbers).to.first(lambda x: x > 20)
        assert_that(False, "no match should raise error")
    except ValueError as e:
        assert_that("no element satisfies" in str(e), f"unexpected error: {e}")


@test("first_or_default without predicate")
def test_first_or_default_basic():
    result = P(sample_numbers).to.first_or_default()
    assert_that(result == 1, f"first_or_default should be 1: {result}")

    result = P([]).to.first_or_default()
    assert_that(result is None, f"empty default should be none: {result}")

    result = P([]).to.first_or_default(default=42)
    assert_that(result == 42, f"custom default should be 42: {result}")


@test("first_or_default with predicate")
def test_first_or_default_with_predicate():
    result = P(sample_numbers).to.first_or_default(lambda x: x > 5)
    assert_that(result == 6, f"first > 5 should be 6: {result}")

    result = P(sample_numbers).to.first_or_default(lambda x: x > 20)
    assert_that(result is None, f"no match should return none: {result}")

    result = P(sample_numbers).to.first_or_default(lambda x: x > 20, default=-1)
    assert_that(result == -1, f"custom default should be -1: {result}")


@test("single without predicate")
def test_single_basic():
    result = P([42]).to.single()
    assert_that(result == 42, f"single element should be 42: {result}")

    try:
        P([]).to.single()
        assert_that(False, "empty should raise error")
    except ValueError as e:
        assert_that("no matching elements" in str(e), f"unexpected error: {e}")

    try:
        P([1, 2]).to.single()
        assert_that(False, "multiple elements should raise error")
    except ValueError as e:
        assert_that("2 matching elements" in str(e), f"unexpected error: {e}")


@test("single with predicate")
def test_single_with_predicate():
    result = P(sample_numbers).to.single(lambda x: x == 5)
    assert_that(result == 5, f"single match should be 5: {result}")

    try:
        P(sample_numbers).to.single(lambda x: x > 20)
        assert_that(False, "no match should raise error")
    except ValueError as e:
        assert_that("no matching elements" in str(e), f"unexpected error: {e}")

    try:
        P(sample_numbers).to.single(lambda x: x > 5)
        assert_that(False, "multiple matches should raise error")
    except ValueError as e:
        assert_that("matching elements" in str(e), f"unexpected error: {e}")


@test("aggregate without seed")
def test_aggregate_basic():
    result = P(sample_numbers).to.aggregate(lambda acc, x: acc + x)
    expected = sum(sample_numbers)
    assert_that(result == expected, f"sum aggregation failed: {result}")

    product = P([1, 2, 3, 4, 5]).to.aggregate(lambda acc, x: acc * x)
    assert_that(product == 120, f"product should be 120: {product}")

    try:
        P([]).to.aggregate(lambda acc, x: acc + x)
        assert_that(False, "empty without seed should raise error")
    except ValueError as e:
        assert_that("empty sequence" in str(e), f"unexpected error: {e}")


@test("aggregate with seed")
def test_aggregate_with_seed():
    result = P(sample_numbers).to.aggregate(lambda acc, x: acc + x, seed=100)
    expected = 100 + sum(sample_numbers)
    assert_that(result == expected, f"seeded sum failed: {result}")

    result = P([]).to.aggregate(lambda acc, x: acc + x, seed=42)
    assert_that(result == 42, f"empty with seed should return seed: {result}")

    concat = P(['a', 'b', 'c']).to.aggregate(lambda acc, x: acc + x, seed='start:')
    assert_that(concat == 'start:abc', f"string concat failed: {concat}")


@test("aggregate_with_selector")
def test_aggregate_with_selector():
    result = P(sample_people).to.aggregate_with_selector(
        seed=0,
        accumulator=lambda acc, person: acc + person.age,
        result_selector=lambda total: f"total age: {total}"
    )

    expected_total = sum(p.age for p in sample_people)
    expected = f"total age: {expected_total}"
    assert_that(result == expected, f"aggregate with selector failed: {result}")


@test("chaining terminal operations")
def test_chaining_terminals():
    ages = P(sample_people).select(lambda p: p.age)

    age_list = ages.to.list()
    age_set = ages.to.set()
    age_count = ages.to.count()

    assert_that(len(age_list) == 5, f"age list wrong length: {len(age_list)}")
    assert_that(len(age_set) == 4, f"age set wrong length: {len(age_set)}")
    assert_that(age_count == 5, f"age count wrong: {age_count}")


@test("mixed data type conversions")
def test_mixed_data_conversions():
    hashable_mixed_data = [2, 'hello', 3.14, False, None, ('key', 'value'), (1, 2, 3)]

    result_list = P(hashable_mixed_data).to.list()
    assert_that(len(result_list) == 7, f"mixed list wrong length: {len(result_list)}")

    result_set = P(hashable_mixed_data).to.set()
    assert_that(len(result_set) == 7, f"mixed set wrong length: {len(result_set)}")

    count = P(hashable_mixed_data).to.count(lambda x: x is not None)
    assert_that(count == 6, f"non-none count wrong: {count}")


@test("empty sequence handling")
def test_empty_sequences():
    empty = P([])

    assert_that(empty.to.list() == [], "empty list should be empty")
    assert_that(empty.to.set() == set(), "empty set should be empty")
    assert_that(empty.to.count() == 0, "empty count should be 0")
    assert_that(not empty.to.any(), "empty any should be false")
    assert_that(empty.to.first_or_default() is None, "empty first_or_default should be none")


@test("performance with large datasets")
def test_performance():
    print()

    large_data = from_range(1, 100000).to.list()

    start = time.perf_counter()
    result = P(large_data).to.count(lambda x: x % 2 == 0)
    duration_count = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    first_big = P(large_data).to.first(lambda x: x > 50000)
    duration_first = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    array_result = P(large_data).to.array()
    duration_array = (time.perf_counter() - start) * 1000

    print(f"    {suite._c.grey}├─> count on 100k items: {duration_count:.2f}ms{suite._c.reset}")
    print(f"    {suite._c.grey}├─> first on 100k items: {duration_first:.2f}ms{suite._c.reset}")
    print(f"    {suite._c.grey}└─> array conversion: {duration_array:.2f}ms{suite._c.reset}")

    assert_that(result == 50000, f"even count should be 50000: {result}")
    assert_that(first_big == 50001, f"first > 50000 should be 50001: {first_big}")
    assert_that(len(array_result) == 100000, f"array length wrong: {len(array_result)}")


@test("complex object conversions")
def test_complex_conversions():
    complex_data = [
        {'users': [{'name': 'alice', 'score': 85}], 'department': 'eng'},
        {'users': [{'name': 'bob', 'score': 92}], 'department': 'sales'},
        {'users': [{'name': 'charlie', 'score': 78}], 'department': 'eng'}
    ]

    dept_dict = P(complex_data).to.dict(lambda x: x['department'])
    assert_that(len(dept_dict) == 2, f"dept dict should have 2 entries: {len(dept_dict)}")

    user_scores = P(complex_data).to.dict(
        lambda x: x['users'][0]['name'],
        lambda x: x['users'][0]['score']
    )

    expected_scores = {'alice': 85, 'bob': 92, 'charlie': 78}
    assert_that(user_scores == expected_scores, f"user scores wrong: {user_scores}")


@test("generated data terminal operations")
def test_with_generated_data():
    schema = {
        'items': [{
            '_qen_items': {
                'id': ('pyint', {'min_value': 1, 'max_value': 1000}),
                'category': {'_qen_provider': 'choice', 'from': ['A', 'B', 'C']},
                'value': ('pyfloat', {'min_value': 0, 'max_value': 100})
            },
            '_qen_count': 50
        }]
    }

    data = from_schema(schema, seed=42).take(1).to.first()['items']

    categories = P(data).select(lambda x: x['category']).to.set()
    assert_that(len(categories) <= 3, f"should have max 3 categories: {categories}")

    high_value_count = P(data).to.count(lambda x: x['value'] > 50)
    assert_that(high_value_count >= 0, f"high value count should be non-negative: {high_value_count}")

    first_a_item = P(data).to.first_or_default(lambda x: x['category'] == 'A')
    if first_a_item:
        assert_that(first_a_item['category'] == 'A', f"category should be A: {first_a_item['category']}")


@test("edge cases and error handling")
def test_edge_cases():
    # test with none values
    none_data = [1, None, 3, None, 5]
    non_none_count = P(none_data).to.count(lambda x: x is not None)
    assert_that(non_none_count == 3, f"non-none count should be 3: {non_none_count}")

    # test single with exact match
    single_none = P(none_data).to.single(lambda x: x == 3)
    assert_that(single_none == 3, f"single match should be 3: {single_none}")

    # test aggregate with mixed types
    mixed_aggregate = P([1, 2, 3]).to.aggregate(
        lambda acc, x: f"{acc},{x}",
        seed="start"
    )
    assert_that(mixed_aggregate == "start,1,2,3", f"mixed aggregate failed: {mixed_aggregate}")


if __name__ == "__main__":
    suite.run(title="pinqy terminal operations test")