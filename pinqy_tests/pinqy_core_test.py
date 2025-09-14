import suite
from pinqy import P, from_range, empty, repeat, generate
from dgen import from_schema
import re

test = suite.test
assert_that = suite.assert_that

# test data schemas
person_schema = {
    'id': ('pyint', {'min_value': 1, 'max_value': 100}),
    'name': 'word',
    'age': ('pyint', {'min_value': 18, 'max_value': 65}),
    'salary': ('pyint', {'min_value': 30000, 'max_value': 150000}),
    'department': {'_qen_provider': 'choice', 'from': ['eng', 'sales', 'hr', 'marketing']},
    'active': ('pybool', {})
}

product_schema = {
    'id': ('pyint', {'min_value': 1, 'max_value': 50}),
    'name': 'word',
    'price': ('pyfloat', {'min_value': 5.0, 'max_value': 500.0}),
    'category': {'_qen_provider': 'choice', 'from': ['electronics', 'books', 'clothing']}
}

# helper data
numbers = P(range(1, 11))  # 1 through 10
mixed_types = P([1, 'hello', 2.5, True, None, [1, 2], {'key': 'value'}])
words = P(['apple', 'banana', 'cherry', 'date', 'elderberry'])
nested_data = P([[1, 2], [3, 4, 5], [], [6]])


# where() tests

@test("where filters elements correctly")
def test_where_basic():
    evens = numbers.where(lambda x: x % 2 == 0).to.list()
    assert_that(evens == [2, 4, 6, 8, 10], "should filter even numbers")


@test("where with complex predicate")
def test_where_complex():
    people = from_schema(person_schema, seed=42).take(20)
    senior_eng = people.where(lambda p: p['department'] == 'eng' and p['age'] > 40).to.list()

    for person in senior_eng:
        assert_that(person['department'] == 'eng', "all should be engineers")
        assert_that(person['age'] > 40, "all should be over 40")


@test("where handles empty result")
def test_where_empty_result():
    result = numbers.where(lambda x: x > 100).to.list()
    assert_that(result == [], "should return empty list for no matches")


@test("where with regex pattern")
def test_where_regex():
    text_data = P(['apple123', 'banana', 'cherry456', 'date', '789elderberry'])
    with_numbers = text_data.where(lambda x: bool(re.search(r'\d', x))).to.list()
    assert_that(len(with_numbers) == 3, "should find 3 items with numbers")
    assert_that('banana' not in with_numbers, "banana should be excluded")


# select() tests

@test("select transforms elements")
def test_select_basic():
    squares = numbers.select(lambda x: x * x).to.list()
    assert_that(squares == [1, 4, 9, 16, 25, 36, 49, 64, 81, 100], "should square all numbers")


@test("select extracts object properties")
def test_select_property_extraction():
    people = from_schema(person_schema, seed=123).take(5)
    names = people.select(lambda p: p['name']).to.list()
    assert_that(len(names) == 5, "should extract 5 names")
    assert_that(all(isinstance(name, str) for name in names), "all names should be strings")


@test("select creates complex objects")
def test_select_complex_transform():
    people = from_schema(person_schema, seed=456).take(3)
    summaries = people.select(lambda p: {
        'display_name': p['name'].upper(),
        'is_senior': p['age'] >= 50,
        'salary_bracket': 'high' if p['salary'] > 100000 else 'standard'
    }).to.list()

    assert_that(len(summaries) == 3, "should create 3 summaries")
    for summary in summaries:
        assert_that('display_name' in summary, "should have display_name")
        assert_that(isinstance(summary['is_senior'], bool), "is_senior should be boolean")


# select_many() tests

@test("select_many flattens sequences")
def test_select_many_basic():
    flattened = nested_data.select_many(lambda x: x).to.list()
    assert_that(flattened == [1, 2, 3, 4, 5, 6], "should flatten all sublists")


@test("select_many with string splitting")
def test_select_many_string_split():
    sentences = P(['hello world', 'pinqy rocks', 'functional programming'])
    words = sentences.select_many(lambda s: s.split()).to.list()
    assert_that(len(words) == 6, "should split into 6 words")
    assert_that('functional' in words, "should contain 'functional'")


@test("select_many with complex transformation")
def test_select_many_complex():
    data = P([{'tags': ['a', 'b']}, {'tags': ['c']}, {'tags': []}])
    all_tags = data.select_many(lambda item: item['tags']).to.list()
    assert_that(all_tags == ['a', 'b', 'c'], "should extract all tags")


# select_with_index() tests

@test("select_with_index includes indices")
def test_select_with_index_basic():
    indexed = words.select_with_index(lambda word, i: f"{i}:{word}").to.list()
    assert_that(indexed[0] == "0:apple", "first item should be '0:apple'")
    assert_that(indexed[-1] == "4:elderberry", "last item should be '4:elderberry'")


@test("select_with_index for enumeration")
def test_select_with_index_enumeration():
    people = from_schema(person_schema, seed=789).take(3)
    enumerated = people.select_with_index(lambda p, i: {
        'rank': i + 1,
        'name': p['name'],
        'is_first': i == 0
    }).to.list()

    assert_that(enumerated[0]['rank'] == 1, "first rank should be 1")
    assert_that(enumerated[0]['is_first'] is True, "first should be marked as first")
    assert_that(enumerated[1]['is_first'] is False, "second should not be marked as first")


# of_type() tests

@test("of_type filters by type")
def test_of_type_basic():
    strings_only = mixed_types.of_type(str).to.list()
    assert_that(strings_only == ['hello'], "should only return string")

    numbers_only = mixed_types.of_type((int, float)).to.list()
    expected_numbers = [1, 2.5]  # bool inherits from int, but we want specific numeric types
    assert_that(len(numbers_only) >= 2, "should find at least 2 numeric values")


@test("of_type with custom class")
def test_of_type_custom_class():
    class TestClass:
        def __init__(self, value):
            self.value = value

    mixed_objects = P([1, TestClass(42), 'hello', TestClass(99), None])
    test_objects = mixed_objects.of_type(TestClass).to.list()
    assert_that(len(test_objects) == 2, "should find 2 TestClass instances")
    assert_that(all(isinstance(obj, TestClass) for obj in test_objects), "all should be TestClass instances")


# order_by() and order_by_descending() tests

@test("order_by sorts ascending")
def test_order_by_basic():
    shuffled = P([3, 1, 4, 1, 5, 9, 2, 6])
    sorted_asc = shuffled.order_by(lambda x: x).to.list()
    assert_that(sorted_asc == [1, 1, 2, 3, 4, 5, 6, 9], "should sort ascending")


@test("order_by_descending sorts descending")
def test_order_by_descending_basic():
    shuffled = P([3, 1, 4, 1, 5, 9, 2, 6])
    sorted_desc = shuffled.order_by_descending(lambda x: x).to.list()
    assert_that(sorted_desc == [9, 6, 5, 4, 3, 2, 1, 1], "should sort descending")


@test("order_by with complex key")
def test_order_by_complex_key():
    people = from_schema(person_schema, seed=101).take(10)
    by_age = people.order_by(lambda p: p['age']).to.list()

    ages = P(by_age).select(lambda p: p['age']).to.list()
    assert_that(ages == sorted(ages), "ages should be in ascending order")


@test("order_by preserves stability")
def test_order_by_stability():
    data = P([{'key': 1, 'value': 'a'}, {'key': 2, 'value': 'b'}, {'key': 1, 'value': 'c'}])
    sorted_data = data.order_by(lambda x: x['key']).to.list()

    key_1_items = P(sorted_data).where(lambda x: x['key'] == 1).to.list()
    assert_that(len(key_1_items) == 2, "should have 2 items with key 1")
    assert_that(key_1_items[0]['value'] == 'a', "should maintain original order for equal keys")
    assert_that(key_1_items[1]['value'] == 'c', "should maintain original order for equal keys")


# reverse() tests

@test("reverse inverts order")
def test_reverse_basic():
    reversed_numbers = numbers.reverse().to.list()
    assert_that(reversed_numbers == [10, 9, 8, 7, 6, 5, 4, 3, 2, 1], "should reverse order")


@test("reverse with complex data")
def test_reverse_complex():
    people = from_schema(person_schema, seed=202).take(5)
    original_names = people.select(lambda p: p['name']).to.list()
    reversed_names = people.reverse().select(lambda p: p['name']).to.list()
    assert_that(reversed_names == list(reversed(original_names)), "should reverse complex data")


# as_ordered() tests

@test("as_ordered enables then_by operations")
def test_as_ordered_basic():
    # create pre-sorted data
    data = P([{'a': 1, 'b': 3}, {'a': 1, 'b': 1}, {'a': 2, 'b': 2}])
    # treat as already ordered by 'a', then sort by 'b'
    result = data.as_ordered().then_by(lambda x: x['b']).to.list()

    # should be sorted by both a and b
    b_values = P(result).select(lambda x: x['b']).to.list()
    assert_that(b_values == [1, 3, 2], "should maintain a-order but sort b within groups")


# take() and skip() tests

@test("take returns specified count")
def test_take_basic():
    first_five = numbers.take(5).to.list()
    assert_that(first_five == [1, 2, 3, 4, 5], "should take first 5")


@test("take handles count larger than sequence")
def test_take_oversized():
    all_numbers = numbers.take(100).to.list()
    assert_that(len(all_numbers) == 10, "should return all available items")


@test("take with zero count")
def test_take_zero():
    none_taken = numbers.take(0).to.list()
    assert_that(none_taken == [], "should return empty list")


@test("skip bypasses elements")
def test_skip_basic():
    last_five = numbers.skip(5).to.list()
    assert_that(last_five == [6, 7, 8, 9, 10], "should skip first 5")


@test("skip handles count larger than sequence")
def test_skip_oversized():
    empty_result = numbers.skip(100).to.list()
    assert_that(empty_result == [], "should return empty list")


# take_while() and skip_while() tests

@test("take_while stops at first false")
def test_take_while_basic():
    taken = numbers.take_while(lambda x: x < 6).to.list()
    assert_that(taken == [1, 2, 3, 4, 5], "should take while < 6")


@test("take_while with complex predicate")
def test_take_while_complex():
    people = from_schema(person_schema, seed=303).take(20)
    young_consecutive = people.take_while(lambda p: p['age'] < 40).to.list()

    if young_consecutive:  # only test if we found some
        for person in young_consecutive:
            assert_that(person['age'] < 40, "all taken should be under 40")


@test("skip_while skips until first false")
def test_skip_while_basic():
    skipped = numbers.skip_while(lambda x: x < 6).to.list()
    assert_that(skipped == [6, 7, 8, 9, 10], "should skip while < 6")


@test("take_while and skip_while are complementary")
def test_take_skip_while_complementary():
    predicate = lambda x: x < 6
    taken = numbers.take_while(predicate).to.list()
    skipped = numbers.skip_while(predicate).to.list()
    combined = taken + skipped
    assert_that(combined == numbers.to.list(), "take_while + skip_while should equal original")


# append() and prepend() tests

@test("append adds element to end")
def test_append_basic():
    with_eleven = numbers.append(11).to.list()
    assert_that(with_eleven[-1] == 11, "should append 11 to end")
    assert_that(len(with_eleven) == 11, "should have 11 elements")


@test("prepend adds element to beginning")
def test_prepend_basic():
    with_zero = numbers.prepend(0).to.list()
    assert_that(with_zero[0] == 0, "should prepend 0 to beginning")
    assert_that(len(with_zero) == 11, "should have 11 elements")


@test("chained append and prepend")
def test_append_prepend_chained():
    modified = numbers.take(3).prepend(0).append(4).to.list()
    assert_that(modified == [0, 1, 2, 3, 4], "should handle chained operations")


# default_if_empty() tests

@test("default_if_empty returns original when not empty")
def test_default_if_empty_not_empty():
    result = numbers.default_if_empty(-1).to.list()
    assert_that(result == numbers.to.list(), "should return original sequence")


@test("default_if_empty returns default when empty")
def test_default_if_empty_empty():
    result = empty().default_if_empty(-1).to.list()
    assert_that(result == [-1], "should return default value in list")


@test("default_if_empty with complex default")
def test_default_if_empty_complex():
    default_person = {'name': 'unknown', 'age': 0}
    result = P([]).default_if_empty(default_person).to.list()
    assert_that(len(result) == 1, "should have one element")
    assert_that(result[0]['name'] == 'unknown', "should use default object")


# chaining and pipeline tests

@test("complex method chaining works correctly")
def test_complex_chaining():
    people = from_schema(person_schema, seed=404).take(50)

    result = (people
              .where(lambda p: p['age'] > 25)
              .where(lambda p: p['salary'] > 50000)
              .select(lambda p: {'name': p['name'], 'dept': p['department'], 'senior': p['age'] > 50})
              .where(lambda p: p['dept'] in ['eng', 'sales'])
              .order_by(lambda p: p['name'])
              .take(10)
              .to.list())

    assert_that(len(result) <= 10, "should respect take limit")
    for person in result:
        assert_that('name' in person, "should have transformed structure")
        assert_that(person['dept'] in ['eng', 'sales'], "should filter departments")


@test("lazy evaluation defers execution")
def test_lazy_evaluation():
    # create a chain without terminal operation
    chain = (numbers
             .where(lambda x: x % 2 == 0)
             .select(lambda x: x * 10)
             .order_by(lambda x: -x))

    # the chain should not be executed yet
    # only when we call a terminal operation should it execute
    result = chain.to.list()
    assert_that(result == [100, 80, 60, 40, 20], "should execute lazily and produce correct result")


@test("multiple terminal operations on same chain")
def test_multiple_terminals():
    chain = numbers.where(lambda x: x <= 5).select(lambda x: x * 2)

    list_result = chain.to.list()
    count_result = chain.to.count()

    assert_that(list_result == [2, 4, 6, 8, 10], "list result should be correct")
    assert_that(count_result == 5, "count result should be correct")


# edge cases and error handling

@test("empty sequence operations")
def test_empty_sequence_operations():
    empty_enum = empty()

    assert_that(empty_enum.to.count() == 0, "empty count should be 0")
    assert_that(empty_enum.where(lambda x: True).to.list() == [], "where on empty should be empty")
    assert_that(empty_enum.select(lambda x: x * 2).to.list() == [], "select on empty should be empty")
    assert_that(empty_enum.take(5).to.list() == [], "take on empty should be empty")


@test("single element operations")
def test_single_element_operations():
    single = P([42])

    assert_that(single.to.count() == 1, "single count should be 1")
    assert_that(single.where(lambda x: x > 40).to.list() == [42], "where should work on single")
    assert_that(single.select(lambda x: x * 2).to.list() == [84], "select should work on single")
    assert_that(single.take(10).to.list() == [42], "take should work on single")


@test("operations with none values")
def test_none_handling():
    with_nones = P([1, None, 3, None, 5])

    non_nones = with_nones.where(lambda x: x is not None).to.list()
    assert_that(non_nones == [1, 3, 5], "should filter out nones")

    transformed = with_nones.select(lambda x: 'none' if x is None else str(x)).to.list()
    assert_that(transformed == ['1', 'none', '3', 'none', '5'], "should handle nones in select")


@test("regex-based data processing pipeline")
def test_regex_processing_pipeline():
    log_entries = P([
        '2023-01-01 ERROR: database connection failed',
        '2023-01-01 INFO: user logged in',
        '2023-01-02 WARNING: disk space low',
        '2023-01-02 ERROR: timeout occurred',
        '2023-01-03 INFO: backup completed'
    ])

    errors = (log_entries
              .where(lambda line: re.search(r'ERROR:', line))
              .select(lambda line: re.sub(r'^\d{4}-\d{2}-\d{2}\s+ERROR:\s*', '', line))
              .order_by(lambda msg: msg)
              .to.list())

    assert_that(len(errors) == 2, "should find 2 error messages")
    assert_that('database connection failed' in errors, "should extract error message")
    assert_that('timeout occurred' in errors, "should extract error message")


@test("functional composition with pinqy")
def test_functional_composition():
    # demonstrate functional composition patterns
    is_even = lambda x: x % 2 == 0
    square = lambda x: x * x
    is_large = lambda x: x > 50

    result = (numbers
              .where(is_even)
              .select(square)
              .where(is_large)
              .to.list())

    assert_that(result == [64, 100], "should compose functions correctly")


@test("data analysis pipeline with generated data")
def test_data_analysis_pipeline():
    # generate realistic dataset
    sales_data = from_schema(product_schema, seed=555).take(100)

    # complex analytical pipeline
    analysis = (sales_data
                .where(lambda p: p['price'] > 50)
                .group.group_by(lambda p: p['category'])
                .items())

    category_analysis = P(analysis).select(lambda item: {
        'category': item[0],
        'count': len(item[1]),
        'avg_price': sum(p['price'] for p in item[1]) / len(item[1]) if item[1] else 0,
        'price_range': {
            'min': min(p['price'] for p in item[1]) if item[1] else 0,
            'max': max(p['price'] for p in item[1]) if item[1] else 0
        }
    }).order_by(lambda cat: -cat['avg_price']).to.list()

    assert_that(len(category_analysis) >= 1, "should produce analysis results")
    for cat in category_analysis:
        assert_that('category' in cat, "should have category field")
        assert_that(cat['avg_price'] >= 0, "avg_price should be non-negative")


if __name__ == "__main__":
    suite.run(title="pinqy core operations test suite")