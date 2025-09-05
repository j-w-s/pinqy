import time
from dgen import from_schema, Generator
from pinqy import Enumerable
import suite

test = suite.test
assert_that = suite.assert_that

@test("raises ValueError for an invalid faker provider")
def test_invalid_faker_provider():
    # unambiguous tuple syntax to force a faker provider call
    # a simple string "not_a_real_faker_method" treated as a literal
    schema = {'bad': ('not_a_real_faker_method', {})}
    g = Generator()
    try:
        g.create(schema)
        assert_that(False, "should have raised ValueError for invalid faker method")
    except ValueError as e:
        assert_that("faker has no provider" in str(e), "error message is incorrect")


@test("generates non-string literal values correctly")
def test_non_string_literals():
    schema = {'id': 123, 'active': True, 'value': None, 'ratio': 0.5}
    g = Generator()
    result = g.create(schema)
    assert_that(result == schema, "literal values should remain unchanged")


@test("generates string literals that are not faker providers")
def test_string_literals_are_not_faker_calls():
    schema = {'status': 'active', 'type': 'user_account', 'message': 'hello world'}
    g = Generator()
    result = g.create(schema)
    assert_that(result['status'] == 'active', "string literal 'active' was processed incorrectly")
    assert_that(result['message'] == 'hello world', "string literal 'hello world' was processed incorrectly")


@test("generates faker strings without arguments")
def test_faker_strings():
    schema = {'name': 'name', 'city': 'city'}
    g = Generator(seed=1)
    result = g.create(schema)
    assert_that(isinstance(result['name'], str) and result['name'] != 'name',
                "faker 'name' should be a generated string")
    assert_that(isinstance(result['city'], str) and result['city'] != 'city',
                "faker 'city' should be a generated string")


@test("generates faker strings with arguments")
def test_faker_with_args():
    schema = {'code': ('pystr', {'min_chars': 8, 'max_chars': 8})}
    g = Generator()
    result = g.create(schema)
    assert_that(isinstance(result['code'], str) and len(result['code']) == 8, "faker 'pystr' should respect arguments")


@test("generates lists with a default count of 5")
def test_list_default_count():
    schema = [{'id': 'uuid4'}]
    g = Generator()
    result = g.create(schema)
    assert_that(isinstance(result, list) and len(result) == 5, f"default list count should be 5, but was {len(result)}")


@test("generates lists with a fixed count via '_qen_count'")
def test_list_fixed_count():
    schema = [{'id': 'uuid4', '_qen_count': 3}]
    g = Generator()
    result = g.create(schema)
    assert_that(isinstance(result, list) and len(result) == 3, "fixed list count should be 3")


@test("generates lists of lists")
def test_list_of_lists():
    schema = [[{'value': ('pyint', {'min_value': 0, 'max_value': 1}), '_qen_count': 2}]]
    g = Generator(seed=123)
    result = g.create(schema)
    assert_that(len(result) == 5, "outer list should have default count 5")
    assert_that(len(result[0]) == 2, "inner list should have count 2")
    assert_that(isinstance(result[0][0]['value'], int), "deeply nested value should be generated")


@test("handles '_qen_provider: choice'")
def test_provider_choice():
    options = ['active', 'inactive', 'pending']
    schema = {'status': {'_qen_provider': 'choice', 'from': options}}
    g = Generator()
    result = g.create(schema)
    assert_that(result['status'] in options, "choice provider should select from the provided list")


@test("handles '_qen_provider: literal' to override faker")
def test_provider_literal_override():
    schema = {'field': {'_qen_provider': 'literal', 'value': 'name'}}
    g = Generator()
    result = g.create(schema)
    assert_that(result['field'] == 'name', "literal provider should return the exact string value")


@test("handles '_qen_provider: ref' for cross-field references")
def test_provider_ref():
    schema = {
        'username': ('pystr', {'min_chars': 5, 'max_chars': 5}),
        'email': {'_qen_provider': 'ref', 'key': 'username', 'format': '{}@test.com'}
    }
    g = Generator(seed=1)
    result = g.create(schema)
    assert_that(result['email'] == f"{result['username']}@test.com",
                "ref provider should correctly reference and format")


@test("handles '_qen_provider: lambda' for dynamic values")
def test_provider_lambda():
    schema = {
        'a': 10,
        'b': 20,
        'sum': {'_qen_provider': 'lambda', 'func': "lambda ctx: ctx['a'] + ctx['b']"}
    }
    g = Generator()
    result = g.create(schema)
    assert_that(result['sum'] == 30, "lambda provider should execute function on context")


@test("handles lambda provider referencing a ref'd value")
def test_provider_lambda_after_ref():
    schema = {
        'first_name': 'first_name',
        'full_name': {'_qen_provider': 'ref', 'key': 'first_name', 'format': '{} doe'},
        'name_len': {'_qen_provider': 'lambda', 'func': 'lambda ctx: len(ctx["full_name"])'}
    }
    g = Generator(seed=42)
    result = g.create(schema)
    expected_len = len(f"{result['first_name']} doe")
    assert_that(result['name_len'] == expected_len, "lambda should correctly use value created by a ref")


@test("generates complex nested objects correctly")
def test_complex_nesting():
    complex_schema = {
        'company': 'company',
        'company_id': ('pystr', {'max_chars': 4}),
        'departments': [{
            '_qen_count': 2,
            'dept_name': {'_qen_provider': 'choice', 'from': ['eng', 'sales']},
            'employees': [{
                '_qen_count': [3, 5],
                'name': 'name',
                'employee_id': {'_qen_provider': 'ref', 'key': 'company_id', 'format': '{}-emp-001'}
            }]
        }]
    }
    g = Generator(seed=42)
    result = g.create(complex_schema)
    assert_that('company' in result and 'departments' in result, "top-level keys should exist")
    assert_that(len(result['departments']) == 2, "departments list should be generated")
    first_dept = result['departments'][0]
    assert_that(3 <= len(first_dept['employees']) <= 5, "employees list should be generated")
    first_emp = first_dept['employees'][0]
    assert_that(first_emp['employee_id'].startswith(result['company_id']), "nested ref should work within its scope")


@test("'from_schema().take(n)' integrates with pinqy")
def test_from_schema_with_pinqy():
    user_schema = {
        'name': 'name',
        'age': ('pyint', {'min_value': 18, 'max_value': 70}),
        'status': {'_qen_provider': 'choice', 'from': ['active', 'inactive']}
    }
    users = from_schema(user_schema, seed=101).take(100)
    assert_that(isinstance(users, Enumerable), "result should be an Enumerable")
    assert_that(users.count() == 100, "enumerable should contain exactly 100 items")
    active_users_over_30 = users.where(lambda u: u['status'] == 'active' and u['age'] > 30)
    assert_that(active_users_over_30.any(), "pinqy .where() should be able to filter generated data")
    avg_age = users.average(lambda u: u['age'])
    assert_that(18 <= avg_age <= 70, "pinqy .average() should work on generated data")


@test("reproduces identical data with the same seed")
def test_reproducibility_with_seed():
    schema = {'id': 'uuid4', 'name': 'name', 'val': ('pyint',)}
    data1 = from_schema(schema, seed=42).take(10).to_list()
    data2 = from_schema(schema, seed=42).take(10).to_list()
    assert_that(data1 == data2, "the same seed should produce identical data")


@test("raises ValueError for a dangling reference key")
def test_dangling_ref():
    schema = {'bad': {'_qen_provider': 'ref', 'key': 'non_existent'}}
    g = Generator()
    try:
        g.create(schema)
        assert_that(False, "should have raised ValueError for dangling ref")
    except ValueError as e:
        assert_that("not found in current context" in str(e), "error message is incorrect")


@test("stress test with large, complex schema")
def test_stress_generation():
    num_records = 30000
    stress_schema = {
        'order_id': 'uuid4',
        'customer': {
            'customer_id': 'uuid4',
            'name': 'name',
            'address': {
                'street': 'street_address',
                'city': 'city',
                'country': 'country'
            }
        },
        'line_items': [{
            '_qen_count': [1, 10],
            'product': {
                'product_id': ('pystr', {'max_chars': 10}),
                'name': 'word',
                'price': ('pydecimal', {'left_digits': 3, 'right_digits': 2, 'positive': True})
            },
            'quantity': ('pyint', {'min_value': 1, 'max_value': 5}),
            'total': {'_qen_provider': 'lambda', 'func': 'lambda ctx: ctx["product"]["price"] * ctx["quantity"]'}
        }]
    }

    print(f"\n    {suite._c.grey}generating {num_records} complex records... (this may take a moment){suite._c.reset}")
    start_time = time.perf_counter()
    data = from_schema(stress_schema, seed=99).take(num_records).to_list()
    duration = time.perf_counter() - start_time
    print(f"    {suite._c.grey}└─> generation finished in {duration:.2f} seconds.{suite._c.reset}")
    assert_that(len(data) == num_records, f"expected {num_records} records, but got {len(data)}")
    last_item = data[-1]
    assert_that('order_id' in last_item and 'customer' in last_item, "toplevel keys are missing in stress test data")
    assert_that(isinstance(last_item['line_items'], list) and len(last_item['line_items']) > 0,
                "nested list is invalid")
    first_line_item = last_item['line_items'][0]
    expected_total = first_line_item['product']['price'] * first_line_item['quantity']
    assert_that(first_line_item['total'] == expected_total, "lambda calculation in nested list is incorrect")


if __name__ == "__main__":
    suite.run(title="qen test suite (comprehensive)")