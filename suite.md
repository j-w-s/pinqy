### suite.py README

a minimal, zero-dependency testing utility for python, designed for simplicity and readability.

## core concepts

`suite` provides a simple way to write and run tests without the overhead of a large framework.

-   **decorator-based**: tests are plain python functions marked with the `@test` decorator.
-   **simple assertions**: uses a straightforward `assert_that` function to provide clean failure messages.
-   **clear output**: provides a color-coded summary of passes, failures, and unexpected errors.
-   **no magic**: no complex test discovery rules. any function decorated with `@test` in a file is added to the run when `run()` is called.

## quick start

create a test file, define your tests, and run them.

```python
# test_math.py
from suite import test, assert_that, run

# a simple function to test
def add(a, b):
    return a + b

# a test that will pass
@test("addition should work correctly")
def test_addition():
    result = add(2, 2)
    assert_that(result == 4, "2+2 should be 4")

# a test that will fail due to a bad assertion
@test("subtraction should be commutative (it's not)")
def test_subtraction_fail():
    assert_that(5 - 3 == 3 - 5, "subtraction is not commutative")

# a test that will fail with an unexpected error
@test("division should handle zero correctly")
def test_division_error():
    result = 1 / 0
    # this test will fail with a zerodivisionerror

# run all decorated tests in this file
if __name__ == "__main__":
    run(title="math function tests")
```

when you execute `python test_math.py`, you will see a formatted report in your terminal detailing the results of each test.

## api reference

#### `@test(description: str)`
a decorator used to register a function as a test case.

-   **`description`**: a human-readable string describing what the test is checking. this description is used in the final report.

```python
# a test function demonstrating the decorator
@test("a user's name should be capitalized")
def test_name_capitalization():
    user_name = "john"
    capitalized = user_name.capitalize()
    assert_that(capitalized == "John")
```

#### `assert_that(condition: any, message: str = "assertion failed")`
a custom assertion function for validating conditions within a test. if the condition is `false`, it raises a special `testassertionerror` which is caught by the runner and reported cleanly.

-   **`condition`**: the expression to evaluate. if `false`, the test fails.
-   **`message`**: the error message to display if the assertion fails.

```python
# usage inside a test function
@test("list should contain expected items")
def test_list_contents():
    my_list = [1, 2, 3]
    assert_that(len(my_list) == 3, "list should have 3 items")
    assert_that(2 in my_list, "list should contain the number 2")
```

#### `run(title: str = "test run")`
executes all tests that have been registered with the `@test` decorator and prints a summary report.

-   **`title`**: an optional title for the test run, printed at the top of the report.

```python
# typically placed at the end of a test file
if __name__ == "__main__":
    run(title="api endpoint tests")
```