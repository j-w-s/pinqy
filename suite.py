import time
import traceback
from functools import wraps
from typing import List, Dict, Any, Callable

_suite_state: Dict[str, List[Dict[str, Any]]] = {
    'tests': [],
    'results': []
}

PASS_FACE = '(^ ω ^)'
FAIL_FACE = '(ﾉಥДಥ)ﾉ'
SUMMARY_FACE = '☆*:.｡.o(≧▽≦)o.｡.:*☆'


class _c:
    """a tiny, silent class for holding color codes."""
    ok = '\033[92m'
    fail = '\033[91m'
    warn = '\033[93m'
    info = '\033[94m'
    grey = '\033[90m'
    reset = '\033[0m'


# --- custom exception for assertions ---

class TestAssertionError(AssertionError):
    """custom error to distinguish assertion failures from other exceptions."""
    pass

# --- public api ---

def test(description: str) -> Callable:
    """decorator to register a function as a test case."""

    def decorator(func: Callable) -> Callable:
        _suite_state['tests'].append({'func': func, 'description': description})

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


def assert_that(condition: Any, message: str = "assertion failed") -> None:
    """custom assertion that raises a specific, catchable error type."""
    if not condition:
        raise TestAssertionError(message)


def run(title: str = "test run") -> None:
    """executes all registered tests and prints a report."""
    print(f"\n{_c.info}--- starting: {title} ---{_c.reset}")
    start_time = time.perf_counter()

    _suite_state['results'] = []
    tests_to_run = _suite_state['tests']

    for test_item in tests_to_run:
        func = test_item['func']
        description = test_item['description']

        passed = False
        error = None

        try:
            func()
            passed = True
        except TestAssertionError as e:
            error = f"assertion failed: {e}"
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            # for debugging unexpected errors, uncomment the line below
            # traceback.print_exc()

        _suite_state['results'].append({'passed': passed, 'description': description, 'error': error})

        if passed:
            print(f"  {_c.ok}✔ pass{_c.reset}  {PASS_FACE}  {description}")
        else:
            print(f"  {_c.fail}✖ fail{_c.reset}  {FAIL_FACE}  {description}")
            print(f"    {_c.grey}└─> {error}{_c.reset}")

    _print_summary(start_time)

    # clear tests after run to allow for multiple, separate suite runs in a single script
    _suite_state['tests'] = []


def _print_summary(start_time: float) -> None:
    """prints the final summary of the test run."""
    duration = (time.perf_counter() - start_time) * 1000
    results = _suite_state['results']

    total = len(results)
    passed_count = sum(1 for r in results if r['passed'])
    failed_count = total - passed_count

    summary_color = _c.ok if failed_count == 0 else _c.fail

    print(f"\n{summary_color}--- summary ---{_c.reset}")
    print(f"  {SUMMARY_FACE}  ran {_c.info}{total}{_c.reset} tests in {_c.warn}{duration:.2f}ms{_c.reset}")
    print(f"  {_c.ok}passed: {passed_count}{_c.reset}, {_c.fail}failed: {failed_count}{_c.reset}")
    print(f"{summary_color}---------------{_c.reset}\n")