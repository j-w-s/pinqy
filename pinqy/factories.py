import typing
from .types import *
from collections import deque

if typing.TYPE_CHECKING:
    from .enumerable import Enumerable

def from_iterable(data: Iterable[T]) -> 'Enumerable[T]':
    """create enumerable from iterable"""
    from .enumerable import Enumerable
    return Enumerable(lambda: list(data))

def from_range(start: int, count: int) -> 'Enumerable[int]':
    """create enumerable from range"""
    from .enumerable import Enumerable
    return Enumerable(lambda: list(range(start, start + count)))

def repeat(item: T, count: int) -> 'Enumerable[T]':
    """create enumerable with repeated item"""
    from .enumerable import Enumerable
    return Enumerable(lambda: [item] * count)

def empty() -> 'Enumerable[Any]':
    """create empty enumerable"""
    from .enumerable import Enumerable
    return Enumerable(lambda: [])

def generate(generator_func: Callable[[], T], count: int) -> 'Enumerable[T]':
    """generate sequence using a function"""
    from .enumerable import Enumerable
    return Enumerable(lambda: [generator_func() for _ in range(count)])

# --- new factory functions ---

def create_tree_from_flat(data: List[T],
                         key_selector: KeySelector[T, K],
                         parent_key_selector: KeySelector[T, K],
                         root_key: K = None) -> 'Enumerable[TreeItem[T, K]]':
    """factory function for creating trees from flat data"""
    return from_iterable(data).tree.build_tree(key_selector, parent_key_selector, root_key)

def recursive_generator(seed: T,
                       child_generator: Callable[[T], Optional[Iterable[T]]],
                       max_depth: int = 100) -> 'Enumerable[T]':
    """factory function for recursive generation with depth limiting"""
    from .enumerable import Enumerable
    def generate_recursive():
        result = []
        queue = deque([(seed, 0)])

        while queue:
            item, depth = queue.popleft()
            if depth >= max_depth:
                continue

            result.append(item)
            children = child_generator(item)
            if children is not None:
                for child in children:
                    queue.append((child, depth + 1))
        return result

    return Enumerable(generate_recursive)

# --- aliases ---
pinqy = from_iterable
P = from_iterable
p = from_iterable