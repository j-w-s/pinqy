"""
'    __________.___ _______   ________ _____.___.
'    \______   \   |\      \  \_____  \\__  |   |
'     |     ___/   |/   |   \  /  / \  \/   |   |
'     |    |   |   /    |    \/   \_/.  \____   |
'     |____|   |___\____|__  /\_____\ \_/ ______|
'                          \/        \__>/
"""

# expose the main classes
from .enumerable import Enumerable, OrderedEnumerable

# expose the factory functions
from .factories import (
    from_iterable,
    from_range,
    repeat,
    empty,
    generate,
    pinqy,
    P,
    # new factory functions
    create_tree_from_flat,
    recursive_generator
)

# expose supporting data classes
from .types import (
    TreeNode,
    ParseResult,
    NestedGroup,
    TreeItem,
    MemoizedEnumerable
)

# define what `import *` does
__all__ = [
    "Enumerable",
    "OrderedEnumerable",
    "from_iterable",
    "from_range",
    "repeat",
    "empty",
    "generate",
    "pinqy",
    "P",
    "create_tree_from_flat",
    "recursive_generator",
    "TreeNode",
    "ParseResult",
    "NestedGroup",
    "TreeItem",
    "MemoizedEnumerable"
]