from __future__ import annotations
import typing
from collections import deque
from ..types import *

if typing.TYPE_CHECKING:
    from ..enumerable import Enumerable


class TreeAccessor(Generic[T]):
    """
    provides recursive traversal, tree operations, and advanced functional composition.
    inspired by functional programming patterns for hierarchical data structures.
    """

    def __init__(self, enumerable_instance: 'Enumerable[T]'):
        self._enumerable = enumerable_instance

    def recursive_select(self,
                         child_selector: Callable[[T], Optional[Iterable[T]]],
                         include_parents: bool = True) -> 'Enumerable[T]':
        """
        recursively traverses a tree structure, yielding elements based on child_selector.
        similar to a depth-first traversal of hierarchical data.
        """
        from ..enumerable import Enumerable
        def recursive_data():
            result = []
            stack = list(self._enumerable._get_data())
            initial_count = len(stack)

            while stack:
                item = stack.pop(0)  # use pop(0) for depth-first order
                result.append(item)

                children = child_selector(item)
                if children is not None:
                    # insert children at the beginning of the stack
                    stack[:0] = list(children)

            return result if include_parents else result[initial_count:]

        return Enumerable(recursive_data)

    def recursive_where(self,
                        child_selector: Callable[[T], Optional[Iterable[T]]],
                        predicate: Predicate[T]) -> 'Enumerable[T]':
        """filter elements recursively through tree structure"""
        return self.recursive_select(child_selector).where(predicate)

    def recursive_select_many(self,
                              child_selector: Callable[[T], Optional[Iterable[T]]],
                              result_selector: Callable[[T], Iterable[U]]) -> 'Enumerable[U]':
        """recursively traverse and flatten results"""
        return self.recursive_select(child_selector).select_many(result_selector)

    def traverse_with_path(self,
                           child_selector: Callable[[T], Optional[Iterable[T]]]) -> 'Enumerable[TreeNode[T]]':
        """
        traverse tree structure while maintaining path context and depth information.
        returns TreeNode objects containing value, path, and depth.
        """
        from ..enumerable import Enumerable
        def traverse_data():
            result = []
            # stack contains (item, path, depth) tuples
            stack = [(item, [item], 0) for item in reversed(self._enumerable._get_data())]

            while stack:
                item, path, depth = stack.pop()
                result.append(TreeNode(item, path, depth))

                children = child_selector(item)
                if children is not None:
                    child_list = list(children)
                    # reverse to maintain order with stack
                    for child in reversed(child_list):
                        new_path = path + [child]
                        stack.append((child, new_path, depth + 1))

            return result

        return Enumerable(traverse_data)

    def select_recursive(self,
                         leaf_selector: Callable[[T], U],
                         child_selector: Callable[[T], Optional[Iterable[T]]],
                         branch_selector: Callable[[T, List[U]], U]) -> 'Enumerable[U]':
        """
        recursive selection with different logic for leaves vs branches.
        applies leaf_selector to leaf nodes, branch_selector to internal nodes.
        """
        from ..enumerable import Enumerable
        def recursive_select_data():
            def process_item(item: T) -> U:
                children = child_selector(item)
                child_list = list(children) if children is not None else []

                if not child_list:
                    # leaf node
                    return leaf_selector(item)
                else:
                    # branch node - process children recursively
                    child_results = [process_item(child) for child in child_list]
                    return branch_selector(item, child_results)

            return [process_item(item) for item in self._enumerable._get_data()]

        return Enumerable(recursive_select_data)

    def reduce_tree(self,
                    child_selector: Callable[[T], Optional[Iterable[T]]],
                    seed: U,
                    accumulator: Callable[[U, T, int], U]) -> U:
        """
        functional reduce operation with tree context.
        accumulator receives (current_value, item, depth).
        """

        def reduce_internal(items: List[T], current_seed: U, depth: int) -> U:
            result = current_seed
            for item in items:
                result = accumulator(result, item, depth)
                children = child_selector(item)
                if children is not None:
                    result = reduce_internal(list(children), result, depth + 1)
            return result

        return reduce_internal(self._enumerable._get_data(), seed, 0)

    def build_tree(self,
                   key_selector: KeySelector[T, K],
                   parent_key_selector: KeySelector[T, K],
                   root_key: K = None) -> 'Enumerable[TreeItem[T, K]]':
        """
        builds hierarchical tree structure from flat data.
        uses parent-child key relationships to construct tree.
        """
        from ..enumerable import Enumerable
        from collections import defaultdict
        def build_tree_data():
            # create lookup for efficient parent-child mapping
            lookup = defaultdict(list)
            for item in self._enumerable._get_data():
                parent_key = parent_key_selector(item)
                lookup[parent_key].append(item)

            def build_tree_internal(items: List[T]) -> List[TreeItem[T, K]]:
                result = []
                for item in items:
                    key = key_selector(item)
                    children = build_tree_internal(lookup.get(key, []))
                    result.append(TreeItem(item, children))
                return result

            return build_tree_internal(lookup.get(root_key, []))

        return Enumerable(build_tree_data)

    def group_by_recursive(self,
                           key_selector: KeySelector[T, K],
                           child_selector: Callable[[T], Optional[Iterable[T]]]) -> 'Enumerable[NestedGroup[K, T]]':
        """recursively group elements maintaining hierarchical relationships"""
        from ..enumerable import Enumerable
        from ..factories import from_iterable
        def recursive_group_data():
            groups = self._enumerable.group.group_by(key_selector)
            result = []

            for key, items in groups.items():
                # collect all children from items in this group
                all_children = []
                for item in items:
                    children = child_selector(item)
                    if children is not None:
                        all_children.extend(children)

                # recursively process children if any exist
                child_groups = []
                if all_children:
                    child_groups = from_iterable(all_children).tree.group_by_recursive(
                        key_selector, child_selector).to.list()

                result.append(NestedGroup(key, items, child_groups))

            return result

        return Enumerable(recursive_group_data)