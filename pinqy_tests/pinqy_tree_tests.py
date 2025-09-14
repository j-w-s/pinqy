import time
import suite
from dgen import from_schema
from pinqy import P, from_iterable, empty

test = suite.test
assert_that = suite.assert_that


# --- test data structures ---

class TreeNode:
    def __init__(self, value, children=None):
        self.value = value
        self.children = children or []

    def __repr__(self):
        return f"TreeNode({self.value})"

    def __eq__(self, other):
        return isinstance(other, TreeNode) and self.value == other.value


# sample tree structure for testing
def build_sample_tree():
    root = TreeNode('root')
    child1 = TreeNode('child1')
    child2 = TreeNode('child2')
    grandchild1 = TreeNode('grandchild1')
    grandchild2 = TreeNode('grandchild2')
    grandchild3 = TreeNode('grandchild3')

    child1.children = [grandchild1, grandchild2]
    child2.children = [grandchild3]
    root.children = [child1, child2]

    return [root]


# filesystem-like structure
def build_filesystem_tree():
    return [
        {'name': 'src', 'type': 'dir', 'size': None, 'children': [
            {'name': 'main.py', 'type': 'file', 'size': 1024, 'children': None},
            {'name': 'utils.py', 'type': 'file', 'size': 512, 'children': None},
            {'name': 'tests', 'type': 'dir', 'size': None, 'children': [
                {'name': 'test_main.py', 'type': 'file', 'size': 256, 'children': None}
            ]}
        ]},
        {'name': 'docs', 'type': 'dir', 'size': None, 'children': [
            {'name': 'readme.md', 'type': 'file', 'size': 128, 'children': None}
        ]}
    ]


# organizational hierarchy
def build_org_hierarchy():
    return [
        {'name': 'CEO', 'level': 1, 'reports': [
            {'name': 'CTO', 'level': 2, 'reports': [
                {'name': 'Dev Lead', 'level': 3, 'reports': [
                    {'name': 'Senior Dev', 'level': 4, 'reports': []},
                    {'name': 'Junior Dev', 'level': 4, 'reports': []}
                ]},
                {'name': 'QA Lead', 'level': 3, 'reports': [
                    {'name': 'QA Tester', 'level': 4, 'reports': []}
                ]}
            ]},
            {'name': 'CFO', 'level': 2, 'reports': [
                {'name': 'Accountant', 'level': 3, 'reports': []}
            ]}
        ]}
    ]


# flat data for tree building
flat_org_data = [
    {'id': 1, 'name': 'CEO', 'parent_id': None},
    {'id': 2, 'name': 'CTO', 'parent_id': 1},
    {'id': 3, 'name': 'CFO', 'parent_id': 1},
    {'id': 4, 'name': 'Dev Lead', 'parent_id': 2},
    {'id': 5, 'name': 'QA Lead', 'parent_id': 2},
    {'id': 6, 'name': 'Senior Dev', 'parent_id': 4},
    {'id': 7, 'name': 'Junior Dev', 'parent_id': 4},
    {'id': 8, 'name': 'QA Tester', 'parent_id': 5},
    {'id': 9, 'name': 'Accountant', 'parent_id': 3}
]


# --- recursive_select tests ---

@test("recursive_select traverses simple tree depth-first")
def test_recursive_select_basic():
    tree = build_sample_tree()
    result = P(tree).tree.recursive_select(lambda node: node.children)
    values = result.select(lambda node: node.value).to.list()

    expected = ['root', 'child1', 'grandchild1', 'grandchild2', 'child2', 'grandchild3']
    assert_that(values == expected, f"expected {expected}, got {values}")


@test("recursive_select works with dict-based trees")
def test_recursive_select_dict():
    fs_tree = build_filesystem_tree()
    result = P(fs_tree).tree.recursive_select(lambda item: item.get('children'))
    names = result.select(lambda item: item['name']).to.list()

    expected = ['src', 'main.py', 'utils.py', 'tests', 'test_main.py', 'docs', 'readme.md']
    assert_that(names == expected, f"filesystem traversal incorrect: {names}")


@test("recursive_select handles exclude_parents option")
def test_recursive_select_exclude_parents():
    tree = build_sample_tree()
    result = P(tree).tree.recursive_select(
        lambda node: node.children,
        include_parents=False
    )
    values = result.select(lambda node: node.value).to.list()

    expected = ['child1', 'grandchild1', 'grandchild2', 'child2', 'grandchild3']
    assert_that(values == expected, f"exclude parents failed: {values}")


@test("recursive_select handles empty children gracefully")
def test_recursive_select_empty_children():
    single_node = [TreeNode('lonely')]
    result = P(single_node).tree.recursive_select(lambda node: node.children)
    values = result.select(lambda node: node.value).to.list()

    assert_that(values == ['lonely'], f"single node traversal failed: {values}")


@test("recursive_select handles none children selector")
def test_recursive_select_none_children():
    tree = [{'name': 'item1'}, {'name': 'item2'}]
    result = P(tree).tree.recursive_select(lambda item: None)
    names = result.select(lambda item: item['name']).to.list()

    assert_that(names == ['item1', 'item2'], f"none children handling failed: {names}")


# --- recursive_where tests ---

@test("recursive_where filters tree nodes by predicate")
def test_recursive_where_basic():
    tree = build_sample_tree()
    result = P(tree).tree.recursive_where(
        lambda node: node.children,
        lambda node: 'child' in node.value
    )
    values = result.select(lambda node: node.value).to.list()

    expected = ['child1', 'grandchild1', 'grandchild2', 'child2', 'grandchild3']
    assert_that(values == expected, f"recursive where filtering failed: {values}")


@test("recursive_where works with complex predicates")
def test_recursive_where_complex():
    fs_tree = build_filesystem_tree()
    files_only = P(fs_tree).tree.recursive_where(
        lambda item: item.get('children'),
        lambda item: item['type'] == 'file'
    )
    names = files_only.select(lambda item: item['name']).to.list()

    expected = ['main.py', 'utils.py', 'test_main.py', 'readme.md']
    assert_that(names == expected, f"file filtering failed: {names}")


# --- recursive_select_many tests ---

@test("recursive_select_many flattens multiple results per node")
def test_recursive_select_many():
    org_tree = build_org_hierarchy()
    # extract all names and levels from each node
    result = P(org_tree).tree.recursive_select_many(
        lambda person: person.get('reports', []),
        lambda person: [person['name'], f"level_{person['level']}"]
    )
    flattened = result.to.list()

    assert_that('CEO' in flattened, "should contain CEO")
    assert_that('level_1' in flattened, "should contain level_1")
    assert_that('Senior Dev' in flattened, "should contain Senior Dev")
    assert_that('level_4' in flattened, "should contain level_4")


# --- traverse_with_path tests ---

@test("traverse_with_path maintains path context")
def test_traverse_with_path():
    tree = build_sample_tree()
    result = P(tree).tree.traverse_with_path(lambda node: node.children)

    # check a specific deep node
    grandchild1_node = result.where(
        lambda tnode: tnode.value.value == 'grandchild1'
    ).to.first()

    path_values = P(grandchild1_node.path).select(lambda node: node.value).to.list()
    expected_path = ['root', 'child1', 'grandchild1']

    assert_that(path_values == expected_path, f"path incorrect: {path_values}")
    assert_that(grandchild1_node.depth == 2, f"depth should be 2, got {grandchild1_node.depth}")


@test("traverse_with_path handles multiple root nodes")
def test_traverse_with_path_multiple_roots():
    fs_tree = build_filesystem_tree()
    result = P(fs_tree).tree.traverse_with_path(lambda item: item.get('children'))

    # verify root nodes have depth 0
    root_nodes = result.where(lambda tnode: tnode.depth == 0).to.list()
    root_names = P(root_nodes).select(lambda tnode: tnode.value['name']).to.list()

    expected_roots = ['src', 'docs']
    assert_that(sorted(root_names) == sorted(expected_roots), f"root detection failed: {root_names}")


# --- select_recursive tests ---

@test("select_recursive applies different logic to leaves vs branches")
def test_select_recursive():
    fs_tree = build_filesystem_tree()
    result = P(fs_tree).tree.select_recursive(
        leaf_selector=lambda item: item['size'],  # files have size
        child_selector=lambda item: item.get('children'),
        branch_selector=lambda item, child_results: sum(r for r in child_results if r is not None)
    )

    sizes = result.to.list()
    # should calculate total sizes: src dir = 1024+512+256=1792, docs dir = 128
    assert_that(1792 in sizes, f"src directory total size missing: {sizes}")
    assert_that(128 in sizes, f"docs directory total size missing: {sizes}")


@test("select_recursive handles pure leaf trees")
def test_select_recursive_leaf_only():
    leaf_data = [
        {'name': 'file1', 'size': 100, 'children': None},
        {'name': 'file2', 'size': 200, 'children': None}
    ]

    result = P(leaf_data).tree.select_recursive(
        leaf_selector=lambda item: item['size'],
        child_selector=lambda item: item.get('children'),
        branch_selector=lambda item, child_results: sum(child_results)
    )

    sizes = result.to.list()
    assert_that(sorted(sizes) == [100, 200], f"leaf-only processing failed: {sizes}")


# --- reduce_tree tests ---

@test("reduce_tree performs functional reduce with depth context")
def test_reduce_tree():
    tree = build_sample_tree()
    # count total nodes and calculate max depth
    result = P(tree).tree.reduce_tree(
        child_selector=lambda node: node.children,
        seed={'count': 0, 'max_depth': 0},
        accumulator=lambda acc, node, depth: {
            'count': acc['count'] + 1,
            'max_depth': max(acc['max_depth'], depth)
        }
    )

    assert_that(result['count'] == 6, f"total node count incorrect: {result['count']}")
    assert_that(result['max_depth'] == 2, f"max depth incorrect: {result['max_depth']}")


@test("reduce_tree calculates weighted sums by depth")
def test_reduce_tree_weighted():
    org_tree = build_org_hierarchy()
    # calculate weighted sum where deeper levels have higher weight
    weight_sum = P(org_tree).tree.reduce_tree(
        child_selector=lambda person: person.get('reports', []),
        seed=0,
        accumulator=lambda acc, person, depth: acc + (person['level'] * (depth + 1))
    )

    assert_that(weight_sum > 0, f"weighted sum should be positive: {weight_sum}")


# --- build_tree tests ---

@test("build_tree constructs hierarchy from flat data")
def test_build_tree():
    result = P(flat_org_data).tree.build_tree(
        key_selector=lambda item: item['id'],
        parent_key_selector=lambda item: item['parent_id'],
        root_key=None
    )

    tree_items = result.to.list()
    assert_that(len(tree_items) == 1, f"should have one root: {len(tree_items)}")

    root = tree_items[0]
    assert_that(root.value['name'] == 'CEO', f"root should be CEO: {root.value['name']}")
    assert_that(len(root.children) == 2, f"CEO should have 2 direct reports: {len(root.children)}")


@test("build_tree handles multiple root scenario")
def test_build_tree_multiple_roots():
    multi_root_data = [
        {'id': 1, 'name': 'Root1', 'parent_id': None},
        {'id': 2, 'name': 'Root2', 'parent_id': None},
        {'id': 3, 'name': 'Child1', 'parent_id': 1}
    ]

    result = P(multi_root_data).tree.build_tree(
        key_selector=lambda item: item['id'],
        parent_key_selector=lambda item: item['parent_id'],
        root_key=None
    )

    roots = result.to.list()
    assert_that(len(roots) == 2, f"should have two roots: {len(roots)}")

    root_names = P(roots).select(lambda r: r.value['name']).to.set()
    assert_that(root_names == {'Root1', 'Root2'}, f"root names incorrect: {root_names}")


@test("build_tree handles empty input")
def test_build_tree_empty():
    result = P([]).tree.build_tree(
        key_selector=lambda x: x['id'],
        parent_key_selector=lambda x: x['parent'],
        root_key=None
    )

    assert_that(result.to.count() == 0, "empty input should produce empty tree")


# --- group_by_recursive tests ---

@test("group_by_recursive maintains hierarchical relationships")
def test_group_by_recursive():
    org_tree = build_org_hierarchy()

    def get_reports(person):
        return person.get('reports', [])

    result = P(org_tree).tree.group_by_recursive(
        key_selector=lambda person: person['level'],
        child_selector=get_reports
    )

    groups = result.to.list()
    level_1_group = P(groups).where(lambda g: g.key == 1).to.first()

    assert_that(len(level_1_group.items) == 1, "level 1 should have 1 person")
    assert_that(level_1_group.items[0]['name'] == 'CEO', "level 1 person should be CEO")
    assert_that(len(level_1_group.children) > 0, "level 1 should have child groups")


# --- integration and chaining tests ---

@test("chaining: recursive traversal with filtering and transformation")
def test_chaining_recursive_ops():
    fs_tree = build_filesystem_tree()

    # find all python files and calculate total size
    python_files = (P(fs_tree)
                    .tree.recursive_select(lambda item: item.get('children'))
                    .where(lambda item: item['type'] == 'file')
                    .where(lambda item: item['name'].endswith('.py'))
                    .select(lambda item: {'name': item['name'], 'size': item['size']}))

    total_size = python_files.stats.sum(lambda f: f['size'])
    file_names = python_files.select(lambda f: f['name']).to.list()

    expected_names = ['main.py', 'utils.py', 'test_main.py']
    expected_total = 1024 + 512 + 256

    assert_that(sorted(file_names) == sorted(expected_names), f"python files: {file_names}")
    assert_that(total_size == expected_total, f"total size: {total_size}")


@test("chaining: tree traversal with path analysis")
def test_chaining_path_analysis():
    tree = build_sample_tree()

    # find deepest nodes and analyze their paths
    deepest_nodes = (P(tree)
                     .tree.traverse_with_path(lambda node: node.children)
                     .where(lambda tnode: tnode.depth == 2)  # grandchildren
                     .select(lambda tnode: {
        'name': tnode.value.value,
        'path_length': len(tnode.path),
        'path_names': P(tnode.path).select(lambda n: n.value).to.list()
    }))

    results = deepest_nodes.to.list()
    assert_that(len(results) == 3, f"should have 3 grandchildren: {len(results)}")

    for result in results:
        assert_that(result['path_length'] == 3, f"path length should be 3: {result}")
        assert_that(result['path_names'][0] == 'root', f"path should start with root: {result}")


@test("chaining: build tree from flat data then traverse")
def test_chaining_build_then_traverse():
    # build tree and immediately traverse to find all leaf nodes
    leaf_employees = (P(flat_org_data)
                      .tree.build_tree(
        key_selector=lambda x: x['id'],
        parent_key_selector=lambda x: x['parent_id'],
        root_key=None
    )
                      .tree.recursive_select(lambda tree_item: tree_item.children)
                      .where(lambda tree_item: len(tree_item.children) == 0)
                      .select(lambda tree_item: tree_item.value['name']))

    leaf_names = leaf_employees.to.list()
    expected_leaves = ['Senior Dev', 'Junior Dev', 'QA Tester', 'Accountant']

    assert_that(sorted(leaf_names) == sorted(expected_leaves), f"leaf employees: {leaf_names}")


# --- complex data scenarios ---

@test("tree operations with generated hierarchical data")
def test_with_generated_data():
    # generate hierarchical company data
    company_schema = {
        'departments': [{
            '_qen_items': {
                'name': {'_qen_provider': 'choice', 'from': ['Engineering', 'Sales', 'Marketing', 'HR']},
                'budget': ('pyint', {'min_value': 50000, 'max_value': 500000}),
                'teams': [{
                    '_qen_items': {
                        'name': 'word',
                        'size': ('pyint', {'min_value': 2, 'max_value': 8}),
                        'members': [{
                            '_qen_items': {
                                'name': 'name',
                                'role': {'_qen_provider': 'choice', 'from': ['Lead', 'Senior', 'Junior']}
                            },
                            '_qen_count': (2, 5)
                        }]
                    },
                    '_qen_count': (1, 3)
                }]
            },
            '_qen_count': 3
        }]
    }

    company = from_schema(company_schema, seed=42).take(1).to.first()

    # traverse to find all team members and analyze by role
    all_members = (P(company['departments'])
                   .tree.recursive_select(
        lambda dept: dept.get('teams', []) if 'teams' in dept
        else dept.get('members', []) if 'members' in dept else None
    )
                   .where(lambda item: 'role' in item)  # filter to actual members
                   .group.group_by(lambda member: member['role']))

    assert_that(len(all_members) > 0, "should find team members")
    for role, members in all_members.items():
        assert_that(role in ['Lead', 'Senior', 'Junior'], f"unexpected role: {role}")
        assert_that(len(members) > 0, f"role {role} should have members")


# --- performance and edge cases ---

@test("tree operations handle deep recursion")
def test_deep_recursion():
    # create a deep linear tree (linked list style)
    def create_deep_tree(depth):
        if depth == 0:
            return []
        return [{'value': depth, 'children': create_deep_tree(depth - 1)}]

    deep_tree = create_deep_tree(100)

    # traverse and count all nodes
    all_nodes = P(deep_tree).tree.recursive_select(lambda item: item.get('children', []))
    node_count = all_nodes.to.count()

    assert_that(node_count == 100, f"deep tree should have 100 nodes: {node_count}")


@test("tree operations with mixed data types")
def test_mixed_data_types():
    mixed_tree = [
        {'type': 'dict', 'children': [
            {'type': 'nested_dict', 'children': []},
            'string_child',  # string as child
            42,  # number as child
            ['list', 'as', 'child']  # list as child
        ]},
        TreeNode('object_root', [TreeNode('object_child')])  # custom object
    ]

    # traverse with flexible child selector
    def flexible_children(item):
        if isinstance(item, dict) and 'children' in item:
            return item['children']
        elif isinstance(item, TreeNode):
            return item.children
        else:
            return []

    all_items = P(mixed_tree).tree.recursive_select(flexible_children)
    count = all_items.to.count()

    assert_that(count >= 6, f"should traverse mixed types: {count}")


@test("tree operations maintain source integrity")
def test_source_integrity():
    tree = build_sample_tree()
    original_root = tree[0]
    original_children_count = len(original_root.children)

    # perform multiple tree operations
    P(tree).tree.recursive_select(lambda node: node.children).to.list()
    P(tree).tree.traverse_with_path(lambda node: node.children).to.list()
    P(tree).tree.recursive_where(
        lambda node: node.children,
        lambda node: True
    ).to.list()

    # verify source is unchanged
    assert_that(len(original_root.children) == original_children_count,
                "source tree should remain unchanged")
    assert_that(tree[0] is original_root, "root reference should be preserved")


@test("tree benchmark on moderate hierarchical data")
def test_tree_performance():
    print()

    # create moderately deep organizational structure
    def build_large_org(levels, branching_factor):
        def build_level(level, max_level, id_counter=[0]):
            if level > max_level:
                return []

            children = []
            for i in range(branching_factor):
                id_counter[0] += 1
                node = {
                    'id': id_counter[0],
                    'name': f'Employee_{id_counter[0]}',
                    'level': level,
                    'reports': build_level(level + 1, max_level, id_counter)
                }
                children.append(node)
            return children

        return build_level(1, levels)

    large_org = build_large_org(6, 3)  # 6 levels, 3 reports each = ~1093 nodes

    start = time.perf_counter()
    all_employees = P(large_org).tree.recursive_select(
        lambda emp: emp.get('reports', [])
    ).to.list()
    duration_traverse = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    managers = P(large_org).tree.recursive_where(
        lambda emp: emp.get('reports', []),
        lambda emp: len(emp.get('reports', [])) > 0
    ).to.list()
    duration_filter = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    with_paths = P(large_org).tree.traverse_with_path(
        lambda emp: emp.get('reports', [])
    ).to.list()
    duration_paths = (time.perf_counter() - start) * 1000

    print(
        f"    {suite._c.grey}├─> recursive_select on {len(all_employees)} nodes took: {duration_traverse:.2f}ms{suite._c.reset}")
    print(
        f"    {suite._c.grey}├─> recursive_where on {len(managers)} managers took: {duration_filter:.2f}ms{suite._c.reset}")
    print(
        f"    {suite._c.grey}└─> traverse_with_path on {len(with_paths)} nodes took: {duration_paths:.2f}ms{suite._c.reset}")

    assert_that(len(all_employees) > 1000, f"should have many employees: {len(all_employees)}")
    assert_that(len(managers) > 0, f"should have managers: {len(managers)}")
    assert_that(len(with_paths) == len(all_employees), "path traversal should match simple traversal")


if __name__ == "__main__":
    suite.run(title="pinqy tree operations test")