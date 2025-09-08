### pinqy - linq-like operations for python

```
    __________.___ _______   ________ _____.___.
    \______   \   |\      \  \_____  \\__  |   |
     |     ___/   |/   |   \  /  / \  \/   |   |
     |    |   |   /    |    \/   \_/.  \____   |
     |____|   |___\____|__  /\_____\ \_/ ______|
                          \/        \__>/
```

A functional programming library that brings linq-style query operations to python with lazy evaluation, numpy optimization, and extensive statistical/analytical capabilities.

---

### Table of Contents

1.  [**Core Concepts & Installation**](#core-concepts--installation)
2.  [**Factory Functions**](#factory-functions)
3.  [**Core `Enumerable` Operations**](#core-enumerable-operations)
4.  [**`OrderedEnumerable` Operations**](#orderedenumerable-operations)
5.  [**Accessor Methods**](#accessor-methods)
    *   [**`.set`**: Set Operations](#set-set-operations)
    *   [**`.group`**: Grouping & Structural](#group-grouping--structural)
    *   [**`.join`**: Joining Sequences](#join-joining-sequences)
    *   [**`.zip`**: Zipping Sequences](#zip-zipping-sequences)
    *   [**`.comb`**: Combinatorics](#comb-combinatorics)
    *   [**`.stats`**: Statistical & Mathematical](#stats-statistical--mathematical)
    *   [**`.util`**: Utility & Functional](#util-utility--functional)
    *   [**`.tree`**: Tree & Hierarchical Data](#tree-tree--hierarchical-data)
6.  [**Terminal Operations (`.to`)**](#terminal-operations-to)

---

### Core Concepts & Installation

*   **Lazy Evaluation**: operations build a pipeline but don't execute until a terminal method (e.g., `.to.list()`) is called. this optimizes performance by processing only the data that is actually needed.
*   **Numpy Optimization**: for enumerables containing numeric data, many operations are automatically accelerated using numpy's vectorized functions.
*   **Accessors**: functionality is organized into logical namespaces like `.set`, `.group`, `.stats`, etc., to keep the api clean and discoverable.

**Installation**:
```bash
pip install -r requirements.txt
# then include the pinqy package in your project
```
---

### Factory Functions

| function(s) | parameters | description | example |
| :--- | :--- | :--- | :--- |
| `pinqy()`<br>`p()`<br>`P()`<br>`from_iterable()` | `data: iterable` | creates an enumerable from any iterable. the primary starting point. | `p([1, 2, 3])`<br>`pinqy(range(10))`<br>`pinqy("hello")` |
| `from_range()` | `start: int`, `count: int` | creates an enumerable of integers from `start` to `start + count - 1`. | `from_range(10, 5).to.list()`<br/>`# [10, 11, 12, 13, 14]` |
| `repeat()` | `item: t`, `count: int` | creates an enumerable containing one item repeated `count` times. | `repeat("hello", 3).to.list()`<br/>`# ["hello", "hello", "hello"]` |
| `empty()` | - | creates an empty enumerable. | `empty().to.count() # 0` |
| `generate()` | `generator_func: callable`, `count: int` | creates an enumerable by calling `generator_func` `count` times. | `import random`<br/>`generate(lambda: random.randint(1, 10), 5)` |
| `create_tree_from_flat()` | `data`, `key`, `parent_key`, `root_key` | builds a tree from a flat list. (see `.tree.build_tree` for example) | `create_tree_from_flat(data, ...)` |
| `recursive_generator()` | `seed`, `child_generator`, `max_depth` | creates a sequence by recursively generating children from a seed. | `recursive_generator(1, lambda n: [n*2, n*2+1] if n<8 else [])` |

---

### Core `Enumerable` Operations

| method | parameters | description | example |
| :--- | :--- | :--- | :--- |
| `.where()` | `predicate` | filters a sequence. *numpy optimized*. | `p([1,2,3,4]).where(lambda x: x % 2 == 0)`<br/>`# yields 2, 4` |
| `.select()` | `selector` | projects each element into a new form. *numpy optimized*. | `people.select(lambda p: p['name'])` |
| `.select_many()` | `selector` | projects to an iterable and flattens the result. | `p(['hello world', 'pinqy lib']).select_many(lambda s: s.split())`<br/>`# yields 'hello', 'world', 'pinqy', 'lib'` |
| `.select_with_index()` | `selector` | projects each element using its zero-based index. | `p(['a','b']).select_with_index(lambda val, i: f"{i}:{val}")`<br/>`# yields '0:a', '1:b'` |
| `.of_type()` | `type_filter` | filters elements based on a specified type. | `p([1, "a", 2.0]).of_type(str)`<br/>`# yields "a"` |
| `.order_by()` | `key_selector` | sorts elements in ascending order. **returns `orderedenumerable`**. | `people.order_by(lambda p: p['age'])` |
| `.order_by_descending()` | `key_selector` | sorts elements in descending order. **returns `orderedenumerable`**. | `people.order_by_descending(lambda p: p['salary'])` |
| `.reverse()` | - | inverts the order of the elements. | `p([1,2,3]).reverse().to.list() # [3, 2, 1]` |
| `.as_ordered()` | - | treats sequence as pre-sorted, allowing `.then_by()`. | `pre_sorted_data.as_ordered().then_by(...)` |
| `.take()` | `count: int` | returns a specified number of elements from the start. | `p(range(100)).take(3).to.list() # [0, 1, 2]` |
| `.skip()` | `count: int` | bypasses a specified number of elements. | `p(range(10)).skip(7).to.list() # [7, 8, 9]` |
| `.take_while()` | `predicate` | returns elements as long as a condition is true, then stops. | `p([1,2,5,1]).take_while(lambda x: x < 4).to.list() # [1, 2]` |
| `.skip_while()` | `predicate` | bypasses elements as long as a condition is true. | `p([1,2,5,1]).skip_while(lambda x: x < 4).to.list() # [5, 1]` |
| `.append()` | `element` | appends a value to the end of the sequence. | `p([1,2]).append(3).to.list() # [1, 2, 3]` |
| `.prepend()` | `element` | adds a value to the beginning of the sequence. | `p([1,2]).prepend(0).to.list() # [0, 1, 2]` |
| `.default_if_empty()` | `default_value` | returns the sequence, or one with the default value if empty. | `p([]).default_if_empty(-1).to.list() # [-1]` |

---

### `OrderedEnumerable` Operations

*these methods are only available after an `order_by`, `order_by_descending`, or `as_ordered` call.*

| method | parameters | description | example |
| :--- | :--- | :--- | :--- |
| `.then_by()` | `key_selector` | performs a subsequent ascending sort. | `users.order_by(lambda u: u.dept).then_by(lambda u: u.age)` |
| `.then_by_descending()` | `key_selector` | performs a subsequent descending sort. | `users.order_by(u.city).then_by_descending(u.salary)` |
| `.find_by_key()` | `*key_prefix` | efficiently finds items matching a key prefix using binary search. | `users.order_by(u.state).then_by(u.city)`<br/>`     .find_by_key("ca", "los angeles")` |
| `.between_keys()` | `lower_bound`, `upper_bound` | gets a slice where sort keys are between bounds. | `products.order_by(p.price).between_keys(10.00, 49.99)` |
| `.merge_with()` | `other` | merges two compatibly sorted sequences (o(n+m)). | `sorted1 = p(d1).order_by(x.id)`<br/>`sorted2 = p(d2).order_by(x.id)`<br/>`merged = sorted1.merge_with(sorted2)` |
| `.lag_in_order()` | `periods`, `fill_value` | shifts elements back `periods` *based on sorted order*. | `prices.order_by(p.date).lag_in_order(1)` |
| `.lead_in_order()` | `periods`, `fill_value` | shifts elements forward `periods` *based on sorted order*. | `prices.order_by(p.date).lead_in_order(1)` |

---

### Accessor Methods

#### `.set`: Set Operations

| method | parameters | description | example |
| :--- | :--- | :--- | :--- |
| `.distinct()` | `key_selector=none`| returns distinct elements. *numpy optimized*. | `p([1,2,1,3]).set.distinct().to.list() # [1, 2, 3]`<br/>`people.set.distinct(lambda p: p['city'])` |
| `.union()` | `other: iterable` | set union (distinct, order-preserved). | `p([1,2]).set.union([2,3,4]).to.list() # [1, 2, 3, 4]` |
| `.intersect()`| `other: iterable` | set intersection, preserving order from first. | `p([1,2,3]).set.intersect([2,4,3]).to.list() # [2, 3]` |
| `.except_()` | `other: iterable` | set difference (items in first but not second). | `p([1,2,3]).set.except_([2,4]).to.list() # [1, 2]` |
| `.symmetric_difference()` | `other` | items in one or the other, but not both. | `p([1,2,3]).set.symmetric_difference([2,4,5]).to.list()`<br/>`# [1, 3, 4, 5]` |
| `.concat()` | `other: iterable` | concatenates sequences, preserving duplicates. | `p([1,2]).set.concat([2,3]).to.list() # [1, 2, 2, 3]` |
| `.multiset_intersect()`| `other` | multiset (bag) intersection, respecting counts. | `p([1,1,2,3]).set.multiset_intersect([1,2,2]).to.list() # [1, 2]` |
| `.except_by_count()`| `other` | multiset (bag) difference, respecting counts. | `p([1,1,2,3]).set.except_by_count([1,2,2]).to.list() # [1, 3]` |
| `.is_subset_of()` | `other` | returns `bool`. | `p([1,2]).set.is_subset_of([1,2,3]) # true` |
| `.is_superset_of()`| `other` | returns `bool`. | `p([1,2,3]).set.is_superset_of([1,2]) # true` |
| `.is_proper_subset_of()`| `other`| returns `bool`. | `p([1,2]).set.is_proper_subset_of([1,2]) # false` |
| `.is_proper_superset_of()`|`other`| returns `bool`. | `p([1,2,3]).set.is_proper_superset_of([1,2]) # true` |
| `.is_disjoint_with()` | `other` | returns `bool`. | `p([1,2]).set.is_disjoint_with([3,4]) # true` |
| `.jaccard_similarity()`| `other` | returns `float` (0.0 to 1.0). | `p([1,2,3]).set.jaccard_similarity([2,3,4]) # 0.5` |

#### `.group`: Grouping & Structural

| method | parameters | description | example |
| :--- | :--- | :--- | :--- |
| `.group_by()` | `key_selector` | groups by key. **returns `dict`**. | `people.group.group_by(lambda p: p['department'])`<br/>`# {'eng': [p1, p2], 'sales': [p3]}` |
| `.group_by_multiple()` | `*key_selectors`| groups by tuple key. **returns `dict`**. | `people.group.group_by_multiple(p.dept, p.level)`<br/>`# {('eng','sr'): [...], ...}` |
| `.group_by_with_aggregate()`| `key, element, result` | groups by key and transforms each group. `result` selector receives `(key, enumerable_group)`. **returns `dict`**. | `people.group.group_by_with_aggregate(`<br/>`  lambda p: p['city'],`<br/>`  lambda p: p['salary'],`<br/>`  lambda city, sals: sals.stats.average())`<br/>`# {'ny': 95000, 'lon': 88000}` |
| `.pivot()` | `row_selector`, `column_selector`, `aggregator` | creates a pivot table. **returns nested `dict`**. | `sales.group.pivot(`<br/>`  row_selector=lambda r: r['year'],`<br/>`  column_selector=lambda r: r['product'],`<br/>`  aggregator=lambda g: g.stats.sum(lambda s: s['sales']))`<br/>`# {2023: {'a': 150, 'b': 150}, ...}` |
| `.partition()` | `predicate` | splits into two lists. **returns `(true_list, false_list)`**. | `evens, odds = nums.group.partition(lambda x: x % 2 == 0)` |
| `.chunk()` | `size` | splits into chunks of specified size. | `p(range(10)).group.chunk(3).to.list()`<br/>`# [[0,1,2], [3,4,5], [6,7,8], [9]]` |
| `.batched()` | `size` | (py 3.12+) batches into tuples. | `p(range(10)).group.batched(3).to.list()`<br/>`# [(0,1,2), (3,4,5), (6,7,8), (9,)]` |
| `.window()` | `size` | creates a sliding window of elements. | `p(range(5)).group.window(3).to.list()`<br/>`# [[0,1,2], [1,2,3], [2,3,4]]` |
| `.pairwise()` | - | returns consecutive overlapping pairs. | `p([1,2,3,4]).group.pairwise().to.list() # [(1,2), (2,3), (3,4)]` |
| `.batch_by()` | `key_selector` | groups consecutive elements with same key. | `p([1,1,2,3,3,2]).group.batch_by(lambda x:x).to.list()`<br/>`# [[1,1], [2], [3,3], [2]]` |
| `.group_by_nested()`| `key_selector`, `sub_key_selector`| creates a two-level nested dict. | `data.group.group_by_nested(x.state, x.city)`<br/>`# {'ca': {'la': [...], 'sf': [...]}}` |

#### `.join`: Joining Sequences

| method | parameters | description | example |
| :--- | :--- | :--- | :--- |
| `.join()` | `inner`, `outer_key_selector`, `inner_key_selector`, `result_selector` | inner join on matching keys. | `people.join.join(orders, p['id'], o['cid'],`<br/>`  lambda p, o: {'name': p['name'], 'total': o['total']})` |
| `.left_join()` | `inner`, ..., `default_inner` | left outer join. includes all outer elements. | `people.join.left_join(orders, p['id'], o['cid'],`<br/>`  lambda p, o: {'name': p.name, 'total': o.total if o else 0})` |
| `.right_join()`| `inner`, ..., `default_outer` | right outer join. includes all inner elements. | `p(o).join.right_join(people, ...)` |
| `.full_outer_join()`|`inner`, ..., `defaults` | full outer join. includes all elements from both. | `p(p).join.full_outer_join(orders, ...)` |
| `.group_join()` | `inner`, ..., `result_selector` | correlates and groups results. `result` selector receives `(outer_item, enumerable_of_inners)`. | `people.join.group_join(orders, p['id'], o['cid'],`<br/>`  lambda p, ords: {'name': p.name, 'orders': ords.to.count()})` |
| `.cross_join()`| `inner: iterable` | computes the cartesian product. | `p(['red','blue']).join.cross_join(['s','m'])` |

#### `.zip`: Zipping Sequences

| method | parameters | description | example |
| :--- | :--- | :--- | :--- |
| `.zip_with()` | `other`, `result_selector`| merges sequences with a function. stops at shorter. | `p([1,2]).zip.zip_with(['a','b'], lambda n, l: f"{n}{l}")`<br/>`# yields "1a", "2b"` |
| `.zip_longest_with()`| `other`, `result_selector`, `default_self`, `default_other` | merges sequences, padding shorter with defaults. | `p([1]).zip.zip_longest_with(['a','b'], ..., default_self=0)`<br/>`# yields "1a", "0b"` |

#### `.comb`: Combinatorics

| method | parameters | description | example |
| :--- | :--- | :--- | :--- |
| `.permutations()` | `r: int = none` | generates all permutations of length `r`. | `p("abc").comb.permutations(2).to.list()`<br/>`# [('a','b'), ('a','c'), ('b','a'), ...]` |
| `.combinations()` | `r: int` | generates unique combinations of length `r`. | `p("abc").comb.combinations(2).to.list()`<br/>`# [('a','b'), ('a','c'), ('b','c')]` |
| `.combinations_with_replacement()`|`r: int` | generates combinations where elements can be re-selected. | `p("ab").comb.combinations_with_replacement(2)` |
| `.power_set()` | - | generates all possible subsets. | `p([1,2]).comb.power_set()`<br/>`# yields (), (1,), (2,), (1,2)` |
| `.cartesian_product()`| `*others` | computes cartesian product with other sequences. | `p([1,2]).comb.cartesian_product(['a','b'])` |
| `.binomial_coefficient()`|`r: int` | calculates "n choose r". **returns `int`**. | `p(range(5)).comb.binomial_coefficient(2) # 10` |

#### `.stats`: Statistical & Mathematical

| category | method | parameters | description & example |
| :--- | :--- | :--- | :--- |
| **aggregates** | `.sum()` | `selector=none` | computes sum. `people.stats.sum(lambda p: p['salary'])` |
| | `.average()` | `selector=none` | computes average. `people.stats.average(lambda p: p['age'])` |
| | `.min()` | `selector=none` | returns min value or *element* with min value. `people.stats.min(p.age)` |
| | `.max()` | `selector=none` | returns max value or *element* with max value. `people.stats.max(p.age)` |
| **statistics** | `.std_dev()` | `selector=none` | calculates population standard deviation. |
| | `.median()` | `selector=none` | calculates median value. `p([1,2,100]).stats.median() # 2` |
| | `.percentile()` | `q: float`, `selector=none`| calculates q-th percentile (0-100). `nums.stats.percentile(75)` |
| | `.mode()` | `selector=none` | finds most frequent element. `p([1,2,2,3]).stats.mode() # 2` |
| **rolling** | `.rolling_window()`| `window_size`, `aggregator`| applies custom aggregator to sliding windows. `nums.stats.rolling_window(3, lambda w: sum(w)/len(w))`|
| | `.rolling_sum()` | `window_size`, `selector=none`| calculates a rolling sum. `p([1,2,3,4]).stats.rolling_sum(2)` |
| | `.rolling_average()`| `window_size`, `selector=none`| calculates a rolling average. |
| **time series** | `.lag()` | `periods=1`, `fill_value=none` | shifts elements back. `p([1,2,3,4]).stats.lag(2, 0).to.list() # [0,0,1,2]` |
| | `.lead()` | `periods=1`, `fill_value=none` | shifts elements forward. `p([1,2,3,4]).stats.lead(2, 0).to.list() # [3,4,0,0]` |
| | `.diff()` | `periods=1` | diff between element and previous. `p([10,12,11,15]).stats.diff().to.list() # [2,-1,4]`|
| **cumulative** | `.scan()` | `accumulator`, `seed` | produces intermediate accumulation values. `p([1,2,3]).stats.scan(op.add, 0).to.list() # [0,1,3,6]` |
| | `.cumulative_sum()`| `selector=none` | calculates cumulative sum. `p([1,2,3,4]).stats.cumulative_sum().to.list() # [1,3,6,10]` |
| | `.cumulative_product()`|`selector=none`| calculates cumulative product. |
| | `.cumulative_max()` | `selector=none` | finds cumulative maximum. `p([1,5,2,6]).stats.cumulative_max()` |
| | `.cumulative_min()` | `selector=none` | finds cumulative minimum. |
| **ranking** | `.rank()` | `selector=none`, `ascending=true`| 1-based rank; same rank for ties (1, 2, 2, 4). `p([10,30,20,30]).stats.rank(ascending=False).to.list() # [4,1,3,1]` |
| | `.dense_rank()` | `selector=none`, `ascending=true`| 1-based rank; no gaps for ties (1, 2, 2, 3). `p([10,30,20,30]).stats.dense_rank(ascending=False).to.list() # [3,1,2,1]` |
| | `.quantile_cut()` | `q`, `selector=none`| bins elements into `q` quantiles. |
| **scaling** | `.normalize()` | `selector=none` | min-max normalization to [0, 1]. `p([0,5,10]).stats.normalize()`|
| | `.standardize()` | `selector=none` | z-score standardization (mean=0, std=1). |
| | `.outliers_iqr()`|`selector=none`, `factor=1.5` | detects outliers using the iqr method. |

#### `.util`: Utility & Functional

| category | method | parameters | description & example |
| :--- | :--- | :--- | :--- |
| **structural**| `.flatten()` | `depth=1` | flattens nested sequences. `p([1,[2,[3]]]).util.flatten(2).to.list() # [1,2,3]`|
| | `.flatten_deep()`| - | recursively flattens a nested sequence completely. |
| | `.transpose()` | - | transposes a matrix. `p([[1,2],[3,4]]).util.transpose()`|
| | `.unzip()` | - | transforms `enumerable[tuple]` to `tuple[enumerable]`. `p([('a',1),('b',2)]).util.unzip()` |
| | `.intersperse()`| `separator` | places separator between elements. `p([1,2,3]).util.intersperse(0).to.list() # [1,0,2,0,3]` |
| | `.run_length_encode()`| - | groups consecutive elements. `p("aabbc").util.run_length_encode().to.list() # [('a',2),('b',2),('c',1)]` |
| **chaining** | `.for_each()` | `action` | performs an action on each element. **eager**: executes immediately and returns the original enumerable. `nums.util.for_each(print).where(...)` |
| | `.side_effect()`| `action` | performs action. *lazy*. `nums.where(...).util.side_effect(print).select(...)` |
| | `.pipe()` | `func`, `*args`, `**kwargs` | pipes enumerable into a function. `data.util.pipe(my_plot_func, title="Plot")` |
| | `.memoize()` | - | returns an enumerable that caches the results of the chain upon first access. **lazy**. |
| **sampling** | `.sample()` | `n`, `replace`, `random_state` | takes a random sample of `n` elements. `data.util.sample(10, random_state=42)` |
| | `.stratified_sample()`|`key_selector`, `samples_per_group` | stratified sampling to ensure representation. |
| | `.bootstrap_sample()`|`n_samples`, `sample_size` | generates multiple bootstrap samples. |
| **functional**| `.pipe_through()`|`*operations` | applies a series of (enumerable->enumerable) functions. `data.util.pipe_through(op1, op2)` |
| | `.apply_if()` | `condition`, `operation` | conditionally applies an operation. `data.util.apply_if(should_sort, lambda en: en.order_by(...))` |
| | `.apply_when()`| `predicate`, `operation` | conditionally applies an op if `pred(enumerable)` is true. |
| | `.lazy_where()`| `contextual_predicate`| filter where predicate receives `(item, all_items_list)`. `nums.util.lazy_where(lambda i, all: i > sum(all)/len(all))` |
| | `.topological_sort()`|`dependency_selector` | performs a topological sort on a dag. |
| | `.unfold()` | `seed_selector`, `unfolder` | generates new sequence by repeatedly applying `unfolder`. |
| | `.try_parse()` | `parser` | parses strings, returning `parseresult`. |
| | `.parse_or_default()`| `parser`, `default_value` | parses strings, using a default value for failures. |
| | `.compose()` | `*functions` | applies functions from left to right. `data.util.compose(op1, op2)`|
| | `.apply_functions()` | `functions` | applies multiple functions to each element. `p([1,2]).util.apply_functions([lambda x: x*2, lambda x: x+1])`|
| | `.memoize_advanced()` | - | advanced lazy memoization with partial caching. |

#### `.tree`: Tree & Hierarchical Data

| method | parameters | description | example |
| :--- | :--- | :--- | :--- |
| `.recursive_select()` | `child_selector`, `include_parents` | depth-first traversal, flattening tree to sequence. | `nodes.tree.recursive_select(lambda n: n.get('children'))` |
| `.recursive_where()` | `child_selector`, `predicate` | filters elements recursively through a tree structure. | `nodes.tree.recursive_where(n.children, lambda n: n.is_active)`|
| `.recursive_select_many()`|`child_selector`, `result_selector`| recursively traverses and flattens the results. | `nodes.tree.recursive_select_many(n.children, n.tags)`|
| `.build_tree()` | `key_selector`, `parent_key_selector`, `root_key`| builds a tree from a flat enumerable. | `flat_data.tree.build_tree(`<br/>`  key_selector=lambda i: i['id'],`<br/>`  parent_key_selector=lambda i: i['parent_id'])` |
| `.traverse_with_path()`|`child_selector` | traverses tree, yielding `treenode` objects with context. | `nodes.tree.traverse_with_path(n.children)` |
| `.select_recursive()`|`leaf_selector`, `child_selector`, `branch_selector`| applies different logic to leaf vs. branch nodes. | `fs_tree.tree.select_recursive(`<br/>`  leaf_selector=lambda leaf: leaf['size'],`<br/>`  child_selector=lambda n: n.get('children'),`<br/>`  branch_selector=lambda br, child_res: sum(child_res))` |
| `.reduce_tree()`|`child_selector`, `seed`, `accumulator`| functional reduce over a tree with depth context. | `tree.tree.reduce_tree(n.children, 0, lambda acc, n, d: acc + n.val)`|
| `.group_by_recursive()`|`key_selector`, `child_selector`| recursively groups elements while maintaining hierarchy. | `tree.tree.group_by_recursive(n.type, n.children)` |

---

### Terminal Operations (`.to`)

*these methods execute the query pipeline and return a final result.*

| method | parameters | description | example |
| :--- | :--- | :--- | :--- |
| `.list()` | - | converts the sequence to a `list`. | `p(range(3)).to.list() # [0, 1, 2]` |
| `.array()` | - | converts to a numpy `ndarray`. | `p([1,2,3]).to.array()` |
| `.set()` | - | converts to a `set`. | `p([1,2,1]).to.set() # {1, 2}` |
| `.dict()` | `key_selector`, `value_selector=none`| converts to a `dict`. | `people.to.dict(lambda p: p['id'], lambda p: p['name'])` |
| `.pandas()` | - | converts to a pandas `series`. | `p([1,2,3]).to.pandas()` |
| `.df()` | - | converts to a pandas `dataframe`. | `users.to.df()` |
| `.count()` | `predicate=none` | counts elements. | `p(range(10)).to.count(lambda x: x>5) # 4` |
| `.any()` | `predicate=none` | checks if any element exists/matches. | `p([1,2]).to.any(lambda x: x > 1) # true` |
| `.all()` | `predicate` | checks if all elements satisfy a predicate. | `p([1,2]).to.all(lambda x: x > 0) # true` |
| `.first()` | `predicate=none` | gets first element. raises `valueerror`. | `p([1,2]).to.first(lambda x: x > 1) # 2` |
| `.first_or_default()`| `predicate=none`, `default=none` | gets first element or default. | `p([]).to.first_or_default(default=-1) # -1` |
| `.single()` | `predicate=none` | gets single element. raises `valueerror`. | `p([5]).to.single() # 5` |
| `.aggregate()` | `accumulator`, `seed=none` | reduce/fold operation. | `p([1,2,3,4]).to.aggregate(lambda acc, x: acc * x) # 24` |
| `.aggregate_with_selector()` | `seed`, `accumulator`, `result_selector` | aggregates with a final transformation. | `p(['a','b']).to.aggregate_with_selector(...)` |