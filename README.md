# pinqy - linq-like operations for python

```
    __________.___ _______   ________ _____.___.
    \______   \   |\      \  \_____  \\__  |   |
     |     ___/   |/   |   \  /  / \  \/   |   |
     |    |   |   /    |    \/   \_/.  \____   |
     |____|   |___\____|__  /\_____\ \_/ ______|
                          \/        \__>/
```

a functional programming library that brings linq-style query operations to python with lazy evaluation, numpy optimization, and extensive statistical/analytical capabilities.

## installation

```bash
pip install -r requirements.txt
```

then include the `pinqy` package in your project.

## quick start

```python
from pinqy import pinqy, p, from_range, repeat, empty, generate

# basic usage
numbers = pinqy([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
result = (numbers
          .where(lambda x: x % 2 == 0)  # filter evens
          .select(lambda x: x ** 2)  # square them
          .take(3)  # take first 3
          .to.list())  # materialize
# result: [4, 16, 36]

# aliases available
p([1, 2, 3]).where(lambda x: x > 1).to.list()  # [2, 3]
```

## core concepts

### lazy evaluation
all operations are lazy - data isn't processed until you call a terminal operation like `.to.list()`, `.to.count()`, or `.to.first()`.

```python
# this creates the pipeline but doesn't execute
pipeline = pinqy(large_dataset).where(expensive_filter).select(complex_transform)

# this actually executes the operations
result = pipeline.take(10).to.list()
```

### numpy optimization
when working with numeric data, pinqy automatically uses numpy vectorization for better performance:

```python
# automatically optimized for numeric operations
large_numbers = pinqy(np.random.randint(1, 1000, 100000))
filtered = large_numbers.where(lambda x: x > 500).to.list()
```

## function reference

### factory functions

#### `from_iterable(data: iterable[t]) -> enumerable[t]`
creates enumerable from any iterable. aliases: `pinqy()`, `p()`

```python
pinqy([1, 2, 3])
pinqy(range(10))
pinqy("hello")  # works with strings too
```

#### `from_range(start: int, count: int) -> enumerable[int]`
creates enumerable from range of integers

```python
from_range(10, 5).to.list()  # [10, 11, 12, 13, 14]
```

#### `repeat(item: t, count: int) -> enumerable[t]`
repeats an item multiple times

```python
repeat("hello", 3).to.list()  # ["hello", "hello", "hello"]
```

#### `empty() -> enumerable[t]`
creates empty enumerable

```python
empty().to.count()  # 0
```

#### `generate(generator_func: callable[[], t], count: int) -> enumerable[t]`
generates sequence using a function

```python
import random
generate(lambda: random.randint(1, 10), 5).to.list()  # [3, 7, 2, 9, 1]
```

### filtering and projection

#### `.where(predicate: predicate[t]) -> enumerable[t]`
filters elements based on condition
- **performance**: numpy optimized for numeric predicates
- **usage**: chain multiple where calls for complex filtering

```python
numbers.where(lambda x: x > 5).where(lambda x: x % 2 == 0)
```

#### `.select(selector: selector[t, u]) -> enumerable[u]`
projects each element to new form
- **performance**: numpy optimized for numeric transformations
- **usage**: can change element type

```python
words.select(lambda w: len(w))  # project to lengths
people.select(lambda p: p['name'])  # extract property
```

#### `.select_many(selector: selector[t, iterable[u]]) -> enumerable[u]`
flattens nested sequences
- **usage**: similar to flatmap

```python
sentences = pinqy(['hello world', 'how are you'])
sentences.select_many(lambda s: s.split()).to.list()
# ['hello', 'world', 'how', 'are', 'you']
```

#### `.select_with_index(selector: callable[[t, int], u]) -> enumerable[u]`
projects each element using its index.

```python
pinqy(['a', 'b', 'c']).select_with_index(lambda item, index: f"{index}:{item}").to.list()
# ['0:a', '1:b', '2:c']
```

#### `.of_type(type_filter: type[u]) -> enumerable[u]`
filters elements based on a specified type.

```python
mixed_list = pinqy([1, "hello", 2.5, "world", 3])
mixed_list.of_type(str).to.list()  # ['hello', 'world']
```

### sorting operations

#### `.order_by(key_selector: keyselector[t, k]) -> orderedenumerable[t]`
sorts elements by key in ascending order. returns an `orderedenumerable`.

```python
people.order_by(lambda p: p['age'])
```

#### `.order_by_descending(key_selector: keyselector[t, k]) -> orderedenumerable[t]`
sorts elements by key in descending order. returns an `orderedenumerable`.

```python
people.order_by_descending(lambda p: p['salary'])
```

#### `.reverse() -> enumerable[t]`
inverts the order of the elements in a sequence.

```python
pinqy([1, 2, 3]).reverse().to.list()  # [3, 2, 1]
```

#### `.as_ordered() -> orderedenumerable[t]`
treats the current sequence as already ordered, allowing `then_by` to be called without performing an initial sort.

```python
# use only when you know the source data is pre-sorted
pre_sorted_data.as_ordered().then_by(lambda x: x.secondary_key)
```

### ordered enumerable operations
these methods are only available on an `orderedenumerable` (i.e., after calling `order_by` or `order_by_descending`).

#### `.then_by(key_selector: keyselector[t, k]) -> orderedenumerable[t]`
secondary sort in ascending order.

```python
people.order_by(lambda p: p['department']).then_by(lambda p: p['age'])
```

#### `.then_by_descending(key_selector: keyselector[t, k]) -> orderedenumerable[t]`
secondary sort in descending order.

```python
people.order_by(lambda p: p['city']).then_by_descending(lambda p: p['salary'])
```

#### .find_by_key(*key_prefix: any) -> enumerable[t]
efficiently finds items matching a key prefix using binary search (o(log n)). supports sequences sorted in ascending, descending, or mixed directions.

```python
users.order_by(lambda u: u.state).then_by(lambda u: u.city).find_by_key("ca", "los angeles")
```

#### .between_keys(lower_bound: union[any, tuple], upper_bound: union[any, tuple]) -> enumerable[t]
efficiently gets a slice of items where the sort key(s) are between the bounds.

```python
# ascending example
products.order_by(lambda p: p.price).between_keys(10.00, 49.99)

# descending example
products.order_by_descending(lambda p: p.price).between_keys(49.99, 10.00)
```

#### `.merge_with(other: orderedenumerable[t]) -> enumerable[t]`
efficiently merges this sorted sequence with another compatible sorted sequence (o(n + m)).
- **edge case**: raises `typeerror` if sort keys and directions are not identical.

```python
sorted1 = pinqy(data1).order_by(lambda x: x.id)
sorted2 = pinqy(data2).order_by(lambda x: x.id)
merged = sorted1.merge_with(sorted2)
```

#### `.lag_in_order(periods: int = 1, fill_value: optional[t] = none) -> enumerable[optional[t]]`
shifts elements by periods based on the sorted order.

#### `.lead_in_order(periods: int = 1, fill_value: optional[t] = none) -> enumerable[optional[t]]`
shifts elements forward by periods based on the sorted order.

### pagination operations

#### `.take(count: int) -> enumerable[t]`
takes first n elements.
- **edge case**: safe with counts larger than sequence.

```python
pinqy([1,2]).take(5).to.list()  # [1, 2]
```

#### `.skip(count: int) -> enumerable[t]`
skips first n elements.

```python
numbers.skip(3).take(2).to.list()  # from [1..10], returns [4, 5]
```

#### `.take_while(predicate: predicate[t]) -> enumerable[t]`
takes elements while condition is true, then stops.

```python
pinqy([1, 2, 6, 3, 7]).take_while(lambda x: x < 5).to.list()  # [1, 2]
```

#### `.skip_while(predicate: predicate[t]) -> enumerable[t]`
skips elements while condition is true, then returns the rest.

```python
pinqy([1, 2, 6, 3, 7]).skip_while(lambda x: x < 5).to.list()  # [6, 3, 7]
```

### set operations

#### `.set.distinct(key_selector: optional[keyselector[t, k]] = none) -> enumerable[t]`
returns unique elements, preserving order of first appearance.
- **performance**: numpy optimized for numeric data.

```python
pinqy([1, 2, 1, 3, 2]).set.distinct().to.list()  # [1, 2, 3]
people.set.distinct(lambda p: p['city'])  # unique people by first city seen
```

#### `.set.union(other: iterable[t]) -> enumerable[t]`
union of two sequences (distinct elements, order preserved).

```python
p([1, 2]).set.union([2, 3, 4]).to.list()  # [1, 2, 3, 4]
```

#### `.set.intersect(other: iterable[t]) -> enumerable[t]`
intersection of sequences, preserving order from the first sequence.

```python
p([1, 2, 3]).set.intersect([2, 4, 3]).to.list()  # [2, 3]
```

#### `.set.except_(other: iterable[t]) -> enumerable[t]`
elements in first sequence but not second (set difference).

```python
p([1, 2, 3]).set.except_([2, 4]).to.list()  # [1, 3]
```

#### `.set.symmetric_difference(other: iterable[t]) -> enumerable[t]`
elements that are in one sequence or the other, but not both.

```python
p([1, 2, 3]).set.symmetric_difference([2, 4, 5]).to.list() # [1, 3, 4, 5]
```

#### `.set.concat(other: iterable[t]) -> enumerable[t]`
concatenates sequences, allowing duplicates.

```python
p([1, 2]).set.concat([2, 3]).to.list()  # [1, 2, 2, 3]
```

#### `.set.multiset_intersect(other: iterable[t]) -> enumerable[t]`
intersection of two multisets (bags), respecting element counts.

```python
p([1, 1, 2, 3]).set.multiset_intersect([1, 2, 2]).to.list() # [1, 2]
```

#### `.set.except_by_count(other: iterable[t]) -> enumerable[t]`
difference of two multisets (bags), respecting element counts.

```python
p([1, 1, 2, 3]).set.except_by_count([1, 2, 2]).to.list() # [1, 3]
```

#### `.set.is_subset_of(other: iterable[t]) -> bool`
determines whether this sequence is a subset of another.

#### `.set.is_superset_of(other: iterable[t]) -> bool`
determines whether this sequence is a superset of another.

#### `.set.is_proper_subset_of(other: iterable[t]) -> bool`
determines whether this sequence is a proper subset of another.

#### `.set.is_proper_superset_of(other: iterable[t]) -> bool`
determines whether this sequence is a proper superset of another.

#### `.set.is_disjoint_with(other: iterable[t]) -> bool`
determines whether this sequence has no elements in common with another.

#### `.set.jaccard_similarity(other: iterable[t]) -> float`
calculates jaccard similarity (intersection over union), a value between 0.0 and 1.0.

### grouping operations

#### `.group.group_by(key_selector: keyselector[t, k]) -> dict[k, list[t]]`
groups elements by key.
- **returns**: dictionary, not an enumerable.
- **performance**: uses `defaultdict` internally for efficiency.

```python
people.group.group_by(lambda p: p['department'])
# {'engineering': [person1, person2], 'sales': [person3]}
```

#### `.group.group_by_multiple(*key_selectors: keyselector[t, any]) -> dict[tuple, list[t]]`
groups by multiple keys into a composite tuple key.

```python
people.group.group_by_multiple(lambda p: p['dept'], lambda p: p['level'])
# {('eng', 'sr'): [...], ('sales', 'jr'): [...]}
```

#### `.group.group_by_with_aggregate(key_selector, element_selector, result_selector) -> dict[k, v]`
groups and transforms each group into a final value.

```python
people.group.group_by_with_aggregate(
    lambda p: p['city'],
    lambda p: p['salary'],
    lambda city, salaries: sum(salaries) / len(salaries)
) # returns {'new york': 95000, 'london': 88000}
```

#### `.group.pivot(row_selector, column_selector, aggregator) -> dict[k, dict[u, v]]`
creates a pivot table, grouping by rows and columns and aggregating cell values.

```python
sales = p([
    {'year': 2023, 'product': 'a', 'sales': 100},
    {'year': 2023, 'product': 'b', 'sales': 150},
    {'year': 2024, 'product': 'a', 'sales': 120},
])
sales.group.pivot(
    row_selector=lambda r: r['year'],
    column_selector=lambda r: r['product'],
    aggregator=lambda g: g.stats.sum(lambda s: s['sales'])
)
# returns: {2023: {'a': 100, 'b': 150}, 2024: {'a': 120}}
```

### join operations

#### `.join.join(inner, outer_key_selector, inner_key_selector, result_selector) -> enumerable[v]`
inner join two sequences.
- **performance**: builds a lookup dictionary for the inner sequence for efficiency.

```python
people.join.join(orders, lambda p: p['id'], lambda o: o['customer_id'],
            lambda p, o: {'name': p['name'], 'total': o['total']})
```

#### `.join.left_join(inner, outer_key_selector, inner_key_selector, result_selector, default_inner=none) -> enumerable[v]`
left outer join; includes all outer elements.

```python
people.join.left_join(orders, lambda p: p['id'], lambda o: o['customer_id'],
                 lambda p, o: {'name': p['name'], 'total': o['total'] if o else 0})
```

#### `.join.right_join(inner, outer_key_selector, inner_key_selector, result_selector, default_outer=none) -> enumerable[v]`
right outer join; includes all inner elements.

#### `.join.full_outer_join(inner, outer_key_selector, inner_key_selector, result_selector, default_outer=none, default_inner=none) -> enumerable[v]`
full outer join; includes all elements from both sequences.

#### `.join.group_join(inner, outer_key_selector, inner_key_selector, result_selector) -> enumerable[v]`
groups inner elements by outer key for each outer element.

```python
people.join.group_join(orders, lambda p: p['id'], lambda o: o['customer_id'],
                  lambda p, orders: {'name': p['name'], 'order_count': len(orders)})
```

#### `.join.cross_join(inner: iterable[u]) -> enumerable[tuple[t, u]]`
cartesian product of two sequences.

```python
colors.join.cross_join(sizes)  # all color-size combinations
```

### zip operations

#### `.zip.zip_with(other, result_selector) -> enumerable[v]`
zips two sequences with a custom result selector.

```python
pinqy().zip.zip_with(['a', 'b'], lambda n, l: f"{n}{l}").to.list()
# ['1a', '2b']
```

#### `.zip.zip_longest_with(other, result_selector, default_self=none, default_other=none) -> enumerable[v]`
zips sequences, padding the shorter one with default values.

```python
p().zip.zip_longest_with(['a'], lambda n,l: f"{n}{l}", default_self=0, default_other='z').to.list()
# ['1a', '2z']
```

### functional & structural operations

#### `.group.partition(predicate: predicate[t]) -> tuple[list[t], list[t]]`
splits elements into two lists based on a predicate.

```python
evens, odds = numbers.group.partition(lambda x: x % 2 == 0)
```

#### .group.chunk(size: int) -> enumerable[list[t]]
splits into chunks of a specified size.

```python
numbers.group.chunk(3).to.list()  # [[1,2,3], [4,5,6], [7,8,9], [10]]
```

### .group.batched(size: int) -> enumerable[tuple[t, ...]]
batches elements into tuples of a specified size (python 3.12+).
code
```python
numbers.group.batched(3).to.list() # [(1,2,3), (4,5,6), (7,8,9), (10,)]
```

#### `.group.window(size: int) -> enumerable[list[t]]`
creates sliding windows of elements.
- **edge case**: returns empty if sequence is shorter than window size.

```python
numbers.group.window(3).to.list()  # [[1,2,3], [2,3,4], [3,4,5], ...]
```

#### `.group.pairwise() -> enumerable[tuple[t, t]]`
returns consecutive pairs of elements.

```python
pinqy([1, 2, 3, 4]).group.pairwise().to.list()  # [(1,2), (2,3), (3,4)]
```

#### `.stats.scan(accumulator, seed) -> enumerable[u]`
produces intermediate accumulation values (cumulative fold).

```python
pinqy([1,2,3]).stats.scan(lambda acc, x: acc + x, 0).to.list()  # [0, 1, 3, 6]
```

#### `.util.flatten(depth: int = 1) -> enumerable[any]`
flattens nested sequences to a specified depth.

```python
nested = pinqy([1, [2, 3], [4, [5]]])
nested.util.flatten(1).to.list() # [1, 2, 3, 4, [5]]
nested.util.flatten(2).to.list() # [1, 2, 3, 4, 5]
```

#### `.util.transpose() -> enumerable[list[t]]`
transposes a matrix-like structure (list of lists).

```python
matrix = pinqy([[1, 2, 3], [4, 5, 6]])
matrix.util.transpose().to.list() # [[1, 4], [2, 5], [3, 6]]
```

#### `.util.unzip() -> tuple[enumerable[any], ...]`
transforms an enumerable of tuples/lists into a tuple of enumerables.

```python
p([('a', 1), ('b', 2)]).util.unzip() # (pinqy(['a', 'b']), pinqy([1, 2]))
```

#### `.util.intersperse(separator: t) -> enumerable[t]`
places a separator element between each element of a sequence.

```python
pinqy([1, 2, 3]).util.intersperse(0).to.list()  # [1, 0, 2, 0, 3]
```

#### `.group.batch_by(key_selector) -> enumerable[list[t]]`
batches consecutive elements that share the same key.

```python
pinqy([1, 1, 2, 3, 3, 3, 2]).group.batch_by(lambda x: x).to.list()
# [[1, 1], [2], [3, 3, 3], [2]]
```

#### `.util.for_each(action: callable[[t], any]) -> enumerable[t]`
performs an action on each element for side-effects, returning the original enumerable.

```python
numbers.util.for_each(print).where(lambda x: x > 5).to.list() # prints 1-10, returns [6,7,8,9,10]
```

### mathematical operations

#### `.stats.sum(selector: optional[selector[...]] = none) -> union[int, float]`
calculates the sum.
- **performance**: uses numpy for numeric data.

```python
numbers.stats.sum()
people.stats.sum(lambda p: p['salary'])
```

#### `.stats.average(selector: optional[selector[...]] = none) -> float`
calculates the average.
- **edge case**: raises `valueerror` on an empty sequence.

```python
numbers.stats.average()
people.stats.average(lambda p: p['age'])
```

#### `.stats.min(selector: optional[selector[...]] = none) -> t`
finds the minimum value or element.
- **usage**: with no selector, finds min value. with selector, returns the *entire element* having the min selected value.

```python
numbers.stats.min() # 1
people.stats.min(lambda p: p['age']) # returns the person object with the minimum age
```

#### `.stats.max(selector: optional[selector[...]] = none) -> t`
finds the maximum value or element.

### extended statistical operations

#### `.stats.std_dev(selector: optional[selector[...]] = none) -> float`
calculates the standard deviation.

#### `.stats.median(selector: optional[selector[...]] = none) -> float`
calculates the median value.

#### `.stats.percentile(q: float, selector: optional[selector[...]] = none) -> float`
calculates the q-th percentile (where q is between 0 and 100).

```python
numbers.stats.percentile(75)  # 75th percentile
```

#### `.stats.mode(selector: optional[selector[t, k]] = none) -> k`
finds the most frequent element (the mode).

### rolling window operations

#### `.stats.rolling_window(window_size: int, aggregator: callable[[list[t]], u]) -> enumerable[u]`
applies a custom aggregator function to rolling windows.

```python
# rolling average
numbers.stats.rolling_window(3, lambda w: sum(w) / len(w))
```

#### `.stats.rolling_sum(window_size: int, selector=none) -> enumerable[union[int, float]]`
calculates a rolling sum over windows.

#### `.stats.rolling_average(window_size: int, selector=none) -> enumerable[float]`
calculates a rolling average over windows.

### time series operations

#### `.stats.lag(periods: int = 1, fill_value: optional[t] = none) -> enumerable[optional[t]]`
shifts elements back by a number of periods, filling the gap.

```python
pinqy([1,2,3,4]).stats.lag(2, 0).to.list() # [0, 0, 1, 2]
```

#### `.stats.lead(periods: int = 1, fill_value: optional[t] = none) -> enumerable[optional[t]]`
shifts elements forward by a number of periods, filling the gap.

```python
pinqy([1,2,3,4]).stats.lead(2, 0).to.list() # [3, 4, 0, 0]
```

#### `.stats.diff(periods: int = 1) -> enumerable[t]`
calculates the difference between an element and a previous element.

```python
prices = pinqy([10, 12, 11, 15])
prices.stats.diff().to.list() # [2, -1, 4]
```

### cumulative operations

#### `.stats.cumulative_sum(selector=none) -> enumerable[union[int, float]]`
calculates the cumulative sum of the sequence.

```python
pinqy([1, 2, 3, 4]).stats.cumulative_sum().to.list()  # [1, 3, 6, 10]
```

#### `.stats.cumulative_product(selector=none) -> enumerable[union[int, float]]`
calculates the cumulative product.

#### `.stats.cumulative_max(selector=none) -> enumerable[k]`
finds the cumulative maximum.

#### `.stats.cumulative_min(selector=none) -> enumerable[k]`
finds the cumulative minimum.

### ranking operations

#### `.stats.rank(selector=none, ascending: bool = true) -> enumerable[int]`
assigns a 1-based rank to each element. ties get the same rank, and the next rank is skipped.

```python
scores = pinqy([10, 30, 20, 30])
scores.stats.rank(ascending=false).to.list() # [4, 1, 3, 1]
```

#### `.stats.dense_rank(selector=none, ascending: bool = true) -> enumerable[int]`
assigns a 1-based rank. ties get the same rank, but no gaps appear in the rank sequence.

```python
scores.stats.dense_rank(ascending=false).to.list() # [3, 1, 2, 1]
```

#### `.stats.quantile_cut(q: int, selector=none) -> enumerable[int]`
bins elements into a specified number of quantiles (e.g., q=4 for quartiles).

### sampling operations

#### `.util.sample(n: int, replace: bool = false, random_state: optional[int] = none) -> enumerable[t]`
takes a random sample of elements from the sequence.

```python
data.util.sample(10, replace=true, random_state=42)
```

#### `.util.stratified_sample(key_selector, samples_per_group: int) -> enumerable[t]`
performs stratified sampling, taking a random sample from each group to ensure representation.

#### `.util.bootstrap_sample(n_samples: int = 1000, sample_size: optional[int] = none) -> enumerable[enumerable[t]]`
generates multiple bootstrap samples (sampling with replacement).

### normalization and outlier detection

#### `.stats.normalize(selector=none) -> enumerable[float]`
applies min-max normalization to scale data to a 0-1 range.

#### `.stats.standardize(selector=none) -> enumerable[float]`
applies z-score standardization to give data a mean of 0 and standard deviation of 1.

#### `.stats.outliers_iqr(selector=none, factor: float = 1.5) -> enumerable[t]`
detects and returns outliers using the interquartile range (iqr) method.

### combinatorial operations

#### `.comb.permutations(r: optional[int] = none) -> enumerable[tuple[t, ...]]`
generates all permutations of the sequence.

```python
pinqy(['a', 'b', 'c']).comb.permutations(2).to.list()
# [('a', 'b'), ('a', 'c'), ('b', 'a'), ('b', 'c'), ('c', 'a'), ('c', 'b')]
```

#### `.comb.combinations(r: int) -> enumerable[tuple[t, ...]]`
generates all unique combinations of a given length.

```python
pinqy(['a', 'b', 'c']).comb.combinations(2).to.list()
# [('a', 'b'), ('a', 'c'), ('b', 'c')]
```

#### `.comb.combinations_with_replacement(r: int) -> enumerable[tuple[t, ...]]`
generates combinations where elements can be re-selected.

#### `.comb.power_set() -> enumerable[tuple[t, ...]]`
generates the power set (all possible subsets) of the sequence.

#### `.comb.cartesian_product(*others) -> enumerable[tuple[any, ...]]`
computes the cartesian product with one or more other sequences.

#### `.comb.binomial_coefficient(r: int) -> int`
calculates "n choose r", where n is the number of items in the sequence.

#### `.util.run_length_encode() -> enumerable[tuple[t, int]]`
encodes the sequence by grouping consecutive identical elements into (element, count) tuples.

```python
pinqy(['a', 'a', 'b', 'c', 'c', 'c']).util.run_length_encode().to.list()
# [('a', 2), ('b', 1), ('c', 3)]
```

### advanced operations

#### `.append(element: t) -> enumerable[t]`
appends a value to the end of the sequence.

#### `.prepend(element: t) -> enumerable[t]`
adds a value to the beginning of the sequence.

#### `.default_if_empty(default_value: t) -> enumerable[t]`
returns the sequence, or a sequence containing only the default value if it's empty.

#### `.util.memoize() -> enumerable[t]`
evaluates the chain and caches the result. subsequent operations start from the cached state.

#### `.util.pipe(func: callable[..., u], *args, **kwargs) -> u`
pipes the enumerable object into an external function, allowing for custom, chainable operations.

```python
data.util.pipe(my_custom_plot_function, title='my data')
```

#### .util.side_effect(action: callable[[t], any]) -> enumerable[t]
performs a side-effect (like printing) for each element without modifying the sequence. this operation is lazy, making it useful for debugging a pipeline without forcing early materialization.

```python
numbers.where(...).util.side_effect(print).select(...).to.list()
```

#### `.util.topological_sort(dependency_selector: callable[[t], iterable[t]]) -> enumerable[t]`
performs a topological sort on the sequence, assuming it represents nodes in a dag.
- **edge case**: raises `valueerror` if a cycle is detected.

### terminal operations

#### `.to.list() -> list[t]`
converts the sequence to a list, materializing the results.

#### `.to.array() -> np.ndarray`
converts to a numpy array.

#### `.to.set() -> set[t]`
converts to a set.

#### `.to.dict(key_selector, value_selector=none) -> dict[k, v]`
converts to a dictionary.

```python
people.to.dict(lambda p: p['id'], lambda p: p['name'])
```

#### `.to.pandas() -> pd.series`
converts to a pandas series.

#### `.to.df() -> pd.dataframe`
converts to a pandas dataframe.

#### `.to.count(predicate: optional[predicate[t]] = none) -> int`
counts elements, optionally matching a predicate.

```python
numbers.to.count()  # 10
numbers.to.count(lambda x: x > 5)  # 5
```

#### `.to.any(predicate: optional[predicate[t]] = none) -> bool`
checks if any element exists, optionally matching a predicate.

```python
numbers.to.any()  # true
numbers.to.any(lambda x: x > 10)  # false
```

#### `.to.all(predicate: predicate[t]) -> bool`
checks if all elements satisfy a condition.

```python
numbers.to.all(lambda x: x > 0) # true
```

#### `.to.first(predicate: optional[predicate[t]] = none) -> t`
gets the first element, optionally matching a predicate.
- **edge case**: raises `valueerror` if no element is found.

#### `.to.first_or_default(predicate=none, default=none) -> optional[t]`
gets the first element or a default value if not found.

#### .to.single(predicate=none) -> t
gets the single element that matches a condition.
- **edge case**: raises `valueerror` if zero or more than one element is found.

#### .to.aggregate(accumulator, seed=none) -> t
applies an accumulator function over the sequence (a fold or reduce operation).

```python
# sum with a seed
numbers.to.aggregate(lambda acc, x: acc + x, 0)
# product without a seed
pinqy([1,2,3,4]).to.aggregate(lambda acc, x: acc * x) # 24
```

#### `.to.aggregate_with_selector(seed, accumulator, result_selector) -> v`
aggregates with a seed and a final transformation of the result.

### advanced functional & tree operations

these functions provide advanced capabilities for working with hierarchical data, building complex functional pipelines, and handling more sophisticated data transformation scenarios.

#### `create_tree_from_flat(data, key_selector, parent_key_selector, root_key=none) -> enumerable[treeitem]`
builds a hierarchical tree structure from a flat list of items based on parent-child key relationships.

```python
from pinqy import create_tree_from_flat

flat_data = [
    {'id': 1, 'parent_id': None, 'name': 'root'},
    {'id': 2, 'parent_id': 1, 'name': 'child a'},
    {'id': 3, 'parent_id': 2, 'name': 'grandchild a1'},
]

tree = create_tree_from_flat(
    flat_data,
    key_selector=lambda item: item['id'],
    parent_key_selector=lambda item: item['parent_id']
)
# returns an enumerable of treeitem objects representing the root(s)
```

#### `recursive_generator(seed, child_generator, max_depth=100) -> enumerable[t]`
creates an enumerable by recursively applying a function to generate children from a starting seed.

```python
from pinqy import recursive_generator

# generate a simple number tree
tree = recursive_generator(
    seed=1,
    child_generator=lambda n: [n*2, n*2+1] if n < 8 else none
)
tree.to.list() # [1, 2, 3, 4, 5, 6, 7]
```

### tree operations (`.tree` accessor)

these methods are designed for working with data that has a natural hierarchical or nested structure.

#### `.tree.recursive_select(child_selector, include_parents=true) -> enumerable[t]`
performs a depth-first traversal of a tree structure, flattening it into a single sequence.

```python
nodes = [
    {'name': 'a', 'children': [{'name': 'b'}, {'name': 'c'}]}
]
pinqy(nodes).tree.recursive_select(
    child_selector=lambda node: node.get('children')
).select(lambda node: node['name']).to.list()
# ['a', 'b', 'c']
```

#### `.tree.build_tree(key_selector, parent_key_selector, root_key=none) -> enumerable[treeitem]`
the method version of the factory function. builds a tree from a flat enumerable.

#### `.tree.traverse_with_path(child_selector) -> enumerable[treenode]`
traverses a tree, yielding `treenode` objects that contain the `value`, `depth`, and `path` from the root.

#### `.tree.select_recursive(leaf_selector, child_selector, branch_selector) -> enumerable[u]`
applies different transformation logic to leaf nodes vs. branch nodes in a tree.

```python
# calculate file sizes in a directory tree
fs_tree = [{'name': 'c:', 'children': [{'name': 'file.txt', 'size': 100}]}]
pinqy(fs_tree).tree.select_recursive(
    leaf_selector=lambda leaf: leaf['size'],
    child_selector=lambda node: node.get('children'),
    branch_selector=lambda branch, child_results: sum(child_results)
).to.list() # [100]
```

#### `.tree.reduce_tree(child_selector, seed, accumulator) -> u`
performs a functional reduce/fold operation over an entire tree structure. the accumulator receives `(current_value, item, depth)`.

### advanced grouping operations (`.group` accessor)

#### `.group.group_by_nested(key_selector, sub_key_selector) -> dict[k, dict[v, list[t]]]`
creates a two-level nested dictionary from groupings.

```python
data.group.group_by_nested(
    lambda x: x['state'],
    lambda x: x['city']
)
# {'ca': {'la': [...], 'sf': [...]}}
```

### advanced utility & functional operations (`.util` accessor)

#### `.util.flatten_deep() -> enumerable[any]`
recursively flattens a nested sequence all the way down to its non-iterable components.

```python
nested = pinqy([1, [2, [3, 4]], 5])
nested.util.flatten_deep().to.list() # [1, 2, 3, 4, 5]
```

#### `.util.pipe_through(*operations) -> enumerable[any]`
applies a series of functions to an enumerable in a sequence. each function must take and return an enumerable.

```python
def double_evens(en):
    return en.where(lambda x: x % 2 == 0).select(lambda x: x * 2)

pinqy([1,2,3,4]).util.pipe_through(double_evens).to.list() # [4, 8]
```

#### `.util.apply_if(condition, operation) -> enumerable[t]`
conditionally applies an operation to the enumerable if a boolean `condition` is true.

```python
should_sort = true
data.util.apply_if(should_sort, lambda en: en.order_by(lambda x: x.name))
```

#### `.util.apply_when(predicate, operation) -> enumerable[t]`
conditionally applies an operation if a `predicate` function (which receives the enumerable itself) returns true.

```python
# only take 10 if there are more than 10 items
data.util.apply_when(lambda en: en.to.count() > 10, lambda en: en.take(10))
```

#### `.util.try_parse(parser) -> parseresult[t]`
attempts to parse a sequence of strings, separating the results into successes and failures.

```python
def parse_int(s):
    try:
        return (true, int(s))
    except valueerror:
        return (false, none)

strings = pinqy(['1', '2', 'a', '3'])
result = strings.util.try_parse(parse_int)
# result.successes -> [1, 2, 3]
# result.failures -> ['a']
```

#### `.util.parse_or_default(parser, default_value=none) -> enumerable[t]`
parses a sequence of strings, using a default value for any that fail.

```python
strings.util.parse_or_default(parse_int, -1).to.list() # [1, 2, -1, 3]
```

#### `.util.lazy_where(contextual_predicate) -> enumerable[t]`
a powerful filter where the predicate function receives both the item and the fully materialized list, allowing for context-aware filtering.

```python
# find all numbers greater than the average of the entire list
numbers = pinqy([1, 2, 3, 10])
numbers.util.lazy_where(
    lambda item, all_items: item > (sum(all_items) / len(all_items))
).to.list() # [10]
```

#### `.util.unfold(seed_selector, unfolder) -> enumerable[v]`
generates a new sequence by repeatedly applying an `unfolder` function to a seed value. the unfolder returns `(value_to_yield, next_seed)` or `none` to stop.

```python
# generate collatz sequence for starting numbers
pinqy([3, 5]).util.unfold(
    seed_selector=lambda x: x,
    unfolder=lambda seed: (seed, seed // 2) if seed % 2 == 0 else ((seed, 3 * seed + 1) if seed > 1 else none)
).to.list()
# [3, 10, 5, 16, 8, 4, 2, 5, 16, 8, 4, 2]
```