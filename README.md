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
pip install numpy pandas
```

then include the `pinqy.py` file in your project.

## quick start

```python
from pinqy import pinqy, p, from_range, repeat, empty, generate

# basic usage
numbers = pinqy([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
result = (numbers
    .where(lambda x: x % 2 == 0)  # filter evens
    .select(lambda x: x ** 2)     # square them
    .take(3)                      # take first 3
    .to_list())                   # materialize
# result: [4, 16, 36]

# aliases available
p([1,2,3]).where(lambda x: x > 1).to_list()  # [2, 3]
```

## core concepts

### lazy evaluation
all operations are lazy - data isn't processed until you call a terminal operation like `.to_list()`, `.count()`, or `.first()`.

```python
# this creates the pipeline but doesn't execute
pipeline = pinqy(large_dataset).where(expensive_filter).select(complex_transform)

# this actually executes the operations
result = pipeline.take(10).to_list()
```

### numpy optimization
when working with numeric data, pinqy automatically uses numpy vectorization for better performance:

```python
# automatically optimized for numeric operations
large_numbers = pinqy(np.random.randint(1, 1000, 100000))
filtered = large_numbers.where(lambda x: x > 500).to_list()
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
from_range(10, 5).to_list()  # [10, 11, 12, 13, 14]
```

#### `repeat(item: t, count: int) -> enumerable[t]`
repeats an item multiple times

```python
repeat("hello", 3).to_list()  # ["hello", "hello", "hello"]
```

#### `empty() -> enumerable[t]`
creates empty enumerable

```python
empty().count()  # 0
```

#### `generate(generator_func: callable[[], t], count: int) -> enumerable[t]`
generates sequence using a function

```python
import random
generate(lambda: random.randint(1, 10), 5).to_list()  # [3, 7, 2, 9, 1]
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
sentences.select_many(lambda s: s.split()).to_list()
# ['hello', 'world', 'how', 'are', 'you']
```

#### `.select_with_index(selector: callable[[t, int], u]) -> enumerable[u]`
projects each element using its index.

```python
pinqy(['a', 'b', 'c']).select_with_index(lambda item, index: f"{index}:{item}").to_list()
# ['0:a', '1:b', '2:c']
```

#### `.of_type(type_filter: type[u]) -> enumerable[u]`
filters elements based on a specified type.

```python
mixed_list = pinqy([1, "hello", 2.5, "world", 3])
mixed_list.of_type(str).to_list()  # ['hello', 'world']
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
pinqy([1, 2, 3]).reverse().to_list()  # [3, 2, 1]
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

#### `.find_by_key(*key_prefix: any) -> enumerable[t]`
efficiently finds items matching a key prefix using binary search (o(log n)).
- **edge case**: raises `notimplementederror` if any searched keys are sorted descending.

```python
users.order_by(lambda u: u.state).then_by(lambda u: u.city).find_by_key("ca", "los angeles")
```

#### `.between_keys(lower_bound: union[any, tuple], upper_bound: union[any, tuple]) -> enumerable[t]`
efficiently gets a slice of items where the sort key(s) are between the bounds.
- **edge case**: raises `notimplementederror` if sorted descending.

```python
products.order_by(lambda p: p.price).between_keys(10.00, 49.99)
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
pinqy([1,2]).take(5).to_list()  # [1, 2]
```

#### `.skip(count: int) -> enumerable[t]`
skips first n elements.

```python
numbers.skip(3).take(2).to_list()  # from [1..10], returns [4, 5]
```

#### `.take_while(predicate: predicate[t]) -> enumerable[t]`
takes elements while condition is true, then stops.

```python
pinqy([1, 2, 6, 3, 7]).take_while(lambda x: x < 5).to_list()  # [1, 2]
```

#### `.skip_while(predicate: predicate[t]) -> enumerable[t]`
skips elements while condition is true, then returns the rest.

```python
pinqy([1, 2, 6, 3, 7]).skip_while(lambda x: x < 5).to_list()  # [6, 3, 7]
```

### set operations

#### `.distinct(key_selector: optional[keyselector[t, k]] = none) -> enumerable[t]`
returns unique elements, preserving order of first appearance.
- **performance**: numpy optimized for numeric data.

```python
pinqy([1, 2, 1, 3, 2]).distinct().to_list()  # [1, 2, 3]
people.distinct(lambda p: p['city'])  # unique people by first city seen
```

#### `.union(other: iterable[t]) -> enumerable[t]`
union of two sequences (distinct elements, order preserved).
- **performance**: numpy optimized when possible.

```python
p([1, 2]).union([2, 3, 4]).to_list()  # [1, 2, 3, 4]
```

#### `.intersect(other: iterable[t]) -> enumerable[t]`
intersection of sequences, preserving order from the first sequence.

```python
p([1, 2, 3]).intersect([2, 4, 3]).to_list()  # [2, 3]
```

#### `.except_(other: iterable[t]) -> enumerable[t]`
elements in first sequence but not second (set difference).

```python
p([1, 2, 3]).except_([2, 4]).to_list()  # [1, 3]
```

#### `.symmetric_difference(other: iterable[t]) -> enumerable[t]`
elements that are in one sequence or the other, but not both.

```python
p([1, 2, 3]).symmetric_difference([2, 4, 5]).to_list() # [1, 3, 4, 5]
```

#### `.concat(other: iterable[t]) -> enumerable[t]`
concatenates sequences, allowing duplicates.

```python
p([1, 2]).concat([2, 3]).to_list()  # [1, 2, 2, 3]
```

#### `.multiset_intersect(other: iterable[t]) -> enumerable[t]`
intersection of two multisets (bags), respecting element counts.

```python
p([1, 1, 2, 3]).multiset_intersect([1, 2, 2]).to_list() # [1, 2]
```

#### `.except_by_count(other: iterable[t]) -> enumerable[t]`
difference of two multisets (bags), respecting element counts.

```python
p([1, 1, 2, 3]).except_by_count([1, 2, 2]).to_list() # [1, 3]
```

#### `.is_subset_of(other: iterable[t]) -> bool`
determines whether this sequence is a subset of another.

#### `.is_superset_of(other: iterable[t]) -> bool`
determines whether this sequence is a superset of another.

#### `.is_proper_subset_of(other: iterable[t]) -> bool`
determines whether this sequence is a proper subset of another.

#### `.is_proper_superset_of(other: iterable[t]) -> bool`
determines whether this sequence is a proper superset of another.

#### `.is_disjoint_with(other: iterable[t]) -> bool`
determines whether this sequence has no elements in common with another.

#### `.jaccard_similarity(other: iterable[t]) -> float`
calculates jaccard similarity (intersection over union), a value between 0.0 and 1.0.

### grouping operations

#### `.group_by(key_selector: keyselector[t, k]) -> dict[k, list[t]]`
groups elements by key.
- **returns**: dictionary, not an enumerable.
- **performance**: uses `defaultdict` internally for efficiency.

```python
people.group_by(lambda p: p['department'])
# {'engineering': [person1, person2], 'sales': [person3]}
```

#### `.group_by_multiple(*key_selectors: keyselector[t, any]) -> dict[tuple, list[t]]`
groups by multiple keys into a composite tuple key.

```python
people.group_by_multiple(lambda p: p['dept'], lambda p: p['level'])
# {('eng', 'sr'): [...], ('sales', 'jr'): [...]}
```

#### `.group_by_with_aggregate(key_selector, element_selector, result_selector) -> dict[k, v]`
groups and transforms each group into a final value.

```python
people.group_by_with_aggregate(
    lambda p: p['city'],
    lambda p: p['salary'],
    lambda city, salaries: sum(salaries) / len(salaries)
) # returns {'new york': 95000, 'london': 88000}
```

### join operations

#### `.join(inner, outer_key_selector, inner_key_selector, result_selector) -> enumerable[v]`
inner join two sequences.
- **performance**: builds a lookup dictionary for the inner sequence for efficiency.

```python
people.join(orders, lambda p: p['id'], lambda o: o['customer_id'],
            lambda p, o: {'name': p['name'], 'total': o['total']})
```

#### `.left_join(inner, outer_key_selector, inner_key_selector, result_selector, default_inner=none) -> enumerable[v]`
left outer join; includes all outer elements.

```python
people.left_join(orders, lambda p: p['id'], lambda o: o['customer_id'],
                 lambda p, o: {'name': p['name'], 'total': o['total'] if o else 0})
```

#### `.right_join(inner, outer_key_selector, inner_key_selector, result_selector, default_outer=none) -> enumerable[v]`
right outer join; includes all inner elements.

#### `.full_outer_join(inner, outer_key_selector, inner_key_selector, result_selector, default_outer=none, default_inner=none) -> enumerable[v]`
full outer join; includes all elements from both sequences.

#### `.group_join(inner, outer_key_selector, inner_key_selector, result_selector) -> enumerable[v]`
groups inner elements by outer key for each outer element.

```python
people.group_join(orders, lambda p: p['id'], lambda o: o['customer_id'],
                  lambda p, orders: {'name': p['name'], 'order_count': len(orders)})
```

#### `.cross_join(inner: iterable[u]) -> enumerable[tuple[t, u]]`
cartesian product of two sequences.

```python
colors.cross_join(sizes)  # all color-size combinations
```

### zip operations

#### `.zip_with(other, result_selector) -> enumerable[v]`
zips two sequences with a custom result selector.

```python
pinqy([1, 2]).zip_with(['a', 'b'], lambda n, l: f"{n}{l}").to_list()
# ['1a', '2b']
```

#### `.zip_longest_with(other, result_selector, default_self=none, default_other=none) -> enumerable[v]`
zips sequences, padding the shorter one with default values.

```python
p([1,2]).zip_longest_with(['a'], lambda n,l: f"{n}{l}", default_self=0, default_other='z').to_list()
# ['1a', '2z']
```

### functional & structural operations

#### `.partition(predicate: predicate[t]) -> tuple[list[t], list[t]]`
splits elements into two lists based on a predicate.

```python
evens, odds = numbers.partition(lambda x: x % 2 == 0)
```

#### `.chunk(size: int) -> enumerable[list[t]]`
splits into chunks of a specified size.

```python
numbers.chunk(3).to_list()  # [[1,2,3], [4,5,6], [7,8,9], [10]]
```

#### `.batched(size: int) -> enumerable[tuple[t, ...]]`
batches elements into tuples of a specified size (python 3.12+).

#### `.window(size: int) -> enumerable[list[t]]`
creates sliding windows of elements.
- **edge case**: returns empty if sequence is shorter than window size.

```python
numbers.window(3).to_list()  # [[1,2,3], [2,3,4], [3,4,5], ...]
```

#### `.pairwise() -> enumerable[tuple[t, t]]`
returns consecutive pairs of elements.

```python
pinqy([1, 2, 3, 4]).pairwise().to_list()  # [(1,2), (2,3), (3,4)]
```

#### `.scan(accumulator, seed) -> enumerable[u]`
produces intermediate accumulation values (cumulative fold).

```python
pinqy([1,2,3]).scan(lambda acc, x: acc + x, 0).to_list()  # [0, 1, 3, 6]
```

#### `.flatten(depth: int = 1) -> enumerable[any]`
flattens nested sequences to a specified depth.

```python
nested = pinqy([1, [2, 3], [4, [5]]])
nested.flatten(1).to_list() # [1, 2, 3, 4, [5]]
nested.flatten(2).to_list() # [1, 2, 3, 4, 5]
```

#### `.transpose() -> enumerable[list[t]]`
transposes a matrix-like structure (list of lists).

```python
matrix = pinqy([[1, 2, 3], [4, 5, 6]])
matrix.transpose().to_list() # [[1, 4], [2, 5], [3, 6]]
```

#### `.unzip() -> tuple[enumerable[any], ...]`
transforms an enumerable of tuples/lists into a tuple of enumerables.

```python
p([('a', 1), ('b', 2)]).unzip() # (pinqy(['a', 'b']), pinqy([1, 2]))
```

#### `.intersperse(separator: t) -> enumerable[t]`
places a separator element between each element of a sequence.

```python
pinqy([1, 2, 3]).intersperse(0).to_list()  # [1, 0, 2, 0, 3]
```

#### `.batch_by(key_selector) -> enumerable[list[t]]`
batches consecutive elements that share the same key.

```python
pinqy([1, 1, 2, 3, 3, 3, 2]).batch_by(lambda x: x).to_list()
# [[1, 1], [2], [3, 3, 3], [2]]
```

#### `.for_each(action: callable[[t], any]) -> enumerable[t]`
performs an action on each element for side-effects, returning the original enumerable.

```python
numbers.for_each(print).where(lambda x: x > 5).to_list() # prints 1-10, returns [6,7,8,9,10]
```

### mathematical operations

#### `.sum(selector: optional[selector[...]] = none) -> union[int, float]`
calculates the sum.
- **performance**: uses numpy for numeric data.

```python
numbers.sum()
people.sum(lambda p: p['salary'])
```

#### `.average(selector: optional[selector[...]] = none) -> float`
calculates the average.
- **edge case**: raises `valueerror` on an empty sequence.

```python
numbers.average()
people.average(lambda p: p['age'])
```

#### `.min(selector: optional[selector[...]] = none) -> t`
finds the minimum value or element.
- **usage**: with no selector, finds min value. with selector, returns the *entire element* having the min selected value.

```python
numbers.min() # 1
people.min(lambda p: p['age']) # returns the person object with the minimum age
```

#### `.max(selector: optional[selector[...]] = none) -> t`
finds the maximum value or element.

### extended statistical operations

#### `.std_dev(selector: optional[selector[...]] = none) -> float`
calculates the standard deviation.

#### `.median(selector: optional[selector[...]] = none) -> float`
calculates the median value.

#### `.percentile(q: float, selector: optional[selector[...]] = none) -> float`
calculates the q-th percentile (where q is between 0 and 100).

```python
numbers.percentile(75)  # 75th percentile
```

#### `.mode(selector: optional[selector[t, k]] = none) -> k`
finds the most frequent element (the mode).

### rolling window operations

#### `.rolling_window(window_size: int, aggregator: callable[[list[t]], u]) -> enumerable[u]`
applies a custom aggregator function to rolling windows.

```python
# rolling average
numbers.rolling_window(3, lambda w: sum(w) / len(w))
```

#### `.rolling_sum(window_size: int, selector=none) -> enumerable[union[int, float]]`
calculates a rolling sum over windows.

#### `.rolling_average(window_size: int, selector=none) -> enumerable[float]`
calculates a rolling average over windows.

### time series operations

#### `.lag(periods: int = 1, fill_value: optional[t] = none) -> enumerable[optional[t]]`
shifts elements back by a number of periods, filling the gap.

```python
pinqy([1,2,3,4]).lag(2, 0).to_list() # [0, 0, 1, 2]
```

#### `.lead(periods: int = 1, fill_value: optional[t] = none) -> enumerable[optional[t]]`
shifts elements forward by a number of periods, filling the gap.

```python
pinqy([1,2,3,4]).lead(2, 0).to_list() # [3, 4, 0, 0]
```

#### `.diff(periods: int = 1) -> enumerable[t]`
calculates the difference between an element and a previous element.

```python
prices = pinqy([10, 12, 11, 15])
prices.diff().to_list() # [2, -1, 4]
```

### cumulative operations

#### `.cumulative_sum(selector=none) -> enumerable[union[int, float]]`
calculates the cumulative sum of the sequence.

```python
pinqy([1, 2, 3, 4]).cumulative_sum().to_list()  # [1, 3, 6, 10]
```

#### `.cumulative_product(selector=none) -> enumerable[union[int, float]]`
calculates the cumulative product.

#### `.cumulative_max(selector=none) -> enumerable[k]`
finds the cumulative maximum.

#### `.cumulative_min(selector=none) -> enumerable[k]`
finds the cumulative minimum.

### ranking operations

#### `.rank(selector=none, ascending: bool = true) -> enumerable[int]`
assigns a 1-based rank to each element. ties get the same rank, and the next rank is skipped.

```python
scores = pinqy([10, 30, 20, 30])
scores.rank(ascending=false).to_list() # [4, 1, 3, 1]
```

#### `.dense_rank(selector=none, ascending: bool = true) -> enumerable[int]`
assigns a 1-based rank. ties get the same rank, but no gaps appear in the rank sequence.

```python
scores.dense_rank(ascending=false).to_list() # [3, 1, 2, 1]
```

#### `.quantile_cut(q: int, selector=none) -> enumerable[int]`
bins elements into a specified number of quantiles (e.g., q=4 for quartiles).

### sampling operations

#### `.sample(n: int, replace: bool = false, random_state: optional[int] = none) -> enumerable[t]`
takes a random sample of elements from the sequence.

```python
data.sample(10, replace=true, random_state=42)
```

#### `.stratified_sample(key_selector, samples_per_group: int) -> enumerable[t]`
performs stratified sampling, taking a random sample from each group to ensure representation.

#### `.bootstrap_sample(n_samples: int = 1000, sample_size: optional[int] = none) -> enumerable[enumerable[t]]`
generates multiple bootstrap samples (sampling with replacement).

### normalization and outlier detection

#### `.normalize(selector=none) -> enumerable[float]`
applies min-max normalization to scale data to a 0-1 range.

#### `.standardize(selector=none) -> enumerable[float]`
applies z-score standardization to give data a mean of 0 and standard deviation of 1.

#### `.outliers_iqr(selector=none, factor: float = 1.5) -> enumerable[t]`
detects and returns outliers using the interquartile range (iqr) method.

### combinatorial operations

#### `.permutations(r: optional[int] = none) -> enumerable[tuple[t, ...]]`
generates all permutations of the sequence.

```python
pinqy(['a', 'b', 'c']).permutations(2).to_list()
# [('a', 'b'), ('a', 'c'), ('b', 'a'), ('b', 'c'), ('c', 'a'), ('c', 'b')]
```

#### `.combinations(r: int) -> enumerable[tuple[t, ...]]`
generates all unique combinations of a given length.

```python
pinqy(['a', 'b', 'c']).combinations(2).to_list()
# [('a', 'b'), ('a', 'c'), ('b', 'c')]
```

#### `.combinations_with_replacement(r: int) -> enumerable[tuple[t, ...]]`
generates combinations where elements can be re-selected.

#### `.power_set() -> enumerable[tuple[t, ...]]`
generates the power set (all possible subsets) of the sequence.

#### `.cartesian_product(*others) -> enumerable[tuple[any, ...]]`
computes the cartesian product with one or more other sequences.

#### `.binomial_coefficient(r: int) -> int`
calculates "n choose r", where n is the number of items in the sequence.

#### `.run_length_encode() -> enumerable[tuple[t, int]]`
encodes the sequence by grouping consecutive identical elements into (element, count) tuples.

```python
pinqy(['a', 'a', 'b', 'c', 'c', 'c']).run_length_encode().to_list()
# [('a', 2), ('b', 1), ('c', 3)]
```

### advanced operations

#### `.append(element: t) -> enumerable[t]`
appends a value to the end of the sequence.

#### `.prepend(element: t) -> enumerable[t]`
adds a value to the beginning of the sequence.

#### `.default_if_empty(default_value: t) -> enumerable[t]`
returns the sequence, or a sequence containing only the default value if it's empty.

#### `.memoize() -> enumerable[t]`
evaluates the chain and caches the result. subsequent operations start from the cached state.

#### `.pipe(func: callable[..., u], *args, **kwargs) -> u`
pipes the enumerable object into an external function, allowing for custom, chainable operations.

```python
data.pipe(my_custom_plot_function, title='my data')
```

#### `.side_effect(action: callable[[t], any]) -> enumerable[t]`
performs a side-effect (like printing) for each element without modifying the sequence. useful for debugging.

```python
numbers.where(...).side_effect(print).select(...).to_list()
```

#### `.topological_sort(dependency_selector: callable[[t], iterable[t]]) -> enumerable[t]`
performs a topological sort on the sequence, assuming it represents nodes in a dag.
- **edge case**: raises `valueerror` if a cycle is detected.

### terminal operations

#### `.to_list() -> list[t]`
converts the sequence to a list, materializing the results.

#### `.to_array() -> np.ndarray`
converts to a numpy array.

#### `.to_set() -> set[t]`
converts to a set.

#### `.to_dict(key_selector, value_selector=none) -> dict[k, v]`
converts to a dictionary.

```python
people.to_dict(lambda p: p['id'], lambda p: p['name'])
```

#### `.to_pandas() -> pd.series`
converts to a pandas series.

#### `.to_df() -> pd.dataframe`
converts to a pandas dataframe.

#### `.count(predicate: optional[predicate[t]] = none) -> int`
counts elements, optionally matching a predicate.

```python
numbers.count()  # 10
numbers.count(lambda x: x > 5)  # 5
```

#### `.any(predicate: optional[predicate[t]] = none) -> bool`
checks if any element exists, optionally matching a predicate.

```python
numbers.any()  # true
numbers.any(lambda x: x > 10)  # false
```

#### `.all(predicate: predicate[t]) -> bool`
checks if all elements satisfy a condition.

```python
numbers.all(lambda x: x > 0) # true
```

#### `.first(predicate: optional[predicate[t]] = none) -> t`
gets the first element, optionally matching a predicate.
- **edge case**: raises `valueerror` if no element is found.

#### `.first_or_default(predicate=none, default=none) -> optional[t]`
gets the first element or a default value if not found.

#### `.single(predicate=none) -> t`
gets the single element that matches a condition.
- **edge case**: raises `valueerror` if zero or more than one element is found.

#### `.aggregate(accumulator, seed=none) -> t`
applies an accumulator function over the sequence (a fold or reduce operation).

```python
# sum with a seed
numbers.aggregate(lambda acc, x: acc + x, 0)
# product without a seed
pinqy([1,2,3,4]).aggregate(lambda acc, x: acc * x) # 24
```

#### `.aggregate_with_selector(seed, accumulator, result_selector) -> v`
aggregates with a seed and a final transformation of the result.