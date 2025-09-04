# PINQY - LINQ-like Operations for Python

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
from pinqy import pinqy, P, from_range, repeat, empty, generate

# basic usage
numbers = pinqy([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
result = (numbers
    .where(lambda x: x % 2 == 0)  # filter evens
    .select(lambda x: x ** 2)     # square them
    .take(3)                      # take first 3
    .to_list())                   # materialize
# result: [4, 16, 36]

# aliases available
P([1,2,3]).where(lambda x: x > 1).to_list()  # [2, 3]
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

#### `from_iterable(data: Iterable[T]) -> Enumerable[T]`
creates enumerable from any iterable. alias: `pinqy()`, `P()`

```python
pinqy([1, 2, 3])
pinqy(range(10))
pinqy("hello")  # works with strings too
```

#### `from_range(start: int, count: int) -> Enumerable[int]`
creates enumerable from range of integers

```python
from_range(10, 5).to_list()  # [10, 11, 12, 13, 14]
```

#### `repeat(item: T, count: int) -> Enumerable[T]`
repeats an item multiple times

```python
repeat("hello", 3).to_list()  # ["hello", "hello", "hello"]
```

#### `empty() -> Enumerable[T]`
creates empty enumerable

```python
empty().count()  # 0
```

#### `generate(generator_func: Callable[[], T], count: int) -> Enumerable[T]`
generates sequence using a function

```python
import random
generate(lambda: random.randint(1, 10), 5).to_list()  # [3, 7, 2, 9, 1]
```

### filtering and projection

#### `.where(predicate: Predicate[T]) -> Enumerable[T]`
filters elements based on condition
- **performance**: numpy optimized for numeric predicates
- **usage**: chain multiple where calls for complex filtering

```python
numbers.where(lambda x: x > 5).where(lambda x: x % 2 == 0)
```

#### `.select(selector: Selector[T, U]) -> Enumerable[U]`
projects each element to new form
- **performance**: numpy optimized for numeric transformations
- **usage**: can change element type

```python
words.select(lambda w: len(w))  # project to lengths
people.select(lambda p: p['name'])  # extract property
```

#### `.select_many(selector: Selector[T, Iterable[U]]) -> Enumerable[U]`
flattens nested sequences
- **usage**: similar to flatmap

```python
sentences.select_many(lambda s: s.split())  # flatten to words
```

### sorting operations

#### `.order_by(key_selector: KeySelector[T, K]) -> OrderedEnumerable[T]`
sorts elements by key in ascending order

```python
people.order_by(lambda p: p['age'])
```

#### `.order_by_descending(key_selector: KeySelector[T, K]) -> OrderedEnumerable[T]`
sorts elements by key in descending order

```python
people.order_by_descending(lambda p: p['salary'])
```

#### `.then_by(key_selector: KeySelector[T, K]) -> OrderedEnumerable[T]`
secondary sort (only on ordered enumerables)

```python
people.order_by(lambda p: p['department']).then_by(lambda p: p['age'])
```

#### `.then_by_descending(key_selector: KeySelector[T, K]) -> OrderedEnumerable[T]`
secondary sort descending

```python
people.order_by(lambda p: p['city']).then_by_descending(lambda p: p['salary'])
```

### pagination operations

#### `.take(count: int) -> Enumerable[T]`
takes first n elements
- **edge case**: safe with counts larger than sequence

```python
pinqy([1,2]).take(5).to_list()  # [1, 2]
```

#### `.skip(count: int) -> Enumerable[T]`
skips first n elements

```python
numbers.skip(3).take(2).to_list()  # skip 3, take next 2
```

#### `.take_while(predicate: Predicate[T]) -> Enumerable[T]`
takes elements while condition is true

```python
numbers.take_while(lambda x: x < 5)  # stops at first >= 5
```

#### `.skip_while(predicate: Predicate[T]) -> Enumerable[T]`
skips elements while condition is true

```python
numbers.skip_while(lambda x: x < 5)  # starts from first >= 5
```

### set operations

#### `.distinct(key_selector: Optional[KeySelector[T, K]] = None) -> Enumerable[T]`
returns unique elements
- **performance**: numpy optimized for numeric data
- **usage**: preserves order

```python
numbers.distinct()  # unique numbers
people.distinct(lambda p: p['city'])  # unique by city
```

#### `.union(other: Iterable[T]) -> Enumerable[T]`
union of two sequences
- **performance**: numpy optimized when possible

```python
list1.union(list2)  # combined unique elements
```

#### `.intersect(other: Iterable[T]) -> Enumerable[T]`
intersection of sequences

```python
list1.intersect(list2)  # common elements
```

#### `.except_(other: Iterable[T]) -> Enumerable[T]`
elements in first sequence but not second

```python
list1.except_(list2)  # elements only in list1
```

#### `.concat(other: Iterable[T]) -> Enumerable[T]`
concatenates sequences (allows duplicates)

```python
list1.concat(list2)  # all elements from both
```

### grouping operations

#### `.group_by(key_selector: KeySelector[T, K]) -> Dict[K, List[T]]`
groups elements by key
- **returns**: dictionary not enumerable
- **performance**: uses defaultdict internally

```python
people.group_by(lambda p: p['department'])
# {'engineering': [person1, person2], 'sales': [person3]}
```

#### `.group_by_multiple(*key_selectors: KeySelector[T, Any]) -> Dict[Tuple, List[T]]`
groups by multiple keys

```python
people.group_by_multiple(
    lambda p: p['department'],
    lambda p: p['seniority']
)
# {('engineering', 'senior'): [...], ('sales', 'junior'): [...]}
```

#### `.group_by_with_aggregate(key_selector, element_selector, result_selector) -> Dict[K, V]`
groups and transforms each group

```python
people.group_by_with_aggregate(
    lambda p: p['city'],          # group by city
    lambda p: p['salary'],        # extract salary
    lambda city, salaries: {      # aggregate
        'city': city,
        'avg_salary': sum(salaries) / len(salaries),
        'count': len(salaries)
    }
)
```

### join operations

#### `.join(inner, outer_key_selector, inner_key_selector, result_selector) -> Enumerable[V]`
inner join two sequences
- **performance**: builds lookup dictionary for inner sequence

```python
people.join(
    orders,
    lambda p: p['id'],           # person id
    lambda o: o['customer_id'],  # order customer_id
    lambda p, o: {               # result
        'person': p['name'],
        'order_amount': o['total']
    }
)
```

#### `.left_join(inner, outer_key_selector, inner_key_selector, result_selector, default_inner=None) -> Enumerable[V]`
left outer join - includes all outer elements

```python
people.left_join(
    orders,
    lambda p: p['id'],
    lambda o: o['customer_id'],
    lambda p, o: {
        'person': p['name'],
        'order_total': o['total'] if o else 0
    }
)
```

#### `.right_join(inner, outer_key_selector, inner_key_selector, result_selector, default_outer=None) -> Enumerable[V]`
right outer join - includes all inner elements

#### `.full_outer_join(inner, outer_key_selector, inner_key_selector, result_selector, default_outer=None, default_inner=None) -> Enumerable[V]`
full outer join - includes all elements from both sequences

#### `.group_join(inner, outer_key_selector, inner_key_selector, result_selector) -> Enumerable[V]`
groups inner elements by outer key

```python
people.group_join(
    orders,
    lambda p: p['id'],
    lambda o: o['customer_id'],
    lambda p, orders_list: {
        'person': p['name'],
        'order_count': len(orders_list),
        'total_spent': sum(o['total'] for o in orders_list)
    }
)
```

#### `.cross_join(inner: Iterable[U]) -> Enumerable[Tuple[T, U]]`
cartesian product of sequences

```python
colors.cross_join(sizes)  # all color-size combinations
```

### zip operations

#### `.zip_with(other, result_selector) -> Enumerable[V]`
zips with custom result selector

```python
numbers.zip_with(letters, lambda n, l: f"{n}{l}")
```

#### `.zip_longest_with(other, result_selector, default_self=None, default_other=None) -> Enumerable[V]`
zips padding shorter sequence with defaults

```python
numbers.zip_longest_with(
    letters,
    lambda n, l: f"{n or 0}{l or 'z'}",
    default_self=0,
    default_other='z'
)
```

### functional operations

#### `.partition(predicate: Predicate[T]) -> Tuple[List[T], List[T]]`
splits into two lists based on predicate

```python
evens, odds = numbers.partition(lambda x: x % 2 == 0)
```

#### `.chunk(size: int) -> Enumerable[List[T]]`
splits into chunks of specified size

```python
numbers.chunk(3).to_list()  # [[1,2,3], [4,5,6], [7,8,9], [10]]
```

#### `.window(size: int) -> Enumerable[List[T]]`
creates sliding windows
- **edge case**: returns empty if sequence shorter than window

```python
numbers.window(3).to_list()  # [[1,2,3], [2,3,4], [3,4,5], ...]
```

#### `.scan(accumulator, seed) -> Enumerable[U]`
produces intermediate accumulation values

```python
numbers.scan(lambda acc, x: acc + x, 0)  # cumulative sums
# [0, 1, 3, 6, 10, 15, ...]
```

#### `.pairwise() -> Enumerable[Tuple[T, T]]`
returns consecutive pairs

```python
numbers.pairwise().to_list()  # [(1,2), (2,3), (3,4), ...]
```

### mathematical operations

#### `.sum(selector: Optional[Selector[T, Union[int, float]]] = None) -> Union[int, float]`
calculates sum
- **performance**: numpy optimized for numeric data

```python
numbers.sum()
people.sum(lambda p: p['salary'])
```

#### `.average(selector: Optional[Selector[T, Union[int, float]]] = None) -> float`
calculates average
- **edge case**: raises error on empty sequence

```python
numbers.average()
people.average(lambda p: p['age'])
```

#### `.min(selector: Optional[Selector[T, Union[int, float]]] = None) -> Union[T, Union[int, float]]`
finds minimum

```python
numbers.min()
people.min(lambda p: p['age'])  # returns person with min age
```

#### `.max(selector: Optional[Selector[T, Union[int, float]]] = None) -> Union[T, Union[int, float]]`
finds maximum

### extended statistical operations

#### `.median(selector: Optional[Selector[T, Union[int, float]]] = None) -> float`
calculates median value

```python
numbers.median()
people.median(lambda p: p['salary'])
```

#### `.std_dev(selector: Optional[Selector[T, Union[int, float]]] = None) -> float`
calculates standard deviation

#### `.percentile(q: float, selector: Optional[Selector[T, Union[int, float]]] = None) -> float`
calculates percentile (q between 0-100)

```python
numbers.percentile(75)  # 75th percentile
```

#### `.mode(selector: Optional[Selector[T, K]] = None) -> K`
finds most frequent element

### rolling window operations

#### `.rolling_window(window_size: int, aggregator: Callable[[List[T]], U]) -> Enumerable[U]`
applies aggregator to rolling windows

```python
numbers.rolling_window(3, lambda w: sum(w) / len(w))  # rolling average
```

#### `.rolling_sum(window_size: int, selector=None) -> Enumerable[Union[int, float]]`
rolling sum over windows

#### `.rolling_average(window_size: int, selector=None) -> Enumerable[float]`
rolling average over windows

### time series operations

#### `.lag(periods: int = 1, fill_value: Optional[T] = None) -> Enumerable[Optional[T]]`
shifts elements by periods

```python
values.lag(2, 0)  # shift right by 2, fill with 0
```

#### `.lead(periods: int = 1, fill_value: Optional[T] = None) -> Enumerable[Optional[T]]`
shifts elements forward

#### `.diff(periods: int = 1) -> Enumerable[T]`
difference with previous elements

```python
prices.diff()  # price changes
```

### cumulative operations

#### `.cumulative_sum(selector=None) -> Enumerable[Union[int, float]]`
cumulative sum

```python
numbers.cumulative_sum().to_list()  # [1, 3, 6, 10, 15, ...]
```

#### `.cumulative_product(selector=None) -> Enumerable[Union[int, float]]`
cumulative product

#### `.cumulative_max(selector=None) -> Enumerable[K]`
cumulative maximum

#### `.cumulative_min(selector=None) -> Enumerable[K]`
cumulative minimum

### ranking operations

#### `.rank(selector=None, ascending: bool = True) -> Enumerable[int]`
rank elements

```python
scores.rank()  # 1-based ranking
```

#### `.dense_rank(selector=None, ascending: bool = True) -> Enumerable[int]`
dense ranking (no gaps for ties)

#### `.quantile_cut(q: int, selector=None) -> Enumerable[int]`
cut into quantile bins

### sampling operations

#### `.sample(n: int, replace: bool = False, random_state: Optional[int] = None) -> Enumerable[T]`
random sampling

```python
data.sample(10, replace=True, random_state=42)
```

#### `.stratified_sample(key_selector, samples_per_group: int) -> Enumerable[T]`
stratified sampling ensuring representation

#### `.bootstrap_sample(n_samples: int = 1000, sample_size: Optional[int] = None) -> Enumerable[Enumerable[T]]`
bootstrap sampling for confidence intervals

### normalization operations

#### `.normalize(selector=None) -> Enumerable[float]`
min-max normalization (0-1 scale)

```python
values.normalize()  # scales to 0-1 range
```

#### `.standardize(selector=None) -> Enumerable[float]`
z-score standardization (mean=0, std=1)

#### `.outliers_iqr(selector=None, factor: float = 1.5) -> Enumerable[T]`
detect outliers using iqr method

### combinatorial operations

#### `.permutations(r: Optional[int] = None) -> Enumerable[List[T]]`
generates permutations

```python
pinqy(['a', 'b', 'c']).permutations(2).to_list()
# [['a', 'b'], ['a', 'c'], ['b', 'a'], ['b', 'c'], ['c', 'a'], ['c', 'b']]
```

#### `.combinations(r: int) -> Enumerable[List[T]]`
generates combinations

```python
pinqy(['a', 'b', 'c']).combinations(2).to_list()
# [['a', 'b'], ['a', 'c'], ['b', 'c']]
```

#### `.cartesian_product(*others) -> Enumerable[List[Union[T, U]]]`
cartesian product with multiple sequences

#### `.binomial_coefficient(r: int) -> int`
calculates n choose r

### structural operations

#### `.flatten(depth: int = 1) -> Enumerable[Any]`
flattens nested sequences

```python
nested.flatten(2)  # flattens 2 levels deep
```

#### `.transpose() -> Enumerable[List[T]]`
transposes matrix-like structure

#### `.intersperse(separator: T) -> Enumerable[T]`
intersperses separator between elements

```python
pinqy([1, 2, 3]).intersperse(0).to_list()  # [1, 0, 2, 0, 3]
```

#### `.batch_by(key_selector) -> Enumerable[List[T]]`
batches consecutive elements with same key

### terminal operations

#### `.to_list() -> List[T]`
converts to list (materializes)

#### `.to_array() -> np.ndarray`
converts to numpy array

#### `.to_set() -> Set[T]`
converts to set

#### `.to_dict(key_selector, value_selector=None) -> Dict[K, V]`
converts to dictionary

```python
people.to_dict(lambda p: p['id'], lambda p: p['name'])
```

#### `.to_pandas() -> pd.Series`
converts to pandas series

#### `.count(predicate: Optional[Predicate[T]] = None) -> int`
counts elements

```python
numbers.count()  # total count
numbers.count(lambda x: x > 5)  # conditional count
```

#### `.any(predicate: Optional[Predicate[T]] = None) -> bool`
checks if any element satisfies condition

```python
numbers.any()  # has any elements?
numbers.any(lambda x: x > 10)  # any greater than 10?
```

#### `.all(predicate: Predicate[T]) -> bool`
checks if all elements satisfy condition

#### `.first(predicate: Optional[Predicate[T]] = None) -> T`
gets first element
- **edge case**: raises error if no element found

```python
numbers.first()  # first element
numbers.first(lambda x: x > 5)  # first > 5
```

#### `.first_or_default(predicate=None, default=None) -> Optional[T]`
gets first element or default

#### `.single(predicate=None) -> T`
gets single element
- **edge case**: raises error if 0 or >1 elements

#### `.aggregate(accumulator, seed=None) -> T`
applies accumulator function

```python
numbers.aggregate(lambda acc, x: acc + x, 0)  # sum with seed
numbers.aggregate(lambda acc, x: acc * x)     # product without seed
```

#### `.aggregate_with_selector(seed, accumulator, result_selector) -> V`
aggregates with final transformation

## usage patterns

### chaining operations

```python
result = (pinqy(data)
    .where(lambda x: x.is_valid)
    .select(lambda x: x.transform())
    .group_by(lambda x: x.category)
    .items())
```

### complex business logic

```python
high_value_customers = (
    pinqy(customers)
    .where(lambda c: c['total_orders'] > 10)
    .where(lambda c: c['avg_order_value'] > 100)
    .join(regions, 
          lambda c: c['region_id'],
          lambda r: r['id'],
          lambda c, r: {
              'customer': c['name'],
              'region': r['name'],
              'value': c['lifetime_value']
          })
    .order_by_descending(lambda x: x['value'])
    .take(10)
    .to_list()
)
```

### statistical analysis

```python
sales_analysis = (
    pinqy(sales_data)
    .select(lambda s: s['amount'])
    .where(lambda a: a > 0)  # remove refunds
)

stats = {
    'mean': sales_analysis.average(),
    'median': sales_analysis.median(),
    'std': sales_analysis.std_dev(),
    'p95': sales_analysis.percentile(95),
    'outliers': sales_analysis.outliers_iqr().count()
}
```

## performance considerations

### numpy optimization
- automatically used for numeric operations on `where`, `select`, `distinct`
- works with lists of `int`, `float`, `complex`
- fallback to pure python if optimization fails

### lazy evaluation
- operations build pipeline without execution
- only terminal operations trigger computation
- enables efficient memory usage with large datasets

### caching
- results cached after first materialization
- subsequent operations on same enumerable reuse cache
- cache invalidated only when creating new enumerable

### memory efficiency
- streaming operations don't load entire dataset
- `take` and `skip` enable pagination
- window operations process chunks, not full dataset

## error handling

### empty sequences
- `average`, `min`, `max`, `first`, `single` raise `ValueError` on empty sequences
- use `first_or_default` for safe access
- `any()` returns `False`, `count()` returns `0` for empty sequences

### type safety
- generic type hints throughout
- runtime type checking minimal for performance
- mixed types handled gracefully where possible

### null handling
- operations generally propagate nulls
- use `.where(lambda x: x is not None)` to filter nulls
- aggregations ignore null values when possible

## examples

### data analysis pipeline

```python
# analyze sales data
sales_summary = (
    pinqy(sales_records)
    .where(lambda s: s['date'] >= start_date)
    .select(lambda s: {
        **s,
        'revenue': s['quantity'] * s['price'],
        'quarter': get_quarter(s['date'])
    })
    .group_by(lambda s: s['quarter'])
)

for quarter, records in sales_summary.items():
    quarter_revenue = sum(r['revenue'] for r in records)
    print(f"Q{quarter}: ${quarter_revenue:,}")
```

### machine learning preprocessing

```python
# normalize features and split data
features_normalized = (
    pinqy(raw_features)
    .select(lambda row: [float(x) for x in row])  # ensure numeric
    .transpose()  # transpose to work column-wise
    .select(lambda col: pinqy(col).normalize().to_list())  # normalize each column
    .transpose()  # transpose back
    .to_array()
)

train_data, test_data = (
    pinqy(features_normalized)
    .zip_with(labels, lambda feat, label: (feat, label))
    .sample(len(features_normalized), random_state=42)
    .partition(lambda _, i=iter(range(len(features_normalized))): next(i) < 0.8 * len(features_normalized))
)
```

### text processing

```python
# word frequency analysis
word_freq = (
    pinqy(documents)
    .select_many(lambda doc: doc.lower().split())
    .where(lambda word: len(word) > 3)
    .group_by(lambda word: word)
)

top_words = (
    pinqy(word_freq.items())
    .select(lambda item: {'word': item[0], 'count': len(item[1])})
    .order_by_descending(lambda item: item['count'])
    .take(10)
    .to_list()
)
```
