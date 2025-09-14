import time
import math
import suite
from dgen import from_schema
from pinqy import P, from_range, Enumerable

# --- setup ---
test = suite.test
assert_that = suite.assert_that

# --- test data & helpers ---
simple_list = ['a', 'b', 'c']
numeric_list = [10, 20, 30, 40]
list_with_dupes = ['x', 'y', 'x']
list_with_more_dupes = [1, 1, 2, 2]
empty_list = []

class TestObject:
    """a hashable, comparable object for testing complex types."""
    def __init__(self, id, name):
        self.id = id
        self.name = name
    def __repr__(self):
        return f"TestObject({self.id}, '{self.name}')"
    def __eq__(self, other):
        return isinstance(other, TestObject) and self.id == other.id
    def __hash__(self):
        return hash(self.id)

obj_list = [TestObject(1, 'A'), TestObject(2, 'B'), TestObject(3, 'C')]
dict_list = [{'id': 1}, {'id': 2}] # unhashable, for testing specific scenarios

# --- core functionality tests ---

@test("permutations generates correct ordered sequences for all valid r values")
def test_permutations_core():
    # standard case with r
    perms_r2 = P(simple_list).comb.permutations(2)
    expected_perms_r2 = {('a', 'b'), ('a', 'c'), ('b', 'a'), ('b', 'c'), ('c', 'a'), ('c', 'b')}
    assert_that(perms_r2.to.count() == 6, "permutations(r=2) count should be n!/(n-r)! = 3!/1! = 6")
    assert_that(perms_r2.to.set() == expected_perms_r2, "permutations(r=2) content is incorrect")

    # full permutations (r is none)
    perms_full = P(simple_list).comb.permutations()
    assert_that(perms_full.to.count() == 6, "full permutations count should be n! = 3! = 6")

    # full permutations (r = n)
    perms_rn = P(simple_list).comb.permutations(3)
    assert_that(perms_rn.to.count() == 6, "permutations with r=n should be same as full permutations")


@test("permutations handles edge cases for r and input lists")
def test_permutations_edges():
    # edge case: r > n
    perms_r_gt_n = P(simple_list).comb.permutations(4)
    assert_that(perms_r_gt_n.to.count() == 0, "permutations with r > n should be empty")

    # edge case: r = 0
    perms_r0 = P(simple_list).comb.permutations(0)
    assert_that(perms_r0.to.list() == [()], "permutations with r=0 should yield one empty tuple")

    # edge case: r = 1
    perms_r1 = P(simple_list).comb.permutations(1)
    assert_that(perms_r1.to.set() == {('a',), ('b',), ('c',)}, "permutations with r=1 should be tuples of single items")

    # edge case: r < 0 (FIXED)
    perms_r_neg = P(simple_list).comb.permutations(-1)
    assert_that(perms_r_neg.to.count() == 0, "permutations with r < 0 should be empty")

    # edge case: empty list, r > 0
    perms_empty_r1 = P(empty_list).comb.permutations(1)
    assert_that(perms_empty_r1.to.count() == 0, "permutations on empty list with r>0 should be empty")

    # edge case: empty list, r = 0
    perms_empty_r0 = P(empty_list).comb.permutations(0)
    assert_that(perms_empty_r0.to.list() == [()], "permutations on empty list with r=0 should yield one empty tuple")


@test("permutations handles duplicates positionally (test logic fixed)")
def test_permutations_with_duplicates():
    # itertools.permutations treats items by position, not by value.
    # input: ['x_pos0', 'y_pos1', 'x_pos2']
    # expected permutations of r=2:
    # (x0, y1), (x0, x2), (y1, x0), (y1, x2), (x2, y1), (x2, x0)
    # which correspond to values:
    # ('x','y'), ('x','x'), ('y','x'), ('y','x'), ('x','y'), ('x','x')
    perms = P(list_with_dupes).comb.permutations(2)
    assert_that(perms.to.count() == 6, "permutations on 3 items (with dupes) should still be 3!/1! = 6")

    grouped = perms.group.group_by(lambda p: p)
    assert_that(len(grouped[('x', 'y')]) == 2, "count for ('x', 'y') should be 2")
    assert_that(len(grouped[('y', 'x')]) == 2, "count for ('y', 'x') should be 2")
    assert_that(len(grouped[('x', 'x')]) == 2, "count for ('x', 'x') should be 2")

    # more complex duplicate case
    perms_complex = P(list_with_more_dupes).comb.permutations(3) # [1,1,2,2], r=3
    assert_that(perms_complex.to.count() == 24, "permutations on 4 items should be 4!/1! = 24")
    # should contain (1,1,2), (1,2,1), (2,1,1), (1,2,2), etc.
    distinct_perms = perms_complex.set.distinct().to.set()
    assert_that((1, 1, 2) in distinct_perms, "distinct permutations should contain (1,1,2)")
    assert_that((1, 2, 1) in distinct_perms, "distinct permutations should contain (1,2,1)")
    assert_that((2, 1, 1) in distinct_perms, "distinct permutations should contain (2,1,1)")
    assert_that((1, 2, 2) in distinct_perms, "distinct permutations should contain (1,2,2)")


@test("combinations generates correct unordered sets for all valid r values")
def test_combinations_core():
    combs_r2 = P(numeric_list).comb.combinations(2)
    expected_combs_r2 = {(10, 20), (10, 30), (10, 40), (20, 30), (20, 40), (30, 40)}
    assert_that(combs_r2.to.count() == 6, "combinations(r=2) count should be n!/(r!(n-r)!) = 4!/(2!2!) = 6")
    assert_that(combs_r2.to.set() == expected_combs_r2, "combinations(r=2) content is incorrect")


@test("combinations handles edge cases for r and input lists")
def test_combinations_edges():
    combs_r_gt_n = P(numeric_list).comb.combinations(5)
    assert_that(combs_r_gt_n.to.count() == 0, "combinations with r > n should be empty")

    combs_r0 = P(numeric_list).comb.combinations(0)
    assert_that(combs_r0.to.list() == [()], "combinations with r=0 should yield one empty tuple")

    combs_r1 = P(numeric_list).comb.combinations(1)
    assert_that(combs_r1.to.set() == {(10,), (20,), (30,), (40,)}, "combinations with r=1 should be tuples of single items")

    combs_rn = P(numeric_list).comb.combinations(4)
    assert_that(combs_rn.to.list() == [(10, 20, 30, 40)], "combinations with r=n should yield one tuple of all items")

    combs_empty = P(empty_list).comb.combinations(1)
    assert_that(combs_empty.to.count() == 0, "combinations on empty list should be empty")

    # new: check negative r
    combs_r_neg = P(numeric_list).comb.combinations(-1)
    assert_that(combs_r_neg.to.count() == 0, "combinations with r < 0 should be empty")


@test("combinations_with_replacement generates correct sets")
def test_combinations_with_replacement():
    # example: 3 flavors, choosing 2 scoops
    flavors = P(['vanilla', 'chocolate', 'strawberry'])
    combos = flavors.comb.combinations_with_replacement(2)
    expected = {
        ('vanilla', 'vanilla'), ('vanilla', 'chocolate'), ('vanilla', 'strawberry'),
        ('chocolate', 'chocolate'), ('chocolate', 'strawberry'),
        ('strawberry', 'strawberry')
    }
    assert_that(combos.to.set() == expected, "cwr for 3 flavors, 2 scoops is incorrect")
    assert_that(combos.to.count() == 6, "cwr count should be (n+r-1)! / (r! * (n-1)!) = 4! / (2! * 2!) = 6")

    # new: check negative r
    combs_wr_neg = flavors.comb.combinations_with_replacement(-1)
    assert_that(combs_wr_neg.to.count() == 0, "combinations_with_replacement with r < 0 should be empty")


@test("power_set generates all subsets of a sequence")
def test_power_set():
    pset = P(['a', 'b']).comb.power_set()
    expected = {(), ('a',), ('b',), ('a', 'b')}
    assert_that(pset.to.count() == 4, "power_set count for n=2 should be 2^2 = 4")
    assert_that(pset.to.set() == expected, "power_set for n=2 has incorrect content")

    # larger set
    assert_that(P(range(4)).comb.power_set().to.count() == 16, "power_set count for n=4 should be 2^4 = 16")

    # empty list
    pset_empty = P(empty_list).comb.power_set()
    assert_that(pset_empty.to.list() == [()], "power_set of empty list should be a list with one empty tuple")


@test("cartesian_product computes the product of multiple iterables")
def test_cartesian_product():
    data1 = P([1, 2])
    data2 = ['x', 'y']
    data3 = from_range(100, 1) # pinqy enumerable as argument
    data4 = ('A',) # tuple as argument

    # with four iterables of different types
    product = data1.comb.cartesian_product(data2, data3, data4)
    expected = {
        (1, 'x', 100, 'A'), (1, 'y', 100, 'A'),
        (2, 'x', 100, 'A'), (2, 'y', 100, 'A')
    }
    assert_that(product.to.count() == 4, "cartesian product of four lists (2x2x1x1) should have 4 elements")
    assert_that(product.to.set() == expected, "cartesian product of four lists has incorrect content")

    # with a middle iterable being empty
    product_empty = data1.comb.cartesian_product(data2, [], data4)
    assert_that(product_empty.to.count() == 0, "cartesian product with a middle empty list should be empty")


@test("binomial_coefficient calculates n-choose-r correctly and matches combinations")
def test_binomial_coefficient():
    data = P(range(10)) # n = 10
    assert_that(data.comb.binomial_coefficient(2) == 45, "10c2 should be 45")
    assert_that(data.comb.binomial_coefficient(0) == 1, "10c0 should be 1")
    assert_that(data.comb.binomial_coefficient(10) == 1, "10c10 should be 1")
    assert_that(data.comb.binomial_coefficient(11) == 0, "nCr where r > n should be 0")

    # new: check negative r
    assert_that(data.comb.binomial_coefficient(-1) == 0, "nCr where r < 0 should be 0")

    # verify against combinations
    count_from_combs = data.comb.combinations(3).to.count()
    coeff = data.comb.binomial_coefficient(3)
    assert_that(coeff == count_from_combs, "binomial coefficient should match count from combinations")


@test("combinatorics works with complex data types like objects and dicts")
def test_combinatorics_with_complex_types():
    # combinations with hashable objects
    combs = P(obj_list).comb.combinations(2)
    first_comb = combs.to.first()
    assert_that(isinstance(first_comb[0], TestObject), "combinations result should contain TestObject instances")
    assert_that(combs.to.set() == {
        (TestObject(1, 'A'), TestObject(2, 'B')),
        (TestObject(1, 'A'), TestObject(3, 'C')),
        (TestObject(2, 'B'), TestObject(3, 'C'))
    }, "combinations content for objects is correct")

    # permutations with unhashable dicts
    perms_dict = P(dict_list).comb.permutations(2)
    assert_that(perms_dict.to.count() == 2, "permutations of 2 dicts should be 2")
    # cannot convert to set, must check list content
    result_list = perms_dict.to.list()
    assert_that(({'id': 1}, {'id': 2}) in result_list, "permutations of dicts contains forward pair")
    assert_that(({'id': 2}, {'id': 1}) in result_list, "permutations of dicts contains reverse pair")


@test("combinatorics methods do not consume or mutate the source enumerable")
def test_source_integrity():
    source_data = from_range(0, 5)
    # perform an operation
    perms = source_data.comb.permutations(2).to.list()
    assert_that(len(perms) == 20, "operation produced expected number of results")
    # check the source again
    source_list_after = source_data.to.list()
    assert_that(source_list_after == [0, 1, 2, 3, 4], "source enumerable remains intact after one operation")
    # perform another operation
    combs = source_data.comb.combinations(3).to.list()
    assert_that(len(combs) == 10, "second operation also produced results")
    assert_that(source_data.to.list() == [0, 1, 2, 3, 4], "source enumerable remains intact after second operation")


@test("chaining: power_set with filtering and aggregation")
def test_chaining_power_set():
    # find all subsets with a sum greater than 6
    data = P([1, 2, 3, 4])
    # (FIXED: changed filter from > 5 to > 6 to match assertion intent)
    large_subsets = (data.comb.power_set()
                     .where(lambda subset: sum(subset) > 6)
                     .to.list())

    assert_that((2, 4) not in large_subsets, "subset with sum=6 should not be in list (sum(subset) > 6)")
    assert_that((1, 2, 3) not in large_subsets, "subset with sum=6 should not be in list") # (FIXED: was `in`)
    assert_that((3, 4) in large_subsets, "subset with sum=7 should be in list")
    assert_that((1, 2, 4) in large_subsets, "subset with sum=7 should be in list")
    assert_that((1, 2, 3, 4) in large_subsets, "subset with sum=10 should be in list")

    # group subsets by their size
    grouped_by_size = (data.comb.power_set()
                       .group.group_by(lambda subset: len(subset)))

    assert_that(len(grouped_by_size[0]) == 1, "there should be 1 subset of size 0")
    assert_that(len(grouped_by_size[1]) == 4, "there should be 4 subsets of size 1")
    assert_that(len(grouped_by_size[2]) == 6, "there should be 6 subsets of size 2")


@test("chaining: complex real-world scenario")
def test_chaining_complex_scenario():
    # scenario: given a list of products, find all 3-product bundles
    # where the total price is under $100 and at least two products are from the 'books' category.
    product_schema = {
        'id': 'uuid4',
        'name': 'word',
        'category': {'_qen_provider': 'choice', 'from': ['books', 'electronics', 'home']},
        'price': ('pyfloat', {'min_value': 10, 'max_value': 60, 'right_digits': 2})
    }
    products = from_schema(product_schema, seed=123).take(10)

    valid_bundles = (
        products.comb.combinations(3)
        .where(lambda bundle: sum(p['price'] for p in bundle) < 100.00)
        .where(lambda bundle: P(bundle).where(lambda p: p['category'] == 'books').to.count() >= 2)
        .select(lambda bundle: {
            'products': P(bundle).select(lambda p: p['name']).to.list(),
            'total_price': sum(p['price'] for p in bundle)
        })
    )

    assert_that(valid_bundles.to.count() > 0, "should find at least one valid bundle")
    first_bundle = valid_bundles.to.first()
    assert_that(isinstance(first_bundle, dict), "result should be transformed into a dictionary")
    assert_that('products' in first_bundle and 'total_price' in first_bundle, "result dict has correct keys")
    assert_that(first_bundle['total_price'] < 100.00, "bundle price should be under 100")


@test("combinatorics benchmark on moderate data size with more context")
def test_combinatorics_performance():
    print() # newline for cleaner test output
    perm_data_size = 9 # 9! = 362,880
    perm_data = from_range(0, perm_data_size)
    start_perms = time.perf_counter()
    perm_count = perm_data.comb.permutations().to.count()
    duration_perms = (time.perf_counter() - start_perms) * 1000
    print(f"    {suite._c.grey}└─> permutations on {perm_data_size} items ({perm_count:,} results) took: {duration_perms:.2f}ms{suite._c.reset}")

    comb_data_size = 25
    comb_data = from_range(0, comb_data_size)
    start_combs = time.perf_counter()
    comb_count = comb_data.comb.combinations(5).to.count() # 25c5 = 53,130
    duration_combs = (time.perf_counter() - start_combs) * 1000
    print(f"    {suite._c.grey}└─> combinations on {comb_data_size}c5 ({comb_count:,} results) took: {duration_combs:.2f}ms{suite._c.reset}")

    pset_data_size = 18 # 2^18 = 262,144
    pset_data = from_range(0, pset_data_size)
    start_pset = time.perf_counter()
    pset_count = pset_data.comb.power_set().to.count()
    duration_pset = (time.perf_counter() - start_pset) * 1000
    print(f"    {suite._c.grey}└─> power_set on {pset_data_size} items ({pset_count:,} results) took: {duration_pset:.2f}ms{suite._c.reset}")

    assert_that(duration_perms > 0 and duration_combs > 0 and duration_pset > 0, "benchmarks should run")


# --- run the suite ---
if __name__ == "__main__":
    suite.run(title="pinqy combinatorics test")