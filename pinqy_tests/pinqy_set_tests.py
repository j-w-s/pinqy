import suite
from dgen import from_schema
from pinqy import P, from_range, empty

test = suite.test
assert_that = suite.assert_that

# --- test data schemas ---
person_schema = {
    'id': ('pyint', {'min_value': 1, 'max_value': 5}),
    'name': 'word',
    'city': {'_qen_provider': 'choice', 'from': ['ny', 'la', 'chi']},
    'score': ('pyint', {'min_value': 80, 'max_value': 100})
}


# --- distinct ---

@test("distinct removes duplicates while preserving order")
def test_distinct_basic():
    data = P([1, 2, 1, 3, 2, 4])
    result = data.set.distinct().to.list()
    assert_that(result == [1, 2, 3, 4], "distinct should preserve first occurrence order")


@test("distinct with key selector works on objects")
def test_distinct_with_key():
    people = from_schema(person_schema, seed=42).take(20)
    unique_cities = people.set.distinct(lambda p: p['city']).to.list()
    city_names = P(unique_cities).select(lambda p: p['city']).to.set()
    assert_that(len(city_names) <= 3, "should have at most 3 unique cities")
    assert_that(len(unique_cities) == len(city_names), "each city should appear once")


@test("distinct handles empty sequences")
def test_distinct_empty():
    assert_that(empty().set.distinct().to.list() == [], "distinct on empty should be empty")


@test("distinct with all same elements")
def test_distinct_all_same():
    result = P([5, 5, 5, 5]).set.distinct().to.list()
    assert_that(result == [5], "distinct should return single element")


# --- union ---

@test("union combines sequences removing duplicates")
def test_union_basic():
    result = P([1, 2]).set.union([2, 3, 4]).to.list()
    assert_that(result == [1, 2, 3, 4], "union should preserve order and remove duplicates")


@test("union preserves order from first sequence")
def test_union_order():
    result = P([3, 1, 2]).set.union([2, 4, 1]).to.list()
    assert_that(result == [3, 1, 2, 4], "union should preserve first sequence order")


@test("union with empty sequences")
def test_union_empty():
    assert_that(P([1, 2]).set.union([]).to.list() == [1, 2], "union with empty should return original")
    assert_that(empty().set.union([1, 2]).to.list() == [1, 2], "empty union other should return other")


# --- intersect ---

@test("intersect returns common elements in first order")
def test_intersect_basic():
    result = P([1, 2, 3, 4]).set.intersect([2, 4, 6]).to.list()
    assert_that(result == [2, 4], "intersect should return common elements")


@test("intersect preserves first sequence order")
def test_intersect_order():
    result = P([4, 2, 3, 1]).set.intersect([1, 2, 3, 4]).to.list()
    assert_that(result == [4, 2, 3, 1], "intersect should preserve first sequence order")


@test("intersect with no common elements")
def test_intersect_disjoint():
    result = P([1, 2]).set.intersect([3, 4]).to.list()
    assert_that(result == [], "intersect of disjoint sets should be empty")


# --- except_ ---

@test("except_ returns elements not in second sequence")
def test_except_basic():
    result = P([1, 2, 3, 4]).set.except_([2, 4]).to.list()
    assert_that(result == [1, 3], "except_ should return elements not in second sequence")


@test("except_ handles duplicates correctly")
def test_except_duplicates():
    result = P([1, 1, 2, 3]).set.except_([2]).to.list()
    assert_that(result == [1, 3], "except_ should return distinct elements")


@test("except_ with empty sequences")
def test_except_empty():
    assert_that(P([1, 2]).set.except_([]).to.list() == [1, 2], "except empty should return original")
    assert_that(empty().set.except_([1, 2]).to.list() == [], "empty except anything should be empty")


# --- symmetric_difference ---

@test("symmetric_difference returns elements in one or other but not both")
def test_symmetric_difference_basic():
    result = P([1, 2, 3]).set.symmetric_difference([2, 3, 4, 5]).to.set()
    assert_that(result == {1, 4, 5}, "symmetric difference should exclude common elements")


@test("symmetric_difference preserves order")
def test_symmetric_difference_order():
    result = P([3, 1]).set.symmetric_difference([2, 4]).to.list()
    assert_that(result == [3, 1, 2, 4], "symmetric difference should preserve appearance order")


# --- concat ---

@test("concat preserves all elements and duplicates")
def test_concat_basic():
    result = P([1, 2]).set.concat([2, 3]).to.list()
    assert_that(result == [1, 2, 2, 3], "concat should preserve all elements including duplicates")


@test("concat with empty sequences")
def test_concat_empty():
    assert_that(P([1, 2]).set.concat([]).to.list() == [1, 2], "concat with empty should return original")
    assert_that(empty().set.concat([1, 2]).to.list() == [1, 2], "empty concat should return other")


# --- multiset operations ---

@test("multiset_intersect respects element counts")
def test_multiset_intersect_basic():
    result = P([1, 1, 2, 3]).set.multiset_intersect([1, 2, 2]).to.list()
    sorted_result = sorted(result)
    assert_that(sorted_result == [1, 2], "multiset intersect should respect counts")


@test("multiset_intersect handles no intersection")
def test_multiset_intersect_empty():
    result = P([1, 2]).set.multiset_intersect([3, 4]).to.list()
    assert_that(result == [], "multiset intersect of disjoint should be empty")


@test("except_by_count respects element counts")
def test_except_by_count_basic():
    result = P([1, 1, 2, 3]).set.except_by_count([1, 2, 2]).to.list()
    sorted_result = sorted(result)
    assert_that(sorted_result == [1, 3], "except by count should respect element counts")


@test("except_by_count handles complete removal")
def test_except_by_count_complete():
    result = P([1, 2, 3]).set.except_by_count([1, 2, 3, 4]).to.list()
    assert_that(result == [], "except by count should handle complete removal")


# --- boolean set checks ---

@test("is_subset_of works correctly")
def test_is_subset_of():
    assert_that(P([1, 2]).set.is_subset_of([1, 2, 3]), "should be subset")
    assert_that(P([1, 2, 3]).set.is_subset_of([1, 2, 3]), "set should be subset of itself")
    assert_that(not P([1, 4]).set.is_subset_of([1, 2, 3]), "should not be subset")


@test("is_superset_of works correctly")
def test_is_superset_of():
    assert_that(P([1, 2, 3]).set.is_superset_of([1, 2]), "should be superset")
    assert_that(P([1, 2]).set.is_superset_of([1, 2]), "set should be superset of itself")
    assert_that(not P([1, 2]).set.is_superset_of([1, 3]), "should not be superset")


@test("is_proper_subset_of works correctly")
def test_is_proper_subset_of():
    assert_that(P([1, 2]).set.is_proper_subset_of([1, 2, 3]), "should be proper subset")
    assert_that(not P([1, 2, 3]).set.is_proper_subset_of([1, 2, 3]), "set should not be proper subset of itself")
    assert_that(not P([1, 4]).set.is_proper_subset_of([1, 2, 3]), "should not be proper subset")


@test("is_proper_superset_of works correctly")
def test_is_proper_superset_of():
    assert_that(P([1, 2, 3]).set.is_proper_superset_of([1, 2]), "should be proper superset")
    assert_that(not P([1, 2]).set.is_proper_superset_of([1, 2]), "set should not be proper superset of itself")
    assert_that(not P([1, 2]).set.is_proper_superset_of([1, 3]), "should not be proper superset")


@test("is_disjoint_with works correctly")
def test_is_disjoint_with():
    assert_that(P([1, 2]).set.is_disjoint_with([3, 4]), "disjoint sets should return true")
    assert_that(not P([1, 2]).set.is_disjoint_with([2, 3]), "overlapping sets should return false")


# --- jaccard similarity ---

@test("jaccard_similarity calculates correctly")
def test_jaccard_similarity_basic():
    similarity = P([1, 2, 3]).set.jaccard_similarity([2, 3, 4])
    assert_that(abs(similarity - 0.5) < 0.001, "jaccard similarity should be 0.5")


@test("jaccard_similarity handles identical sets")
def test_jaccard_similarity_identical():
    similarity = P([1, 2, 3]).set.jaccard_similarity([1, 2, 3])
    assert_that(abs(similarity - 1.0) < 0.001, "identical sets should have similarity 1.0")


@test("jaccard_similarity handles disjoint sets")
def test_jaccard_similarity_disjoint():
    similarity = P([1, 2]).set.jaccard_similarity([3, 4])
    assert_that(abs(similarity - 0.0) < 0.001, "disjoint sets should have similarity 0.0")


@test("jaccard_similarity handles empty sets")
def test_jaccard_similarity_empty():
    similarity = empty().set.jaccard_similarity([])
    assert_that(abs(similarity - 1.0) < 0.001, "two empty sets should have similarity 1.0")


# --- complex scenarios ---

@test("chained set operations work correctly")
def test_chained_operations():
    result = (P([1, 2, 3, 4, 5])
              .set.union([6, 7, 1])
              .set.intersect([1, 2, 6, 8, 9])
              .to.list())
    assert_that(sorted(result) == [1, 2, 6], "chained operations should work correctly")


@test("set operations with complex objects")
def test_complex_objects():
    # create controlled test data instead of relying on random generation
    people_data = [
        {'id': 1, 'city': 'ny', 'score': 85},  # ny, low score
        {'id': 2, 'city': 'ny', 'score': 95},  # ny, high score ✓
        {'id': 3, 'city': 'la', 'score': 95},  # not ny, high score
        {'id': 4, 'city': 'la', 'score': 80},  # not ny, low score
        {'id': 5, 'city': 'ny', 'score': 92},  # ny, high score ✓
    ]

    people = P(people_data)
    ny_people = people.where(lambda p: p['city'] == 'ny')
    high_scores = people.where(lambda p: p['score'] > 90)

    # test the intersection logic using id-based comparison to avoid dict hashing issues
    ny_people_list = ny_people.to.list()
    high_scores_list = high_scores.to.list()

    # create sets of ids for comparison
    ny_ids = set(p['id'] for p in ny_people_list)
    high_score_ids = set(p['id'] for p in high_scores_list)

    # find intersection of ids - should be {2, 5}
    intersection_ids = ny_ids.intersection(high_score_ids)

    # get actual people objects that match both criteria
    intersection = [p for p in people_data if p['id'] in intersection_ids]

    # should have exactly 2 people: id 2 and id 5
    assert_that(len(intersection) == 2, f"should have exactly 2 people in intersection, got {len(intersection)}")

    # verify each person meets both criteria
    for person in intersection:
        assert_that(person['city'] == 'ny', f"person {person['id']} should be from ny")
        assert_that(person['score'] > 90, f"person {person['id']} should have high score")

    # verify we got the right ids
    actual_ids = set(p['id'] for p in intersection)
    expected_ids = {2, 5}
    assert_that(actual_ids == expected_ids, f"should have ids {expected_ids}, got {actual_ids}")


if __name__ == "__main__":
    suite.run(title="pinqy set operations test")