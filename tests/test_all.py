"""Correctness tests.  Run with:  python -m pytest -q   (or python tests/test_all.py)"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quicksort import (  # noqa: E402
    deterministic_quicksort,
    randomized_quicksort,
    randomized_quicksort_2way,
)
from hash_table import HashTableChaining  # noqa: E402

SORTERS = [randomized_quicksort, deterministic_quicksort, randomized_quicksort_2way]


# ----------------------------- Quicksort ---------------------------------- #
def _cases():
    rng = random.Random(0)
    yield []
    yield [1]
    yield [2, 1]
    yield [1, 1, 1, 1, 1]
    yield list(range(200))
    yield list(range(200, 0, -1))
    yield [rng.randint(0, 5) for _ in range(500)]
    yield [rng.randint(-10**6, 10**6) for _ in range(1000)]
    yield [rng.random() for _ in range(300)]
    yield ["pear", "apple", "fig", "apple", "kiwi"]


def test_sorts_match_builtin():
    for sorter in SORTERS:
        for case in _cases():
            expected = sorted(case)
            assert sorter(list(case)) == expected, (sorter.__name__, case[:10])


def test_sorts_in_place_and_returns_same_list():
    data = [3, 1, 2]
    out = randomized_quicksort(data)
    assert out is data and data == [1, 2, 3]


def test_large_sorted_input_no_recursion_error():
    # Deterministic worst case: must be slow-but-correct, never crash.
    data = list(range(3000))
    assert deterministic_quicksort(data[:]) == data
    assert randomized_quicksort(data[:]) == data


def test_randomized_many_duplicates_is_fast():
    stats = {}
    randomized_quicksort([7] * 20000, stats=stats)
    # 3-way partition finishes all-equal input in a single pass.
    assert stats["comparisons"] == 19999


# ----------------------------- Hash table --------------------------------- #
def test_insert_search_delete():
    t = HashTableChaining(seed=1)
    t.insert("a", 1)
    t.insert("b", 2)
    assert t.search("a") == 1 and t.search("b") == 2
    assert t.search("zzz") is None
    t.insert("a", 10)                 # update, not duplicate
    assert t.search("a") == 10 and len(t) == 2
    assert t.delete("a") is True
    assert t.delete("a") is False
    assert "a" not in t and len(t) == 1


def test_matches_dict_under_random_operations():
    rng = random.Random(123)
    t, ref = HashTableChaining(seed=7), {}
    for _ in range(20000):
        k = rng.randint(0, 2000)
        op = rng.random()
        if op < 0.5:
            v = rng.random()
            t.insert(k, v)
            ref[k] = v
        elif op < 0.8:
            assert t.search(k) == ref.get(k)
        else:
            assert t.delete(k) == (k in ref)
            ref.pop(k, None)
    assert len(t) == len(ref)
    assert dict(t.items()) == ref


def test_resizing_keeps_load_factor_bounded():
    t = HashTableChaining(capacity=8, seed=3)
    for i in range(10000):
        t.insert(i, i)
        assert t.load_factor <= t.max_load
    assert t.capacity >= 10000
    for i in range(10000):
        t.delete(i)
    assert len(t) == 0 and t.capacity == 8


def test_mixed_key_types_and_dunder_methods():
    t = HashTableChaining(seed=0)
    t[(1, 2)] = "tuple"
    t[3.5] = "float"
    t[-17] = "negative"
    assert t[(1, 2)] == "tuple" and t[3.5] == "float" and t[-17] == "negative"
    del t[3.5]
    try:
        _ = t[3.5]
        raise AssertionError("expected KeyError")
    except KeyError:
        pass


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
