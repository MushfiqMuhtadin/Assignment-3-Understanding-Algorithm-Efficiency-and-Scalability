"""
Randomized Quicksort and Deterministic Quicksort.

Two sorting functions live here:

* randomized_quicksort(arr)
    - Pivot is chosen *uniformly at random* from the subarray being partitioned.
    - Uses 3-way partitioning ("Dutch National Flag"), which splits the subarray
      into  < pivot | == pivot | > pivot.  All copies of the pivot are finished in
      one pass, so arrays with many repeated elements stay O(n log n) instead of
      degrading to O(n^2).

* deterministic_quicksort(arr)
    - Classic textbook version: the FIRST element of the subarray is the pivot,
      with a standard 2-way (Lomuto-style) partition.
    - Worst case O(n^2) on already-sorted, reverse-sorted and all-equal inputs.

Both functions:
* sort in place AND return the list (so `sorted_list = randomized_quicksort(x)` works);
* handle empty lists and single-element lists immediately;
* recurse on the SMALLER side and loop on the larger side, so the call-stack depth
  is at most O(log n) even when the running time is O(n^2).  This means the
  deterministic version never crashes with RecursionError on sorted input - it is
  just slow, which is exactly what we want to measure.

An optional `stats` dict can be passed; if given, stats["comparisons"] is increased
by the number of elements compared against a pivot.  This lets the benchmark relate
measured work to the theoretical 2 n ln n bound.
"""

from __future__ import annotations

import random
from typing import Any, List, MutableSequence, Optional

__all__ = [
    "randomized_quicksort",
    "deterministic_quicksort",
    "randomized_quicksort_2way",
]


# --------------------------------------------------------------------------- #
# Partition helpers
# --------------------------------------------------------------------------- #
def _partition_3way(a: MutableSequence, lo: int, hi: int, pivot_index: int):
    """
    Partition a[lo..hi] (inclusive) around a[pivot_index] into three regions:

        a[lo .. lt-1]   <  pivot
        a[lt .. gt]     == pivot
        a[gt+1 .. hi]   >  pivot

    Returns (lt, gt).  Runs in Theta(hi - lo + 1) time.
    """
    pivot = a[pivot_index]
    lt, i, gt = lo, lo, hi
    while i <= gt:
        x = a[i]
        if x < pivot:
            a[lt], a[i] = x, a[lt]
            lt += 1
            i += 1
        elif x > pivot:
            a[gt], a[i] = x, a[gt]
            gt -= 1
        else:
            i += 1
    return lt, gt


def _partition_lomuto(a: MutableSequence, lo: int, hi: int) -> int:
    """
    CLRS-style Lomuto partition using a[lo] as the pivot.

    After the call:  a[lo..p-1] <= pivot,  a[p] == pivot,  a[p+1..hi] > pivot.
    Returns p.  Runs in Theta(hi - lo + 1) time.
    """
    pivot = a[lo]
    i = lo
    for j in range(lo + 1, hi + 1):
        if a[j] <= pivot:
            i += 1
            a[i], a[j] = a[j], a[i]
    a[lo], a[i] = a[i], a[lo]
    return i


# --------------------------------------------------------------------------- #
# Randomized Quicksort (the main implementation)
# --------------------------------------------------------------------------- #
def randomized_quicksort(
    arr: List[Any],
    rng: Optional[random.Random] = None,
    stats: Optional[dict] = None,
) -> List[Any]:
    """
    Sort `arr` in place with Randomized Quicksort and return it.

    Expected running time: O(n log n) for EVERY input (the expectation is over the
    algorithm's own random pivot choices, not over the input).
    Extra space: O(log n) for the call stack.

    >>> randomized_quicksort([3, 1, 2, 3, 0])
    [0, 1, 2, 3, 3]
    >>> randomized_quicksort([])
    []
    """
    if arr is None:
        raise TypeError("arr must be a list, not None")
    n = len(arr)
    if n < 2:
        return arr
    rand = (rng or random).randint
    _rqs(arr, 0, n - 1, rand, stats)
    return arr


def _rqs(a, lo, hi, rand, stats):
    while lo < hi:
        p = rand(lo, hi)                       # uniform random pivot in a[lo..hi]
        if stats is not None:
            stats["comparisons"] = stats.get("comparisons", 0) + (hi - lo)
        lt, gt = _partition_3way(a, lo, hi, p)
        # Recurse on the smaller side, iterate on the larger one -> O(log n) stack.
        if (lt - lo) < (hi - gt):
            _rqs(a, lo, lt - 1, rand, stats)
            lo = gt + 1
        else:
            _rqs(a, gt + 1, hi, rand, stats)
            hi = lt - 1


# --------------------------------------------------------------------------- #
# Deterministic Quicksort (first element as pivot) - the baseline
# --------------------------------------------------------------------------- #
def deterministic_quicksort(
    arr: List[Any], stats: Optional[dict] = None
) -> List[Any]:
    """
    Sort `arr` in place using the first element of each subarray as the pivot.

    Average case (random input): O(n log n).
    Worst case (sorted / reverse-sorted / all-equal input): Theta(n^2).

    >>> deterministic_quicksort([5, 4, 3, 2, 1])
    [1, 2, 3, 4, 5]
    """
    if arr is None:
        raise TypeError("arr must be a list, not None")
    n = len(arr)
    if n < 2:
        return arr
    _dqs(arr, 0, n - 1, stats)
    return arr


def _dqs(a, lo, hi, stats):
    while lo < hi:
        if stats is not None:
            stats["comparisons"] = stats.get("comparisons", 0) + (hi - lo)
        p = _partition_lomuto(a, lo, hi)
        if (p - lo) < (hi - p):
            _dqs(a, lo, p - 1, stats)
            lo = p + 1
        else:
            _dqs(a, p + 1, hi, stats)
            hi = p - 1


# --------------------------------------------------------------------------- #
# Ablation variant: random pivot but ordinary 2-way partition
# --------------------------------------------------------------------------- #
def randomized_quicksort_2way(
    arr: List[Any],
    rng: Optional[random.Random] = None,
    stats: Optional[dict] = None,
) -> List[Any]:
    """
    Random pivot + Lomuto 2-way partition.  NOT the recommended version - it is
    included only to show (in the benchmark) that random pivots alone do not fix
    inputs with many duplicates: when every key is equal, every split is n-1 / 0.
    """
    n = len(arr)
    if n < 2:
        return arr
    rand = (rng or random).randint

    def go(lo, hi):
        while lo < hi:
            r = rand(lo, hi)
            arr[lo], arr[r] = arr[r], arr[lo]      # move random pivot to front
            if stats is not None:
                stats["comparisons"] = stats.get("comparisons", 0) + (hi - lo)
            p = _partition_lomuto(arr, lo, hi)
            if (p - lo) < (hi - p):
                go(lo, p - 1)
                lo = p + 1
            else:
                go(p + 1, hi)
                hi = p - 1

    go(0, n - 1)
    return arr


if __name__ == "__main__":
    demo = [9, 3, 7, 3, 1, 8, 2, 7, 0, 3]
    print("input:        ", demo)
    print("randomized:   ", randomized_quicksort(demo[:]))
    print("deterministic:", deterministic_quicksort(demo[:]))
