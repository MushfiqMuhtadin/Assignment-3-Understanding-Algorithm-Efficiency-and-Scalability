"""
Hash table with separate chaining and a universal hash function.

Design
------
* Table = a Python list of `m` buckets; each bucket is a list ("chain") of
  [key, value] pairs.  Colliding keys simply share a chain.

* Hash function - the Carter-Wegman universal family

        h_{a,b}(k) = ((a * k + b) mod p) mod m

  with p = 2**61 - 1 (a Mersenne prime larger than any key we map to),
  a chosen uniformly from {1, ..., p-1} and b from {0, ..., p-1}.
  For any two distinct keys x != y, Pr[h(x) == h(y)] <= 1/m, no matter how the
  keys were chosen.  A fresh (a, b) is drawn every time the table is resized, so
  an adversary cannot pick keys that are known to collide.

  Non-integer keys (strings, tuples, ...) are first turned into an integer with
  Python's built-in hash() and then fed to the universal function.

* Dynamic resizing
    - load factor  alpha = n / m
    - if alpha would exceed MAX_LOAD (default 1.0)   -> double m and rehash
    - if alpha drops below MIN_LOAD (default 0.25)   -> halve m (never below the
      initial capacity) and rehash
  Doubling makes insertion O(1) *amortized*; the gap between 0.25 and 1.0
  prevents "thrashing" (grow/shrink/grow...) around a single threshold.

Operations
----------
insert(key, value)   O(1 + alpha) expected; updates the value if key exists
search(key)          O(1 + alpha) expected; returns value or None (see get())
delete(key)          O(1 + alpha) expected; returns True if a key was removed
"""

from __future__ import annotations

import random
from typing import Any, Hashable, Iterator, List, Optional, Tuple

__all__ = ["HashTableChaining"]

_MERSENNE_PRIME = (1 << 61) - 1
_MISSING = object()


class HashTableChaining:
    def __init__(
        self,
        capacity: int = 8,
        max_load: float = 1.0,
        min_load: float = 0.25,
        resize: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if not (0 < min_load < max_load):
            raise ValueError("need 0 < min_load < max_load")
        self._rng = random.Random(seed)
        self._initial_capacity = capacity
        self._m = capacity
        self._n = 0
        self.max_load = max_load
        self.min_load = min_load
        self.resize_enabled = resize
        self.resize_count = 0
        self._buckets: List[List[list]] = [[] for _ in range(self._m)]
        self._new_hash_params()

    # ------------------------------------------------------------------ #
    # Hashing
    # ------------------------------------------------------------------ #
    def _new_hash_params(self) -> None:
        """Draw a fresh function h_{a,b} from the universal family."""
        self._a = self._rng.randrange(1, _MERSENNE_PRIME)
        self._b = self._rng.randrange(0, _MERSENNE_PRIME)

    def _index(self, key: Hashable) -> int:
        k = key if isinstance(key, int) else hash(key)
        k %= _MERSENNE_PRIME                       # map into [0, p)
        return ((self._a * k + self._b) % _MERSENNE_PRIME) % self._m

    # ------------------------------------------------------------------ #
    # Core operations
    # ------------------------------------------------------------------ #
    def insert(self, key: Hashable, value: Any) -> None:
        """Add (key, value); if key is already present, overwrite its value."""
        chain = self._buckets[self._index(key)]
        for pair in chain:
            if pair[0] == key:
                pair[1] = value
                return
        chain.append([key, value])
        self._n += 1
        if self.resize_enabled and self._n > self.max_load * self._m:
            self._rehash(self._m * 2)

    def search(self, key: Hashable, default: Any = None) -> Any:
        """Return the value stored for key, or `default` if key is absent."""
        for k, v in self._buckets[self._index(key)]:
            if k == key:
                return v
        return default

    def delete(self, key: Hashable) -> bool:
        """Remove key. Returns True if it was present, False otherwise."""
        chain = self._buckets[self._index(key)]
        for i, pair in enumerate(chain):
            if pair[0] == key:
                # O(1) removal: swap with last element of the chain, then pop.
                chain[i] = chain[-1]
                chain.pop()
                self._n -= 1
                if (
                    self.resize_enabled
                    and self._m > self._initial_capacity
                    and self._n < self.min_load * self._m
                ):
                    self._rehash(max(self._initial_capacity, self._m // 2))
                return True
        return False

    def _rehash(self, new_m: int) -> None:
        old = self._buckets
        self._m = new_m
        self._buckets = [[] for _ in range(new_m)]
        self._new_hash_params()
        for chain in old:
            for pair in chain:
                self._buckets[self._index(pair[0])].append(pair)
        self.resize_count += 1

    # ------------------------------------------------------------------ #
    # Instrumentation (used by the benchmark / report)
    # ------------------------------------------------------------------ #
    def probes_for(self, key: Hashable) -> Tuple[bool, int]:
        """(found?, number of chain elements examined) for a search of key."""
        examined = 0
        for k, _ in self._buckets[self._index(key)]:
            examined += 1
            if k == key:
                return True, examined
        return False, examined

    def chain_lengths(self) -> List[int]:
        return [len(c) for c in self._buckets]

    @property
    def load_factor(self) -> float:
        return self._n / self._m

    @property
    def capacity(self) -> int:
        return self._m

    # ------------------------------------------------------------------ #
    # Pythonic conveniences
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return self._n

    def __contains__(self, key: Hashable) -> bool:
        return self.search(key, _MISSING) is not _MISSING

    def __getitem__(self, key: Hashable) -> Any:
        v = self.search(key, _MISSING)
        if v is _MISSING:
            raise KeyError(key)
        return v

    def __setitem__(self, key: Hashable, value: Any) -> None:
        self.insert(key, value)

    def __delitem__(self, key: Hashable) -> None:
        if not self.delete(key):
            raise KeyError(key)

    def items(self) -> Iterator[Tuple[Hashable, Any]]:
        for chain in self._buckets:
            for k, v in chain:
                yield k, v

    def __repr__(self) -> str:
        return (
            f"HashTableChaining(n={self._n}, m={self._m}, "
            f"load_factor={self.load_factor:.2f})"
        )


if __name__ == "__main__":
    t = HashTableChaining(seed=42)
    for word in ["apple", "banana", "cherry", "date", "elderberry"]:
        t.insert(word, len(word))
    print(t)
    print("search('cherry') ->", t.search("cherry"))
    print("delete('banana') ->", t.delete("banana"))
    print("search('banana') ->", t.search("banana"))
    print(t)
