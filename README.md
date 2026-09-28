# Algorithm Efficiency and Scalability

**Assignment 3: Randomized Quicksort and Hashing with Chaining**

This repository contains Python implementations of **Randomized Quicksort** and a **hash table with chaining** (universal hashing and dynamic resizing), a benchmark suite that reproduces every experiment, and a full report with the theoretical analysis and a discussion of the results.

📄 **Full report: [REPORT.docx](REPORT.docx)** (Word document; on GitHub click the file, then **Download raw file** to open it)

## Repository layout

```
.
├── README.md              ← you are here
├── REPORT.docx            ← theoretical analysis, experiments, discussion
├── benchmark.py           ← runs all experiments, writes results/
├── requirements.txt
├── src/
│   ├── quicksort.py       ← randomized_quicksort, deterministic_quicksort
│   └── hash_table.py      ← HashTableChaining
├── tests/
│   └── test_all.py        ← correctness tests for both parts
└── results/               ← CSV data + figures produced by benchmark.py
```

## How to run

Requires **Python 3.8+**. The algorithms themselves use only the standard library; `matplotlib` is needed only to draw the figures.

```bash
git clone https://github.com/<your-username>/algorithm-efficiency-scalability.git
cd algorithm-efficiency-scalability
pip install -r requirements.txt

# 1. Correctness tests (either command works)
python tests/test_all.py
python -m pytest -q

# 2. Quick demos
python src/quicksort.py
python src/hash_table.py

# 3. Reproduce all experiments and figures (about 20 seconds)
python benchmark.py
python benchmark.py --quick     # smaller sizes, a few seconds
```

### Using the code

```python
import sys; sys.path.insert(0, "src")
from quicksort import randomized_quicksort
from hash_table import HashTableChaining

randomized_quicksort([5, 3, 8, 3, 1])      # -> [1, 3, 3, 5, 8] (sorts in place, also returns the list)

t = HashTableChaining()
t.insert("apple", 3)
t.search("apple")    # -> 3
t.delete("apple")    # -> True
t.search("apple")    # -> None
```

## Implementation highlights

**Randomized Quicksort** (`src/quicksort.py`)
- The pivot is chosen uniformly at random from the current subarray.
- A 3-way partition (`< | == | >`) handles repeated elements in linear time.
- It recurses on the smaller side, so stack depth is O(log n) and there is no `RecursionError`.
- Empty, single-element, sorted, reverse-sorted and all-equal arrays are all handled and tested.

**Hash table with chaining** (`src/hash_table.py`)
- Uses the universal hash function h(k) = ((a·k + b) mod p) mod m with p = 2⁶¹ − 1, and picks a fresh random (a, b) on every resize.
- `insert`, `search` and `delete` run in expected O(1 + α) time.
- The table doubles when the load factor α > 1 and halves when α < ¼, giving O(1) amortized cost per operation.

## Summary of findings

**Quicksort** (median times, n = 8,000):

| Input | Randomized | Deterministic (first pivot) | Randomized advantage |
|---|---:|---:|---:|
| Random | 9.4 ms | 5.5 ms | 0.6× (deterministic faster) |
| Already sorted | 9.2 ms | 564 ms | **61×** |
| Reverse sorted | 8.8 ms | 1,085 ms | **123×** |
| Repeated (10 distinct values) | 2.0 ms | 141 ms | **70×** |

- Randomized Quicksort ran in about the same time on every input order, as the expected **O(n log n)** bound (≈ 2n ln n comparisons) predicts for every input. The measured comparison counts were within 3% of the exact formula 2(n+1)Hₙ − 4n.
- Deterministic Quicksort became **quadratic** on sorted and reverse-sorted input, taking ×4 time each time n doubled.
- On random input the deterministic version was about 1.5–2× faster. That is a **constant-factor** effect (the cost of calling `random.randint` and the extra comparison in the 3-way partition), not a difference in growth rate.
- With many repeated keys, a random pivot alone did **not** help (random pivot with a 2-way partition was just as slow). The **3-way partition** is what makes that case fast.

**Hash table:**
- The number of keys examined per search matched the theory almost exactly: **1 + α/2** for successful searches and **α** for unsuccessful ones. Cost grows linearly with the load factor.
- With dynamic resizing, α stays ≤ 1 and operations stay O(1). At 256,000 keys, a fixed 1,024-slot table (α = 250) had **12× slower searches** and **14× slower deletes**.
- On adversarial keys (all multiples of m), `k mod m` put all 1,024 keys in **one** chain. The universal hash function's longest chain was **2**.

See [REPORT.docx](REPORT.docx) for the step-by-step proofs, all figures, and the full discussion.
