"""
Run every experiment in the report and write results to ./results/.

    python benchmark.py            # full run  (~2-4 minutes)
    python benchmark.py --quick    # smaller sizes, for a fast sanity check

Outputs
    results/quicksort_times.csv
    results/quicksort_comparisons.csv
    results/hash_load_factor.csv
    results/hash_resizing.csv
    results/hash_adversarial.csv
    results/fig1_quicksort_by_distribution.png
    results/fig2_quicksort_comparisons.png
    results/fig3_hash_load_factor.png
    results/fig4_hash_resizing.png
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import platform
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from quicksort import (  # noqa: E402
    deterministic_quicksort,
    randomized_quicksort,
    randomized_quicksort_2way,
)
from hash_table import HashTableChaining  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SEED = 2026

# Colours (fixed categorical order; blue = randomized / main, orange = baseline)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# =========================================================================== #
# Input generators
# =========================================================================== #
def gen_random(n, rng):
    return [rng.randint(0, 10 * n) for _ in range(n)]


def gen_sorted(n, rng):
    return list(range(n))


def gen_reverse(n, rng):
    return list(range(n, 0, -1))


def gen_repeated(n, rng):
    # only 10 distinct values -> each value appears ~n/10 times
    return [rng.randint(0, 9) for _ in range(n)]


DISTRIBUTIONS = {
    "random": gen_random,
    "sorted": gen_sorted,
    "reverse": gen_reverse,
    "repeated": gen_repeated,
}


def time_it(fn, data, repeats):
    """Median wall-clock seconds of fn(copy of data) over `repeats` runs."""
    times = []
    for _ in range(repeats):
        a = data[:]
        t0 = time.perf_counter()
        fn(a)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


# =========================================================================== #
# Part 1 - Quicksort experiments
# =========================================================================== #
def quicksort_experiments(sizes, repeats):
    rng = random.Random(SEED)
    rows = []
    algos = {
        "randomized": lambda a: randomized_quicksort(a, rng=rng),
        "deterministic": deterministic_quicksort,
        "randomized_2way": lambda a: randomized_quicksort_2way(a, rng=rng),
    }
    for dist, gen in DISTRIBUTIONS.items():
        for n in sizes:
            data = gen(n, rng)
            for name, fn in algos.items():
                # The 2-way ablation is only interesting on duplicate-heavy input.
                if name == "randomized_2way" and dist != "repeated":
                    continue
                t = time_it(fn, data, repeats)
                rows.append({"distribution": dist, "n": n, "algorithm": name,
                             "seconds": round(t, 6)})
                print(f"  quicksort  {dist:9s} n={n:>7,}  {name:16s} {t*1000:10.2f} ms")
    _write_csv("quicksort_times.csv", rows)
    return rows


def quicksort_scaling(sizes, repeats):
    """Randomized quicksort on large random arrays + comparison counts vs theory."""
    rng = random.Random(SEED + 1)
    rows = []
    for n in sizes:
        data = [rng.random() for _ in range(n)]      # distinct keys
        counts, times = [], []
        for _ in range(repeats):
            stats = {}
            a = data[:]
            t0 = time.perf_counter()
            randomized_quicksort(a, rng=rng, stats=stats)
            times.append(time.perf_counter() - t0)
            counts.append(stats["comparisons"])
        # exact expected comparisons for distinct keys: 2(n+1)H_n - 4n
        H = sum(1.0 / k for k in range(1, n + 1))
        expected = 2 * (n + 1) * H - 4 * n
        rows.append({
            "n": n,
            "mean_comparisons": round(statistics.mean(counts)),
            "expected_2(n+1)H_n-4n": round(expected),
            "2n_ln_n": round(2 * n * math.log(n)),
            "n_log2_n": round(n * math.log2(n)),
            "median_seconds": round(statistics.median(times), 6),
        })
        print(f"  scaling    n={n:>8,}  comparisons={statistics.mean(counts):>12,.0f}"
              f"  expected={expected:>12,.0f}")
    _write_csv("quicksort_comparisons.csv", rows)
    return rows


# =========================================================================== #
# Part 2 - Hash table experiments
# =========================================================================== #
def hash_load_factor_experiment(m, alphas, queries):
    """Fixed number of slots m (resizing OFF); vary n = alpha * m."""
    rng = random.Random(SEED + 2)
    rows = []
    for alpha in alphas:
        n = int(alpha * m)
        t = HashTableChaining(capacity=m, resize=False, seed=SEED)
        keys = rng.sample(range(10**9), n + queries)
        present, absent = keys[:n], keys[n:]
        for k in present:
            t.insert(k, k)

        hit_keys = [rng.choice(present) for _ in range(queries)]
        miss_keys = absent[:queries]

        succ = statistics.mean(t.probes_for(k)[1] for k in hit_keys)
        unsucc = statistics.mean(t.probes_for(k)[1] for k in miss_keys)

        t0 = time.perf_counter()
        for k in hit_keys:
            t.search(k)
        hit_ns = (time.perf_counter() - t0) / queries * 1e9
        t0 = time.perf_counter()
        for k in miss_keys:
            t.search(k)
        miss_ns = (time.perf_counter() - t0) / queries * 1e9

        lens = t.chain_lengths()
        rows.append({
            "alpha": alpha, "n": n, "m": m,
            "avg_probes_successful": round(succ, 3),
            "theory_successful_1+a/2": round(1 + alpha / 2 - alpha / (2 * n), 3),
            "avg_probes_unsuccessful": round(unsucc, 3),
            "theory_unsuccessful_a": alpha,
            "search_hit_ns": round(hit_ns, 1),
            "search_miss_ns": round(miss_ns, 1),
            "max_chain": max(lens),
            "empty_slots_pct": round(100 * lens.count(0) / m, 1),
        })
        print(f"  hash α={alpha:<5} probes hit={succ:6.2f} miss={unsucc:6.2f}"
              f"  hit={hit_ns:7.0f} ns  miss={miss_ns:7.0f} ns  max chain={max(lens)}")
    _write_csv("hash_load_factor.csv", rows)
    return rows


def hash_resizing_experiment(sizes):
    """Average insert / search / delete cost as n grows: resizing vs fixed m."""
    rng = random.Random(SEED + 3)
    rows = []
    for n in sizes:
        keys = rng.sample(range(10**9), n)
        for label, kwargs in [("dynamic resizing (α ≤ 1)", dict(capacity=8)),
                              ("fixed 1,024 slots", dict(capacity=1024, resize=False))]:
            t = HashTableChaining(seed=SEED, **kwargs)
            t0 = time.perf_counter()
            for k in keys:
                t.insert(k, k)
            ins = (time.perf_counter() - t0) / n * 1e9
            final_alpha = t.load_factor
            final_m = t.capacity
            # random sample, so we do not only probe the (front-of-chain) earliest keys
            sample = rng.sample(keys, min(n, 20000))
            t0 = time.perf_counter()
            for k in sample:
                t.search(k)
            srch = (time.perf_counter() - t0) / len(sample) * 1e9
            t0 = time.perf_counter()
            for k in sample:
                t.delete(k)
            dele = (time.perf_counter() - t0) / len(sample) * 1e9
            rows.append({"n": n, "variant": label,
                         "insert_ns_per_op": round(ins, 1),
                         "search_ns_per_op": round(srch, 1),
                         "delete_ns_per_op": round(dele, 1),
                         "final_slots": final_m,
                         "final_load_factor": round(final_alpha, 3),
                         "resizes": t.resize_count})
            print(f"  resize n={n:>7,} {label:26s} insert={ins:7.0f} ns search={srch:7.0f} ns"
                  f" delete={dele:7.0f} ns  α={final_alpha:.2f}")
    _write_csv("hash_resizing.csv", rows)
    return rows


def hash_adversarial_experiment(m=1024, n=1024):
    """Keys that are all multiples of m: fatal for h(k)=k mod m, harmless for universal."""
    keys = [i * m for i in range(n)]
    naive_buckets = [0] * m
    for k in keys:
        naive_buckets[k % m] += 1
    rows = [{"hash_function": "division  h(k) = k mod m",
             "max_chain": max(naive_buckets),
             "non_empty_slots": sum(1 for c in naive_buckets if c)}]
    t = HashTableChaining(capacity=m, resize=False, seed=SEED)
    for k in keys:
        t.insert(k, k)
    lens = t.chain_lengths()
    rows.append({"hash_function": "universal ((a*k+b) mod p) mod m",
                 "max_chain": max(lens),
                 "non_empty_slots": sum(1 for c in lens if c)})
    for r in rows:
        print(f"  adversarial {r['hash_function']:34s} max chain={r['max_chain']:5d}"
              f"  non-empty slots={r['non_empty_slots']}")
    _write_csv("hash_adversarial.csv", rows)
    return rows


# =========================================================================== #
# Plotting
# =========================================================================== #
def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def make_plots(qs_rows, cmp_rows, lf_rows, rs_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "#fcfcfb",
                         "axes.facecolor": "#fcfcfb"})

    # ---- Fig 1: quicksort per distribution (log-log) ---------------------- #
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    series = [("randomized", "Randomized (random pivot, 3-way)", BLUE, "o"),
              ("deterministic", "Deterministic (first-element pivot)", ORANGE, "s"),
              ("randomized_2way", "Randomized, 2-way partition (ablation)", AQUA, "^")]
    titles = {"random": "Random array", "sorted": "Already sorted",
              "reverse": "Reverse sorted", "repeated": "Repeated elements (10 distinct values)"}
    for ax, dist in zip(axes.flat, DISTRIBUTIONS):
        for key, label, color, marker in series:
            pts = [(r["n"], r["seconds"] * 1000) for r in qs_rows
                   if r["distribution"] == dist and r["algorithm"] == key]
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=color, linewidth=2, marker=marker, markersize=6,
                    label=label)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        _style(ax, titles[dist], "input size n (log scale)", "time, ms (log scale)")
    handles, labels = axes.flat[3].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Randomized vs deterministic Quicksort — median running time",
                 x=0.02, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.02, 0.935, "On log–log axes, slope ≈ 1 means ~n log n growth; "
             "slope ≈ 2 means quadratic growth.", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(os.path.join(OUT, "fig1_quicksort_by_distribution.png"), dpi=150)
    plt.close(fig)

    # ---- Fig 2: comparisons vs theory ------------------------------------ #
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ns = [r["n"] for r in cmp_rows]
    meas = [r["mean_comparisons"] / r["n"] for r in cmp_rows]
    theo = [r["expected_2(n+1)H_n-4n"] / r["n"] for r in cmp_rows]
    ax.plot(ns, theo, color=INK2, linewidth=2, linestyle="--",
            label="Theory: 2(n+1)Hₙ − 4n  (≈ 2n ln n)")
    ax.plot(ns, meas, color=BLUE, linewidth=2, marker="o", markersize=7,
            label="Measured (randomized quicksort)")
    ax.set_xscale("log", base=2)
    _style(ax, "Comparisons per element grow like log n",
           "input size n (log scale)", "comparisons ÷ n")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_quicksort_comparisons.png"), dpi=150)
    plt.close(fig)

    # ---- Fig 3: hash load factor ----------------------------------------- #
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    al = [r["alpha"] for r in lf_rows]
    a1.plot(al, [r["avg_probes_successful"] for r in lf_rows], color=BLUE,
            marker="o", linewidth=2, markersize=7, label="measured, successful search")
    a1.plot(al, [r["avg_probes_unsuccessful"] for r in lf_rows], color=ORANGE,
            marker="s", linewidth=2, markersize=7, label="measured, unsuccessful search")
    a1.plot(al, [r["theory_successful_1+a/2"] for r in lf_rows], color=INK,
            linestyle=(0, (3, 2)), linewidth=1.2, label="theory: 1 + α/2 (hit), α (miss)")
    a1.plot(al, [r["theory_unsuccessful_a"] for r in lf_rows], color=INK,
            linestyle=(0, (3, 2)), linewidth=1.2)
    _style(a1, "Keys examined per search", "load factor α = n / m",
           "chain elements examined")
    a1.legend(frameon=False, fontsize=8)
    a2.plot(al, [r["search_hit_ns"] for r in lf_rows], color=BLUE, marker="o",
            linewidth=2, markersize=6, label="successful search")
    a2.plot(al, [r["search_miss_ns"] for r in lf_rows], color=ORANGE, marker="s",
            linewidth=2, markersize=6, label="unsuccessful search")
    _style(a2, "Wall-clock time per search", "load factor α = n / m", "ns per search")
    a2.set_ylim(bottom=0)
    a2.legend(frameon=False, fontsize=8)
    fig.suptitle("Hashing with chaining: cost grows linearly with the load factor",
                 x=0.02, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT, "fig3_hash_load_factor.png"), dpi=150)
    plt.close(fig)

    # ---- Fig 4: resizing -------------------------------------------------- #
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for variant, color, marker in [("dynamic resizing (α ≤ 1)", BLUE, "o"),
                                   ("fixed 1,024 slots", ORANGE, "s")]:
        pts = [r for r in rs_rows if r["variant"] == variant]
        xs = [r["n"] for r in pts]
        a1.plot(xs, [r["insert_ns_per_op"] for r in pts], color=color, marker=marker,
                linewidth=2, markersize=6, label=variant)
        a2.plot(xs, [r["search_ns_per_op"] for r in pts], color=color, marker=marker,
                linewidth=2, markersize=6, label=variant)
    for ax, what in [(a1, "insert"), (a2, "search")]:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        _style(ax, f"Average {what} time", "number of keys n (log scale)",
               f"ns per {what} (log scale)")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Dynamic resizing keeps operations O(1) as the table grows",
                 x=0.02, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT, "fig4_hash_resizing.png"), dpi=150)
    plt.close(fig)


# =========================================================================== #
def _write_csv(name, rows):
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small sizes, fast run")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    sys.setrecursionlimit(10000)

    if args.quick:
        qs_sizes, reps = [250, 500, 1000, 2000], 3
        scale_sizes = [1000, 4000, 16000]
        hash_m, hash_q = 1024, 2000
        rs_sizes = [1000, 4000, 16000]
    else:
        qs_sizes, reps = [500, 1000, 2000, 4000, 8000], 3
        scale_sizes = [1000, 4000, 16000, 64000, 256000]
        hash_m, hash_q = 4096, 20000
        rs_sizes = [1000, 4000, 16000, 64000, 256000]

    print(f"Python {platform.python_version()} on {platform.platform()}")
    print("\n[1/5] Quicksort: randomized vs deterministic")
    qs = quicksort_experiments(qs_sizes, reps)
    print("\n[2/5] Quicksort: comparison counts vs theory")
    cmp_rows = quicksort_scaling(scale_sizes, 5)
    print("\n[3/5] Hash table: effect of load factor")
    lf = hash_load_factor_experiment(hash_m, [0.25, 0.5, 1, 2, 4, 8, 16], hash_q)
    print("\n[4/5] Hash table: dynamic resizing")
    rs = hash_resizing_experiment(rs_sizes)
    print("\n[5/5] Hash table: adversarial keys")
    hash_adversarial_experiment()
    make_plots(qs, cmp_rows, lf, rs)
    print(f"\nDone. Results and figures written to {OUT}")


if __name__ == "__main__":
    main()
