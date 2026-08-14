"""Small benchmarking script using only existing gridsfm API calls (predict,
load_model, load_pyg_json) -- no new data/format code.

1. Cache warm-up: repeated predict() on the SAME topology (warm cache after
   the first call) vs cycling across DIFFERENT topologies each call (always
   cold), to quantify the LRU-cache speedup described in the capability doc.
2. Per-graph inference latency vs grid size: unbatched predict() timing
   across all 53 shipped samples, to see how single-graph latency scales
   with bus count (complements infer_samples.py's single batched timing).

Run from the `model/` directory: python scripts_experiments/cache_and_scaling.py
"""

from __future__ import annotations

import statistics as stats
import time
from pathlib import Path

import torch

from gridsfm import load_model, predict

ROOT = Path(__file__).parent.parent
CKPT = ROOT / "checkpoints" / "gridsfm_open_v1.1.pt"
SAMPLES_DIR = ROOT / "samples"


def time_call(fn) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    model = load_model(str(CKPT), device=device)

    samples = sorted(SAMPLES_DIR.glob("*.pyg.json"))
    print(f"{len(samples)} samples available\n")

    # --- Experiment 1: cache warm-up ---
    warm_sample = samples[len(samples) // 2]  # a mid-size case, arbitrary
    print(f"=== Cache warm-up on {warm_sample.stem} ===")
    n_repeats = 10
    warm_times = []
    for i in range(n_repeats):
        t = time_call(lambda: predict(model, str(warm_sample)))
        warm_times.append(t)
        print(f"  call {i}: {t * 1000:.2f} ms" + ("  (cold)" if i == 0 else ""))

    cold_first = warm_times[0]
    warm_rest = warm_times[1:]
    print(
        f"  first call (cold): {cold_first * 1000:.2f} ms, "
        f"subsequent (warm) mean: {stats.mean(warm_rest) * 1000:.2f} ms, "
        f"speedup: {cold_first / stats.mean(warm_rest):.2f}x",
    )

    print(f"\n=== Always-cold: cycling through {min(10, len(samples))} distinct topologies ===")
    cold_cycle = samples[:10]
    cold_times = []
    for s in cold_cycle:
        t = time_call(lambda s=s: predict(model, str(s)))
        cold_times.append(t)
    print(
        f"  mean latency across distinct topologies: {stats.mean(cold_times) * 1000:.2f} ms "
        f"(vs {stats.mean(warm_rest) * 1000:.2f} ms warm-cache-hit on a repeated topology)",
    )

    # --- Experiment 2: per-graph latency vs grid size ---
    print("\n=== Per-graph latency vs grid size (unbatched predict(), warm run) ===")
    import json

    results = []
    for s in samples:
        with open(s) as f:
            n_bus = len(json.load(f)["grid"]["nodes"]["bus"])
        # warm-up call (build caches for this topology) then time a second call
        predict(model, str(s))
        t = time_call(lambda s=s: predict(model, str(s)))
        results.append((s.stem, n_bus, t))

    results.sort(key=lambda r: r[1])
    print(f"{'case':30s}  {'n_bus':>6s}  {'latency_ms':>10s}")
    for name, n_bus, t in results:
        print(f"{name:30s}  {n_bus:6d}  {t * 1000:10.2f}")

    sizes = [r[1] for r in results]
    times_ms = [r[2] * 1000 for r in results]
    print(f"\nSmallest ({sizes[0]} bus): {times_ms[0]:.2f} ms")
    print(f"Largest  ({sizes[-1]} bus): {times_ms[-1]:.2f} ms")
    print(f"Ratio: {times_ms[-1] / times_ms[0]:.1f}x latency for {sizes[-1] / sizes[0]:.1f}x buses")


if __name__ == "__main__":
    main()
