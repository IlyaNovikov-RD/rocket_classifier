#!/usr/bin/env python3
"""Download production model artifacts from GitHub Release.

Downloads model.pkl, train_medians.npy, and threshold_biases.npy from
the v1.0.0 GitHub Release into the project root. Required before running
the inference pipeline (``python -m rocket_classifier.main``).

Usage:
    python download_weights.py
    # or: make download-weights
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

RELEASE_BASE = (
    "https://github.com/IlyaNovikov-RD/rocket_classifier"
    "/releases/download/v1.0.0"
)

ARTIFACTS = [
    "model.pkl",
    "train_medians.npy",
    "threshold_biases.npy",
]

# Feature caches — large parquet files (~15 MB total).
# Optional: skip if you have data/ and want to recompute from scratch.
CACHE_ARTIFACTS = [
    "cache_train_features.parquet",
    "cache_test_features.parquet",
]

ROOT = Path(__file__).parent


def main(include_caches: bool = False) -> None:
    targets = ARTIFACTS + (CACHE_ARTIFACTS if include_caches else [])
    for name in targets:
        dest = ROOT / name
        if dest.exists():
            print(f"  {name} — already exists, skipping")
            continue
        url = f"{RELEASE_BASE}/{name}"
        print(f"  Downloading {name} ...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, str(dest))
            size_mb = dest.stat().st_size / 1_048_576
            print(f"OK ({size_mb:.1f} MB)")
        except Exception as exc:
            print(f"FAILED: {exc}")
            sys.exit(1)
    print("All artifacts ready.")


if __name__ == "__main__":
    caches = "--with-caches" in sys.argv
    main(include_caches=caches)
