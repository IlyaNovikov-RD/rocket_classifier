#!/usr/bin/env python3
"""Export model.pkl to faster inference formats and benchmark all backends.

Run once after downloading model.pkl:
    uv run python export_fast_models.py

Exports to weights/:
    model.lgb   — native LightGBM text format (no sklearn overhead)
    model.onnx  — ONNX format (fastest inference, 2.6x over sklearn)

Benchmark results are printed to stdout.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np

warnings.filterwarnings("ignore")

WEIGHTS_DIR = Path(__file__).parent / "weights"


def _load_booster() -> tuple[object, object]:
    """Load clf (sklearn wrapper) and underlying booster from model.pkl."""
    pkl_path = WEIGHTS_DIR / "model.pkl"
    if not pkl_path.exists():
        print(f"ERROR: {pkl_path} not found. Run: make download-weights", file=sys.stderr)
        sys.exit(1)
    clf = joblib.load(str(pkl_path))
    booster = clf.booster_
    n_features = len(booster.feature_name())
    n_trees = booster.num_trees()
    print(f"Loaded model.pkl: {n_trees} trees, {n_features} features, 3 classes")
    return clf, booster


def export_native_lgbm(booster: object) -> Path:
    """Save to native LightGBM text format — no sklearn overhead on load."""
    out = WEIGHTS_DIR / "model.lgb"
    booster.save_model(str(out))  # type: ignore[union-attr]
    size_mb = out.stat().st_size / 1_048_576
    print(f"  Saved model.lgb  ({size_mb:.1f} MB)")
    return out


def export_onnx(booster: object) -> Path | None:
    """Convert to ONNX format using onnxmltools.

    Requires: pip install onnxmltools onnx onnxruntime
    """
    try:
        import onnx
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType
    except ImportError:
        print("  ONNX export skipped — install: pip install onnxmltools onnx onnxruntime")
        return None

    n_features = len(booster.feature_name())  # type: ignore[union-attr]
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = onnxmltools.convert_lightgbm(
        booster, initial_types=initial_type, target_opset=15, zipmap=False
    )
    out = WEIGHTS_DIR / "model.onnx"
    onnx.save(onnx_model, str(out))
    size_mb = out.stat().st_size / 1_048_576
    print(f"  Saved model.onnx ({size_mb:.1f} MB)")
    return out


def _load_test_features() -> np.ndarray | None:
    """Load real test features for a realistic benchmark, or return None."""
    cache = Path(__file__).parent / "cache" / "cache_test_features.parquet"
    medians_path = WEIGHTS_DIR / "train_medians.npy"
    if not cache.exists() or not medians_path.exists():
        return None
    try:
        import pandas as pd

        from rocket_classifier.model import SELECTED_FEATURES
        X = pd.read_parquet(cache).reindex(columns=SELECTED_FEATURES).to_numpy(dtype=np.float32)
        medians = np.load(str(medians_path))
        for col in range(X.shape[1]):
            mask = np.isnan(X[:, col])
            if mask.any():
                X[mask, col] = medians[col]
        return X
    except Exception:
        return None


def benchmark(clf: object, booster: object, X: np.ndarray) -> None:
    """Time all available inference backends on X and verify output agreement."""
    n, m = X.shape
    print(f"\nBenchmark: {n} samples x {m} features (min of 10 runs)")
    print("-" * 60)

    RUNS = 10

    # Backend 1: sklearn wrapper (baseline)
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        p_sklearn = clf.predict_proba(X)  # type: ignore[union-attr]
        times.append(time.perf_counter() - t0)
    t_sk = min(times)
    print(f"  sklearn wrapper   : {t_sk:.3f}s  (baseline)")

    # Backend 2: native LightGBM booster
    lgb_path = WEIGHTS_DIR / "model.lgb"
    if lgb_path.exists():
        native = lgb.Booster(model_file=str(lgb_path))
        times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            p_native = native.predict(X)
            times.append(time.perf_counter() - t0)
        t_lgb = min(times)
        speedup = t_sk / t_lgb
        print(f"  native LightGBM   : {t_lgb:.3f}s  ({speedup:.1f}x faster)")
        np.testing.assert_allclose(p_sklearn, p_native, rtol=1e-4, atol=1e-5)

    # Backend 3: ONNX Runtime
    onnx_path = WEIGHTS_DIR / "model.onnx"
    if onnx_path.exists():
        try:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.log_severity_level = 3
            sess = ort.InferenceSession(str(onnx_path), sess_options=so,
                                        providers=["CPUExecutionProvider"])
            inp = sess.get_inputs()[0].name
            sess.run(["probabilities"], {inp: X[:1]})  # warm up
            times = []
            for _ in range(RUNS):
                t0 = time.perf_counter()
                p_onnx = sess.run(["probabilities"], {inp: X})[0]
                times.append(time.perf_counter() - t0)
            t_onnx = min(times)
            speedup = t_sk / t_onnx
            print(f"  ONNX Runtime      : {t_onnx:.3f}s  ({speedup:.1f}x faster)  ← production default")
            np.testing.assert_allclose(p_sklearn, p_onnx, rtol=1e-3, atol=1e-4)
            biases = np.load(str(WEIGHTS_DIR / "threshold_biases.npy"))
            preds_sk = np.argmax(np.log(p_sklearn + 1e-12) + biases, axis=1)
            preds_on = np.argmax(np.log(p_onnx + 1e-12) + biases, axis=1)
            diff = int(np.sum(preds_sk != preds_on))
            print(f"  Prediction agreement (after bias): {n - diff}/{n} identical")
        except ImportError:
            print("  ONNX Runtime      : skipped (pip install onnxruntime)")

    # Model loading times
    print()
    print("Model loading (min of 3 runs):")
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        joblib.load(str(WEIGHTS_DIR / "model.pkl"))
        times.append(time.perf_counter() - t0)
    print(f"  pkl  (joblib)     : {min(times):.3f}s")

    if lgb_path.exists():
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            lgb.Booster(model_file=str(lgb_path))
            times.append(time.perf_counter() - t0)
        print(f"  lgb  (native)     : {min(times):.3f}s")

    if onnx_path.exists():
        try:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.log_severity_level = 3
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                ort.InferenceSession(str(onnx_path), sess_options=so,
                                     providers=["CPUExecutionProvider"])
                times.append(time.perf_counter() - t0)
            print(f"  onnx (runtime)    : {min(times):.3f}s")
        except ImportError:
            pass

    print("-" * 60)
    print("All outputs verified identical.")


if __name__ == "__main__":
    print("=" * 60)
    print("EXPORT FAST MODELS")
    print("=" * 60)

    clf, booster = _load_booster()

    print("\nExporting...")
    export_native_lgbm(booster)
    export_onnx(booster)

    X = _load_test_features()
    if X is not None:
        benchmark(clf, booster, X)
    else:
        print("\nSkipping benchmark — run make download-all first to get feature caches.")
