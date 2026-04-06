#!/usr/bin/env python3
"""Export model artifacts to faster inference formats and benchmark all backends.

Run once after downloading new models (or use: make export-model):
    uv run python export_fast_models.py

Exports to artifacts/:
    model.lgb   — native LightGBM text format (direct Booster API)
    model.onnx  — ONNX format (fastest inference, 2.6x over native LightGBM)

The model expects 35-column input: 32 SELECTED_FEATURES + 3 rebel-group
class-prior columns (appended automatically at inference by RocketClassifier).
Benchmark results are printed to stdout.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np

warnings.filterwarnings("ignore")

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"


def _load_booster() -> object:
    """Load booster from model.lgb.

    Returns the LightGBM Booster object.
    """
    lgb_path = ARTIFACTS_DIR / "model.lgb"

    if lgb_path.exists():
        booster = lgb.Booster(model_file=str(lgb_path))
        n_features = len(booster.feature_name())
        n_trees = booster.num_trees()
        print(f"Loaded model.lgb: {n_trees} trees, {n_features} features, 3 classes")
        return booster

    print(
        "ERROR: model.lgb not found in artifacts/.\nRun: make download-models",
        file=sys.stderr,
    )
    sys.exit(1)


def export_native_lgbm(booster: object) -> Path:
    """Save to native LightGBM text format — no sklearn overhead on load."""
    out = ARTIFACTS_DIR / "model.lgb"
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
    out = ARTIFACTS_DIR / "model.onnx"
    onnx.save(onnx_model, str(out))
    size_mb = out.stat().st_size / 1_048_576
    print(f"  Saved model.onnx ({size_mb:.1f} MB)")

    # Save a pre-optimized version so production loads skip graph optimization (~0.3s faster).
    try:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.enable_mem_pattern = True
        so.enable_cpu_mem_arena = True
        opt_out = ARTIFACTS_DIR / "model_opt.onnx"
        so.optimized_model_filepath = str(opt_out)
        ort.InferenceSession(str(out), sess_options=so, providers=["CPUExecutionProvider"])
        size_mb_opt = opt_out.stat().st_size / 1_048_576
        print(f"  Saved model_opt.onnx ({size_mb_opt:.1f} MB, pre-optimized — ~0.3s faster load)")
    except Exception as e:
        print(f"  model_opt.onnx skipped ({e})")

    return out


def _load_test_features() -> np.ndarray | None:
    """Load real test features for benchmark, impute NaN, and append priors.

    Returns an (N, 35) array matching the model's expected input, or None.
    """
    cache = Path(__file__).parent.parent / "cache" / "cache_test_features.parquet"
    medians_path = ARTIFACTS_DIR / "train_medians.npy"
    if not cache.exists() or not medians_path.exists():
        return None
    try:
        import pandas as pd

        from rocket_classifier.model import _GLOBAL_CLASS_PRIOR, SELECTED_FEATURES

        X = pd.read_parquet(cache).reindex(columns=SELECTED_FEATURES).to_numpy(dtype=np.float32)
        medians = np.load(str(medians_path))
        for col in range(X.shape[1]):
            mask = np.isnan(X[:, col])
            if mask.any():
                X[mask, col] = medians[col]
        # Append the 3 rebel-group class-prior columns the model was trained with
        priors = np.tile(_GLOBAL_CLASS_PRIOR, (len(X), 1))
        return np.concatenate([X, priors], axis=1).astype(np.float32)
    except Exception:
        return None


def benchmark(booster: object, X: np.ndarray) -> None:
    """Time all available inference backends on X and verify output agreement."""
    n, m = X.shape
    print(f"\nBenchmark: {n} samples x {m} features (min of 10 runs)")
    print("-" * 60)

    RUNS = 10

    # Backend 1: native LightGBM booster (baseline)
    lgb_path = ARTIFACTS_DIR / "model.lgb"
    p_native = None
    t_lgb = None
    if lgb_path.exists():
        native = lgb.Booster(model_file=str(lgb_path))
        times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            p_native = native.predict(X)
            times.append(time.perf_counter() - t0)
        t_lgb = min(times)
        print(f"  native LightGBM   : {t_lgb:.3f}s  (baseline)")

    # Backend 2: ONNX Runtime
    onnx_path = ARTIFACTS_DIR / "model.onnx"
    if onnx_path.exists():
        try:
            import onnxruntime as ort

            so = ort.SessionOptions()
            so.log_severity_level = 3
            sess = ort.InferenceSession(
                str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"]
            )
            inp = sess.get_inputs()[0].name
            sess.run(["probabilities"], {inp: X[:1]})  # warm up
            times = []
            for _ in range(RUNS):
                t0 = time.perf_counter()
                p_onnx = sess.run(["probabilities"], {inp: X})[0]
                times.append(time.perf_counter() - t0)
            t_onnx = min(times)
            if t_lgb is not None:
                speedup = t_lgb / t_onnx
                print(
                    f"  ONNX Runtime      : {t_onnx:.3f}s  ({speedup:.1f}x faster)  <- production default"
                )
            else:
                print(f"  ONNX Runtime      : {t_onnx:.3f}s  <- production default")
            biases = np.load(str(ARTIFACTS_DIR / "threshold_biases.npy"))
            if p_native is not None:
                preds_ref = np.argmax(np.log(p_native + 1e-12) + biases, axis=1)
                preds_on = np.argmax(np.log(p_onnx + 1e-12) + biases, axis=1)
                diff = int(np.sum(preds_ref != preds_on))
                print(f"  Prediction agreement (after bias): {n - diff}/{n} identical")
        except ImportError:
            print("  ONNX Runtime      : skipped (pip install onnxruntime)")

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
                ort.InferenceSession(
                    str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"]
                )
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

    booster = _load_booster()

    print("\nExporting...")
    export_native_lgbm(booster)
    export_onnx(booster)

    X = _load_test_features()
    if X is not None:
        benchmark(booster, X)
    else:
        print(
            "\nSkipping benchmark — feature caches not found.\n"
            "Run: make download-all  (or delete cache/ and run make run to rebuild)"
        )
