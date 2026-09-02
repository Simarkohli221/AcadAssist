"""
train_faithfulness.py
──────────────────────
Trains the Logistic Regression faithfulness classifier consumed by
faithfulness.py (FaithfulnessVerifier.predict()).

WHAT this script does
──────────────────────
    1. Loads data/faithfulness/dataset.csv (produced by
       create_faithfulness_dataset.py).
    2. Extracts the 3 model features EXCLUSIVELY via the project's existing
       `faithfulness.FaithfulnessVerifier` — semantic_similarity,
       keyword_jaccard, numeric_consistency. The feature formulas
       themselves are NOT reimplemented here; this script only calls
       into faithfulness.py, so training and inference are guaranteed to
       compute features identically (PRD §4.7.2 — feature/inference parity).
    3. Splits train/test using a GROUP-aware split (sklearn
       GroupShuffleSplit) keyed on the dataset's `group_key` column, so
       that rows sharing the same underlying context chunk can never be
       split across train and test — avoiding the leakage that a plain
       random split would allow (PRD §26 — no silent correctness gaps).
    4. Trains sklearn.linear_model.LogisticRegression on the 3 features.
    5. Reports Accuracy, Precision, Recall, F1, and the confusion matrix
       on the held-out test set.
    6. Saves the trained model to models/faithfulness_lr.joblib as a dict
       of the exact shape FaithfulnessVerifier._load_model() expects:
           {"model": <sklearn estimator>, ...metadata}

Embedder note
─────────────
FaithfulnessVerifier.semantic_similarity() needs an Embedder. This script
builds ONE Embedder instance, fits it once on the full dataset's answer +
context texts (required for the LSA fallback backend; a no-op for the
SentenceTransformer backend if it's installed), and reuses that single
instance for every row — exactly the reuse pattern documented in
faithfulness.py, so feature values are computed with a consistent encoder
across the whole run.

Usage
─────
    python src/train_faithfulness.py
    python src/train_faithfulness.py --dataset data/faithfulness/dataset.csv \\
        --model-out models/faithfulness_lr.joblib --test-size 0.2 --seed 42
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from embedder import Embedder
from faithfulness import FaithfulnessVerifier, FEATURE_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("raka.train_faithfulness")

# embedder.py logs an INFO line on every single encode_texts() call, which
# is unreadable across ~1000+ row-by-row feature extractions. This only
# raises the log LEVEL for that logger in this process — it does not touch
# embedder.py itself.
logging.getLogger("embedder").setLevel(logging.WARNING)

# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "faithfulness" / "dataset.csv"
DEFAULT_MODEL_OUT = PROJECT_ROOT / "models" / "faithfulness_lr.joblib"

SEED = 42
TEST_SIZE = 0.2


# ── Feature extraction (delegates entirely to faithfulness.py) ──────────────
def _extract_feature_matrix(
    df: pd.DataFrame,
    verifier: FaithfulnessVerifier,
) -> np.ndarray:
    """
    Build the (n, 3) feature matrix by calling
    FaithfulnessVerifier.extract_features() for every row.

    Feature order is taken from FEATURE_NAMES (imported from faithfulness.py)
    — the single source of truth shared with prediction time — never
    hardcoded here.
    """
    rows: list[list[float]] = []
    t0 = time.time()
    for i, (answer, context) in enumerate(zip(df["answer"], df["context"])):
        features = verifier.extract_features(str(answer), str(context))
        rows.append(features.as_list())
        if (i + 1) % 200 == 0:
            logger.info("  extracted features for %d/%d rows …", i + 1, len(df))
    elapsed = time.time() - t0
    logger.info(
        "Feature extraction complete: %d rows in %.1fs (%.1f rows/s) — features=%s",
        len(df), elapsed, len(df) / max(elapsed, 1e-6), FEATURE_NAMES,
    )
    return np.asarray(rows, dtype=np.float64)


# ── Group-aware split ─────────────────────────────────────────────────────────
def _group_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split (X, y) into train/test such that no `group` value appears in
    both splits — prevents leakage from rows that share the same
    underlying context chunk (see create_faithfulness_dataset.py's
    `group_key` column).
    """
    from sklearn.model_selection import GroupShuffleSplit

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    logger.info(
        "Group-aware split: %d train rows (%d groups) / %d test rows (%d groups)",
        len(train_idx), len(set(groups[train_idx])),
        len(test_idx), len(set(groups[test_idx])),
    )
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ── Training ──────────────────────────────────────────────────────────────────
def _train_model(X_train: np.ndarray, y_train: np.ndarray, seed: int):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",   # robust to any residual class skew
        random_state=seed,
    )
    model.fit(X_train, y_train)
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────
def _evaluate(model, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    )

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy":  float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "f1":        float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
    }
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    metrics["confusion_matrix"] = cm.tolist()  # [[TN, FP], [FN, TP]]
    return metrics


def _print_report(metrics: dict[str, Any], n_train: int, n_test: int, coefs, intercept) -> None:
    cm = metrics["confusion_matrix"]
    tn, fp = cm[0]
    fn, tp = cm[1]

    print(f"\n{'═'*62}")
    print("  RAKA — Faithfulness Classifier: Training Report")
    print(f"{'═'*62}")
    print(f"  Train examples     : {n_train}")
    print(f"  Test examples      : {n_test}")
    print(f"\n  Accuracy           : {metrics['accuracy']:.4f}")
    print(f"  Precision (class 1): {metrics['precision']:.4f}")
    print(f"  Recall    (class 1): {metrics['recall']:.4f}")
    print(f"  F1        (class 1): {metrics['f1']:.4f}")
    print(f"\n  Confusion matrix (rows=actual, cols=predicted; labels=[0,1]):")
    print(f"                 pred 0   pred 1")
    print(f"    actual 0  |  {tn:>6}   {fp:>6}   (TN, FP)")
    print(f"    actual 1  |  {fn:>6}   {tp:>6}   (FN, TP)")
    print(f"\n  Feature weights (order = {FEATURE_NAMES}):")
    for name, coef in zip(FEATURE_NAMES, coefs[0]):
        print(f"    ├─ {name:<22}: {coef:+.4f}")
    print(f"  intercept            : {intercept[0]:+.4f}")
    print(f"{'═'*62}\n")


# ── Save ──────────────────────────────────────────────────────────────────────
def _save_model(
    model,
    model_out: Path,
    metrics: dict[str, Any],
    n_train: int,
    n_test: int,
    seed: int,
) -> None:
    import joblib

    model_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,                      # required key — read by faithfulness.py
        "feature_names": list(FEATURE_NAMES),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_size": n_train,
        "test_size": n_test,
        "seed": seed,
        "test_metrics": {k: v for k, v in metrics.items() if k != "confusion_matrix"},
        "confusion_matrix": metrics["confusion_matrix"],
    }
    joblib.dump(payload, model_out)
    logger.info("Model saved → %s", model_out)


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run(
    dataset_path: Path,
    model_out: Path,
    test_size: float = TEST_SIZE,
    seed: int = SEED,
) -> dict[str, Any]:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. "
            f"Run create_faithfulness_dataset.py first."
        )

    df = pd.read_csv(dataset_path)
    required_cols = {"answer", "context", "label", "group_key"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"dataset.csv missing required columns: {missing}")

    df = df.dropna(subset=["answer", "context", "label"]).reset_index(drop=True)
    logger.info("Loaded dataset: %d rows from %s", len(df), dataset_path)
    logger.info(
        "Class balance: %d faithful (1) / %d unfaithful (0)",
        int((df["label"] == 1).sum()), int((df["label"] == 0).sum()),
    )

    # ── Build ONE shared embedder, fit once on the full text corpus ─────────
    # (Fitting is a no-op for the SentenceTransformer backend; required for
    # the pure-sklearn LSA fallback. Reusing a single fitted instance for
    # every feature extraction call matches faithfulness.py's documented
    # "loaded once, reused" pattern and keeps embedding space consistent.)
    logger.info("Building shared Embedder for feature extraction …")
    embedder = Embedder()
    all_texts = pd.concat([df["answer"], df["context"]]).astype(str).tolist()
    embedder.fit_corpus(all_texts)
    logger.info(
        "Embedder ready: backend=%s dim=%d", embedder.backend_name, embedder.dim,
    )

    # FaithfulnessVerifier with model_path=None → feature extraction only,
    # no trained classifier needed yet (that's what we're about to produce).
    verifier = FaithfulnessVerifier(model_path=None, embedder=embedder)

    X = _extract_feature_matrix(df, verifier)
    y = df["label"].astype(int).to_numpy()
    groups = df["group_key"].astype(str).to_numpy()

    X_train, X_test, y_train, y_test = _group_train_test_split(
        X, y, groups, test_size=test_size, seed=seed,
    )

    model = _train_model(X_train, y_train, seed=seed)
    metrics = _evaluate(model, X_test, y_test)

    _print_report(metrics, len(X_train), len(X_test), model.coef_, model.intercept_)
    _save_model(model, model_out, metrics, len(X_train), len(X_test), seed)

    return {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "metrics": metrics,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the faithfulness Logistic Regression classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--model-out", default=str(DEFAULT_MODEL_OUT))
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run(
        dataset_path=Path(args.dataset),
        model_out=Path(args.model_out),
        test_size=args.test_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()