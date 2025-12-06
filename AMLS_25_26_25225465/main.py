### main.py

from __future__ import annotations
import logging
import os
from pathlib import Path
import argparse
from pprint import pformat
import numpy as np
from dataclasses import dataclass
from typing import Any, cast, Optional

from A import SVMHOGClassifier
from A import run_data_analysis, remove_row_from_dataset, BREASTMNIST_SVM_HOG_PARAMS, display_results

from B import CNNClassifier


# --------------------- logging ---------------------

def configure_logging(level: int = logging.INFO, Model_name="Model-X") -> logging.Logger:
    logger = logging.getLogger()
    # Avoid duplicate handlers when running in notebooks or reloads
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logging.getLogger(Model_name)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",
                   choices=["grid", "learned"],
                   default="learned",
                   help="grid = run GridSearchCV (current flow); direct = set fixed params and fit once")
    return p.parse_args()

# --------------------- data loading ---------------------

def load_breastmnist(root_dir: str = "Datasets"):
    """Try MedMNIST loader first; fallback to generic npz/CSV if medmnist is absent."""
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    # Try MedMNIST (official splits)
    try:
        from medmnist import BreastMNIST

        def collect_labels(ds_z):
            labels_dict = {}
            try:
                from medmnist.info import INFO
                labels_dict = INFO["breastmnist"]["label"]
            except Exception:
                return {}
            return labels_dict

        def to_np(ds):
            X = np.empty((len(ds), 28, 28, 1), dtype=np.float32)
            y = np.empty((len(ds),), dtype=np.int64)
            for i in range(len(ds)):
                img, lab = ds[i]
                arr = np.asarray(img, dtype=np.float32)
                if arr.ndim == 2:
                    arr = arr[..., None]
                if arr.max() > 1:
                    arr = arr / 255.0
                y[i] = int(np.array(lab).squeeze())
                X[i] = arr
            return X, y

        train_ds = BreastMNIST(split="train", root=str(root), download=False)
        val_ds   = BreastMNIST(split="val",   root=str(root), download=False)
        test_ds  = BreastMNIST(split="test",  root=str(root), download=False)
        return {"train": to_np(train_ds), "val": to_np(val_ds), "test": to_np(test_ds), 
                "labels": collect_labels(train_ds + val_ds + test_ds)}
    except Exception:
        raise RuntimeError("MedMNIST not available; please install it via 'pip install medmnist'.")


# --------------------- load data ----------------------------------
def load_data():
    print("Loading BreastMNIST from official splits (Ensure ROOT_DIR (Datasets/breastmnist.npz)exists)...")
    splits = load_breastmnist()
    Xtr, ytr = splits["train"]
    Xva, yva = splits["val"]
    Xte, yte = splits["test"]
    labels_dict = splits["labels"]
    return Xtr, ytr, Xva, yva, Xte, yte, labels_dict

# --------------------- training / eval script ---------------------

def run_model_A(Xtr, ytr, Xva, yva, Xte, yte, labels_dict):
    args = parse_args()
    logger_a = configure_logging(Model_name="Model-A")
    #splits = load_breastmnist()
    #Xtr, ytr = splits["train"]
    #Xva, yva = splits["val"]
    #Xte, yte = splits["test"]
    #labels_dict = splits["labels"]
    logger_a.info("Loaded: train=%d, val=%d, test=%d, labels=%d", len(Xtr), len(Xva), len(Xte), len(labels_dict))

    data_processing_hints = run_data_analysis(Xtr, ytr, Xva, yva, Xte, yte, labels_dict, logger=logger_a)
    if 'train_duplicates_indices' in data_processing_hints:
        train_dup_indices = data_processing_hints['train_duplicates_indices']
        logger_a.info(f"Removing {len(train_dup_indices)} duplicate samples from training set...")
        Xtr = np.asarray(remove_row_from_dataset(Xtr, train_dup_indices))
        ytr = np.asarray(remove_row_from_dataset(ytr, train_dup_indices))
        logger_a.info(f"New training set size: {len(Xtr)}")

    # Combine train+val for CV; keep test untouched
    Xtrva = np.concatenate([Xtr, Xva], axis=0)
    ytrva = np.concatenate([ytr, yva], axis=0)
    logger_a.info("CV dataset size (train+val): %d", len(Xtrva))

    # --- branch: grid vs direct ---
    mode = getattr(args, "mode", "learned")  # default to "learned" if arg not present
    if mode == "grid":
        logger_a.info("Training SVM+HOG via GridSearchCV…")
        model = SVMHOGClassifier(
            random_state=0, cv_splits=5, n_jobs=-1,
            param_grid=None,
            logger=logger_a
        )
        model.fit(Xtrva, ytrva, refit_metric="f1")
        best = model.best_params()
        logger_a.info("Selected hyperparameters: %s", best)

    else:  # mode == "learned"
        logger_a.info("Training SVM+HOG in DIRECT mode with fixed params learnt earlier by grid search…")
        fixed = {k: (v[0] if isinstance(v, (list, tuple, np.ndarray)) else v)
                 for k, v in BREASTMNIST_SVM_HOG_PARAMS.items()}

        model = SVMHOGClassifier(
            random_state=0, cv_splits=5, n_jobs=-1,
            param_grid=BREASTMNIST_SVM_HOG_PARAMS,  # ensure no grid search path is taken
            logger=logger_a
        )
        # set and fit the pipeline once on train+val
        model.pipeline.set_params(**fixed)
        model.fit(Xtrva, ytrva, refit_metric="f1")
        # keep attributes for consistency with the grid path
        try:
            # Cast to Any so static type checkers (pyright/mypy) allow setting these attributes dynamically.
            cast(Any, model).best_params_ = dict(fixed)
            cast(Any, model).best_estimator_ = model.pipeline
        except Exception:
            pass
        best = fixed
        logger_a.info("Used fixed hyperparameters: %s", best)

    # Evaluate on test (unchanged)
    results = model.evaluate(Xte, yte)
    display_results(results, logger=logger_a)
    


def run_model_B(Xtr, ytr, Xva, yva, Xte, yte, labels_dict):
    # CNN model-B
    # Instantiate Model B (small CNN)
    # class weights (keep)
    logger_b = configure_logging(Model_name="Model-B")
    counts = np.bincount(ytr)
    class_weight = {i: (len(ytr)/(2.0*counts[i])) for i in range(len(counts))}

    modelB = CNNClassifier(
        input_shape=(28, 28, 1),
        batch_size=64,
        epochs=10,
        patience_es=3,
        patience_lr=2,
        lr=1e-3,
        logger=logger_b,
    )

    # Train, using val set for early stopping + threshold selection
    modelB.fit(Xtr, ytr, Xva, yva, class_weight=class_weight, verbose=1)

    print(f"Best threshold (VAL F1) for Model B: {modelB.best_threshold_:.3f}")

    # Evaluate on VAL and TEST in the same style as SVMHOGClassifier
    metrics_val_B = modelB.evaluate(Xva, yva, split_name="VAL")
    metrics_test_B = modelB.evaluate(Xte, yte, split_name="TEST")
    display_results(metrics_test_B, logger=logger_b)
    modelB.display_training_history()

if __name__ == "__main__":
    print("Starding main.py...")
    Xtr, ytr, Xva, yva, Xte, yte, labels_dict = load_data()
    run_model_A(Xtr, ytr, Xva, yva, Xte, yte, labels_dict)
    run_model_B(Xtr, ytr, Xva, yva, Xte, yte, labels_dict)
