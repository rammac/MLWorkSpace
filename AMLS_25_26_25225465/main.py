### main.py

from __future__ import annotations
import logging
import os
from pathlib import Path
import numpy as np

from A import SVMHOGClassifier
from A import run_data_analysis, remove_row_from_dataset, BREASTMNIST_SVM_HOG_PARAMS


# --------------------- logging ---------------------

def configure_logging(level: int = logging.INFO) -> logging.Logger:
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
    return logging.getLogger("ModelA")


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


# --------------------- training / eval script ---------------------

def run():
    logger = configure_logging()
    logger.info("Loading BreastMNIST from official splits (Ensure ROOT_DIR (Datasets/breastmnist.npz)exists)...")
    splits = load_breastmnist()
    Xtr, ytr = splits["train"]
    Xva, yva = splits["val"]
    Xte, yte = splits["test"]
    labels_dict = splits["labels"]
    logger.info("Loaded: train=%d, val=%d, test=%d, labels=%d", len(Xtr), len(Xva), len(Xte), len(labels_dict))

    data_processing_hints = run_data_analysis(splits, logger=logger)
    if 'train_duplicates_indices' in data_processing_hints:
        train_dup_indices = data_processing_hints['train_duplicates_indices']
        logger.info(f"Removing {len(train_dup_indices)} duplicate samples from training set...")
        Xtr = np.asarray(remove_row_from_dataset(Xtr, train_dup_indices))
        ytr = np.asarray(remove_row_from_dataset(ytr, train_dup_indices))
        logger.info(f"New training set size: {len(Xtr)}")

    # Combine train+val for CV; keep test untouched
    Xtrva = np.concatenate([Xtr, Xva], axis=0)
    ytrva = np.concatenate([ytr, yva], axis=0)
    logger.info("CV dataset size (train+val): %d", len(Xtrva))

    logger.info("Training SVM + HOG classifier with BestHyperparameters/ set param_grid=None to select again")
    model = SVMHOGClassifier(random_state=0, cv_splits=5, n_jobs=-1, param_grid=BREASTMNIST_SVM_HOG_PARAMS, logger=logger)
    model.fit(Xtrva, ytrva, refit_metric="f1")

    best = model.best_params()
    logger.info("Selected hyperparameters: %s", best)

    results = model.evaluate(Xte, yte)
    logger.info(
        "Test metrics — Acc %.4f | Prec %.4f | Rec %.4f | F1 %.4f | ROC-AUC %s",
        results.accuracy,
        results.precision,
        results.recall,
        results.f1,
        f"{results.roc_auc:.4f}" if results.roc_auc is not None else "n/a"
    )
    logger.info("Confusion matrix:%s", results.confusion)


if __name__ == "__main__":
    run()