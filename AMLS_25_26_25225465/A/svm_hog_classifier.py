### A/svm_hog_classifier.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import logging
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)
from joblib import Memory
from .hog_transformer import HOGTransformer
from .augument import RandomAugmenter


@dataclass
class EvalResults:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float]
    confusion: np.ndarray


class SVMHOGClassifier:
    """Encapsulates HOG → Scaler → PCA → SVC with CV grid search and evaluation.

    Logging: pass a logger or it will create one named after the class. The
    class only logs high-level progress; detailed CV tables are accessible via
    .cv_table().
    """

    def __init__(
        self,
        random_state: int = 0,

        cv_splits: int = 5,
        n_jobs: int = -1,
        cache_dir: Optional[str] = None,
        param_grid: Optional[Dict] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.random_state = random_state
        self.cv_splits = cv_splits
        self.n_jobs = n_jobs
        self.memory = Memory(location=cache_dir or None, verbose=0)
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        # Base pipeline (HOG first so all downstream sees features)
        self.pipeline = Pipeline(
            steps=[
                ("aug", RandomAugmenter(enabled=True, random_state=random_state,
                             rot_deg=15.0, hflip_p=0.5, blur_p=0.2, noise_p=0.2, gamma_p=0.2)),
                ("hog", HOGTransformer()),
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("pca", PCA(svd_solver="full", random_state=random_state)),
                ("svc", SVC(
                        kernel="rbf",
                        probability=False,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ],
            memory=self.memory,
        )

        # defaults for 28x28 + HOG features
        if param_grid is None:
            self.param_grid = {
                "pca__n_components": [None, 32, 64, 128],
                "pca__whiten": [False, True],
                "svc__C": np.logspace(-2, 2, 9),        # 0.01 .. 100
                "svc__gamma": np.logspace(-4, -1, 8),   # 1e-4 .. 1e-1
                # Optional (expensive):
                 "hog__orientations": [8, 9],
                 "hog__pixels_per_cell": [(4, 4), (6, 6)],
                 "hog__cells_per_block": [(2, 2)],
                # Augmentation params handled in augmenter
                "aug__enabled": [False, True],               # toggles augmentation
                "aug__rot_deg": [0, 10, 15],                 # medical-friendly
                "aug__hflip_p": [0.0, 0.5],
                # still optional but expensive:
                "aug__blur_p": [0.0, 0.2],
                "aug__noise_p": [0.0, 0.2],
                "aug__gamma_p": [0.0, 0.2],
            }
            logger.info("Using default param_grid with %d combinations",
                        np.prod([len(v) for v in self.param_grid.values()]))
        else:
            self.param_grid = param_grid

        self.grid_: Optional[GridSearchCV] = None
        self.best_model_: Optional[Pipeline] = None

    # --------------------- training / CV ---------------------
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, *, refit_metric: str = "f1") -> None:
        self.logger.info(
            "Starting CV grid search (cv_splits=%d, refit=%s, n=%d) (Usualy takes about 15 minutes / Apple M3/ Single thread mpl lib)... )",
            self.cv_splits, refit_metric, len(X_train)
        )
        cv = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=self.random_state)
        grid = GridSearchCV(
            estimator=self.pipeline,
            param_grid=self.param_grid,
            scoring={
                "f1": "f1",
                "accuracy": "accuracy",
                "roc_auc": "roc_auc",
            },
            refit=refit_metric,
            cv=cv,
            n_jobs=self.n_jobs,
            verbose=0,
            return_train_score=False,
        )
        grid.fit(X_train, y_train)
        self.grid_ = grid
        self.best_model_ = grid.best_estimator_
        self.logger.info("Best params: %s", grid.best_params_)
        self.logger.info("Best CV %s: %.4f", refit_metric, grid.best_score_)

    # --------------------- inference ---------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.best_model_.predict(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.best_model_.decision_function(X)

    # --------------------- evaluation ---------------------
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> EvalResults:
        self._check_fitted()
        self.logger.info("Evaluating on held-out test (n=%d)", len(X_test))
        y_pred = self.predict(X_test)
        try:
            scores = self.decision_function(X_test)
            roc = float(roc_auc_score(y_test, scores))
        except Exception:
            roc = None
        acc = float(accuracy_score(y_test, y_pred))
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", zero_division=0
        )
        prec, rec, f1 = float(prec), float(rec), float(f1)
        conf = confusion_matrix(y_test, y_pred)
        return EvalResults(acc, prec, rec, f1, roc, conf)

    def best_params(self) -> Dict:
        self._check_fitted()
        if self.grid_ is None:
            raise RuntimeError("Grid search not complete. Need to execute .fit(...)")
        return self.grid_.best_params_

    # --------------------- internals ---------------------
    def _check_fitted(self) -> None:
        if self.best_model_ is None:
            raise RuntimeError("Model not fitted. Call .fit(...) first.")