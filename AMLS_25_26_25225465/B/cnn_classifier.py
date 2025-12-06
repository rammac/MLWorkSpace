import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    average_precision_score, brier_score_loss,
    confusion_matrix, precision_score,recall_score
)
from EvalResults import EvalResults, plot_history_loss

# Reuse your existing data_augmentation (or define it here)
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomTranslation(0.03, 0.03),
    ],
    name="data_augmentation",
)


class CNNClassifier:
    """
    Model B: small CNN for BreastMNIST, wrapped similarly to SVMHOGClassifier.

    Interface:
        - fit(X_train, y_train, X_val, y_val, class_weight=None)
        - predict_proba(X)
        - predict(X, threshold=None)
        - evaluate(X, y, threshold=None) -> dict of metrics
    """

    def __init__(
        self,
        input_shape=(28, 28, 1),
        batch_size=64,
        epochs=10,
        patience_es=3,
        patience_lr=2,
        lr=1e-3,
        seed=42,
        logger=None,
    ):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience_es = patience_es
        self.patience_lr = patience_lr
        self.lr = lr
        self.seed = seed
        self.logger=logger

        # will be set after fit()
        self.model = None
        self.history_ = None
        self.best_threshold_ = 0.5  # default if you call predict() before tuning

        self._build_model()

    def _build_model(self):
        # core architecture = your model_1
        inputs = keras.Input(shape=self.input_shape, name="input_gray28")
        x = data_augmentation(inputs)
        x = layers.Conv2D(25, kernel_size=3, activation="relu", padding="valid", name="conv1")(x)
        x = layers.MaxPool2D(pool_size=1, name="pool1")(x)  # effectively a no-op
        x = layers.Flatten(name="flatten")(x)
        x = layers.Dense(100, activation="relu", name="fc1")(x)
        outputs = layers.Dense(1, activation="sigmoid", name="pred")(x)

        self.model = keras.Model(inputs, outputs, name="modelB_small_cnn")

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.lr),
            loss="binary_crossentropy",
            metrics=["accuracy", keras.metrics.AUC(name="auc")],
        )

    # -------------------
    # Core API
    # -------------------

    def fit(self, X_train, y_train, X_val, y_val, class_weight=None, verbose=1):
        """Train the CNN and pick best F1 threshold on the validation set."""
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=self.patience_es,
                restore_best_weights=True,
                monitor="val_loss",
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=self.patience_lr,
                verbose=verbose,
                monitor="val_loss",
                #min_lr=1e-6,
            ),
        ]

        self.history_ = self.model.fit( # type: ignore
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=verbose,
            class_weight=class_weight,
        )

        # ---- choose threshold on VAL by F1 (non-degenerate) ----
        val_prob = self.predict_proba(X_val)
        ts = np.linspace(0.05, 0.95, 181)

        best_f1 = -1.0
        best_t = 0.5

        for t in ts:
            yhat = (val_prob >= t).astype(int)
            # skip degenerate thresholds that predict only one class
            if yhat.min() == yhat.max():
                continue
            f1 = f1_score(y_val, yhat)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)

        self.best_threshold_ = best_t
        return self

    def predict_proba(self, X, verbose=0):
        """Return probabilities P(y=1|x)."""
        prob = self.model.predict(X, verbose=verbose).ravel() # type: ignore
        return prob

    def predict(self, X, threshold=None, verbose=0):
        """Class labels using given threshold (or best_threshold_ if None)."""
        if threshold is None:
            threshold = self.best_threshold_
        prob = self.predict_proba(X, verbose=verbose)
        return (prob >= threshold).astype(int)

    def display_training_history(self):
        if self.history_ is None:
            raise ValueError("No training history found.Need to call fit() first.")

        plot_history_loss(self.history_, logger=self.logger)
    

    def evaluate(self, X, y, threshold=None, verbose=0, split_name="TEST"):
        if threshold is None:
            threshold = self.best_threshold_
        prob = self.predict_proba(X, verbose=verbose)
        yhat = (prob >= threshold).astype(int)

        acc  = accuracy_score(y, yhat)
        prec = precision_score(y, yhat, zero_division=0)
        rec  = recall_score(y, yhat, zero_division=0)
        f1   = f1_score(y, yhat)

        try:
            auc = roc_auc_score(y, prob)
        except ValueError:
            auc = None  # e.g. only one class present in y

        cm = confusion_matrix(y, yhat)

        acc, prec, rec, f1, auc = float(acc),float(prec), float(rec), float(f1), float(auc if auc is not None else float('nan'))
        return EvalResults(
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1=f1,
            roc_auc=auc,
            confusion=cm,
        )
    