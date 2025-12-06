import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    average_precision_score, brier_score_loss,
    confusion_matrix, precision_score,recall_score
)
from EvalResults import EvalResults, plot_history_loss  

AUTOTUNE = tf.data.AUTOTUNE
BATCH = 64
IMG_SIZE = 224  # EfficientNetB0 default input size

def get_weights_counts(y):
    """Utility to compute class weights from labels y."""
    counts = np.bincount(y)
    total = len(y)
    class_weight = {i: (total / (2.0 * counts[i])) for i in range(len(counts))}
    return class_weight, counts

# tf.data pipelines
def make_dataset(X, y, training=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(4096, seed=42)
    ds = ds.batch(BATCH).prefetch(AUTOTUNE)
    return ds

# -------------------------
# Data augmentation layers
# -------------------------
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomZoom(0.05),
    ],
    name="data_augmentation",
)


class CNNEfficientNetB0:
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
        self.img_size = IMG_SIZE

        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test =  None, None, None, None, None, None

    def set_data(self, X_train, y_train, X_val, y_val, X_test, y_test):
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test =  X_train, y_train, X_val, y_val, X_test, y_test
        self.train_ds = make_dataset(X_train, y_train, training=True)
        self.val_ds   = make_dataset(X_val,   y_val,   training=False)
        self.test_ds  = make_dataset(X_test,  y_test,  training=False)
        self.class_weight, self.counts = get_weights_counts(y_train)

    def build_effnet_b0_breastmnist(
            self,
            img_size=IMG_SIZE,
            base_trainable=False,
            label_smoothing=0.0,
            dropout_rate=0.4,
    ):
    # Pretrained EfficientNetB0 base
        base_model = EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(img_size, img_size, 3),
            pooling="avg",
        )
        base_model.trainable = base_trainable

        # Input is 28x28x1 (grayscale)
        inputs = keras.Input(shape=(28, 28, 1), name="input_gray28")

        # 28x28x1 -> 224x224x3
        x = layers.Resizing(img_size, img_size, interpolation="bilinear", name="resize")(inputs)
        x = layers.Concatenate(axis=-1, name="gray_to_rgb")([x, x, x])  # (H,W,3)

        # Augment only in training
        x = data_augmentation(x)

        # EfficientNet preprocess: expects [0,255] then normalizes
        x = layers.Rescaling(255.0, name="to_255")(x)
        x = layers.Lambda(preprocess_input, name="effnet_preproc")(x)

        # Backbone
        x = base_model(x)

        # Classification head
        x = layers.Dropout(dropout_rate, name="dropout")(x)
        outputs = layers.Dense(1, activation="sigmoid", name="pred")(x)

        model = keras.Model(inputs, outputs, name="effnetb0_breastmnist")

        loss = keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=loss,
            metrics=[
                "accuracy",
                keras.metrics.AUC(name="auc"),
                keras.metrics.AUC(name="pr_auc", curve="PR"),
            ],
        )
        return model, base_model

    def _build_model(self):
        self.model, self.base_model = self.build_effnet_b0_breastmnist(
            img_size=self.img_size,
            base_trainable=False,
            label_smoothing=0.0,
            dropout_rate=0.4,
        )


    def report(self, split_name: str, y_true, prob, t: float) -> EvalResults:
        """
        split_name: "VAL" or "TEST" (for logging only)
        y_true: labels for this split
        prob: predicted probabilities for this split
        t: threshold to binarise prob
        """
        yhat = (prob >= t).astype(int)

        acc  = accuracy_score(y_true, yhat)
        prec = precision_score(y_true, yhat, zero_division=0)
        rec  = recall_score(y_true, yhat, zero_division=0)
        f1   = f1_score(y_true, yhat)

        try:
            auc = roc_auc_score(y_true, prob)
        except ValueError:
            auc = None  # e.g. only one class present

        cm = confusion_matrix(y_true, yhat)
        self.logger.info(f"\n[{split_name}] @ threshold={t:.3f}")
        acc, pr, rec, f1, auc = float(acc),float(prec), float(rec), float(f1), float(auc if auc is not None else float('nan'))
        return EvalResults(
            accuracy=acc,
            precision=pr,
            recall=rec,
            f1=f1,
            roc_auc=auc,
            confusion=cm,
        )

    def fit(self):
        """Train the CNN and pick best F1 threshold on the validation set."""
        self._build_model()
        callbacks_phase1 = [
            keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=4, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, verbose=1, min_lr=1e-6
        ),
        keras.callbacks.ModelCheckpoint(
        "breastmnist_effb0_phase1_best.keras",
        monitor="val_auc", mode="max", save_best_only=True,
        save_weights_only=False,
        ),
        ]
        self.history_1 = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=15,
            class_weight=self.class_weight,
            callbacks=callbacks_phase1,
            verbose=1,
        )

        # -----------------------------
        # Phase 2: fine-tune top layers
        # -----------------------------
        # Unfreeze top 30% of EfficientNetB0 layers (except BatchNorm)
        fine_tune_at = int(len(self.base_model.layers) * 0.7)
        for i, layer in enumerate(self.base_model.layers):
            if i < fine_tune_at:
                layer.trainable = False
            else:
                # often better to keep BN layers frozen
                if isinstance(layer, layers.BatchNormalization):
                    layer.trainable = False
                else:
                    layer.trainable = True

        self.logger.info(f"\nUnfreezing from layer {fine_tune_at} / {len(self.base_model.layers)}")

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss=keras.losses.BinaryCrossentropy(label_smoothing=0.03),
            metrics=[
                "accuracy",
                keras.metrics.AUC(name="auc"),
                keras.metrics.AUC(name="pr_auc", curve="PR"),
            ],
        )

        callbacks_phase2 = [
                keras.callbacks.EarlyStopping(
                    monitor="val_auc", mode="max", patience=4, restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=2, verbose=1, min_lr=1e-6
                ),
                keras.callbacks.ModelCheckpoint(
                    "breastmnist_effb0_phase2_best.keras",
                    monitor="val_auc", mode="max", save_best_only=True,
                    save_weights_only=False,
                ),
            ]

        self.logger.info("\n===== Phase 2: fine-tune top EfficientNetB0 layers =====")
        self.history_2 = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=15,
            class_weight=self.class_weight,
            callbacks=callbacks_phase2,
            verbose=1,
        )


        val_prob  = self.model.predict(self.X_val,  verbose=0).ravel() # type: ignore
        test_prob = self.model.predict(self.X_test, verbose=0).ravel() # type: ignore

        ts = np.linspace(0.05, 0.95, 181)
        val_f1s = [f1_score(self.y_val, (val_prob >= t).astype(int)) for t in ts]
        t_best = float(ts[int(np.argmax(val_f1s))])

        self.logger.info(f"\nBest threshold on VAL (by F1): {t_best:.3f}")
        self.report("VAL",  self.y_val,  val_prob,  t_best)
        self.report("TEST", self.y_test, test_prob, t_best)

    def display_training_history(self):
        plot_history_loss(self.history_2, width=80, height=20, logger=self.logger)    

