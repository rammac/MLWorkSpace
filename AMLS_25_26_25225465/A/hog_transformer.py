### A/hog_transformer.py

from __future__ import annotations
from typing import Iterable, Tuple, Optional
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from skimage.feature import hog
    from skimage.color import rgb2gray
except Exception as e:
    raise ImportError(
        "scikit-image is required. Install with `pip install scikit-image`."
    ) from e


class HOGTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that converts images into HOG feature vectors.

    Accepts X in shape (N, H, W) or (N, H, W, C). If C==1, the channel is squeezed.
    If C==3, the image is converted to grayscale.

    Parameters mirror skimage.feature.hog. Keep them modest for 28x28 inputs.
    """

    def __init__(
        self,
        orientations: int = 9,
        pixels_per_cell: Tuple[int, int] = (4, 4),
        cells_per_block: Tuple[int, int] = (2, 2),
        block_norm: str = "L2-Hys",
        transform_sqrt: bool = True,
        feature_vector: bool = True,
        # When True, inputs are assumed in [0,1]; if False we rescale if max>1
        assume_scaled: bool = True,
    ) -> None:
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
        self.transform_sqrt = transform_sqrt
        self.feature_vector = feature_vector
        self.assume_scaled = assume_scaled

    # No fitting needed for a pure feature transform
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self._check_X_shape(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._prepare_X(X)
        # Compute HOG per-sample
        feats = [
            hog(
                img,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                block_norm=self.block_norm,
                visualize=False,
                transform_sqrt=self.transform_sqrt,
                feature_vector=self.feature_vector,
                channel_axis=None,
            )
            for img in X
        ]
        return np.asarray(feats, dtype=np.float32)

    # --------------------- helpers ---------------------
    def _check_X_shape(self, X: np.ndarray) -> None:
        if not isinstance(X, np.ndarray):
            raise TypeError("HOGTransformer expects a numpy.ndarray.")
        if X.ndim not in (3, 4):
            raise ValueError(
                f"Expected X with 3 or 4 dims (N,H,W[,C]), got shape {X.shape}."
            )

    def _prepare_X(self, X: np.ndarray) -> np.ndarray:
        self._check_X_shape(X)
        Xp = X
        # Ensure float in [0,1]
        Xp = Xp.astype("float32", copy=False)
        if not self.assume_scaled and Xp.max() > 1.0:
            Xp = Xp / 255.0

        # Squeeze channel if present
        if Xp.ndim == 4:
            if Xp.shape[-1] == 1:
                Xp = Xp[..., 0]
            elif Xp.shape[-1] == 3:
                # Convert each sample to grayscale
                Xp = np.stack([rgb2gray(x) for x in Xp], axis=0).astype("float32")
            else:
                raise ValueError(
                    f"Unsupported channel count {Xp.shape[-1]}; expected 1 or 3."
                )
        return Xp
    
    
# End of A/hog_transformer.py