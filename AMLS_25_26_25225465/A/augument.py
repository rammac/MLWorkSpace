# A/augmenter.py
from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from skimage.transform import rotate
from skimage.filters import gaussian
from skimage.util import random_noise
from skimage.exposure import adjust_gamma

class RandomAugmenter(BaseEstimator, TransformerMixin):
    """Apply light medical-friendly augmentations at fit-time only.
    Keeps sample count the same /one random augmentation per sample"""
    def __init__(self,
                 enabled: bool = True,
                 rot_deg: float = 15.0,
                 hflip_p: float = 0.5,
                 blur_p: float = 0.2,
                 noise_p: float = 0.2,
                 gamma_p: float = 0.2,
                 random_state: int | None = 0):
        self._fit_rng = None
        self.enabled = enabled
        self.rot_deg = rot_deg
        self.hflip_p = hflip_p
        self.blur_p  = blur_p
        self.noise_p = noise_p
        self.gamma_p = gamma_p
        self.random_state = random_state

    def _rng(self):
        return np.random.RandomState(self.random_state) if self.random_state is not None else np.random

    def _one(self, img: np.ndarray, rng) -> np.ndarray:
        # img shape: (H,W,1) or (H,W); values assumed in [0,1]
        a = img[..., 0] if img.ndim == 3 else img

        # horizontal flip
        if self.hflip_p > 0 and rng.rand() < self.hflip_p:
            a = np.fliplr(a)

        # small rotation (edge fill)
        if self.rot_deg > 0:
            ang = rng.uniform(-self.rot_deg, self.rot_deg)
            a = rotate(a, ang, mode="edge", preserve_range=True)

        # optional gaussian blur
        if self.blur_p > 0 and rng.rand() < self.blur_p:
            sig = rng.uniform(0.5, 1.0)
            a = gaussian(a, sigma=sig, preserve_range=True)

        # optional gaussian noise
        if self.noise_p > 0 and rng.rand() < self.noise_p:
            var = rng.uniform(0.001, 0.01)
            a = random_noise(a, mode="gaussian", var=var, clip=True)

        # brightness/contrast via gamma jitter (≈ brightness)
        if self.gamma_p > 0 and rng.rand() < self.gamma_p:
            gamma = rng.uniform(0.9, 1.1)
            a = adjust_gamma(a, gamma=gamma)

        a = np.clip(a, 0.0, 1.0).astype(np.float32)
        return a[..., None]  # back to (H,W,1)

    # --- sklearn API ---
    def fit(self, X, y=None):
        self._fit_rng = self._rng()
        return self

    def fit_transform(self, X:np.ndarray, y: np.ndarray | None = None, **fit_params) -> np.ndarray:
        self.fit(X, y)
        if not self.enabled:
            return X
        rng = self._fit_rng
        X = X.astype(np.float32)
        if X.max() > 1:  # just in case
            X = X / 255.0
        out = np.empty_like(X, dtype=np.float32)
        for i in range(len(X)):
            out[i] = self._one(X[i], rng)
        return out

    def transform(self, X):
        # identity on val/test (no augmentation!)
        return X
