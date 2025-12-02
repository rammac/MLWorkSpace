import numpy as np
import pandas as pd
import hashlib
from tabulate import tabulate
from typing import List

from .svm_hog_classifier import SVMHOGClassifier, EvalResults
from .hog_transformer import HOGTransformer
from .augument import RandomAugmenter
__all__ = ["SVMHOGClassifier", "EvalResults", "HOGTransformer", "RandomAugmenter"]

import torch
from torch.utils.data import Subset
from torch.utils.data import Dataset # Import for type hinting

def remove_row_from_dataset(dataset: Dataset, index_to_remove: List[int]) -> Dataset:
    original_length = len(dataset)
    if index_to_remove and not (0 <= index_to_remove[0] < original_length):
        raise IndexError(
            f"Index {index_to_remove} is out of bounds for dataset of size {original_length}."
        )
    all_indices = range(original_length)
    new_indices = [i for i in all_indices if i not in index_to_remove]
    new_dataset = Subset(dataset, new_indices)
    return new_dataset


def ds_to_df(ds_x, ds_y, labels_dict, split_name, max_rows=None):
    rows = []
    n = len(ds_x) if max_rows is None else min(max_rows, len(ds_y))
    for i in range(n):
        img = ds_x[i]  
        x = np.array(img)  
        y =  ds_y[i]# int(label if np.isscalar(label) else label[0]) # label is array of size 1 hence take first element 
        # scale-agnostic stats (works whether pixels are 0-255 or 0-1)
        x_float = x.astype(np.float32)
        rows.append({
            "split": split_name,
            "idx": i,
            "label": y,
            "label_name": labels_dict[str(y)] if labels_dict else str(y), # LABELS.get(y, str(y)),
            "mean": float(x_float.mean()),
            "std": float(x_float.std()),
            "min": float(x_float.min()),
            "max": float(x_float.max()),
            "zero_frac": float((x_float == 0).mean()),
            "hash": hashlib.md5(x.tobytes()).hexdigest(),  # duplicate check
        })
    return pd.DataFrame(rows)

def run_data_analysis(splits, logger = None):
    data_post_processing_hints = {}
    Xtr, ytr = splits["train"]
    Xva, yva = splits["val"]
    Xte, yte = splits["test"]
    labels_dict = splits["labels"]

    train_df = ds_to_df(Xtr, ytr, labels_dict, "train")
    val_df   = ds_to_df(Xva, yva, labels_dict, "val")
    test_df  = ds_to_df(Xte, yte, labels_dict, "test")

    df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Summary stats
    if logger:
        l_df = df.head()
        table_string = tabulate(l_df, headers="keys", tablefmt="psql", showindex=False)
        logger.info(f"\n{table_string}")

    # image mean intensity stats
    image_intensity = (df
    .groupby(["split","label_name"])
    .agg(n=("label","size"),
         mean_of_means=("mean","mean"),
         mean_std=("std","mean"),
         min_pixel=("min","min"),
         max_pixel=("max","max"),
         zero_frac_mean=("zero_frac","mean"))
    .round(3))
    if logger:
        table_string = tabulate(image_intensity, headers="keys", tablefmt="psql", showindex=True)
        logger.info(f"Image mean intensity stats by split and label:\n{table_string}")

    # Check for duplicates across splits
    hash_counts = df["hash"].value_counts()
    dup_hashes = hash_counts[hash_counts > 1].index.tolist()
    if dup_hashes:
        dup_rows = df[df["hash"].isin(dup_hashes)]
        splits_with_dup = dup_rows["split"].unique().tolist()
        if logger:
            logger.warning(f"Found duplicate images across splits {splits_with_dup} with same hash {dup_hashes}")
            hash_row_string = tabulate(dup_rows, headers="keys", tablefmt="psql", showindex=False)
            logger.warning(f"\n{hash_row_string}")
            train_dup_indices = dup_rows[dup_rows['split'] == 'train'].index.tolist()
            logger.warning(f"Option to remove duplicates from training set, using indices: {train_dup_indices}")
            data_post_processing_hints['train_duplicates_indices'] = train_dup_indices
    return  data_post_processing_hints

# Pre-Learned Hyperparameters for SVM-HOG on BreastMNIST
BREASTMNIST_SVM_HOG_PARAMS = {
    "pca__n_components": [64],
    "pca__whiten": [False],
    "svc__C": [100],
    "svc__gamma": [0.0019306977288832496],   
    "hog__orientations": [8],
    "hog__pixels_per_cell": [(4, 4)],
    "hog__cells_per_block": [(2, 2)],
    "aug__enabled": [True],               # toggles augmentation
    "aug__rot_deg": [0],                 # medical-friendly
    "aug__hflip_p": [0.0],
    "aug__blur_p": [0.2],
    "aug__noise_p": [0.2],
    "aug__gamma_p": [0.2],
}   