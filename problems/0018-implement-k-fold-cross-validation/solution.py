import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds (default 5)
        shuffle: Whether to shuffle indices before splitting (default True)
    
    Returns:
        List of (train_indices, test_indices) tuples
    """
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)

    quotient = n_samples // k
    remainder = n_samples % k
    fold_sizes = [quotient for _ in range(k)]
    if remainder != 0:
        for i in range(remainder):
            fold_sizes[i] += 1

    fold_indices = np.cumsum([0] + fold_sizes)

    output = []
    for idx1, idx2 in zip(fold_indices, fold_indices[1:]):
        output.append(
            (
                np.concatenate([indices[0:idx1], indices[idx2:n_samples]]).tolist(),
                indices[idx1:idx2].tolist()
            )
        )
    return output