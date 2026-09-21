import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.
    
    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return
    
    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """
    # standardization
    means = np.mean(data, axis=0, keepdims=True)
    stds = np.std(data, axis=0, keepdims=True)
    data = np.divide(
        data - means,
        stds,
        out=np.zeros_like(data, dtype=float),
        where=(stds!=0.0)
    )

    _, _, Vh = np.linalg.svd(data)
    principle_components = Vh[:k, :].T
    n_features = principle_components.shape[0]
    for col_idx in range(k):
        for row_idx in range(n_features):
            elem = principle_components[row_idx, col_idx]
            if np.abs(elem) > 1e-10:
                if elem < 0.0:
                    principle_components[:, col_idx] *= -1
                break

    return np.round(principle_components, 4)