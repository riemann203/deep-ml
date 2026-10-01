import numpy as np

def train(X, y, W, b):
    """
    Train linear regression weights on standardized data.
    
    Args:
        X: numpy array of shape (n_samples, n_features) -- standardized features
        y: numpy array of shape (n_samples,) -- standardized targets
        W: numpy array of shape (n_features,) -- initial random weights
        b: float -- initial bias (0.0)
    
    Returns:
        W: numpy array of shape (n_features,) -- trained weights
        b: float -- trained bias
    """
    n_samples = X.shape[0]
    ones = np.ones(shape=(n_samples, 1))
    design_matrix = np.hstack((ones, X))
    beta = np.linalg.pinv(design_matrix) @ y
    b = beta[0]
    W = beta[1:]
    return W, b