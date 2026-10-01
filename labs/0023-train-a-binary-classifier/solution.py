import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y_val, p_val) -> float:
    return -float(np.mean(y_val * np.log(p_val) + (1.0 - y_val) * np.log(1.0 - p_val)))


def train(X_train, y_train, X_val, y_val):
    """
    Train a binary classifier.
    
    Args:
        X_train: numpy array of shape (n_samples, n_features) -- standardized features
        y_train: numpy array of shape (n_samples,) -- binary labels (0 or 1)
        X_val:   numpy array of shape (n_val, 30) -- standardized
        y_val:   numpy array of shape (n_val,) -- validation labels
    
    Returns:
        predict: callable that takes X (n, 30) and returns y_pred (n,) of 0s and 1s
    """
    num_epochs = 1000
    lr = 0.001
    n_features = X_train.shape[1]
    W = np.zeros(n_features)
    b = 0.0
    best_W = W.copy()
    best_b = b
    best_val_loss = np.inf

    for _ in range(num_epochs):
        p = sigmoid(X_train @ W + b)

        grad_W = X_train.T @ (p - y_train) / len(y_train)
        grad_b = np.mean(p - y_train)

        W -= lr * grad_W
        b -= lr * grad_b

        p_val = sigmoid(X_val @ W + b)
        val_loss = binary_cross_entropy(y_val, p_val)

        if val_loss < best_val_loss:
            best_W = W.copy()
            best_b = float(b)
            best_val_loss = val_loss

    return (
        lambda X_test: np.where(
            sigmoid(X_test @ best_W + best_b) > 0.5,
            1,
            0
        )
    )

