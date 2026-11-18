import numpy as np

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Calculate the Expected Calibration Error (ECE).
    
    Args:
        y_true: List or array of true binary labels (0 or 1)
        y_prob: List or array of predicted probabilities for the positive class
        n_bins: Number of bins for grouping predictions (default: 10)
    
    Returns:
        float: ECE value rounded to 3 decimal places
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bin_edges = np.linspace(0.0, 1.0, num = n_bins + 1)

    bin_indices = np.searchsorted(
        bin_edges[1:-1],
        y_prob,
        side="left"
    )

    bin_errors = np.bincount(
        bin_indices,
        weights=y_true - y_prob,
        minlength=n_bins
    )

    ece = np.sum(np.abs(bin_errors)) / len(y_true)
    return float(np.round(ece, 3))