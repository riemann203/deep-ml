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
    bin_indices = np.digitize(y_prob, bin_edges, right=True)
    bin_indices = np.clip(bin_indices, 1, n_bins)

    ece = 0.
    for i in range(1, n_bins + 1):
        mask = (bin_indices == i)
        if np.all(~mask):
            continue
        acc = np.mean(y_true[mask])
        confidence = np.mean(y_prob[mask])
        weight = np.sum(mask) / len(y_true)
        ece += weight * abs(acc - confidence)
    return float(np.round(ece, 3))