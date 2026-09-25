def min_max(x: list[float]) -> list[float]:
    """
    Perform Min-Max normalization to scale values to [0, 1].
    
    Args:
        x: A list of numerical values
    
    Returns:
        A new list with values normalized to [0, 1]
    """
    min_val = min(x)
    max_val = max(x)
    return [
        (elem - min_val) / (max_val - min_val) for elem in x
    ]
