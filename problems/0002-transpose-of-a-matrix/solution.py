def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    """
    Transpose a 2D matrix by swapping rows and columns.
    
    Args:
        a: A 2D matrix of shape (m, n)
    
    Returns:
        The transposed matrix of shape (n, m)
    """
    num_rows = len(a[0])
    at = [[] for _ in range(num_rows)]
    for row in a:
        for j, elem in enumerate(row):
            at[j].append(elem)

    return at