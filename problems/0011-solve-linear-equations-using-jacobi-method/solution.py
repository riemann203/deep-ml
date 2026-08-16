import numpy as np
def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    num_rows = A.shape[0]
    diagonal = np.diag(A)
    x = np.zeros(num_rows)

    for _ in range(n):
        x = (b - A @ x + diagonal * x) / diagonal

    return x.round(4).tolist()