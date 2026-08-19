import math
import numpy as np


def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    """
    Compute the SVD of a real 2x2 matrix using one Jacobi rotation.

    Returns:
        U, S, Vt such that

            A ≈ U @ np.diag(S) @ Vt

        where:
        - U  is 2x2 orthogonal
        - S  is a length-2 array of singular values in descending order
        - Vt is 2x2 orthogonal
    """
    A = np.asarray(A, dtype=float)

    if A.shape != (2, 2):
        raise ValueError("A must be a 2x2 matrix")

    # 1. Form the symmetric matrix A^T A
    B = A.T @ A

    # 2. Find a Jacobi rotation that diagonalizes B
    theta = 0.5 * math.atan2(
        2 * B[0, 1],
        B[0, 0] - B[1, 1]
    )

    c = math.cos(theta)
    s = math.sin(theta)

    V = np.array([
        [c, -s],
        [s,  c]
    ])

    # 3. Eigenvalues of A^T A are squared singular values
    D = V.T @ B @ V

    # Clamp tiny negative values caused by floating-point error
    eigenvalues = np.clip(np.diag(D), 0.0, None)
    S = np.sqrt(eigenvalues)

    # 4. Singular values conventionally appear in descending order
    order = np.argsort(S)[::-1]
    S = S[order]
    V = V[:, order]

    # 5. Zero matrix
    if S[0] == 0.0:
        U = np.eye(2)
        return U, S, V.T

    # First left singular vector:
    # u1 = A v1 / sigma1
    u1 = (A @ V[:, 0]) / S[0]

    # Treat a numerically tiny second singular value as zero
    tol = 2 * np.finfo(float).eps * S[0]

    if S[1] <= tol:
        S[1] = 0.0

        # Choose any unit vector orthogonal to u1
        u2 = np.array([-u1[1], u1[0]])
    else:
        # u2 = A v2 / sigma2
        u2 = (A @ V[:, 1]) / S[1]

    # 6. Put u1 and u2 into the columns of U
    U = np.column_stack((u1, u2))

    return U, S, V.T