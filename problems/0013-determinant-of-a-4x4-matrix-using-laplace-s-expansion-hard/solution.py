import numpy as np


def determinant_4x4(matrix: list[list[int|float]]) -> float:
    matrix = np.array(matrix, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError("matrix must be a 4x4 matrix.")

    def determinant_3x3(m: np.ndarray) -> float:
        return m[0, 0] * (
            m[1, 1] * m[2, 2] - m[2, 1] * m[1, 2]
        ) + m[0, 1] * (
            m[1, 2] * m[2, 0] - m[1, 0] * m[2, 2]
        ) + m[0, 2] * (
            m[1, 0] * m[2, 1] - m[2, 0] * m[1, 1]
        )

    determinant = 0.0
    for j in range(4):
        minor = np.delete(np.delete(matrix, 0, axis=0), j, axis=1)
        determinant += (-1) ** j * matrix[0, j] * determinant_3x3(minor)
    return determinant