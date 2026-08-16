def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:

    num_row_a = len(a)
    num_col_a = len(a[0])
    num_row_b = len(b)
    num_col_b = len(b[0])

    if num_col_a != num_row_b:
        return -1

    return [
        [sum(a[i][k] * b[k][j] for k in range(num_col_a)) for j in range(num_col_b)]
        for i in range(num_row_a)
    ]