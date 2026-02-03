import numpy as np


# LU разложение с перестановками
def LU_with_pivoting(A):
    n = len(A)
    L = np.eye(n)
    U = np.copy(A).astype(float)
    P = np.eye(n)
    for i in range(n):
        # Строка с максимальным элементом в i столбце
        max_row = np.argmax(np.abs(U[i:n, i])) + i
        if i != max_row:
            U[[i, max_row], :] = U[[max_row, i], :]
            L[[i, max_row], :i] = L[[max_row, i], :i]
            P[[i, max_row], :] = P[[max_row, i], :]

        for j in range(i + 1, n):
            L[j][i] = U[j][i] / U[i][i]  # Вычисляем элементы L
            U[j, i:n] -= L[j][i] * U[i, i:n]

    return P, L, U  # Возвращаем матрицы P, L и U


# Решает Ly = b (прямой ход)
def forward_substitution(L, b):
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - sum(L[i][k] * y[k] for k in range(i))
    return y


# Решает обратный ход Ux = y
def backward_substitution(U, y):
    n = len(y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):  # Идём снизу вверх
        if U[i][i] == 0:
            raise ValueError(f"Нулевая диагональ в U на шаге {i}, невозможно продолжить решение!")
        x[i] = (y[i] - sum(U[i][k] * x[k] for k in range(i + 1, n))) / U[i][i]
    return x


# Нахождение обратной матрицы через LU-разложение
def inverse_matrix(A):
    n = len(A)
    P, L, U = LU_with_pivoting(A)
    I = np.eye(n)
    inv_A = np.zeros_like(A)

    for i in range(n):
        # Решаем систему P * L * U * x = I_i (i-й столбец единичной матрицы)
        y = forward_substitution(L, np.dot(P, I[:, i]))  # Ly = P * I_i
        inv_A[:, i] = backward_substitution(U, y)  # Ux = y => x = U^-1 * y

    return inv_A


A = np.array([[4, 8, 7], [1, 2, 2], [2, 3, 1]], dtype=float)

inv_A = inverse_matrix(A)

print("Обратная матрица A^{-1}:\n", inv_A)

