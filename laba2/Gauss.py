import numpy as np


def select_lead_elem(matrix):
    matrix = matrix.astype(float)
    num_rows, num_cols = matrix.shape

    for col in range(min(num_rows, num_cols)):  # Проход по столбцам
        max_index = col + np.argmax(abs(matrix[col:, col]))  # Находим индекс максимального элемента в столбце
        if max_index != col:  # Если ведущий элемент не на месте, меняем строки
            matrix[[col, max_index]] = matrix[[max_index, col]]
    return matrix


def direct_Gausse(matrix):
    matrix = select_lead_elem(matrix)  # Упорядочиваем строки по ведущему элементу
    lead_elem = 0

    for row_1 in matrix:
        for i in range(lead_elem + 1, len(matrix)):
            sub_k = matrix[i][lead_elem] / row_1[lead_elem]
            matrix[i] = np.subtract(matrix[i], row_1 * sub_k)
        lead_elem += 1
        if lead_elem == len(matrix) - 1:
            break
    return matrix


def redirect_Gause(matrix):
    root = np.zeros(len(matrix))
    for i in range(len(matrix) - 1, -1, -1):
        sum_ax = sum(matrix[i][j] * root[j] for j in range(i + 1, len(matrix)))
        root[i] = (matrix[i][-1] - sum_ax) / matrix[i][i]
    return root


def Gausse(matrix):
    matrix = select_lead_elem(matrix)  # Упорядочиваем строки по ведущему элементу

    matrix = direct_Gausse(matrix)
    root = redirect_Gause(matrix)
    return root


'''
Решить систему линейных уравнений 4-го порядка методом Гаусса с точностью e = 10^(-3):
Уравнение системы:
            0,17*x1-0,13*x2-0,11*x3-0,12*x4=0,22             
            1,00*x1-1,00*x2-0,13*x3+0,13*x4=0,11             
            0,35*x1+0,33*x2+0,12*x3+0,13*x4=0,12             
            0,13*x1+0,11*x2-0,13*x3-0,11*x4=1,00
            '''
A = np.array([[0.17, -0.13, -0.11, -0.12, 0.22], [1, -1, -0.13, 0.13, 0.11], [0.35, 0.33, 0.12, 0.13, 0.12],
              [0.13, 0.11, -0.13, -0.11, 1.00]])
X = Gausse(A)
i = 0
print("Корни методом Гаусса:")
for x in X:
    i += 1
    print(f"x{i}: {x}")

