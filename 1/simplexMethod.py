import numpy as np
from typing import List,Union,Tuple
def simplex_method(table: np.ndarray, num_constraints: int, num_vars: int) -> Union[str, np.ndarray]:
    """Прямой симплекс-метод с правилом Блэнда."""
    iteration_limit = 1000  # Ограничение на количество итераций
    iteration = 0
    previous_pivots = set()  # Храним историю ведущих элементов

    while iteration < iteration_limit:
        last_row = table[-1, :-1]
        if np.all(last_row >= 0):
            break  # Оптимум достигнут


        pivot_col = np.where(last_row < 0)[0][0]  # Первый отрицательный коэффициент
        column = table[:num_constraints, pivot_col]

        if np.all(column <= 0):
            return "Задача не имеет ограниченного решения."

        ratios = []
        for i in range(num_constraints):
            if column[i] > 0:
                ratios.append(table[i, -1] / column[i])
            else:
                ratios.append(np.inf)

        pivot_row = np.argmin(ratios)
        if ratios[pivot_row] == np.inf:
            return "Задача не имеет ограниченного решения."

        # Проверка на вырожденность
        pivot_key = (pivot_row, pivot_col)
        if pivot_key in previous_pivots:
            return "Обнаружено зацикливание в симплекс-методе."
        previous_pivots.add(pivot_key)

        pivot_value = table[pivot_row, pivot_col]
        if abs(pivot_value) < 1e-6:
            return "Ошибка: ведущий элемент слишком мал."

        table[pivot_row] /= pivot_value
        for i in range(len(table)):
            if i != pivot_row:
                table[i] -= table[i, pivot_col] * table[pivot_row]

        iteration += 1

    if iteration >= iteration_limit:
        return "Достигнут лимит итераций в симплекс-методе."
    return table

def create_basis_matrix(coeff_system: List[List[float]],
                        target_function: List[float],
                        constraints: List[float],
                        sign_vector: List[str]) -> np.ndarray:
    num_vars = len(target_function)
    num_constraints = len(constraints)

    A = np.array(coeff_system, dtype=float)
    b = np.array(constraints, dtype=float).reshape(-1, 1)

    slack_columns = []
    for i, sign in enumerate(sign_vector):
        col = [0.0] * num_constraints
        if sign == "LESS":
            col[i] = 1.0
        elif sign == "MORE":
            col[i] = -1.0
        else:
            col[i] = 0.0
        slack_columns.append(col)

    # Формируем расширенную матрицу с искусственными переменными
    slack_matrix = np.array(slack_columns).T
    table = np.hstack((A, slack_matrix, b))

    # Целевая функция
    total_vars = num_vars + slack_matrix.shape[1]
    c_extension = np.zeros(total_vars + 1)
    for i in range(num_vars):
        c_extension[i] = -target_function[i]

    table = np.vstack((table, c_extension.reshape(1, -1)))

    return table


def search_negative_b(table: np.ndarray, num_constraints: int) -> Tuple[bool, int]:
    """Находит строку с отрицательным свободным членом."""
    has_negative = False
    pivot_row = -1
    for i in range(num_constraints):
        if table[i, -1] < 0:
            has_negative = True
            if pivot_row == -1 or table[i, -1] < table[pivot_row, -1]:
                pivot_row = i
    return has_negative, pivot_row




def detect_infeasibility(table: np.ndarray, num_constraints: int, num_vars: int) -> Union[str, None]:
    """Проверка на несовместность ограничений."""
    for i in range(num_constraints):
        if np.all(table[i, :-1] == 0) and table[i, -1] != 0:
            return "Задача не имеет решений (ограничения несовместны)."
    return None


def extract_solution(table: np.ndarray, num_constraints: int, num_vars: int, target_function: List[float]) -> Tuple[float, List[float]]:
    """Извлекает оптимальное решение и значение функции."""
    solution = [0.0] * num_vars
    for col in range(num_vars):
        column = table[:num_constraints, col]
        if np.count_nonzero(column == 1) == 1 and np.count_nonzero(column == 0) == num_constraints - 1:
            row = np.where(column == 1)[0][0]
            solution[col] = table[row, -1]

    z = sum(solution[i] * target_function[i] for i in range(num_vars))
    return z, solution


def has_multiple_optimal_solutions(table: np.ndarray, num_vars: int, num_constraints: int) -> bool:
    """Проверяет наличие альтернативных оптимальных решений."""
    last_row = table[-1, :num_vars + num_constraints]
    for j in range(num_vars + num_constraints):
        if last_row[j] == 0:
            col = table[:num_constraints, j]
            if np.count_nonzero(col == 1) == 1 and np.count_nonzero(col == 0) == num_constraints - 1:
                return True
    return False

if __name__ == "__main__":
    c = [1, 1]  # Целевая функция: max x1 + x2
    A = [[7, 3], [4, 6], [2, -2]]  # Ограничения
    b = [3, 9, 1]
    signs = ["MORE", "LESS", "LESS"]

    table = create_basis_matrix(A, c, b, signs)
    num_constraints = len(b)
    num_vars = len(c)
    print(table)

    # Используем прямой симплекс-метод вместо дуального
    table_result = simplex_method(table.copy(), num_constraints, num_vars)

    if isinstance(table_result, str):
        print(table_result)
    else:
        table = table_result
        infeasibility_msg = detect_infeasibility(table, num_constraints, num_vars)
        if infeasibility_msg:
            print(infeasibility_msg)
        else:
            if has_multiple_optimal_solutions(table, num_vars, num_constraints):
                print("Задача имеет несколько оптимальных решений.")
            z, solution = extract_solution(table, num_constraints, num_vars, c)
            print(f"\nОптимальное значение целевой функции: {z}")
            print(f"Оптимальный вектор переменных: {solution}")
            print("Table:")
            print(table)
