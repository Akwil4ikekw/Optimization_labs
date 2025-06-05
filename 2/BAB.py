
import math
import copy

def normalize_table(table, epsilon=1e-9):
    """
    Нормализация таблицы: заменяет числа, близкие к нулю, на 0.
    """
    for i in range(len(table)):
        for j in range(len(table[i])):
            if abs(table[i][j]) < epsilon:
                table[i][j] = 0


def print_table(table):
    """
    Вывод таблицы в формате.
    """
    for row in table:
        formatted_row = " ".join(f"{value:10.4f}" for value in row)
        print(formatted_row)


def simplex_method(table, basis):
    """
    Пересчёт таблицы симплекс-методом.
    """
    while True:
        # Вывод текущей таблицы
        print("Текущая таблица (симплекс-метод):")

        print_table(table)
        print(f"Базисные переменные: {basis}")

        print("-" * 50)

        # Поиск разрешающего столбца (столбец с минимальным коэффициентом в строке целевой функции)
        referenceColumn = -1
        min_value = 0
        for j in range(len(table[0]) - 1):
            if table[len(table) - 1][j] < min_value:
                min_value = table[len(table) - 1][j]
                referenceColumn = j

        # Если все коэффициенты >= 0, решение найдено
        if referenceColumn == -1:
            break

        # Поиск разрешающей строки (по минимальному отношению b / a)
        referenceRow = -1
        min_r = float('inf')
        for i in range(len(table) - 1):
            if table[i][referenceColumn] > 0:
                r = table[i][len(table[i]) - 1] / table[i][referenceColumn]
                if r < min_r:
                    min_r = r
                    referenceRow = i

        # Если разрешающая строка не найдена, функция не ограничена
        if referenceRow == -1:
                return "Задача не имеет ограниченного решения (цел. функция стремится к бесконечности)."

        # Обновление массива базисных переменных
        basis[referenceRow] = referenceColumn + 1

        # Нормализация разрешающей строки
        pivot_value = table[referenceRow][referenceColumn]
        for j in range(len(table[referenceRow])):
            table[referenceRow][j] /= pivot_value

        # Приведение остальных строк таблицы к каноническому виду
        for i in range(len(table)):
            if i != referenceRow:
                factor = table[i][referenceColumn]
                for j in range(len(table[i])):
                    table[i][j] -= factor * table[referenceRow][j]

        normalize_table(table)

        # Проверка на бесконечно много решений
    zero_count = 0
    for j in range(len(table[0]) - 1):  # Проходим по всем столбцам, кроме последнего (свободных членов)
        if table[len(table) - 1][j] == 0:  # Проверяем, является ли коэффициент в последней строке нулём
            zero_count += 1
    if zero_count > len(basis):  # Если количество нулей больше числа базисных переменных
        return "Задача имеет бесконечно много решений. \nВ строке метрик нулей больше, чем базисных переменных."

    # Формируем решение
    solution = [0] * (len(table[0]) - 1)
    for i in range(len(basis)):
        solution[basis[i] - 1] = table[i][len(table[i]) - 1]

        # Возвращаем оптимальное значение и значения переменных
    return table[len(table) - 1][len(table[0]) - 1], solution


def dual_simplex_method(table, basis):
    """
    Пересчёт таблицы двойственным симплекс-методом.
    """
    while True:
        # Вывод текущей таблицы
        print("Текущая таблица (двойственный симплекс-метод):")
        print_table(table)

        print(f"Базисные переменные: {basis}")
        print("-" * 50)

        # Поиск разрешающей строки (строка с минимальным свободным членом b)
        referenceRow = -1
        min_value = 0
        for i in range(len(table) - 1):
            if table[i][len(table[i]) - 1] < min_value:
                min_value = table[i][len(table[i]) - 1]
                referenceRow = i

        # Если все свободные члены >= 0, решение найдено
        if referenceRow == -1:
            # Если все свободные члены >= 0, но в строке целевой функции есть отрицательные элементы
            all_non_negative = True
            for row in table[:-1]:
                if row[len(row) - 1] < 0:
                    all_non_negative = False
                    break
            any_negative = any(value < 0 for value in table[len(table) - 1][:-1])
            if all_non_negative and any_negative:
                print("Переключение на симплекс-метод для завершения оптимизации.")
                return simplex_method(table, basis)
            break

        # Поиск разрешающего столбца (по минимальному отношению c / a)
        referenceColumn = -1
        min_r = float('inf')
        for j in range(len(table[0]) - 1):
            if table[referenceRow][j] < 0:
                r = abs(table[len(table) - 1][j] / table[referenceRow][j])
                if r < min_r:
                    min_r = r
                    referenceColumn = j

        # Если разрешающий столбец не найден "нет решений"
        if referenceColumn == -1:
            return "Задача не имеет решений "

        # Обновление массива базисных переменных
        basis[referenceRow] = referenceColumn + 1

        # Нормализация разрешающей строки
        pivot_value = table[referenceRow][referenceColumn]
        for j in range(len(table[referenceRow])):
            table[referenceRow][j] /= pivot_value

        # Приведение остальных строк таблицы к каноническому виду
        for i in range(len(table)):
            if i != referenceRow:
                factor = table[i][referenceColumn]
                for j in range(len(table[i])):
                    table[i][j] -= factor * table[referenceRow][j]

        normalize_table(table)  # Нормализация таблицы

    # Формируем решение
    solution = [0] * (len(table[0]) - 1)
    for i in range(len(basis)):
        solution[basis[i] - 1] = table[i][len(table[i]) - 1]

    # Возвращаем оптимальное значение и значения переменных
    return table[len(table) - 1][len(table[0]) - 1], solution


def choose_method(c, A, b):
    """
    Выбор метода решения (симплекс или двойственный симплекс).
    """
    # Создание симплекс-таблицы
    table = []
    basis = []

    # Формируем строки ограничений
    for i in range(len(b)):
        row = A[i] + [0] * len(b) + [b[i]]
        row[len(c) + i] = 1
        table.append(row)
        basis.append(len(c) + i + 1)

    # Проверяем, есть ли отрицательные свободные члены
    has_negative_b = any(row[-1] < 0 for row in table[:-1])

    # Если есть отрицательные свободные члены, используем двойственный симплекс-метод
    if has_negative_b:
        table.append([-x for x in c] + [0] * (len(b) + 1))
        return dual_simplex_method(table, basis)
    else:
        table.append([-x for x in c] + [0] * (len(b) + 1))
        return simplex_method(table, basis)




def is_integer(value, epsilon=1e-5):
    """
    Проверяет, является ли значение целым (с учетом допустимой погрешности).
    """
    return abs(value - round(value)) < epsilon


def is_solution_integer(solution):
    """
    Проверяет, является ли весь вектор решения целочисленным.
    """
    return all(is_integer(x) for x in solution)


def branch_and_bound(c, A, b):
    """
    Метод ветвей и границ для целочисленного линейного программирования.
    Основан на двойственном симплекс-методе.
    """
    stack = [(c, A, b)]  # стек задач
    best_solution = None
    best_value = float('-inf')
    all_solutions = []

    while stack:
        current_c, current_A, current_b = stack.pop()

        result = choose_method(current_c, current_A, current_b)

        if isinstance(result, str):
            if result == "Задача имеет бесконечно много решений. ":

                # Получаем коэффициенты целевой функции
                cx, cy = c[0], c[1]

                # Получаем ограничения
                points = []

                # Ищем ограничение, параллельное целевой функции
                def is_parallel(v1, v2):
                    return abs(v1[0] * v2[1] - v1[1] * v2[0]) < 1e-8

                parallel_line = None
                rhs = None
                for i, row in enumerate(A):
                    if is_parallel(row, c):
                        parallel_line = row
                        rhs = b[i]
                        break

                if parallel_line is None:
                    print("Не найдено ограничений, параллельных целевой функции.")
                    continue

                # Уравнение прямой: cx * x + cy * y = rhs

                # Пересечение с осью x (y = 0): x = rhs / cx
                if abs(cx) > 1e-8:
                    x = rhs / cx
                    y = 0
                    points.append((x, y))

                # Пересечение с осью y (x = 0): y = rhs / cy
                if abs(cy) > 1e-8:
                    x = 0
                    y = rhs / cy
                    points.append((x, y))

                # Пересечение с другими ограничениями
                for i, (a, bi) in enumerate(zip(A, b)):
                    if a == parallel_line:
                        continue  # не пересекаем сами с собой

                    # Решаем систему: cx*x + cy*y = rhs и a[0]*x + a[1]*y = bi
                    det = cx * a[1] - cy * a[0]
                    if abs(det) > 1e-8:
                        x = (rhs * a[1] - bi * cy) / det
                        y = (cx * bi - a[0] * rhs) / det
                        points.append((x, y))

                if len(points) < 2:
                    print("Недостаточно точек пересечения.")
                    continue

                # Сортировка по x для построения отрезка
                points.sort()
                x1, y1 = points[0]
                x2, y2 = points[-1]

                # Перебираем все целые точки на прямой между x1 и x2
                solutions = []
                for x in range(math.ceil(min(x1, x2)), math.floor(max(x1, x2)) + 1):
                    y = (rhs - cx * x) / cy
                    if abs(y - round(y)) < 1e-6:
                        # Проверим, удовлетворяет ли точка всем ограничениям
                        satisfies_all = all(
                            A[i][0] * x + A[i][1] * round(y) <= b[i] + 1e-6 for i in range(len(A))
                        )
                        if satisfies_all:
                            solutions.append((x, round(y)))

                if solutions:
                    print("Бесконечно много решений")
                    print("Целочисленные решения:")
                    for sol in solutions:
                        print(f"x1 = {sol[0]}\nx2 = {sol[1]}")

                    return ""
                else:
                    print("Нет целочисленных решений.")
            else:
                print("Нет допустимого решения:", result)
            continue

        optimal_value, solution = result

        # Обрезка по верхней границе
        if best_solution is not None and optimal_value < best_value:
            # print("Обрезаем по границе: текущее значение < лучшего найденного.")
            continue

        # if best_solution is not None and optimal_value == best_value:
        #     print("=== Оптимальное решение исходной задачи: ===")
        #     print(f"\nОптимальное значение (целочисленное): {optimal_value}")
        #     print("Целочисленные решения:")
        #     s = [best_solution]
        #     for sol in s:
        #         for _ in range(len(c)):
        #             print(f"x{_+1} = ", round(sol[_]))

        if is_solution_integer(solution):
            print("Целочисленное решение:")
            if optimal_value > best_value:
                best_value = optimal_value
                best_solution = solution
                all_solutions = [solution]
            elif math.isclose(optimal_value, best_value):
                if solution not in all_solutions:
                    all_solutions.append(solution)
            continue

        # Нецелое решение → разбиение задачи
        for i, x in enumerate(solution):
            if not is_integer(x):
                floor_val = math.floor(x)
                ceil_val = math.ceil(x)
                print(f"Ветка по x{i + 1} = {x} → {floor_val}, {ceil_val}")

                # Ветка x_i ≤ floor(x_i)
                A1 = copy.deepcopy(current_A)
                b1 = copy.deepcopy(current_b)
                new_row1 = [0] * len(current_c)
                new_row1[i] = 1
                A1.append(new_row1)
                b1.append(floor_val)

                # Ветка x_i ≥ ceil(x_i)
                A2 = copy.deepcopy(current_A)
                b2 = copy.deepcopy(current_b)
                new_row2 = [0] * len(current_c)
                new_row2[i] = -1
                A2.append(new_row2)
                b2.append(-ceil_val)

                stack.append((current_c, A1, b1))
                stack.append((current_c, A2, b2))
                break  # Ветвим только по первой нецелой переменной

    if best_solution is None:
        return "Нет целочисленного решения."

    return best_value, all_solutions

def standart_basis(c,A,b,signs):
    for i in range(len(signs)):
        if signs[i] == "MORE":
            b[i] *= -1
            for j in range(len(A[i])):
                A[i][j] *= -1

    return c,A,b


if __name__ == "__main__":

    # c = [1, 1]
    # A = [[7, 3], [4, 6],[2,-2]]
    # b = [3, 9,1]

    c = [1, 1]  # Целевая функция: max x1 + x2
    A = [[7, 3], [4, 6],[2,-2]]  # Матрица ограничений
    b = [3, 9,1]  # Свободные члены
    signs = ["MORE", "LESS","LESS"]  # Знаки ограничений: ≤, ≥, ≥
    #

    # signs = ["MORE", "MORE","MORE"]
    c,A,b = standart_basis(c,A,b,signs)
    result = branch_and_bound(c, A, b)
    if isinstance(result, str):
        print(result)
    else:
        optimal_value, solutions = result
        print("=== Оптимальное решение исходной задачи: ===")
        print(f"\nОптимальное значение (целочисленное): {optimal_value}")
        print("Целочисленные решения:")
        for sol in solutions:
            for _ in range(len(c)):
                print(f"x{_ + 1} = ", round(sol[_]))

