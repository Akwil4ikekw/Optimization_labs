import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Определение функции и её производной
def f(x):
    return 3 * x - np.exp(x)


def df(x):
    return 3 - np.exp(x)


def ddf(x):
    return (np.exp(x)) * (-1)


# Проверка условия сходимости
def check_convergence(a, b):
    m = min(abs(df(a)), abs(df(b)))  # Берем минимум по модулю
    M = max(abs(ddf(a)), abs(ddf(b)))  # В идеале, найти максимум на всем интервале

    print(f"a: {a}, b: {b}")
    print(f"df(a): {df(a)}, df(b): {df(b)}")
    print(f"ddf(a): {ddf(a)}, ddf(b): {ddf(b)}")
    print(f"m = {m}, M = {M}, Условие M <= 2m: {M <= 2 * m}")

    return M <= 2 * m, m, M

# Метод Ньютона
def newton_method(x0):
    return x0 - f(x0) / df(x0)


# Метод хорд
def chord_method(x0, x1):
    return x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))


# Метод простых итераций
def iteration_method(x0, tol=1e-5, max_iter=26):
    def phi(x):
        return np.log(3 * x)  # Итерационная функция

    arr_iteration = []
    x = x0
    for _ in range(max_iter):
        x_new = phi(x)
        arr_iteration.append(x_new)
        if abs(x_new - x) < tol:
            return arr_iteration
        x = x_new
    return arr_iteration


def combined_method(x0, x1, tol=1e-5):
    iteration_newton = []
    iteration_chord = []
    conv, m, M = check_convergence(x0, x1)

    iteration_chord.append(x0)
    iteration_newton.append(x1)
  #  print("Условие M < 2m выполнено")

    while abs(f(x1)) / m > tol:  # Новый критерий остановки
        x0 = chord_method(x0, x1)
        x1 = newton_method(x0)
        iteration_chord.append(x0)
        iteration_newton.append(x1)
    return iteration_newton, iteration_chord, m, M


# Вызываем функции и получаем корни для разных методов
root_newton, root_chord, m, M = combined_method(x0=1.4, x1=1.6, tol=1e-5)
root_iteration = iteration_method(x0=1.1, tol=1e-5)

if root_newton:
    print(f"Приближенное решение (комбинированный метод): {root_newton[-1]}")
    print(root_newton)
    print(root_chord)
if root_iteration:
    print(f"Приближенное решение (метод простых итераций): {root_iteration[-1]}")
print(f"m = {m}, M = {M}")

# Вывод таблицы для метода итераций
if root_iteration:
    data_iteration = pd.DataFrame({
        "Значение функции": [f(x) for x in root_iteration],
        "Значение корня": root_iteration
    })
    print(data_iteration)

# Объединяем итерации хорд и Ньютона

if root_chord:
    data_combination = pd.DataFrame({
        "Значение корня методом хорд": [f(x) for x in root_chord],
        "Значение методом Ньютона": [f(x) for x in root_newton],
        "Значение корня": root_chord
    })
    print(data_combination)

# График функции
x_range = np.linspace(1.4, 1.6, 400)
y_range = f(x_range)

plt.figure(figsize=(10, 5))

# График метода хорд и Ньютона
plt.subplot(1, 2, 1)
plt.plot(x_range, y_range, label="f(x) = 3x - e^x", color='black')
plt.axhline(0, color="black", linewidth=1, linestyle="--")
for i in range(min(2, len(root_chord))):
    x_c = root_chord[i]
    x_n = root_newton[i]

    plt.scatter(x_c, f(x_c), color='blue', label="Хорда" if i == 0 else "")
    plt.scatter(x_n, f(x_n), color='red', label="Ньютон" if i == 0 else "")

    # Правильная касательная (линию строим в небольшом диапазоне вокруг x_n)
    tangent_x = np.linspace(x_n - 0.05, x_n + 0.05, 100)
    tangent_y = f(x_n) + df(x_n) * (tangent_x - x_n)
    plt.plot(tangent_x, tangent_y, color='purple', linestyle='--', label="Касательная" if i == 0 else "")

    # Хорда
    plt.plot([x_c, root_chord[i + 1]], [f(x_c), 0], color='blue', linestyle='--', label="Хорда" if i == 0 else "")

plt.legend()
plt.title("Методы хорд и Ньютона")
plt.grid()

# График корней
plt.subplot(1, 2, 2)
plt.plot(x_range, y_range, label="f(x) = 3x - e^x")
plt.axhline(0, color="black", linewidth=1)
if root_newton:
    plt.axvline(root_newton[-1], color="red", linestyle="--", label=f"Корень Ньютона ≈ {root_newton[-1]:.5f}")
if root_iteration:
    plt.axvline(root_iteration[-1], color="green", linestyle="--", label=f"Корень итераций ≈ {root_iteration[-1]:.5f}")
plt.legend()
plt.grid()
plt.show()
