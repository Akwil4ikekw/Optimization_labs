import numpy as np


# Определяем функции и их производные
def f1(x, y):
    return 30 * x ** 2 + 7 * y ** 2 - 1


def f2(x, y):
    return np.sin(4 * x - 0.5 * y) + 5 * x


def df1_dx(x, y):
    return 60 * x


def df1_dy(x, y):
    return 14 * y


def df2_dx(x, y):
    return 4 * np.cos(4 * x - 0.5 * y) + 5


def df2_dy(x, y):
    return -2 * np.cos(4 * x - 0.5 * y)


# Метод Ньютона для решения системы
def newton_method(x0, y0, tol=1e-3, max_iter=100):
    x, y = x0, y0
    for _ in range(max_iter):
        F1 = f1(x, y)
        F2 = f2(x, y)

        J11 = df1_dx(x, y)
        J12 = df1_dy(x, y)
        J21 = df2_dx(x, y)
        J22 = df2_dy(x, y)

        det_J = J11 * J22 - J12 * J21

        J_inv = (1 / det_J) * np.array([[J22, -J12], [-J21, J11]])

        F = np.array([F1, F2])

        delta = J_inv.dot(F)
        x, y = x - delta[0], y - delta[1]

        if np.linalg.norm(F, ord=2) < tol:
            break
#Работает за 8 итераций
    return x, y
x0, y0 = 0.1, 0.1

solution = newton_method(x0, y0)
print()
print(f"Решение: x = {solution[0]}, y = {solution[1]}")

