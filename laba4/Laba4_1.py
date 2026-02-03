import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 1 / np.sqrt(x**2 + 1.2)

def f_second_derivative(x):
    return 3 * x**2 / (x**2 + 1.2)**(5/2) - 1 / (x**2 + 1.2)**(3/2)

def trapezoidal_rule(a, b, n, func):
    h = (b - a) / n
    integral = 0.5 * (func(a) + func(b))
    for i in range(1, n):
        integral += func(a + i * h)
    integral *= h
    return integral

# Пределы интегрирования и точность
a, b = 1.2, 2.0
epsilon = 0.0001

# =============================================
# Способ 1: Через оценку второй производной
# =============================================
x_vals = np.linspace(a, b, 1000)
max_f_double_prime = max(abs(f_second_derivative(x)) for x in x_vals)
n_derivative = int(np.sqrt((b - a)**3 * max_f_double_prime / (12 * epsilon))) + 1
integral_deriv = trapezoidal_rule(a, b, n_derivative, f)

# =============================================
# Способ 2: Через принцип Рунге (автоматический подбор n)
# =============================================
n_runge = 2  # Начинаем с минимального n
while True:
    integral_n = trapezoidal_rule(a, b, n_runge, f)
    integral_2n = trapezoidal_rule(a, b, 2 * n_runge, f)
    error_estimate = abs(integral_2n - integral_n) / 3  # Для метода трапеций
    if error_estimate < epsilon:
        break
    n_runge *= 2

integral_runge = trapezoidal_rule(a, b, n_runge, f)

# =============================================
# Вывод результатов
# =============================================
print(f"Способ 1 (через производную):")
print(f"  n = {n_derivative}, интеграл = {integral_deriv:.6f}")
print(f"  Оценка погрешности: {(b-a)**3 * max_f_double_prime / (12 * n_derivative**2):.6f}")

print(f"\nСпособ 2 (принцип Рунге):")
print(f"  n = {n_runge}, интеграл = {integral_runge:.6f}")
print(f"  Оценка погрешности: {error_estimate:.6f}")

# =============================================
# График функции
# =============================================
plt.figure(figsize=(10, 5))
plt.title(r"График функции $f(x) = \dfrac{1}{\sqrt{x^2 + 1.2}}$")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x_vals, [f(x) for x in x_vals], label="Функция")
plt.grid(True)
plt.legend()
plt.show()