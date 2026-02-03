import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.log10(x**2 + 3) / (2 * x)

def simpson_rule(a, b, n, func):
    """Вычисление интеграла методом Симпсона с контролем чётности n"""
    if n % 2 != 0:
        n += 1  # Гарантируем чётное n
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = func(x)
    integral = h/3 * (y[0] + y[-1] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-1:2]))
    return integral

# Параметры интегрирования
a, b = 1.2, 2.0
epsilon = 0.0001

# Визуализация функции
x_vals = np.linspace(a, b, 200)
y_vals = f(x_vals)

plt.figure(figsize=(10, 5))
plt.plot(x_vals, y_vals, label=r'$f(x) = \frac{\log_{10}(x^2 + 3)}{2x}$')
plt.xlabel('Аргумент функции')
plt.ylabel('Значение функции')
plt.title('График подынтегральной функции')
plt.grid(True)
plt.legend()
plt.show()

# Адаптивное вычисление интеграла
n = 4
I_prev = simpson_rule(a, b, n, f)
I_next = simpson_rule(a, b, 2*n, f)
error = abs(I_next - I_prev)/15

history = []  # Для хранения истории вычислений

while error > epsilon:
    n *= 2
    I_prev = I_next
    I_next = simpson_rule(a, b, 2*n, f)
    error = abs(I_next - I_prev)/15
    history.append((2*n, I_next, error))

# Вывод результатов

for n, val, err in history[-3:]:  # Показываем последние 3 итерации
    print(f"{n:>8} {val:.8f} {err:.2e}")

final_n = 2*n
final_integral = I_next
final_error = error

print("\nИтоговый результат:")
print(f"Число разбиений: n = {final_n}")
print(f"Значение интеграла: {final_integral:.8f}")
print(f"Оценка погрешности: {final_error:.2e}")

# Проверка с удвоенным n
check_n = 2*final_n
check_integral = simpson_rule(a, b, check_n, f)
difference = abs(final_integral - check_integral)

print("\nПроверка точности:")
print(f"При n = {check_n}: {check_integral:.8f}")
print(f"Разница с предыдущим: {difference:.2e}")