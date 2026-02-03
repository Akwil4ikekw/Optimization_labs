import numpy  as np
import matplotlib.pyplot as plt

'''
2. По таблице с равноотстоящими значениями аргумента вычислить значения функции для заданных значений аргументов,
     используя первую и вторую интерполяционные формулы Ньютона. Точность E<=0.000001.
Задание:

X1=0,1243; X2=0,492; X3=0,0024; X4=0,660;

0,01      0,991824 
0,06      0,951935 
0,11      0,913650 
0,16      0,876905 
0,21      0,841638 
0,26      0,807789 
0,31      0,775301 
0,36      0,744120 
0,41      0,714198 
0,46      0,685470 
0,51      0,657902 
0,56      0,631442 
'''
import numpy as np
import matplotlib.pyplot as plt


def divided_differences(y):
    """Вычисление конечных разностей."""
    n = len(y)
    coef = [[0] * n for _ in range(n)]
    for i in range(n):
        coef[i][0] = y[i]

    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1])

    return coef


def newton_forward(x, y, x_val, h):
    """Интерполяция по Ньютону вперёд."""
    n = len(x)
    q = (x_val - x[0]) / h
    coef = divided_differences(y)
    result = coef[0][0]
    product_term = 1

    for i in range(1, n):
        product_term *= (q - (i - 1)) / i
        result += coef[0][i] * product_term

    return result


def newton_backward(x, y, x_val, h):
    """Интерполяция по Ньютону назад."""
    n = len(x)
    q = (x_val - x[-1]) / h
    coef = divided_differences(y)
    result = coef[-1][0]
    product_term = 1

    for i in range(1, n):
        product_term *= (q + (i - 1)) / i
        result += coef[n - i - 1][i] * product_term

    return result


# Исходные данные
x_vals = [0.01, 0.06, 0.11, 0.16, 0.21, 0.26, 0.31, 0.36, 0.41, 0.46, 0.51, 0.56]
y_vals = [0.991824, 0.951935, 0.913650, 0.876905, 0.841638, 0.807789, 0.775301, 0.744120, 0.714198, 0.685470, 0.657902,
          0.631442]
x_interps = [0.1243, 0.492, 0.0024, 0.660]
h = x_vals[1] - x_vals[0]

# Определяем середину диапазона для выбора метода
x_mid = (min(x_vals) + max(x_vals)) / 2

# Вычисляем интерполяцию для заданных точек
for x_interp in x_interps:
    if x_interp < x_mid:
        result = newton_forward(x_vals, y_vals, x_interp, h)
        method = "вперёд"
    else:
        result = newton_backward(x_vals, y_vals, x_interp, h)
        method = "назад"
    print(f"Интерполяция {method} в точке {x_interp}: {result}")

# Создаем плотную сетку для построения графика
x_plot = np.linspace(min(x_vals), max(x_interps), 500)  # 500 точек между min(x_vals) и 0.660
y_plot = []

for x in x_plot:
    if x < x_mid:
        y_plot.append(newton_forward(x_vals, y_vals, x, h))
    else:
        y_plot.append(newton_backward(x_vals, y_vals, x, h))

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_plot, 'r-', label='Интерполяционный полином Ньютона', linewidth=2)
plt.scatter(x_interps,
            [newton_forward(x_vals, y_vals, x, h) if x < x_mid else newton_backward(x_vals, y_vals, x, h) for x in
             x_interps],
            color='green', label='Точки интерполяции', s=100, zorder=5)
plt.title('Интерполяция по Ньютону (вперёд/назад)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()