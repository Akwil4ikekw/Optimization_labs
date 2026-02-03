import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return x**2 - y**2

def runge_kutta_2(f, x0, y0, h, x_end):
    x = np.arange(x0, x_end + h, h)
    y = np.zeros(len(x))
    y[0] = y0
    for i in range(1, len(x)):
        k1 = h * f(x[i-1], y[i-1])
        k2 = h * f(x[i-1] + h, y[i-1] + k1)
        y[i] = y[i-1] + 0.5 * (k1 + k2)
    return x, y

x_h, y_h = runge_kutta_2(f, 0, 0, 0.1, 1)
x_h2, y_h2 = runge_kutta_2(f, 0, 0, 0.05, 1)

# Вывод таблиц
print("Таблица (h=0.1):")
for xi, yi in zip(x_h, y_h):
    print(f"{xi:.1f} | {yi:.6f}")

print("\nТаблица (h=0.05):")
for xi, yi in zip(x_h2[::2], y_h2[::2]):  # Выводим каждую вторую точку для сравнения
    print(f"{xi:.1f} | {yi:.6f}")

# График

plt.plot(x_h2, y_h2,label='h=0.05')
plt.plot(x_h, y_h,  label='h=0.1')
plt.scatter(x_h,y_h, c = "yellow",label = "Точки с шагом h")
plt.scatter(x_h2,y_h2, c ="blue",label = "Точки с шагом h/2")

plt.xlabel('x')
plt.ylabel('y(x)')
plt.legend()
plt.grid()
plt.show()