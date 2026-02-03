import numpy as np
import matplotlib.pyplot as plt

def Lagrange(x, x_arr, y_arr):
    polinom = 0
    for i in range(len(x_arr)):
        l = 1
        for j in range(len(x_arr)):
            if j != i:
                l *= (x - x_arr[j]) / (x_arr[i] - x_arr[j])
        polinom += l * y_arr[i]
    return polinom

x_arr = np.array([0.62, 0.67 , 0.74 , 0.80 , 0.87, 0.96, 0.99])
y_arr = np.array([0.537944, 0.511709, 0.477114, 0.449329, 0.418952, 0.382893, 0.371577])
x = 0.692
lagr = round(Lagrange(x, x_arr, y_arr), 6)


x_smooth = np.linspace(min(x_arr), max(x_arr), 100)
y_smooth = [Lagrange(xi, x_arr, y_arr) for xi in x_smooth]


plt.plot(x_smooth, y_smooth, label="Полином Лагранжа", color="blue")  # Гладкий график полинома
plt.scatter(x_arr, y_arr, color="blue", label="Исходные точки")  # Узлы интерполяции
plt.scatter(x, lagr, color="red", label=f"Точка интерполяции ({x}, {round(lagr, 6)})")  # Интерполированная точка

plt.title("Интерполяция полиномом Лагранжа")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()
