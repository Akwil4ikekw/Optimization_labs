import numpy as np
import matplotlib.pyplot as plt

def gauss_quadrature(f, a, b, n):

    # Узлы и веса для квадратур Гаусса
    if n == 4:
        nodes = [-0.86114, -0.33998, 0.33998, 0.86114]
        weights = [0.34785, 0.65215, 0.65215, 0.34785]
    elif n == 7:
        nodes = [-0.949107912, -0.741531186, -0.405845151, 0.0, 0.405845151, 0.741531186, 0.949107912]
        weights = [0.129484966, 0.279705391, 0.381830051, 0.417959184, 0.381830051, 0.279705391, 0.129484966]
    else:
        raise ValueError("Поддерживаются только 4 или 7 узлов.")
    
    # Преобразование пределов интегрирования
    transform = lambda x: 0.5 * (b - a) * x + 0.5 * (b + a)
    
    # Приближенное вычисление интеграла
    integral = sum(w * f(transform(x)) for x, w in zip(nodes, weights)) * (b - a) / 2
    return integral

# Функция под интегралом
def func(x):
    return x / np.sqrt(x**2 + 2.5)

# Пределы интегрирования
a, b = 1.4, 2.6

x_value = np.linspace(a,b,200)
y_value = [func(x) for x in x_value ]

# Вычисление интеграла методом Гаусса с 4 и 7 узлами
result_4 = gauss_quadrature(func, a, b, 4)
result_7 = gauss_quadrature(func, a, b, 7)

print(f"Приближенное значение интеграла (4 узла): {result_4}")
print(f"Приближенное значение интеграла (7 узлов): {result_7}")

plt.title("Исходная функция")
plt.plot(x_value,y_value)
plt.show()