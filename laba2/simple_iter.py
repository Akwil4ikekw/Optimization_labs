import numpy as np

def norm_infinity(P):
    return max(np.sum((P), axis=0))

# Функция для вычисления нормы по максимальному элементу в строках
def norm_onee(P):
    for i in range(len(P)):
        if max(np.sum((P),axis = i):

    return max(np.sum((P), axis=1))

# Функция для вычисления евклидовой нормы
def norm_twoo(P):
    return max(np.sqrt(np.sum(P**2, axis=1)))


def simple_iteration(P, g, error=0.001):
    n = len(g)
    x = [0] * n

    norm_inf = norm_infinity(P)
    norm_one = norm_onee(P)
    norm_two = norm_twoo(P)

    print(f"Норма по максимальному элементу в столбцах: {norm_inf}")
    print(f"Норма по максимальному элементу в строках: {norm_one}")
    print(f"Норма по квадратам: {norm_two}")

    while(True):
        x_new = [sum(P[i][j] * x[j] for j in range(n)) + g[i] for i in range(n)]

        if max(abs(x_new[i] - x[i]) for i in range(n)) < error:
            return x_new

        x = x_new[:]


P = [
    [0.23, -0.04, 0.21, -0.18],
    [0.45, -0.23, 0.06, 0],
    [0.26, 0.34, -0.11, 0],
    [0.05, -0.26, 0.34, -0.11]
]

g = [1.24, -0.88, 0.62, -1.17]


solution = simple_iteration(np.array(P), g)
print("Решение:", solution)
