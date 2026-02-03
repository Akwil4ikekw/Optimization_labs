
import numpy as np
'''

2. Решить систему линейных уравнений 4-го порядка с точностью  e = 10^(-3): 
методом простой итерации.
Уравнение системы:

                   x1=0,23*x1-0,04*x2+0,21*x3-0,18*x4+1,24           
                   x2=0,45*x1-0,23*x2+0,06*x3-0,88                   
                   x3=0,26*x1+0,34*x2-0,11*x3+0,62                   
                   x4=0,05*x1-0,26*x2+0,34*x3-0,12*x4-1,17
'''


def jacobi(A, tol=1e-10, max_iterations=100):
    n = len(A)
    b = A[:, -1]  # Последний столбец - свободные члены
    A = A[:, :-1]  # Оставшаяся часть - коэффициенты
    x = np.zeros(n)  # Начальное приближение (нулевой вектор)
    x_new = np.zeros_like(x)
    
    for _ in range(max_iterations):
        for i in range(n):
            sum_terms = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - sum_terms) / A[i][i]
        
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new.copy()
    
    raise ValueError("Метод Якоби не сошелся за {} итераций".format(max_iterations))

# Пример использования
A = np.array([[10, 2, -1, 9],
              [-2, 8, -1, 5],
              [1, -1, 4, 6]], dtype=float)

solution = jacobi(A)
print("Решение:", solution)
