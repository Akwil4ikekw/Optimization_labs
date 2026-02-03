# 0,1        1,91   
# 0,2        3,03   
# 0,3        3,98   
# 0,4        4,82   
# 0,5        5,59   
# 0,6        6,31   
# 0,7        7,00   
# 0,8        7,65   
# 0,9        8,27   
# 1,0        8,88
# '''

import matplotlib.pyplot as plt
import numpy as np


# def abolute_error(value_x,value_y,predict_x,predict_y):
#     distance = [0]*len(value_x)
#     for i in range(len(value_x)):
#         distance[i] = np.sqrt((value_x[i]-predict_x)**2 + (value_y[i]-predict_y)**2 )
#     min_dist = min(distance)
#     return distance.index(min_dist)




# x_value = np.array([0.1,.02,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],float)
# y_value = np.array([1.91,3.03,3.98,4.82,5.59,6.31,7.00,7.065,8.27,8.88],float)

# x_avg = x_value[0]+x_value[-1]/2
# y_avg = y_value[0]+y_value[-1]/2

# x_geom = np.sqrt(x_value[0]*x_value[-1])
# y_geom = np.sqrt(y_value[0]*y_value[-1])

# x_garm = (2*x_value[0]*x_value[-1])/(x_value[0]+x_value[-1])
# y_garm = (2*y_value[0]*y_value[-1])/(x_value[0]+y_value[-1])


# index_y1 = (abolute_error(x_value,y_value,x_avg,y_avg))
# index_y2 = (abolute_error(x_value,y_value,x_geom,y_geom))
# index_y3 = (abolute_error(x_value,y_value,x_garm,y_garm))

# y1 = y_value[index_y1]
# y2 = y_value[index_y2]
# y3 = y_value[index_y3]

# arr_y = [y1,y2,y3]

# y_predict = [y_avg,y_geom,y_garm]
# def calculcate_error(arr_y, y_predict):
#     e=[]
#     for y in arr_y:
#         e1,e2,e3 = [abs(y - y_pred) for y_pred in y_predict]
#         e.append((e1,e2,e3)) 
#     return e
# error = (calculcate_error(arr_y,y_predict))
# error_min = np.array([min(e) for e in error])
# index_min_e = np.argmin(error_min)
# #index_min_e = 0 => зависимость линейная 



# Данные
x_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], float)
y_value = np.array([1.91, 3.03, 3.98, 4.82, 5.59, 6.31, 7.00, 7.65, 8.27, 8.88], float)

# Вычисление коэффициентов линейной регрессии
n = len(x_value)
sum_x = np.sum(x_value)
sum_y = np.sum(y_value)
sum_xy = np.sum(x_value * y_value)
sum_x2 = np.sum(x_value ** 2)

a = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
b = (sum_y - a * sum_x) / n

def linear_function(x):
    return a * x + b

# Вычисление среднеквадратичного отклонения
errors = y_value - linear_function(x_value)
print(errors)
sigma = np.sqrt(np.sum(errors ** 2) / n)

# Построение графика
plt.scatter(x_value, y_value, color='red', label='Экспериментальные точки')
plt.plot(x_value, linear_function(x_value), color='blue', label=f'Линейная модель: y = {a:.3f}x + {b:.3f}')
plt.xlabel("Аргументы функции")
plt.ylabel("Значения функции")
plt.legend()
plt.grid()
plt.show()

# Вывод коэффициентов и среднеквадратичного отклонения
print(f"Коэффициенты линейной модели: a = {a:.3f}, b = {b:.3f}")
print(f"Среднеквадратичное отклонение: σ = {sigma:.4f}")