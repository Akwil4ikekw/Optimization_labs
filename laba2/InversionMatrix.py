import numpy as np


def determinant(arr):
    return arr[0][0] * arr[1][1] * arr[2][2] - arr[2][0] * arr[1][1] * arr[0][2] + arr[0][1]*arr[1][2]*arr[2][0] + arr[0][1]*arr[1][2] * arr[2][1] \
    - arr[0][1]*arr[0][2]*arr[2][2]- arr[0][0]*arr[1][2]*[2][1]
A = [[4,8,7],[1,2,1],[2,3,1]]
print(determinant(A))

"""
                                    4*x1+8*x2+7*x3=9              
                                    x1+2*x2+2*x3=2                
                                    2*x1+3*x2+x3=9
                                       """
