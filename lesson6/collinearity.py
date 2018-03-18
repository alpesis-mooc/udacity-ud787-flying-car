"""
6.5 Collinearity

"""

# Define Points (feel free to change these)
# By default these will be cast as int64 arrays
import numpy as np
p1 = np.array([1, 2, 1])
p2 = np.array([2, 3, 1])
p3 = np.array([3, 4, 1])

def collinearity_3D(p1, p2, p3, epsilon=1e-6): 
    collinear = False
    # TODO: Create the matrix out of three points
    # TODO: Calculate the determinant of the matrix. 
    # TODO: Set collinear to True if the determinant is less than epsilon

    return collinear


def collinearity_2D(p1, p2, p3): 
    collinear = False
    # TODO: Calculate the determinant of the matrix using integer arithmetic 
    # TODO: Set collinear to True if the determinant is equal to zero

    return collinear


import time
t1 = time.time()
collinear = collinearity_3D(p1, p2, p3)
t_3D = time.time() - t1

t1 = time.time()
collinear = collinearity_2D(p1, p2, p3)
t_2D = time.time() - t1
print(t_3D/t_2D)



