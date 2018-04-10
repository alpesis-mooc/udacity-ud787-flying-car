import time
import numpy as np


def collinearity_3D(p1, p2, p3, epsilon=1e-6):
    """
    area = x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)
    """
    collinear = False
    mat = np.vstack((p1, p2, p3))
    det = np.linalg.det(mat)
    if np.abs(det) < epsilon: collinear = True
    return collinear


def collinearity_2D(p1, p2, p3):
    collinear = False
    area = p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])
    if area == 0: collinear = True
    return collinear


if __name__ == '__main__':
    p1 = np.array([1, 2, 1])
    p2 = np.array([2, 3, 1])
    p3 = np.array([3, 4, 1])

    t1 = time.time()
    collinear = collinearity_3D(p1, p2, p3)
    t_3D = time.time() - t1

    t1 = time.time()
    collinear = collinearity_2D(p1, p2, p3)
    t_2D = time.time() - t1
    print(t_3D / t_2D)
