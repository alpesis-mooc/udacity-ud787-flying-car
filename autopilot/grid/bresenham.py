import numpy as np
#from bresenham import bresenham
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] == [12, 12]


def bresenham_raw(p1, p2):
    """
    ref: https://en.wikipedia.org/wiki/Bresenham's_line_algorithm
    Yield integer coordinates on the line from (x1, y1) to (x2, y2).
    Conditions:
    - x1 < x2
    - y1 < y2
    """

    x1, y1 = p1
    x2, y2 = p2
    cells = []

    dx = x2 - x1
    dy = y2 - y1
    distance = 2 * dy - dx
    y = y1
    for x in range(x1, x2+1):
        cells.append((x, y))
        if distance > 0:
            y = y + 1
            distance = distance - 2 * dx
        distance = distance + 2 * dy

    return np.array(cells)


def plot(p1, p2, cells):
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]])

    for q in cells:
        plt.plot([q[0], q[0] + 1], [q[1], q[1]], 'k')
        plt.plot([q[0], q[0] + 1], [q[1] + 1, q[1]], 'k')
        plt.plot([q[0], q[0]], [q[1], q[1] + 1], 'k')
        plt.plot([q[0] + 1, q[0] + 1], [q[1], q[1] + 1], 'k')

    plt.grid()
    plt.axis('equal')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Integer based Bresenham algorithm")
    plt.show()


if __name__ == '__main__':

    p1 = (0, 0)
    p2 = (7, 5)

    cells = bresenham_raw(p1, p2)
    print(cells)
    plot(p1, p2, cells)

    #line = (0, 0, 7, 5)
    #cells = list(bresenham(line[0], line[1], line[2], line[3]))
    #print(cells)
    #plot(p1, p2, cells)
