import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from voronoi import create_grid_and_edges

plt.rcParams['figure.figsize'] = [12, 12]


def plot(grid, edges):
    plt.imshow(grid, origin='lower', cmap='Greys')
    for e in edges:
        p1 = e[0]
        p2 = e[1]
        plt.plot([p1[1], p2[1]], [p1[0], p2[0]], 'b-')
    plt.plot(start[1], start[0], 'rx')
    plt.plot(goal[1], goal[0], 'rx')

    plt.xlabel('EAST')
    plt.ylabel('NORTH')
    plt.show()


if __name__ == '__main__':
    filename = 'colliders.csv'
    data = np.loadtxt(filename, delimiter=',', dtype='Float64', skiprows=2)
    print(data)

    start = (25, 100)
    goal = (750., 370.)
    drone_altitude = 5
    grid, edges = create_grid_and_edges(data, drone_altitude)
    print(len(edges))
    plot(grid, edges) 
