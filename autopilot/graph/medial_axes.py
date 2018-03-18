import numpy as np
import matplotlib.pyplot as plt
from skimage.util import invert
from skimage.morphology import medial_axis

from grid import create_grid
from planning import a_star

plt.rcParams["figure.figsize"] = [12, 12]


def find_start_goal(skel, start, goal):
    near_start = None
    near_goal = None
    for x in skel:
        print(x)        
        
    return near_start, near_goal


def plot(grid, skeleton, start, goal):
    plt.imshow(grid, cmap="Greys", origin="lower")
    plt.imshow(skeleton, cmap="Greys", origin="lower", alpha=0.7)

    plt.plot(start[1], start[0], 'rx')
    plt.plot(goal[1], goal[0], 'rx')

    plt.xlabel('EAST')
    plt.ylabel('NORTH')

    plt.show()


if __name__ == '__main__':
    filename = "colliders.csv"
    data = np.loadtxt(filename, delimiter=',', dtype='Float64', skiprows=2)
    print(data)

    start = (25, 100)
    goal = (650, 500)
    drone_altitude = 5
    safety_distance = 2
    grid = create_grid(data, drone_altitude, safety_distance)
    skeleton = medial_axis(invert(grid))
    plot(grid, skeleton, start, goal)

    skel_start, skel_goal = find_start_goal(skeleton, start, goal)
    print(start, goal)
    print(skel_start, skel_goal)
