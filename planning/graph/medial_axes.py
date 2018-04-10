import numpy as np
import matplotlib.pyplot as plt
from skimage.util import invert
from skimage.morphology import medial_axis

from grid import create_grid
from planning import a_star

plt.rcParams["figure.figsize"] = [12, 12]


def distance(x, y):
    return np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)


def heuristic(position, goal_position):
    return np.abs(position[0] - goal_position[0]) + np.abs(position[1] - goal_position[1])


def find_start_goal(skel, start, goal):
    skel_true = np.nonzero(skel)
    skel_trans = np.transpose(skel_true)
    distance_start_min = distance(skel_trans[0], start)
    distance_goal_min = distance(skel_trans[0], goal)
    near_start = (skel_trans[0][0], skel_trans[0][1])
    near_goal = (skel_trans[0][0], skel_trans[0][1])
    for x in skel_trans[1:]:
        this_distance_start = distance(x, start)
        this_distance_goal = distance(x, goal)
        if this_distance_start < distance_start_min:
            distance_start_min = this_distance_start
            near_start = (x[0], x[1])
        if this_distance_goal < distance_goal_min:
            distance_goal_min = this_distance_goal
            near_goal = (x[0], x[1])
    return near_start, near_goal


def plot(grid, skeleton, start, goal, skel_start, skel_goal):
    plt.imshow(grid, cmap="Greys", origin="lower")
    plt.imshow(skeleton, cmap="Greys", origin="lower", alpha=0.7)

    plt.plot(start[1], start[0], 'rx')
    plt.plot(goal[1], goal[0], 'rx')
    plt.plot(skel_start[1], skel_start[0], 'x')
    plt.plot(skel_goal[1], skel_goal[0], 'x')

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

    skel_start, skel_goal = find_start_goal(skeleton, start, goal)
    print(start, goal)
    print(skel_start, skel_goal)
    plot(grid, skeleton, start, goal, skel_start, skel_goal)
