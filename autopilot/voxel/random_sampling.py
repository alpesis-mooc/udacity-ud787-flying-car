import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from shapely.geometry import Polygon, Point

from grid import create_grid

plt.rcParams['figure.figsize'] = [12, 12]


def extract_polygons(data):
    polygons = []
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        
        obstacle = [north - d_north, north + d_north, east - d_east, east + d_east]
        corners = [(obstacle[0], obstacle[2]),
                   (obstacle[0], obstacle[3]),
                   (obstacle[1], obstacle[3]),
                   (obstacle[1], obstacle[2])]

        height = alt + d_alt

        p = Polygon(corners)
        polygons.append((p, height))

    return polygons        


def collides(polygons, point):
    for (p, height) in polygons:
        if p.contains(Point(point)) and height >= point[2]:
            return True
    return False


def random_sampling(data, polygons, num_samples, zmin, zmax):
    xmin = np.min(data[:, 0] - data[:, 3])
    xmax = np.max(data[:, 0] + data[:, 3])
    ymin = np.min(data[:, 1] - data[:, 4])
    ymax = np.max(data[:, 1] + data[:, 4])

    print("X")
    print("min = {0}, max = {1}\n".format(xmin, xmax))
    print("Y")
    print("min = {0}, max = {1}\n".format(ymin, ymax))
    print("Z")
    print("min = {0}, max = {1}\n".format(zmin, zmax))

    xvals = np.random.uniform(xmin, ymax, num_samples)
    yvals = np.random.uniform(ymin, ymax, num_samples)
    zvals = np.random.uniform(zmin, zmax, num_samples)
    samples = list(zip(xvals, yvals, zvals))

    t0 = time.time()
    to_keep = []
    points = []
    for point in samples:
        if not collides(polygons, point):
            to_keep.append(point)
    time_taken = time.time() - t0
    print("Time taken {0} seconds...", time_taken)

    return to_keep


def plot(grid, to_keep):
    fig = plt.figure()
    plt.imshow(grid, cmap="Greys", origin="lower")
    
    nmin = np.min(data[:, 0])
    emin = np.min(data[:, 1])

    # draw points
    all_pts = np.array(to_keep)
    north_vals = all_pts[:, 0]
    east_vals = all_pts[:, 1]
    plt.scatter(east_vals - emin, north_vals - nmin, c='red')

    plt.ylabel('NORTH')
    plt.xlabel('EAST')

    plt.show()


if __name__ == '__main__':
    filename = 'colliders.csv'
    data = np.loadtxt(filename, delimiter=',', dtype='Float64', skiprows=2)
    print(data)

    polygons = extract_polygons(data)
    print(len(polygons))
    num_samples = 100
    zmin = 0
    zmax = 10
    to_keep = random_sampling(data, polygons, num_samples, zmin, zmax) 
    print(len(to_keep))

    grid = create_grid(data, zmax, 1)
    plot(grid, to_keep) 
