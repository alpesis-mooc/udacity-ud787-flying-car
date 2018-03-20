import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from shapely.geometry import Polygon, Point

plt.rcParams['figure.figsize'] = [12, 12]


def extract_polygons(data):
    polygons = []
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        


if __name__ == '__main__':
    filename = 'colliders.csv'
    data = np.loadtxt(filename, delimiter=',', dtype='Float64', skiprows=2)
    print(data)
 
