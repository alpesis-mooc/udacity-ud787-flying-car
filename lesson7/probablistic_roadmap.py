import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString
from queue import PriorityQueue

# Again, ugly but we need the latest version of networkx!
import sys
!{sys.executable} -m pip install -Iv networkx==2.1
import pkg_resources
pkg_resources.require("networkx==2.1")
import networkx as nx


plt.rcParams['figure.figsize'] = 14, 14

# This is the same obstacle data from the previous lesson.
filename = 'colliders.csv'
data = np.loadtxt(filename, delimiter=',', dtype='Float64', skiprows=2)
print(data)

# TODO: sample points randomly
# then use KDTree to find nearest neighbor polygon
# and test for collision

# TODO: connect nodes
# Suggested method
    # 1) cast nodes into a graph called "g" using networkx
    # 2) write a method "can_connect()" that:
        # casts two points as a shapely LineString() object
        # tests for collision with a shapely Polygon() object
        # returns True if connection is possible, False otherwise
    # 3) write a method "create_graph()" that:
        # defines a networkx graph as g = Graph()
        # defines a tree = KDTree(nodes)
        # test for connectivity between each node and 
            # k of it's nearest neighbors
        # if nodes are connectable, add an edge to graph
    # Iterate through all candidate nodes!

# Create a grid map of the world
from grid import create_grid
# This will create a grid map at 1 m above ground level
grid = create_grid(data, 1, 1)

fig = plt.figure()

plt.imshow(grid, cmap='Greys', origin='lower')

nmin = np.min(data[:, 0])
emin = np.min(data[:, 1])

# If you have a graph called "g" these plots should work
# Draw edges
#for (n1, n2) in g.edges:
#    plt.plot([n1[1] - emin, n2[1] - emin], [n1[0] - nmin, n2[0] - nmin], 'black' , alpha=0.5)

# Draw all nodes connected or not in blue
#for n1 in nodes:
#    plt.scatter(n1[1] - emin, n1[0] - nmin, c='blue')
    
# Draw connected nodes in red
#for n1 in g.nodes:
#    plt.scatter(n1[1] - emin, n1[0] - nmin, c='red')
    


plt.xlabel('NORTH')
plt.ylabel('EAST')

plt.show()

def heuristic(n1, n2):
    # TODO: complete
 
def a_star(graph, heuristic, start, goal):
    """Modified A* to work with NetworkX graphs."""
    
    # TODO: complete


fig = plt.figure()

# draw nodes
for n1 in g.nodes:
    plt.scatter(n1[1], n1[0], c='red')
    
# draw edges
for (n1, n2) in g.edges:
    plt.plot([n1[1], n2[1]], [n1[0], n2[0]], 'black')
    
# TODO: add code to visualize the path

nmin = 0
nmax = 0
for n1 in g.nodes:
    nmin = min(n1[1], nmin)
    nmax = max(n1[1], nmax)
ax.set_xlim(nmax, nmin)

plt.xlabel('NORTH')
plt.ylabel('EAST')

plt.show()

 
