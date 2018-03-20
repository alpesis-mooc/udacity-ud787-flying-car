# OK this might look a little ugly but...
# Just waiting on eng to install these in the backend!
import sys
!{sys.executable} -m pip install -Iv networkx==2.0
import pkg_resources
pkg_resources.require("networkx==2.0")
import networkx as nx


import numpy as np
import matplotlib.pyplot as plt
from grid import create_grid_and_edges
import numpy.linalg as LA
%matplotlib inline 

plt.rcParams['figure.figsize'] = 12, 12

nx.__version__

plt.rcParams['figure.figsize'] = 12, 12

# This is the same obstacle data from the previous lesson.
filename = 'colliders.csv'
data = np.loadtxt(filename, delimiter=',', dtype='Float64', skiprows=2)
print(data)

start_ne = (25,  100)
goal_ne = (750., 370.)

# Static drone altitude (metres)
drone_altitude = 5

# This is now the routine using Voronoi
grid, edges = create_grid_and_edges(data, drone_altitude)
print(len(edges))

# equivalent to
# plt.imshow(np.flip(grid, 0))
plt.imshow(grid, origin='lower', cmap='Greys') 

for e in edges:
    p1 = e[0]
    p2 = e[1]
    plt.plot([p1[1], p2[1]], [p1[0], p2[0]], 'b-')

    
plt.plot(start_ne[1], start_ne[0], 'rx')
plt.plot(goal_ne[1], goal_ne[0], 'rx')

plt.xlabel('EAST')
plt.ylabel('NORTH')
plt.show()

# TODO: create the graph with the weight of the edges
# set to the Euclidean distance between the points


from queue import PriorityQueue

def heuristic(n1, n2):
    #TODO: define a heuristic
    return 0

###### THIS IS YOUR OLD GRID-BASED A* IMPLEMENTATION #######
###### With a few minor modifications it can work with graphs! ####
#TODO: modify A* to work with a graph
def a_star(graph, heuristic, start, goal):
    
    path = []
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        current_cost = item[0]
        current_node = item[1]

        if current_node == goal:        
            print('Found a path.')
            found = True
            break

        else:
            for action in valid_actions(grid, current_node):
                # get the tuple representation
                da = action.delta
                cost = action.cost
                next_node = (current_node[0] + da[0], current_node[1] + da[1])
                new_cost = current_cost + cost + h(next_node, goal)
                
                if next_node not in visited:                
                    visited.add(next_node)               
                    queue.put((new_cost, next_node))
                    
                    branch[next_node] = (new_cost, current_node, action)
             
    path = []
    path_cost = 0
    if found:
        
        # retrace steps
        path = []
        n = goal
        path_cost = branch[n][0]
        while branch[n][1] != start:
            path.append(branch[n][2])
            n = branch[n][1]
        path.append(branch[n][2])
            
    return path[::-1], path_cost
