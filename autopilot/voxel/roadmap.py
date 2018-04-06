import time
from queue import PriorityQueue

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString
from sklearn.neighbors import KDTree

from grid import create_grid
from random_sampling import extract_polygons, random_sampling

plt.rcParams['figure.figsize'] = [14, 14]


def can_connect(polygons, node1, node2):
    l = LineString([node1, node2])
    for (p, height) in polygons:
        if p.crosses(l) and height >= min(node1[2], node2[2]):
            return False
    return True


def create_graph(polygons, nodes, k):
    g = nx.Graph()
    tree = KDTree(nodes)
    for node1 in nodes:
        # try to connect to k nearest nodes
        idxs = tree.query([node1], k, return_distance=False)[0]
        for idx in idxs:
            node2 = nodes[idx]
            if node2 == node1:
                continue
            if can_connect(polygons, node1, node2):
                g.add_edge(node1, node2, weight=1)
    return g


def heuristic(node1, node2):
    # TODO: finish
    return np.linalg.norm(np.array(node2) - np.array(node1))


def a_star(graph, heuristic, start, goal):
    """Modified A* to work with NetworkX graphs."""
    
    # TODO: complete

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
            for next_node in graph[current_node]:
                cost = graph.edges[current_node, next_node]['weight']
                new_cost = current_cost + cost + heuristic(next_node, goal)
                
                if next_node not in visited:                
                    visited.add(next_node)               
                    queue.put((new_cost, next_node))
                    
                    branch[next_node] = (new_cost, current_node)
             
    path = []
    path_cost = 0
    if found:
        
        # retrace steps
        path = []
        n = goal
        path_cost = branch[n][0]
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
            
    return path[::-1], path_cost


def plot(data, nodes, g):
    fig = plt.figure()
    plt.imshow(grid, cmap='Greys', origin='lower')

    nmin = np.min(data[:, 0])
    emin = np.min(data[:, 1])

    # draw edges
    for (n1, n2) in g.edges:
        plt.plot([n1[1] - emin, n2[1] - emin], [n1[0] - nmin, n2[0] - nmin], 'black' , alpha=0.5)

    # draw all nodes
    for n1 in nodes:
        plt.scatter(n1[1] - emin, n1[0] - nmin, c='blue')
    
    # draw connected nodes
    for n1 in g.nodes:
        plt.scatter(n1[1] - emin, n1[0] - nmin, c='red')

    plt.xlabel('NORTH')
    plt.ylabel('EAST')

    plt.show()

def plot_path(data, g, path):
    fig = plt.figure()

    plt.imshow(grid, cmap='Greys', origin='lower')

    nmin = np.min(data[:, 0])
    emin = np.min(data[:, 1])

    # draw nodes
    for n1 in g.nodes:
        plt.scatter(n1[1] - emin, n1[0] - nmin, c='red')
    
    # draw edges
    for (n1, n2) in g.edges:
        plt.plot([n1[1] - emin, n2[1] - emin], [n1[0] - nmin, n2[0] - nmin], 'black')
    
    # TODO: add code to visualize the path
    path_pairs = zip(path[:-1], path[1:])
    for (n1, n2) in path_pairs:
        plt.plot([n1[1] - emin, n2[1] - emin], [n1[0] - nmin, n2[0] - nmin], 'green')


    plt.xlabel('NORTH')
    plt.ylabel('EAST')

    plt.show()


if __name__ == '__main__':

    filename = 'colliders.csv'
    data = np.loadtxt(filename, delimiter=',', dtype='Float64', skiprows=2)
    print(data)

    num_samples = 300
    zmin = 0
    zmax = 10
    polygons = extract_polygons(data)
    nodes = random_sampling(data, polygons, num_samples, zmin, zmax) 
    print(len(nodes))

    t0 = time.time()
    g = create_graph(polygons, nodes, 10)
    print('graph took {0} seconds to build'.format(time.time()-t0))
    print("Number of edges", len(g.edges))

    grid = create_grid(data, zmax, 1)
    plot(data, nodes, g)

    start = list(g.nodes)[0]
    k = np.random.randint(len(g.nodes))
    print(k, len(g.nodes))
    goal = list(g.nodes)[k]

    path, cost = a_star(g, heuristic, start, goal)
    print(len(path), path)
    path_pairs = zip(path[:-1], path[1:])
    for (n1, n2) in path_pairs:
        print(n1, n2)
    plot_path(data, g, path)
