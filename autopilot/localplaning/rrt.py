import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree


plt.rcParams['figure.figsize'] = 12, 12


class RRT:
    def __init__(self, x_init):
        # A tree is a special case of a graph with
        # directed edges and only one path to any vertex
        self.tree = nx.DiGraph()
        self.tree.add_node(x_init)


    def add_vertex(self, x_new):
        self.tree.add_node(tuple(x_init))


    def add_edge(self, x_near, x_new, u):
        self.tree.add_edge(tuple(x_near), tuple(x_new), orientation=u)


    @property
    def vertices(self):
        return self.tree.nodes


    @property
    def edges(self):
        return self.tree.edges



def create_grid():
    grid = np.zeros((100, 100))
    # build same obstacles
    grid[10:20, 10:20] = 1
    grid[63:80, 10:20] = 1
    grid[43:60, 30:40] = 1
    grid[71:86, 38:50] = 1
    grid[10:20, 55:67] = 1
    grid[80:90, 80:90] = 1
    grid[75:90, 80:90] = 1
    grid[30:40, 60:82] = 1
    return grid


def sample_state(grid):
    x = np.random.uniform(0, grid.shape[0])
    y = np.random.uniform(0, grid.shape[1])
    return (x, y)


def nearest_neighbor(x_rand, rrt):
    closest_dist = 100000
    closest_vertex = None
    x_rand = np.array(x_rand)
    
    for v in rrt.vertices:
        d = np.linalg.norm(x_rand - np.array(v[:2]))
        if d < closest_dist:
            closest_dist = d
            closest_vertex = v
    return closest_vertex


def select_input(x_rand, x_near):
    return np.arctan2(x_rand[1] - x_near[1], x_rand[0] - x_near[0])


def new_state(x_near, u, dt):
    nx = x_near[0] + np.cos(u)*dt
    ny = x_near[1] + np.sin(u)*dt
    return [nx, ny]


def generate_rrt(grid, x_init, num_vertices, dt):
    rrt = RRT(x_init)
    for _ in range(num_vertices):
        x_rand = sample_state(grid)
        # sample states until a free state is found
        while grid[int(x_rand[0]), int(x_rand[1])] == 1:
            x_rand = sample_state(grid)

        x_near = nearest_neighbor(x_rand, rrt)
        u = select_input(x_rand, x_near)
        x_new = new_state(x_near, u, dt)

        if grid[int(x_new[0]), int(x_new[1])] == 0:
            # the orientation u will be added as metadata to the edge
            rrt.add_edge(x_near, x_new, u)
    return rrt


if __name__ == '__main__':

    grid = create_grid()
    plt.imshow(grid, cmap="Greys", origin='upper')
    plt.show()

    num_vertices = 300
    dt = 1
    x_init = (50, 50)
    rrt = generate_rrt(grid, x_init, num_vertices, dt)

    plt.imshow(grid, cmap='Greys', origin='lower')
    plt.plot(x_init[0], x_init[1], 'ro')
    for (v1, v2) in rrt.edges:
        plt.plot([v1[1], v2[1]], [v1[0], v2[0]], 'y-')
    plt.show()
