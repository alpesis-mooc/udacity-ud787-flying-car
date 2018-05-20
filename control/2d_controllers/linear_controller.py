import numpy as np
from matplotlib import pyplot as plt

def linearize(function, operating_point):
    """
    Returns a linearized version of the function that will
    be valid within some window around the operating point.
    """
    f_at_op = function(operating_point)
    slope_at_op = get_function_slope_at_point(function, operating_point)
    
    def linearized(x):
        return f_at_op + slope_at_op * (x - operating_point)
    
    return linearized


def get_function_slope_at_point(f, x):
    """
    Calculates the slope of a function f at the value x.
    """
    dx = 0.00001
    vertical_change = f(x+dx) - f(x-dx)
    horizontal_change = 2*dx
    return vertical_change / horizontal_change

def plot_compare(f1, f2, centered_at=0.0, zoom=1.0):
    """
    Compares the graphs of two functions. Increase zoom 
    to zoom in.
    """
    window_width = 8 / zoom
    left = centered_at - window_width/2
    right =centered_at + window_width/2
    t = np.linspace(left, right, 1000)
    plt.plot(t, f1(t))
    plt.plot(t, f2(t))
    plt.axis('equal')
    plt.scatter([centered_at], [f1(centered_at)])
    plt.legend(["$f(x)$", "$f_{lin}(x)$"])
    plt.show()
    
    print("Comparing", f1.__name__, "to its linear approximation")
    print("at x =", centered_at, "from", left, "to", right)
    print("Increase the zoom to see a better approximation")


def x_square(n): 
    return n * n


def polynomial(x):
    return -2 + 4*x - 2*x**2 - x**3



def test1():
    f    = np.sin
    op   = 0.0 
    zoom = 1.0

    f_lin = linearize(f, op)
    plot_compare(f, f_lin, centered_at=op, zoom=zoom)


def test2():
    f    = np.cos
    op   = 0.0
    zoom = 1.0

    f_lin = linearize(f, op)
    plot_compare(f, f_lin, centered_at=op, zoom=zoom)


def test3():
    f    = x_square
    op   = 1.0
    zoom = 1.0

    f_lin = linearize(f, op)
    plot_compare(f, f_lin, centered_at=op, zoom=zoom)


def test4():
    f    = polynomial
    op   = 0.5
    zoom = 1.0

    f_lin = linearize(f, op)
    plot_compare(f, f_lin, centered_at=op, zoom=zoom)


if __name__ == '__main__':
    test1()
    test2()
    test3()
    test4()
