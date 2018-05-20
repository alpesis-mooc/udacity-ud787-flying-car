from matplotlib import pyplot as plt
import numpy as np


class UniformDistribution:
    
    def __init__(self,a,b):
        self.a = a
        self.b = b
        
    def f(self, x):
        """Probability Density Function (PDF) f(x)"""
        if x < self.a or x > self.b: 
            return 0.0
        return 1.0 / (self.b - self.a)
    
    def F(self,x):
        "Cumulative Density Function (CDF) F(x)"
        if x < self.a: return 0.0
        if x > self.b: return 1.0
        return (x - self.a) / (self.b - self.a)
    
    def p(self, x1, x2):
        """Calculates the probability that x1 < x < x2"""
        return self.F(x2) - self.F(x1)
    
    def plot_f(self, x_min, x_max):
        """Plots the PDF"""
        p = []
        X = np.linspace(x_min, x_max, 1000)
        for x in X:
            p.append(self.f(x))
        plt.plot(X, p)
        
    def plot_F(self, x_min, x_max):
        """Plots the CDF"""
        p = []
        X = np.linspace(x_min, x_max, 1000)
        for x in X:
            p.append(self.F(x))
        plt.plot(X, p)


a, b = -1, 2
uniform = UniformDistribution(a,b)

uniform.plot_f(a-1, b+1)
uniform.plot_F(a-1, b+1)
plt.legend(["PDF of uniform distribution", 
            "CDF of uniform distribution"])
plt.show()
