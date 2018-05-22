import numpy as np 
import math
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from ipywidgets import interactive
from CoaxialDrone import CoaxialCopter
from PIDcontroller import PIDController_with_ff
from PathGeneration import flight_path
from DronewithPIDControllerParameters import DronewithPID

pylab.rcParams['figure.figsize'] = 10, 10

class IMU:
    def __init__(self,
                 z_hat, # initial estimated value
                 alpha  # alpha value how fast to update the estimated value
                ):
        '''
        Initializing the IMU object with initial altitude estimation and the alpha value for the exponential averaging. 
        '''
        self.z_hat = z_hat
        self.alpha = alpha
        
    def measure(self, 
                z,                 # True altitude
                sigma = 0.01       # Error sigma value 
               ):
        '''
        Simulating the sensor measurement
        '''
        # TODO: Simulate the measurement of the altitude by adding the error associated
        # with the measurement to the true method.
        return z + np.random.normal(0.0, sigma)


    def estimate(self, z_t):
        '''
        Estimates the drone altitude using the weighted average method. 
        '''
        # TODO: Estimate the drone altitude using the weighted average method. 
        self.z_hat = (1 - self.alpha) * z_t + self.alpha * self.z_hat
        return self.z_hat

total_time = 10.0   # Total Flight time 
dt = 0.01           # A time interval between measurements 

t, z_path, z_dot_path, z_dot_dot_path =  flight_path(total_time, dt,'constant' )

FlyingDrone = DronewithPID(z_path, z_dot_path, z_dot_dot_path, t, dt, IMU)

interactive_plot = interactive(FlyingDrone.PID_controler_with_measured_values, 
                               k_p=(5.0, 35.0, 1),
                               k_d=(0.0, 10, 0.5), 
                               k_i=(0.0, 10, 0.5), 
                               mass_err =(0.7, 1.31, 0.01),
                               sigma=(0.0, 0.1, 0.001))
output = interactive_plot.children[-1]
output.layout.height = '800px'
interactive_plot

interactive_plot = interactive(FlyingDrone.PID_controler_with_estimated_values, 
                               k_p=(5.0, 35.0, 1),
                               k_d=(0.0, 10, 0.5), 
                               k_i=(0.0, 10, 0.5), 
                               mass_err =(0.7, 1.31, 0.01),
                               sigma = (0.0, 0.1, 0.001),
                               alpha = (0.51, 0.99, 0.01))
output = interactive_plot.children[-1]
output.layout.height = '800px'
interactive_plot
