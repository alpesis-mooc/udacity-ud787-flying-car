%matplotlib inline 
%config InlineBackend.figure_format = 'retina'

%load_ext autoreload
%autoreload 2

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from simplified_monorotor import Monorotor
import plotting
import testing
import trajectories

pylab.rcParams['figure.figsize'] = 10,10

class PIDController:
    
    def __init__(self, k_p, k_d, k_i, m):
        self.k_p = k_p
        self.k_d = k_d
        self.k_i = k_i
        self.vehicle_mass = m
        self.g = 9.81
        self.integrated_error = 0.0
        
    def thrust_control(self,
                z_target, 
                z_actual, 
                z_dot_target, 
                z_dot_actual,
                dt=0.1,
                z_dot_dot_ff=0.0):
        #
        # TODO
        #   implement a PID controller according to the math
        #   shown above. Remember that the integral accumulates
        #   error by adding "rectangles" of error with area
        #   equal to err * dt
        #
        #   don't forget to return a thrust command
        #
    
        return u
    
testing.pid_controller_test(PIDController)

MASS_ERROR = 1.5
K_P = 20.0
K_D = 10.0
K_I = 0.0 # TODO - increase to 0.5, 1, 2, etc...

# preparation 
drone = Monorotor()
perceived_mass = drone.m * MASS_ERROR
controller    = PIDController(K_P, K_D, K_I, perceived_mass)

t, z_path, z_dot_path = trajectories.step(duration=10.0)

dt = t[1] - t[0]

# run simulation
history = []
for z_target, z_dot_target in zip(z_path, z_dot_path):
    z_actual = drone.z
    z_dot_actual = drone.z_dot

    u = controller.thrust_control(z_target, z_actual, 
                                  z_dot_target, z_dot_actual,
                                  dt)
    
    drone.thrust = u
    drone.advance_state(dt)
    history.append(drone.X)
    
# generate plots
z_actual = [h[0] for h in history]
plotting.compare_planned_to_actual(z_actual, z_path, t)   


