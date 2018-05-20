import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from simplified_monorotor import Monorotor
import plotting
import testing
import trajectories

class PDController:
    
    def __init__(self, k_p, k_d, m):
        self.k_p = k_p
        self.k_d = k_d
        self.vehicle_mass = m
        self.g = 9.81
    
    def thrust_control(self,
                z_target, 
                z_actual, 
                z_dot_target, 
                z_dot_actual,
                z_dot_dot_ff=0.0): # IGNORE this for now.
        
        err = z_target - z_actual
        err_dot = z_dot_target - z_dot_actual
        
        p_term_thrust = self.k_p * err
        d_term_thrust = self.k_d * err_dot
        
        u_bar = p_term_thrust + d_term_thrust 
        u = self.vehicle_mass * (self.g - u_bar)
        
        return u


if __name__ == '__main__':
    testing.pd_controller_test(PDController, feed_forward=False)

    MASS_ERROR = 1.5
    K_P = 20.0
    K_D = 0.0

    # preparation
    drone = Monorotor()
    perceived_mass = drone.m * MASS_ERROR
    controller = PDController(K_P, K_D, perceived_mass)

    # generate trajectory
    total_time = 3.0
    dt = 0.001
    t=np.linspace(0.0,total_time,int(total_time/dt))
    z_path= -np.ones(t.shape[0])
    z_dot_path = np.zeros(t.shape[0])

    # run simulation
    history = []
    for z_target, z_dot_target in zip(z_path, z_dot_path):
        z_actual = drone.z
        z_dot_actual = drone.z_dot
        u = controller.thrust_control(z_target, z_actual, 
                                  z_dot_target, z_dot_actual)
        drone.thrust = u
        drone.advance_state(dt)
        history.append(drone.X)
    
    # generate plots
    z_actual = [h[0] for h in history]
    plotting.compare_planned_to_actual(z_actual, z_path, t)  
