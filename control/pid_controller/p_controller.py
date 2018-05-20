import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from simplified_monorotor import Monorotor
import plotting
import testing
import trajectories


MASS_ERROR = 1.01
K_P = 2.0

class PController:

    def __init__(self, k_p, m):
        self.k_p = k_p
        self.vehicle_mass = m
        self.g = 9.81


    def thrust_control(self, z_target, z_actual):
        err = z_target - z_actual
        u_bar = self.k_p * err
        u = self.vehicle_mass * (self.g - u_bar)
        return u


if __name__ == '__main__':
    testing.p_controller_test(PController)

    # preparation
    drone = Monorotor()
    perceived_mass = drone.m * MASS_ERROR
    controller = PController(K_P, perceived_mass)

    # generate trajectory
    total_time = 10.0
    dt = 0.001
    t = np.linspace(0.0, total_time, int(total_time/dt))
    z_path = -np.ones(t.shape[0])

    # run simulation
    history = []
    for z_target in z_path:
        z_actual = drone.z
        u = controller.thrust_control(z_target, z_actual)
        drone.thrust = u
        drone.advance_state(dt)
        history.append(drone.X)

    # generate plots
    z_actual = [h[0] for h in history]
    plotting.compare_planned_to_actual(z_actual, z_path, t)
