import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

pylab.rcParams['figure.figsize'] = 10, 10


class CoaxialCopter:

    def __init__(self,
                 k_f = 0.1,  # value of the thrust coefficient
                 k_m = 0.1,  # value of the angular torque coefficient
                 m = 0.5,    # mass of the vehicle
                 i_z = 0.2,  # moment of inertia around the z-axis
                ):

        self.k_f = k_f
        self.k_m = k_m
        self.m = m
        self.i_z = i_z

        self.omega_1 = 0.0
        self.omega_2 = 0.0
        self.g = 9.81


    @property
    def z_dot_dot(self):
        """Calculates current vertical acceleration."""

        force_1 = self.k_f * self.omega_1**2
        force_2 = self.k_f * self.omega_2**2
        force_g = self.m * self.g
        force_total = - force_1 - force_2 + force_g

        vertical_acceleration = force_total / self.m

        return vertical_acceleration


    @property
    def psi_dot_dot(self):
        """Calculates current rotational acceleration."""

        # T = k_m * w^2
        cw_torque = self.k_m * self.omega_1**2
        ccw_torque = self.k_m * self.omega_2**2
        net_torque = cw_torque - ccw_torque

        # acc = T / Iz
        angular_acceleration = net_torque / self.i_z

        return angular_acceleration


    def set_rotors_angular_velocities(self, linear_acc, angular_acc):
        """
        Sets the turn rates for the rotors so that the drone achieves
        the desired linear_acc and angular_acc.
        """

        vertical = self.m * (-linear_acc + self.g) / (2 * self.k_f)
        angular = self.i_z * angular_acc / (2 * self.k_m)

        self.omega_1 = - math.sqrt(vertical + angular)
        self.omega_2 = math.sqrt(vertical - angular)

        return self.omega_1, self.omega_2



if __name__ == '__main__':

    bi = CoaxialCopter()
    stable_omega_1, stable_omega_2 = bi.set_rotors_angular_velocities(0.0, 0.0)
    print('Drone achieves stable hover with angular velocity of %5.2f' % stable_omega_1,
          'for the first propeller and %5.2f' % stable_omega_2,
          'for the second propeller.')

    bi.omega_1 = stable_omega_1 * math.sqrt(1.1)
    bi.omega_2 = stable_omega_2 * math.sqrt(1.1)
    vertical_acceleration = bi.z_dot_dot
    print('Increase by %5.2f' % math.sqrt(1.1),
          'of the propeller angular velocity will result in',
          '%5.2f' % vertical_acceleration,
          'm/(s*s) vertical acceleration.' )

    bi.omega_1 = stable_omega_1 * math.sqrt(1.1)
    bi.omega_2 = stable_omega_2 * math.sqrt(0.9)
    ang_acceleration = bi.psi_dot_dot
    print('Increase in %5.2f'%math.sqrt(1.1),' of the angular velocity for the first propellr and',
          ' decrease of the angular velocity of the second propellr by %f.2f'%math.sqrt(0.9),' will result in',
          '%5.2f'%ang_acceleration, 'rad/(s*s) angular acceleration.' )
