import numpy as np 
from math import sin, cos, tan
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from mpl_toolkits.mplot3d import Axes3D
import random
from calibrating_object import ObjectInThreeD
from misalignment_calibration import flight_path

pylab.rcParams['figure.figsize'] = 10, 10

t, dt, x, x_dot, x_dot_dot, y, y_dot, y_dot_dot, z, z_dot, z_dot_dot, phi, phi_dot, theta, theta_dot = flight_path()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(x, y, z)
plt.title('Calibration path').set_fontsize(20)
ax.set_xlabel('$x$ [$m$]').set_fontsize(20)
ax.set_ylabel('$y$ [$m$]').set_fontsize(20)
ax.set_zlabel('$z$ [$m$]').set_fontsize(20)
plt.legend(['Executed path'],fontsize = 14)

plt.show()

class IMU:
    
    def __init__(self,
                 sigma_a = 0.0001,          # An error of acceleration measurement
                ):
        '''
        Initializing the IMU object with the sigma values associated 
        to the measuring the accelerations in the body frame.
        '''
        self.sigma_a  = sigma_a
        self.g = 9.81 * np.array([[0], [0], [-1]]) # The opposite of the gravity thus directed opposite of the z-axis. 
    
    def rotation_matrix(self,phi,theta):
        '''
        Returns the rotation matrix for the given roll, pitch and yaw angles 
        '''
        
        psi = 0.0 
        r_x = np.array([[1, 0, 0],
                        [0, np.cos(phi), -np.sin(phi)],
                        [0, np.sin(phi), np.cos(phi)]])

        r_y = np.array([[np.cos(theta), 0, np.sin(theta)],
                        [0, 1, 0],
                        [-np.sin(theta), 0, np.cos(theta)]])

        r_z = np.array([[np.cos(psi), -np.sin(psi), 0],
                        [np.sin(psi), np.cos(psi), 0],
                        [0,0,1]])

        r = np.matmul(r_z,np.matmul(r_y,r_x))

        return r 
    
    @property
    def scaling_cross_coupling_matrix(self):
        '''
        scaling cross-coupling matrix which can characterize the accelerometer measurements
        '''

        m = np.array([[0.0, 0.001, 0.001],
                      [0.001, 0.0, 0.001],
                      [0.001, 0.001, 0.0]])

        return m
    
    def accelerometer_measurement(self, actual_a, phi, theta):
        '''
        Simulates the measurements of the accelerations in the body frame 
        based on the actual linear acceleration.
        We kept the accelerometer measurement implementation developed in the previous exercise.
        '''
        
        actual_a = actual_a.reshape(3, 1)
        linear_acc_bodyframe = np.matmul(self.rotation_matrix(phi, theta), actual_a)
        
        gravity_component = np.matmul(self.rotation_matrix(phi, theta), self.g)
        
        error_component = np.random.normal(0.0, self.sigma_a, (3, 1))

        # TODO: calculate the measured acceleration using the scaling cross-coupling matrix
        measured_acceleration = None
    

        return measured_acceleration
   
MisalignedIMU =IMU(sigma_a=0.0001)
CalibrationObject = ObjectInThreeD(dt)
CalibrationObject.X = np.array([x[0], y[0], z[0], theta[0], phi[0], x_dot[0], y_dot[0], z_dot[0]])

state_histroy = CalibrationObject.X 
actual_a = np.vstack((np.vstack((x_dot_dot, y_dot_dot)), z_dot_dot))
measured_acceleration = MisalignedIMU.accelerometer_measurement(actual_a[:,0], phi[0], theta[0])
measured_acceleration_history=measured_acceleration

for i in range(phi.shape[0]):
    measured_acceleration = MisalignedIMU.accelerometer_measurement(actual_a[:,i], phi[i], theta[i])
    p, q = 0, 0 
    
    state_of_drone = CalibrationObject.advance_state(measured_acceleration, p, q, dt)
    state_histroy = np.vstack((state_histroy,state_of_drone))
    measured_acceleration_history = np.hstack((measured_acceleration_history,measured_acceleration))

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(x, y, z,color ='red', marker='.')
ax.plot(state_histroy[:,0], state_histroy[:,1], state_histroy[:,2], color ='blue')
plt.title('Calibration path').set_fontsize(20)
ax.set_xlabel('$x$ [$m$]').set_fontsize(20)
ax.set_ylabel('$y$ [$m$]').set_fontsize(20)
ax.set_zlabel('$z$ [$m$]').set_fontsize(20)
plt.legend(['Executed path','Estimated path'], fontsize = 14)

plt.show()

estimated_scaling_cross_coupling_matrix_and_bias=np.zeros((3,4))
# TODO: Add a row of 1s to the actual acceleration matrix.
acceleration_vectors = None

# TODO: Perform linear regression for each acceleration components. 
for i in range(3):
    estimated_scaling_cross_coupling_matrix_and_bias[i,:] = None
    
print('(I+M) =  \n',estimated_scaling_cross_coupling_matrix_and_bias) 
