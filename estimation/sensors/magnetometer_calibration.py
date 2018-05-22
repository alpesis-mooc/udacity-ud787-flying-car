import numpy as np 
import math
from math import sin, cos, tan
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from mpl_toolkits.mplot3d import Axes3D
import random
from magnetic_field_measurements import measured_field

pylab.rcParams['figure.figsize'] = 10, 10

orientation, field_strength = measured_field()

plt.scatter(field_strength[0,:], field_strength[1,:])
plt.axis('equal')
plt.grid()
plt.title('Magnetic field measurement').set_fontsize(20)
plt.xlabel('$m_x$').set_fontsize(20)
plt.ylabel('$m_y$').set_fontsize(20)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.show()

attitude = np.vstack([orientation, np.ones(orientation.shape[1])])
transformation_matrix = np.zeros((2,3))
for i in range(2):
    # TODO: calculate the transformation_matrix containing bias and scailing and cross-correlation elements 
    transformation_matrix[i,:] = np.linalg.lstsq(attitude.T,field_strength[i,:])[0]

# TODO: Normalize the measured magnetic field. 
m = np.matmul(np.linalg.inv(transformation_matrix[:, :2]),
              (field_strength - np.reshape(transformation_matrix[:, -1],(2, 1))))


plt.scatter(field_strength[0,:], field_strength[1,:])
plt.scatter(m[0,:], m[1,:])
plt.legend(['Measured','Calibrated'],fontsize = 14)
plt.axis('equal')
plt.grid()
plt.title('Magnetic field measurement').set_fontsize(20)
plt.xlabel('$m_x$').set_fontsize(20)
plt.ylabel('$m_y$').set_fontsize(20)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.show()

delta_psi_1 = np.arctan2(transformation_matrix[1,0],transformation_matrix[0,0])/np.pi*180
print('First estimation of the relative yaw angle is ',delta_psi_1)

delta_psi_2 = np.arctan2(-transformation_matrix[0,1],transformation_matrix[1,1])/np.pi*180
print('Second estimation of the relative yaw angle is ',delta_psi_2)
estimated_delta_psi = (delta_psi_1+delta_psi_2)/2
print('Average relative yaw angle is= ',estimated_delta_psi)

sample_number = int(np.random.uniform(0,field_strength.shape[1]))

sample_measurement = field_strength[:,sample_number]


normalized_measurement = np.matmul(np.linalg.inv(transformation_matrix[:, :2]),
             (np.reshape(sample_measurement,(2,1)) - np.reshape(transformation_matrix[:, -1],(2, 1))))


# TODO: Calculate the yaw angle relative to the pre-assumed zero direction and add the correction factor 
yaw_relative_to_introduced_zero = np.arctan2(-normalized_measurement[1],normalized_measurement[0])/np.pi*180
yaw_relative_to_magnetic_north= yaw_relative_to_introduced_zero + estimated_delta_psi

print('Yaw relative to the pre-assumed zero  =',yaw_relative_to_introduced_zero)
print('Yaw relative to the magnetic north = ',yaw_relative_to_magnetic_north)


plt.scatter(field_strength[0,:], field_strength[1,:])
plt.scatter(m[0,:], m[1,:])
plt.scatter(sample_measurement[0], sample_measurement[1],s=300,color='red',marker = 'o')
plt.scatter(normalized_measurement[0], normalized_measurement[1],s=300,color='green',marker = 'o')
plt.legend(['Measured','Calibrated','Sample measurement','Normalized sample'],fontsize = 14)
plt.title('Magnetic field measurement').set_fontsize(20)
plt.xlabel('$m_x$').set_fontsize(20)
plt.ylabel('$m_y$').set_fontsize(20)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.axis('equal')
plt.grid()
plt.show()
