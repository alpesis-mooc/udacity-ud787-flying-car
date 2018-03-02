"""
5.12  Quaternion

- Solution: https://view3679c992.udacity-student-workspaces.com/notebooks/Quaternions-Solution.ipynb
"""

import numpy as np

def euler_to_quaternion(angles):
    roll = angles[0]
    pitch = angles[1]
    yaw = angles[2]
    
    # TODO: complete the conversion
    # and return a numpy array of
    # 4 elements representing a quaternion [a, b, c, d]

def quaternion_to_euler(quaternion):
    a = quaternion[0]
    b = quaternion[1]
    c = quaternion[2]
    d = quaternion[3]
    
    # TODO: complete the conversion
    # and return a numpy array of
    # 3 element representing the euler angles [roll, pitch, yaw]

euler = np.array([np.deg2rad(90), np.deg2rad(30), np.deg2rad(0)])

q = euler_to_quaternion(euler) # should be [ 0.683  0.683  0.183 -0.183]
print(q)

assert np.array_equal(euler, quaternion_to_euler(q))
