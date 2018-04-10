import numpy as np


def euler_to_quaternion(angles):
    """
    ref: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    """
    roll = angles[0]
    pitch = angles[1]
    yaw = angles[2]

    sp = np.sin(pitch * 0.5)
    cp = np.cos(pitch * 0.5)
    sr = np.sin(roll * 0.5)
    cr = np.cos(roll * 0.5)
    sy = np.sin(yaw * 0.5)
    cy = np.cos(yaw * 0.5)

    a = cr * cp * cy + sr * sp * sy
    b = sr * cp * cy - cr * sp * sy
    c = cr * sp * cy + sr * cp * sy
    d = cr * cp * sy - sr * sp * cy
    
    return np.array([a, b, c, d]) 


def quaternion_to_euler(quaternion):
    """
    ref: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    """
    a = quaternion[0]
    b = quaternion[1]
    c = quaternion[2]
    d = quaternion[3]

    roll = np.arctan2(2.0 * (a * b + c * d), 1.0 - 2.0 * (b**2 + c**2))
    pitch = np.arcsin(2.0 * (a * c - d * b))
    yaw = np.arctan2(2.0 * (a * d + b * c), 1.0 - 2.0 * (c**2 + d**2))
    
    return np.array([roll, pitch, yaw])


if __name__ == '__main__':
    euler = np.array([np.deg2rad(90),
                      np.deg2rad(30),
                      np.deg2rad(0)])

    q = euler_to_quaternion(euler)
    print(q)
    e = quaternion_to_euler(q)
    print(e)
    assert np.array_equal(euler, e)
