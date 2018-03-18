import numpy as np
from enum import Enum

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams["figure.figsize"] = [12, 12]
np.set_printoptions(precision=3, suppress=True)


class Rotation(Enum):
    ROLL = 0
    PITCH = 1
    YAW = 2


class EulerRotation:

    def __init__(self, rotations):
        """
        rotations:  [(Rotation.ROLL, 45), (Rotation.YAW, 32), (Rotation.PITCH, 55)]
        """
        self._rotations = rotations
        self._rotation_map = {Rotation.ROLL: self.roll,
                              Rotation.PITCH: self.pitch,
                              Rotation.YAW: self.yaw}


    def roll(self, phi):
        return np.array([[1, 0, 0],
                         [0, np.cos(phi), -np.sin(phi)],
                         [0, np.sin(phi), np.cos(phi)]])

    def pitch(self, theta):
        return np.array([[np.cos(theta), 0, np.sin(theta)],
                         [0, 1, 0],
                         [-np.sin(theta), 0, np.cos(theta)]])


    def yaw(self, psi):
        return np.array([[np.cos(psi), -np.sin(psi), 0],
                         [np.sin(psi), np.cos(psi), 0],
                         [0, 0, 1]])


    def rotate(self):
        """
        t = r(roll) * r(pitch) * r(yaw)
        """
        t = np.eye(3)
        for r in self._rotations:
            angle = r[1] * np.pi / 180
            t = np.dot(t, self._rotation_map[r[0]](angle))
        return t


def visualize(v, rv1, rv2, rv3):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    # axes
    ax.quiver(0, 0, 0, 1.5, 0, 0, color='black', arrow_length_ratio=0.15)
    ax.quiver(0, 0, 0, 0, 1.5, 0, color='black', arrow_length_ratio=0.15)
    ax.quiver(0, 0, 0, 0, 0, 1.5, color='black', arrow_length_ratio=0.15)

    # original vector
    ax.quiver(0, 0, 0, v[0], v[1], v[2], color='blue', arrow_length_ratio=0.15)

    # rotated vectors
    ax.quiver(0, 0, 0, rv1[0], rv1[1], rv1[2], color='red', arrow_length_ratio=0.15)
    ax.quiver(0, 0, 0, rv2[0], rv2[1], rv2[2], color='purple', arrow_length_ratio=0.15)
    ax.quiver(0, 0, 0, rv3[0], rv3[1], rv3[2], color='green', arrow_length_ratio=0.15)

    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(1, -1)
    ax.set_zlim3d(1, -1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


if __name__ == '__main__':

    rotations = [
        (Rotation.ROLL, 25),
        (Rotation.PITCH, 75),
        (Rotation.YAW, 90)
    ]

    R = EulerRotation(rotations).rotate()
    print("Rotation matrix ...")
    print(R)

    # calculate 3 rotation matrics
    # rotation: [roll, yaw, pitch]
    rotation1 = [rotations[0], rotations[2], rotations[1]]
    # rotation: [pitch, yaw, roll]
    rotation2 = [rotations[1], rotations[2], rotations[0]]
    # rotation: [yaw, pitch, roll]
    rotation3 = [rotations[2], rotations[1], rotations[0]]
    R1 = EulerRotation(rotation1).rotate()
    R2 = EulerRotation(rotation2).rotate()
    R3 = EulerRotation(rotation3).rotate()

    # unit vector along x-axis
    v = np.array([1, 0, 0])
    rv1 = np.dot(R1, v)
    rv2 = np.dot(R2, v)
    rv3 = np.dot(R3, v)
    visualize(v, rv1, rv2, rv3)
