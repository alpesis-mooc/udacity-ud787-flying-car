import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = 12, 12


# steering angle range
MAX_STEERING_ANGLE = np.deg2rad(30)
# km/h
MAX_VELOCITY = 1


def dubins_car_model(state, angle, v, dt):
    x = state[0]
    y = state[1]
    theta = state[2]

    nx = x + v * np.cos(theta) * dt
    ny = y + v * np.sin(theta) * dt
    ntheta = theta + v * np.tan(angle) * dt

    return [nx, ny, ntheta]


def steer(x1, x2):
    angle = np.arctan2(x2[1] - x1[1], x2[0] - x1[0])
    if angle > MAX_STEERING_ANGLE: angle = MAX_STEERING_ANGLE

    velocity = np.sin(angle) * MAX_VELOCITY
    if velocity > MAX_VELOCITY: velocity = MAX_VELOCITY
    return angle, velocity


if __name__ == '__main__':
    dt = 0.1
    total_time = 50

    # initial state
    start = [0, 0, 0]
    # the goal location
    goal = [10, -15, 0]
    states = [start]

    for _ in np.arange(0, total_time, dt):
        current_state = states[-1]
        angle, velocity = steer(current_state, goal)
        state = dubins_car_model(current_state, angle, velocity, dt)
        states.append(state)

    states = np.array(states)
    plt.plot(states[:, 0], states[:, 1], color='blue')
    plt.axis('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
