import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['figure.figsize'] = 12, 12


# limit the steering angle range
STEERING_ANGLE_MAX = np.deg2rad(30)


def sample_steering_angle():
    return np.random.uniform(-STEERING_ANGLE_MAX, STEERING_ANGLE_MAX)


def dubins_car_model(state, v, dt):
    x = state[0]
    y = state[1]
    theta = state[2]    
    
    angle = sample_steering_angle()

    nx = x + v*np.cos(theta)*dt
    ny = y + v*np.sin(theta)*dt
    ntheta = theta + v*np.tan(angle)*dt
    
    return [nx, ny, ntheta]


if __name__ == '__main__':
    v = 5
    dt = 0.1
    total_time = 50

    states = [[0, 0, 0]]
    for _ in np.arange(0, total_time, dt):
        state = dubins_car_model(states[-1], v, dt)
        states.append(state)


    states = np.array(states)
    
    plt.plot(states[:, 0], states[:, 1], color='blue')
    plt.axis('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
