import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = 12, 12

# limit the steering angle range
STEERING_ANGLE_MAX = np.deg2rad(30)

def sample_steering_angle():
    return np.random.uniform(-STEERING_ANGLE_MAX, STEERING_ANGLE_MAX)

def simulate(state, v, dt):
    # TODO: implement the dubin's car model
    # return the next state
    return [0, 0, 0]

# feel free to play around with these
v = 1
dt = 0.1
total_time = 10

# initial state
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


