import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

plt.rcParams['figure.figsize'] = 12, 12

def simulate(y, dt):
    return y - y * dt

total_time = 10
# TODO: change the value of dt
dt = 0.1
timesteps = np.arange(0, total_time, dt)

# Initial value of y is 1
ys = [1]

for _ in timesteps:
    y = simulate(ys[-1], dt)
    ys.append(y)
    
plt.plot(timesteps, ys[:-1], color='blue')
plt.ylabel('Y')
plt.xlabel('Time')
plt.title('Dynamics Model')
plt.legend()
plt.show()

