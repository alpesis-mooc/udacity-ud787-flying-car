import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = 12, 12


def simulate(y, dt):
    return y - y * dt


if __name__ == '__main__':
    total_time = 10
    dt = 0.1
    timesteps = np.arange(0, total_time, dt)
    
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
