import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
from matplotlib.patches import Ellipse
from matplotlib.patches import Arrow

plt.rcParams['figure.figsize'] = 12, 12

def create_ellipse(mu, angle, width, height):
    # angle is plotted such that it starts from 180 and goes
    # clockwise instead of from 0 degrees and goes counter-clockwise
    # so 135 degrees -> 45 degrees
    return Ellipse(xy=mu, width=width*2, height=height*2, angle=angle,
                      facecolor='none', linestyle='dotted', edgecolor='r')

def plot_ellipse(mu, sigma):  
    (w, v) = LA.eig(sigma)
    angle = np.degrees(np.arctan2(v[1, 0], v[0, 0]))
    
    ax = plt.gca()
    ax.set_xlim(mu[0] + -1.3, mu[0] + 1.3)
    ax.set_ylim(mu[1] + -1.3, mu[1] + 1.3)
    
    ellipse = create_ellipse(mu, angle, w[0], w[1])
    ellipse.set_edgecolor('k')
    ax.add_patch(ellipse)
    plt.plot(mu[0], mu[1], 'ro')

# Constants

# number of samples
n_samples = 1000

# ground truth state
x = np.array([-0.85, 0.25]).reshape(2, 1)

# state -> measurement map
H = np.array([1, -1]).reshape(1, 2)

# noise
R = np.eye(1) * 0.01

x_hat0 = np.array([0, 0]).reshape(-1, 1)
Q0 = np.eye(len(x_hat0))

plot_ellipse(x_hat0, Q0)

def recursive_estimation(x_hat0, Q0, n_samples):
    x_hat = np.copy(x_hat0)
    Q = np.copy(Q0)
    
    for _ in range(n_samples):
        y_obs = H @ x + np.random.multivariate_normal([0.], R)

        Q = LA.pinv(LA.pinv(Q) + H.T @ LA.pinv(R) @ H)
        x_hat = x_hat + Q @ H.T @ LA.pinv(R) @ (y_obs - H @ x_hat)
        
    return x_hat, Q

x_hat, Q = recursive_estimation(x_hat0, Q0, n_samples)
    
print("x =", x.squeeze())
print("x̂ =", x_hat.squeeze())
print("Hx =", H @ x.squeeze())
print("Hx̂ =", H @ x_hat.squeeze())

(w, v) = LA.eig(Q)
# sync angle with plotted ellipse
angle = np.pi - np.arctan2(v[1, 0], v[0, 0])
print("eigen values", w)
print("eigen vectors", v)
print("angle", np.degrees(angle))

inputs = []
outputs = []
dt = 0.25

for v in np.arange(-1, 1+dt, dt):
    s = np.array([x_hat[0] + np.cos(angle) * v, x_hat[1] + np.sin(angle) * v])
    outputs.append(H @ s)
    inputs.append(s.reshape(1, -1))
inputs = np.array(inputs).squeeze()

plt.plot(inputs[:, 0], inputs[:, 1], 'bo')
# plot x_hat
plot_ellipse(x_hat, Q)

print(outputs)
