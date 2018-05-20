import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA

# number of samples
# the larger this value the more
# accurate the x hat will be.
n_samples = 100

# size of state
n = 4

def make_H(m, degree, t):
    """
    Creates a matrix where
    each row is of the form:
    
        [t**degree, t**(degree-1), ..., 1]
    """
    H = np.zeros((m, degree))
    for i in np.arange(degree-1, -1, -1):
        H[:, -i-1] = t**i
    return H

# known constants
t = np.random.uniform(-5, 5, n_samples)
H = make_H(n_samples, n, t)

# state, unknown in practice
x = np.random.randn(n) * 2

# TODO: collect m noisy observations, the noise distribution should be gaussian
y_obs = H @ x + np.random.normal(0, 1, size=(n_samples))

plt.plot(t, y_obs, 'bx')
plt.title("Noisy Observations")

x_hat = LA.pinv(H.T @ H) @ H.T @ y_obs

print(x_hat)
print(x)


