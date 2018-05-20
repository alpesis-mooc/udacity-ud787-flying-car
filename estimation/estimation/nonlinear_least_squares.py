import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA

# TODO: complete implementation
def f_range(x):
    """
    Distance of x from the origin.
    """
    return LA.norm(x)

# TODO: complete implementation
def f_bearing(x):
    """
    atan2(x_2, x_1)
    """
    return np.arctan2(x[1], x[0])

def h(x):
    return np.array([f_range(x), f_bearing(x)])


# TODO: complete jacobian of h(x)
def jacobian_of_h(x):
    t = (1/2) * (x[0]**2 + x[1]**2) ** (-1/2)
    return np.array([        
        [t*2*x[0], t*2*x[1]],
        
        # atan2(x, y)
        # ( y / (x^2 + y^2), ( -x / (x^2 + y^2)
        # atan2(x, y)
        # ( -x / (x^2 + y^2), ( $y / (x^2 + y^2)
        [-x[0] / (x[0]**2 + x[1]**2), x[1] / (x[0]**2 + x[1]**2)]
    ]).squeeze()


n_samples = 1000

# Covariance matrix
# added noise for range and bearing functions
#
# NOTE: these are set to low variance values
# to start with, if you increase them you
# might more samples to get
# a good estimate.
R = np.eye(2)
R[0, 0] = 0.01
R[1, 1] = np.radians(1) 

# ground truth state
x = np.array([1.5, 1])

x_hat0 = np.array([3., 3]).reshape(-1, 1)
Q0 = np.eye(len(x_hat0))

# TODO: Recursive Estimation
def recursive_estimation(x_hat0, Q0, n_samples):
    x_hat = np.copy(x_hat0)
    Q = np.copy(Q0)

    for _ in range(n_samples):

        # TODO: sample a measurement
        y_obs = h(x) + np.random.multivariate_normal([0, 0], R)

        # TODO: compute the jacobian of h(x_hat)
        H = jacobian_of_h(x_hat)

        # TODO: update Q and x_hat
        Q = LA.pinv(LA.pinv(Q) + H.T @ LA.pinv(R) @ H)
        x_hat = x_hat + (Q @ H.T @ LA.pinv(R) @ (y_obs - h(x_hat))).reshape(2, 1)
        
    return x_hat, Q

print("x̂0 =", x_hat0.squeeze())

x_hat, Q = recursive_estimation(x_hat0, Q0, n_samples)
    
print("x =", x.squeeze())
print("x̂ =", x_hat.squeeze())
print("Hx =", h(x))
print("Hx̂ =", h(x_hat))

errors = []
Ns = np.arange(0, 201, 5)
for n in Ns:
    x_hat, Q = recursive_estimation(x_hat0, Q0, n)
    errors.append(LA.norm(x.squeeze() - x_hat.squeeze()))

plt.plot(Ns, errors)
plt.xlabel('Number of samples')
plt.ylabel('Error')
