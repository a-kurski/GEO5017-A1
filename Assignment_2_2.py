# imports
import numpy as np
import matplotlib.pyplot as plt

#Variables
D = np.array(
    [
        [2, 0, 1],
        [1.08, 1.68, 2.38],
        [-0.83, 1.82, 2.49],
        [-1.97, 0.28, 2.15],
        [-1.31, -1.51, 2.59],
        [0.57, -1.91, 4.32]
    ]
)

T = np.array([1, 2, 3, 4, 5, 6], dtype=float)

itter = 100
tol = 0.1
learn_rate = 0.001

x = D[:, 0]
y = D[:, 1]
z = D[:, 2]

# Error sum
def error_function(data, t, params):
    a, b = params
    E = 0
    for i in range(len(data)):
        E += (data[i] - (a * t[i] + b)) ** 2
    return E

#
def gradient_function(data, t, params):
    a, b = params
    dEa = 0
    dEb = 0

    for i in range(len(data)):
        error = data[i] - (a * t[i] + b)
        dEa += -2 * t[i] * error
        dEb += -2 * error

    return np.array([dEa, dEb])

# -------------------------------
def gradient_descent(start, gradient, learn_rate, max_iter, tol=0.01):
    params = start.copy()
    for _ in range(max_iter):
        diff = learn_rate * gradient(params)
        if np.linalg.norm(diff) < tol:
            break
        params = params - diff
    return params

# -------------------------------
def run_regression_all_dims(D, T):
    learn_rate = 0.01
    max_iter = 1000
    tol = 0.01
    velocities = []
    intercepts = []
    total_error = 0
    # Loop over x, y, z (columns 0,1,2)
    for dim in range(3):
        data = D[:, dim]
        start = np.array([0.0, 0.0])  # initial guess [a, b]
        grad = lambda params: gradient_function(data, T, params)

        optimal_params = gradient_descent(
            start,
            grad,
            learn_rate,
            max_iter,
            tol
        )

        a_opt, b_opt = optimal_params

        velocities.append(a_opt)
        intercepts.append(b_opt)
        total_error += error_function(data, T, optimal_params)

    velocity_vector = np.array(velocities)
    intercepts = np.array(intercepts)

    print("Estimated velocity vector [vx, vy, vz]:")
    print(velocity_vector)

    print("Total estimated velocity [m/s]:")
    print(np.sqrt(velocity_vector[0] ** 2 + velocity_vector[1] ** 2 + velocity_vector[2] ** 2))

    print("\nTotal squared error:")
    print(total_error)

    return velocity_vector, total_error, intercepts


# -------------------------------
velocity, error, intercepts = run_regression_all_dims(D, T)



