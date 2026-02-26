# imports
import numpy as np
import matplotlib.pyplot as plt

#Variables
#xyz value matrix
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

#time value vector
T = np.array([1, 2, 3, 4, 5, 6], dtype=float)

#gradient descent variables
itter = 100
tol = 0.1
learn_rate = 0.001

x = D[:, 0]
y = D[:, 1]
z = D[:, 2]

# Error function
def error_quadratic(data, t, params):
    #current estimates for a and b
    a, b, c = params
    #Initialize error
    E = 0
    #Summation
    for i in range(len(data)):
        E += (data[i] - (a * t[i]**2 + b * t[i] + c)) ** 2
    return E

#Gradient function (a and b derivative of error function
def gradient_function(data, t, params):
    #current estimates for a and b
    a, b, c = params
    #Initialization of gradient A and B errors
    dEa = 0
    dEb = 0
    dEc = 0

    #Sum of gradients errors
    for i in range(len(data)):
        error = data[i] - (a * t[i]**2 + b * t[i] + c)
        dEa += -2 * t[i]**2 * error
        dEb += -2 * t[i] * error
        dEc += -2 * error

    #returns calculated gradient error for a and b estimate
    return np.array([dEa, dEb, dEc])

# -------------------------------
def gradient_descent(start, data,t, learn_rate, max_iter, tol=0.01):
    params = start.copy()
    #loops for n itterations or until descent value is less than tollerance
    for _ in range(max_iter):
        diff = learn_rate * gradient_function(data,t,params)
        #tollerance
        if np.linalg.norm(diff) < tol:
            break
        #new parameters based on gradient and learning rate
        params = params - diff
    return params

# -------------------------------
def run_regression_all_dims(D, T):
    learn_rate = 1e-4
    max_iter = 50000
    tol = 1e-10
    acceleration = []
    velocities = []
    intercepts = []
    total_error = 0
    # Loop over x, y, z (columns 0,1,2)
    for dim in range(3):
        #retreives values for given dimension
        data = D[:, dim]
        start = np.array([2.0, 2.0, 2.0])  # initial guess [a, b]

        optimal_params = gradient_descent(
            start,
            data,
            T,
            learn_rate,
            max_iter,
            tol
        )

        a_opt, b_opt, c_opt = optimal_params

        acceleration.append(a_opt)
        velocities.append(b_opt)
        intercepts.append(c_opt)
        total_error += error_quadratic(data, T, optimal_params)

    acceleration_vector = np.array(acceleration)
    velocity_vector = np.array(velocities)
    intercepts = np.array(intercepts)

    print("Estimated acceleration vector [ax, ay, az]:")
    print(acceleration_vector)

    print("Total estimated acceleration vector [ax, ay, az]:")
    print(np.sqrt(acceleration_vector[0]**2 + acceleration_vector[1]**2 + acceleration_vector[2]**2))
    print("\n")

    print("Estimated velocity vector [vx, vy, vz]:")
    print(velocity_vector)

    print("Total estimated velocity [m/s]:")
    print(np.sqrt(velocity_vector[0] ** 2 + velocity_vector[1] ** 2 + velocity_vector[2] ** 2))

    print("\nTotal squared error:")
    print(np.sqrt(total_error))

    return velocity_vector, acceleration_vector, total_error, intercepts


# -------------------------------
velocity, acceleration, error, intercepts = run_regression_all_dims(D, T)