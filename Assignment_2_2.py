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
T_a = np.linalg.matrix_transpose([T,[1,1,1,1,1,1]])
print(T_a)
#least squares solution
D_T = np.transpose(T_a)
D_inv = np.linalg.inv(np.matmul(D_T,T_a))
D_dot = np.matmul(D_T,D)
a = np.matmul(D_inv, D_dot)
print("LSS:", a)

# Error function
def error_linear(data, t, params):
    #current estimates for a and b
    a, b = params
    #Initialize error
    E = 0
    #Summation
    for i in range(len(data)):
        E += (data[i] - (a * t[i] + b)) ** 2
    return E

#Gradient function (a and b derivative of error function)
def gradient_function(data, t, params):
    #current estimates for a and b
    a, b = params
    #Initialization of gradient A and B errors
    dEa = 0
    dEb = 0

    #Sum of gradients errors
    for i in range(len(data)):
        error = data[i] - (a * t[i] + b)
        dEa += -2 * t[i] * error
        dEb += -2 * error
    #returns calculated gradient error for a and b estimate
    return np.array([dEa, dEb])

# -------------------------------
def gradient_descent(start, data,t, learn_rate, max_iter, tol=0.01):
    params = start.copy()
    #loops for n iterations or until descent value is less than tolerance
    for i in range(max_iter):
        diff = learn_rate * gradient_function(data,t,params)
        #tolerance
        if np.linalg.norm(diff) < tol:
            print("itteration: ",_)
            break
        #new parameters based on gradient and learning rate
        params = params - diff
    return params

# -------------------------------
def run_regression_all_dims(D, T):
    learn_rate = 1e-4
    max_iter = 80000
    tol = 1e-8
    velocities = []
    intercepts = []
    total_error = 0
    # Loop over x, y, z (columns 0,1,2)
    for dim in range(3):
        #retreives values for given dimension
        data = D[:, dim]
        start = np.array([2.0, 2.0])  # initial guess [a, b]

        optimal_params = gradient_descent(
            start,
            data,
            T,
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
    total_error = total_error**2

    print("Estimated velocity vector [vx, vy, vz]:")
    print(velocity_vector)
    print(intercepts)

    print("Total estimated velocity [m/s]:")
    print(np.sqrt(velocity_vector[0] ** 2 + velocity_vector[1] ** 2 + velocity_vector[2] ** 2))

    print("\nTotal squared error:")
    print(np.sqrt(total_error))

    return velocity_vector, total_error, intercepts


# -------------------------------
velocity, error, intercepts = run_regression_all_dims(D, T)