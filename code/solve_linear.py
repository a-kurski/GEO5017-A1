import numpy as np
from tqdm import tqdm

def error_linear(data, t, params):
    """
    calculates the error for linear estimation of the function

    :param data: array of dependent variables x, y, z = f(t)
    :param t: array of values for independent variable t
    :param params: tuple of parameters a and b for estimated linear regression
    :return e: sum of squared errors
    """
    # current estimates for a and b
    a, b = params
    # initialize error
    e = 0
    for i in range(len(data)):
        # add error in the i-th dimension
        e += (data[i] - (a * t[i] + b)) ** 2
    return e

def gradient_function(data, t, params):
    """
    calculates the error for current estimation of the function in a given dimensions

    :param data: array of values for a given dependent variable
    :param t: array of values for independent variable t
    :param params: tuple of parameters a and b for estimated linear regression f(x) = ax + b
    :return errors: gradient error for current set of parameters a, b
    """
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

    errors = np.array([dEa, dEb])
    return errors

def gradient_descent(start, data,t, learn_rate, max_iter, tol=0.01):
    """
    performs gradient descent for linear estimation of the function in a given dimension

    :param start: starting tuple of parameters (a, b)
    :param data: array of values for a given dependent variable
    :param t: array of values for independent variable t
    :param learn_rate: learning rate for gradient descent
    :param max_iter: maximum number of iterations for gradient descent
    :param tol: tolerance
    :return: tuple of parameters (a, b)
    """
    params = start.copy()


    #loop for n iterations or until descent value is less than tolerance
    for i in tqdm(range(max_iter)):
        diff = learn_rate * gradient_function(data,t,params)

        #check tolerance
        if np.linalg.norm(diff) < tol:
            break

        #new parameters based on gradient and learning rate
        params = params - diff
    return params

def solve(D, T, learn_rate=1e-4, max_iter=80000, tol=1e-8):
    """
    performs regression for all dimensions

    :param D: array of dependent variables x, y, z = f(t)
    :param T: array of values for independent variable t
    :param learn_rate: learning rate for gradient descent
    :param max_iter: maximum number of iterations for gradient descent
    :param tol: tolerance
    :return velocity_vector: vector denoting the velocity (x, y, z)
    :return total_error: total error
    :return intercepts: vector denoting the position at t = 0 (x, y, z)
    """
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
        total_error += error_linear(data, T, optimal_params)**2

    velocity_vector = np.array(velocities)
    intercepts = np.array(intercepts)

    print("Estimated velocity vector [vx, vy, vz]:")
    print(velocity_vector)

    print("Total estimated velocity [m/s]:")
    print(np.sqrt(velocity_vector[0] ** 2 + velocity_vector[1] ** 2 + velocity_vector[2] ** 2))

    print("\nTotal squared error:")
    print(np.sqrt(total_error))

    return velocity_vector, total_error, intercepts

def main():
    pass