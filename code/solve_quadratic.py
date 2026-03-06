# imports
import numpy as np
import matplotlib.pyplot as plt


def error_quadratic(data, t, params):
    """
    calculates the error for quadratic estimation of the function

    :param data: array of dependent variables x, y, z = f(t)
    :param t: array of values for independent variable t
    :param params: tuple of parameters a and b for estimated quadratic function
    :return e: sum of squared errors
    """
    #current estimates for a, b, c
    a, b, c = params
    #Initialize error
    e = 0
    #Summation
    for i in range(len(data)):
        e += (data[i] - (0.5 * a * t[i]**2 + b * t[i] + c)) ** 2
    return e


def gradient_function(data, t, params):
    """
    calculates the error for current estimation of the function in a given dimensions

    :param data: array of values for a given dependent variable
    :param t: array of values for independent variable t
    :param params: tuple of parameters a and b for estimated quadratic f(t) = (at^2)/2 + bt + c
    :return errors: gradient error for current set of parameters a, b
    """
    #current estimates for a and b
    a, b, c = params
    #Initialization of gradient A and B errors
    dEa = 0
    dEb = 0
    dEc = 0

    #Sum of gradients errors
    for i in range(len(data)):
        error = data[i] - (0.5 * a * t[i]**2 + b * t[i] + c)
        dEa += -2 * t[i]**2 * error
        dEb += -2 * t[i] * error
        dEc += -2 * error

    #returns calculated gradient error for a and b estimate
    return np.array([dEa, dEb, dEc])

def gradient_descent(start, data,t, learn_rate, max_iter, tol):
    """
        performs gradient descent for quadratic estimation of the function in a given dimension

        :param start: starting tuple of parameters (a, b, c)
        :param data: array of values for a given dependent variable
        :param t: array of values for independent variable t
        :param learn_rate: learning rate for gradient descent
        :param max_iter: maximum number of iterations for gradient descent
        :param tol: tolerance
        :return: tuple of parameters (a, b, c)
        """
    params = start.copy()
    #loops for n iterations or until descent value is less than tolerance
    for i in range(max_iter):
        diff = learn_rate * gradient_function(data,t,params)
        #tolerance
        if np.linalg.norm(diff) < tol:
            break
        #new parameters based on gradient and learning rate
        params = params - diff
    return params

# -------------------------------
def solve(D, T, learn_rate=1e-4, max_iter=80000, tol=1e-8):
    """
    performs regression for all dimensions on linear function f(x) = ax + b

    :param D: array of dependent variables x, y, z = f(t)
    :param T: array of values for independent variable t
    :param learn_rate: learning rate for gradient descent
    :param max_iter: maximum number of iterations for gradient descent
    :param tol: tolerance
    :return velocity_vector: vector denoting the velocity (x, y, z)
    :return acceleration_vector: vector denoting the acceleration (x, y, z)
    :return total_error: total error
    :return intercepts: vector denoting the position at t = 0 (x, y, z)
    """
    acceleration = []
    velocities = []
    intercepts = []
    total_error = 0
    # Loop over x, y, z (columns 0,1,2)
    for dim in range(3):
        #retreives values for given dimension
        data = D[:, dim]
        start = np.array([4.0, 4.0, 4.0])  # initial guess [a, b]

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
    total_error = total_error ** 2

    return velocity_vector, acceleration_vector, total_error, intercepts

def main():

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
    # array of input vars
    T = np.array([1, 2, 3, 4, 5, 6], dtype=float)

    velocity_vector, acceleration_vector, total_error, intercepts = solve(D, T)

    print("Estimated acceleration vector [ax, ay, az]:")
    print(acceleration_vector)

    print("Total estimated acceleration vector [ax, ay, az]:")
    print(np.sqrt(acceleration_vector[0] ** 2 + acceleration_vector[1] ** 2 + acceleration_vector[2] ** 2))
    print("\n")

    print("Estimated velocity vector [vx, vy, vz]:")
    print(velocity_vector)

    print("Total estimated velocity [m/s]:")
    print(np.sqrt(velocity_vector[0] ** 2 + velocity_vector[1] ** 2 + velocity_vector[2] ** 2))

    print("\nTotal squared error:")
    print(np.sqrt(total_error))

if __name__ == "__main__":
    main()