import numpy as np
import matplotlib.pyplot as plt
from basic_plot import plot_basic
import solve_linear
import solve_quadratic
import LSS_linear
import LSS_quadratic

import argparse

def main():
    #array of output vars
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
    #array of input vars
    T = np.array([1, 2, 3, 4, 5, 6], dtype=float)

    velocity_linear, error_linear, intercepts_linear = solve_linear.solve(D, T, learn_rate=1e-4, max_iter=80000, tol=1e-8)

    print("Results of estimation with gradient descent on linear function:")

    print("Estimated velocity vector [vx, vy, vz]:")
    print(velocity_linear)
    print("Total estimated velocity [m/s]:")
    print(np.sqrt(velocity_linear[0] ** 2 + velocity_linear[1] ** 2 + velocity_linear[2] ** 2))
    print("Total squared error:")
    print(np.sqrt(error_linear))

    # lss_l = LSS_linear.lss_linear(D, T)
    # print("\nLSS estimate of linear function parameters:")
    # print(lss_l)

    velocity_quadratic, acceleration_quadratic, error_quadratic, intercepts_quadratic = solve_quadratic.solve(D, T, learn_rate=1e-4, max_iter=80000, tol=1e-8)

    print("\nResults of estimation with gradient descent on quadratic function:")

    print("Estimated acceleration vector [ax, ay, az]:")
    print(acceleration_quadratic)

    print("Total estimated acceleration [ax, ay, az]:")
    print(np.sqrt(acceleration_quadratic[0] ** 2 + acceleration_quadratic[1] ** 2 + acceleration_quadratic[2] ** 2))

    print("Estimated velocity vector [vx, vy, vz]:")
    print(velocity_quadratic)

    print("Total estimated velocity [m/s]:")
    print(np.sqrt(velocity_quadratic[0] ** 2 + velocity_quadratic[1] ** 2 + velocity_quadratic[2] ** 2))

    print("Total squared error:")
    print(np.sqrt(error_quadratic))

    # lss_q = LSS_quadratic.lss_quadratic(D, T)
    # print("\nLSS estimate of acceleration, velocity, position per dimension:")
    # print(lss_q)

    p = intercepts_quadratic + 7 * velocity_quadratic + acceleration_quadratic * 49 / 2

    print("\nDrone's estimated position at t = 7 is:")
    print(p)

    original = plot_basic(D, 'given positions')
    original.savefig('../figures/original.png')

    predicted_positions = np.append(D, np.array([p]), 0)
    #np.append(predicted_positions, p)

    predicted = plot_basic(predicted_positions, 'predicted positions', 'brown')
    predicted.savefig('../figures/predicted.png')


if __name__ == '__main__':
    main()