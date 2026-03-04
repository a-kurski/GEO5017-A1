import numpy as np
from basic_plot import plot_basic
import solve_linear
import solve_quadratic

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

    velocity_linear, error_linear, intercepts_linear = solve_linear.solve(D, T)

    print("Results of estimation with gradient descent on linear function:")

    print("Estimated velocity vector [vx, vy, vz]:")
    print(velocity_linear)
    print("Total estimated velocity [m/s]:")
    print(np.sqrt(velocity_linear[0] ** 2 + velocity_linear[1] ** 2 + velocity_linear[2] ** 2))
    print("Total squared error:")
    print(np.sqrt(error_linear))

    velocity_quadratic, acceleration_quadratic, error_quadratic, intercepts_quadratic = solve_quadratic.solve(D, T)

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

    plot_basic(D)



if __name__ == '__main__':
    main()