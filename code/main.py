import numpy as np
import matplotlib.pyplot as plt
from basic_plot import plot_basic
from predicted_plot import plot_predicted
import solve_linear
import solve_quadratic
import LSS_linear
import LSS_quadratic
import argparse

def main():
    #initialise argparse
    parser = argparse.ArgumentParser(description='GEO5017 Assignment 1 Solver')
    # define arguments
    parser.add_argument("input", help='input file (required)')
    parser.add_argument('--lr', type=float, help='learning rate', default=0.0001)
    parser.add_argument('--iter', type=int, help='maximum number of iterations', default=80000)
    parser.add_argument('--tol', type=float, help='tolerance', default=1e-8)
    #run function
    run_with_args(parser.parse_args())

def run_with_args(args):
    """
    runs the program with given arguments
    :param args: list of user-defined arguments
    :return: None
    """
    # csv to np array
    inputfile = np.genfromtxt(args.input, delimiter=',')

    # split np array into independent (T) and dependent variables
    T = inputfile[:,0]
    D = inputfile[:,1:]

    # plot as given
    original = plot_basic(D)

    # solve linear fit
    velocity_linear, error_linear, intercepts_linear = solve_linear.solve(D, T, learn_rate=args.lr, max_iter=args.iter, tol=args.tol)

    print("Results of estimation with gradient descent on linear function:")

    print("Estimated position at t = 0 [x, y, z]:")
    print(intercepts_linear)
    print("Estimated velocity vector [vx, vy, vz]:")
    print(velocity_linear)
    print("Total estimated velocity [m/s]:")
    print(np.sqrt(velocity_linear[0] ** 2 + velocity_linear[1] ** 2 + velocity_linear[2] ** 2))
    print("Total squared error:")
    print(np.sqrt(error_linear))

    # lss_l = LSS_linear.lss_linear(D, T)
    # print("\nLSS estimate of linear function parameters:")
    # print(lss_l)

    # solve quadratic fit
    velocity_quadratic, acceleration_quadratic, error_quadratic, intercepts_quadratic = solve_quadratic.solve(D, T, learn_rate=args.lr, max_iter=args.iter, tol=args.tol)

    print("\nResults of estimation with gradient descent on quadratic function:")

    print("Estimated position at t = 0 [x, y, z]:")
    print(intercepts_linear)

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

    # calculate predicted position
    p = intercepts_quadratic + 7 * velocity_quadratic + acceleration_quadratic * 49 / 2

    print("\nDrone's estimated position at t = 7 is:")
    print(p)

    # plot predicted position with measured positions
    predicted = plot_predicted(D, p)


if __name__ == '__main__':
    main()