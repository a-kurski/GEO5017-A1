import numpy as np
from basic_plot import plot_basic

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

def linear_estimation(independent, dependent, learning_rate, num_iterations):
    pass

def quadratic_estimation(independent, dependent, learning_rate, num_iterations):
    pass

def main():
    plot_basic(D)

if __name__ == '__main__':
    main()