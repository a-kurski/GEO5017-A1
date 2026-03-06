import numpy as np

def lss_linear(D, T):
    T_a = np.linalg.matrix_transpose([T,[1,1,1,1,1,1]])
    #least squares solution
    D_T = np.transpose(T_a)
    D_inv = np.linalg.inv(np.matmul(D_T,T_a))
    D_dot = np.matmul(D_T,D)
    a = np.matmul(D_inv, D_dot)
    return a

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

    # time value vector
    T = np.array([1, 2, 3, 4, 5, 6], dtype=float)

    lss = lss_linear(D, T)

    print("Linear LSS solution:", lss)

    # 1. Recreate the design matrix used in the solver
    T_a = np.transpose([T, [1, 1, 1, 1, 1, 1]])

    # 2. Calculate the predicted positions (Design matrix multiplied by the LSS parameters)
    D_pred = np.matmul(T_a, lss)

    # 3. Calculate the sum of squared differences between predictions and actual data
    total_error = np.sum((D_pred - D) ** 2)

    print("Total squared error:", total_error)

if __name__ == "__main__":
    main()