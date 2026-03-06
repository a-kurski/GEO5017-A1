import numpy as np

def lss_quadratic(D, T):
    # Quadratic design matrix
    T_a = np.column_stack([
        0.5 * T ** 2,  # acceleration term
        T,  # velocity term
        np.ones_like(T)  # intercept (starting position)
    ])
    # least squares solution
    D_T = T_a.T
    D_inv = np.linalg.inv(D_T @ T_a)
    D_dot = D_T @ D
    params = D_inv @ D_dot

    return params

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

    #time value vector
    T = np.array([1, 2, 3, 4, 5, 6], dtype=float)

    params = lss_quadratic(D, T)
    print("Acceleration, Velocity, Position per dimension:")
    print(params)

    # 1. Recreate the quadratic design matrix used in the solver
    T_a = np.column_stack([
        0.5 * T ** 2,
        T,
        np.ones_like(T)
    ])

    # 2. Calculate the predicted positions (Design matrix @ parameters)
    D_pred = T_a @ params

    # 3. Calculate the sum of squared differences
    total_error = np.sum((D_pred - D) ** 2)

    print("Total squared error:", total_error)

if __name__ == "__main__":
    main()