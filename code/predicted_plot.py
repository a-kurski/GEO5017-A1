# imports
import numpy as np
import matplotlib.pyplot as plt

def plot_predicted(D, p):
    x = D[:, 0]
    y = D[:, 1]
    z = D[:, 2]

    xp = np.append(D[-1,0], p[0])
    yp = np.append(D[-1,1], p[1])
    zp = np.append(D[-1,2], p[2])

    # -- 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax3d = fig.add_subplot(221, projection='3d')
    ax3d.plot(x, y, z, color='blue', label="Measured",
              marker='s')
    ax3d.plot(xp, yp, zp, color='brown', label="Predicted",
              marker='o')
    ax3d.set_title("3D Trajectory")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.legend()

    # --- XY projection ---
    ax_xy = fig.add_subplot(222)
    ax_xy.plot(x, y, marker='s', label="Measured")
    ax_xy.plot(xp, yp, marker='o', label="Predicted")
    ax_xy.set_title("XY projection")
    ax_xy.set_xlabel("x")
    ax_xy.set_ylabel("y")
    ax_xy.legend()

    # --- YZ projection ---
    ax_yz = fig.add_subplot(223)
    ax_yz.plot(y, z, marker='s', label="Measured")
    ax_yz.plot(yp, zp, marker='o', label="Predicted")
    ax_yz.set_title("YZ projection")
    ax_yz.set_xlabel("y")
    ax_yz.set_ylabel("z")
    ax_yz.legend()

    # --- XZ projection ---
    ax_xz = fig.add_subplot(224)
    ax_xz.plot(x, z, marker='s', label="Measured")
    ax_xz.plot(xp, zp, marker='o', label="Predicted")
    ax_xz.set_title("XZ projection")
    ax_xz.set_xlabel("x")
    ax_xz.set_ylabel("z")
    ax_xz.legend()

    plt.tight_layout()

    plt.show()

if __name__ == '__main__':
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

    plot_basic(D, "title").show()