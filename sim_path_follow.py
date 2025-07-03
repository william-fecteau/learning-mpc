import stat
import sys
import matplotlib
from matplotlib.colors import Normalize
import matplotlib.patches
import numpy as np
import casadi as cas
import matplotlib.pyplot as plt

from frenet import cartesian_to_frenet
from path_follow_single_shooting import create_solver


wheelbase = 0.5

def draw_robot(ax, state_vector) -> None:
    x, y, yaw = state_vector

    circle = matplotlib.patches.Circle((x, y), wheelbase / 2, color="yellow")

    wheel_width = 0.1
    wheel_height = wheelbase * 0.8

    left_wheel_x = x - wheelbase / 2 - wheel_width
    left_wheel_y = y - wheel_height / 2

    right_wheel_x = x + wheelbase / 2
    right_wheel_y = y - wheel_height / 2

    wheel_rotation_deg = np.rad2deg(yaw) + 90
    left_wheel = matplotlib.patches.Rectangle(
        (left_wheel_x, left_wheel_y),
        wheel_width,
        wheel_height,
        rotation_point=(x, y),
        angle=wheel_rotation_deg,
        color="black",
    )
    right_wheel = matplotlib.patches.Rectangle(
        (right_wheel_x, right_wheel_y),
        wheel_width,
        wheel_height,
        rotation_point=(x, y),
        angle=wheel_rotation_deg,
        color="black",
    )

    ax.add_patch(circle)
    ax.add_patch(left_wheel)
    ax.add_patch(right_wheel)

if __name__ == "__main__":

    path = np.loadtxt('traj.np')
    points = path[:, :2]

    state_vector = path[0]

    v_max = 0.5
    v_omega_max = 0.5

    dt = 1.0/10.0
    N = 50 # Horizon
    n_controls = 2
    Q = np.diag([1.0, 1.0, 0.5]) # State weighting
    R = np.diag([0.05, 0.1]) # Input weighting

    solver, ff = create_solver(N, dt, Q, R, path)

    sol_X = []

    # Visualization loop
    try:
        plt.ion()
        fig, ax = plt.subplots()
        while True:
            s, d = cartesian_to_frenet(points, state_vector[:2])
            s = int(s)

            lbx = np.zeros(n_controls*N)
            lbx[0::n_controls] = -v_max
            lbx[1::n_controls] = -v_omega_max

            ubx = np.zeros(n_controls*N)
            ubx[0::n_controls] = v_max
            ubx[1::n_controls] = v_omega_max
            
            path_horizon = np.array([path[(s+i)%(len(path))] for i in range(N)])
            params = cas.horzcat(*state_vector, *path_horizon.flatten())

            solution = solver(
                x0=np.zeros(n_controls*N), # Initial guess
                lbx=lbx, # Lower bound x
                ubx=ubx, # Upper bound x
                lbg=-cas.inf, # Lower bound g
                ubg=cas.inf, # Upper bound g
                p=params # Parameters
            )

            sol_U = cas.reshape(solution['x'], n_controls, N)
            sol_X = ff(sol_U, params)

            command_vector = np.array(sol_U[:, 0])
            state_vector = np.array(sol_X[:, 1]).flatten() # type: ignore

            # Drawing
            ax.clear()

            x_pos = np.array(sol_X[0, :-1]) # type: ignore
            y_pos = np.array(sol_X[1, :-1]) # type: ignore
            theta = np.array(sol_X[2, :-1]) # type: ignore
            v = np.array(sol_U[0, :])

            uu = np.cos(theta)
            vv = np.sin(theta)

            draw_robot(ax, state_vector.flatten())
            quiver = plt.quiver(x_pos, y_pos, uu, vv, v, angles='xy', scale_units='xy', scale=1, cmap='viridis', norm=Normalize(-v_max, vmax=v_max))
            cbar = plt.colorbar(quiver)
            cbar.set_label('Linear Speed')


            plt.plot(path[:, 0], path[:, 1], color='red', label='Target')
            plt.plot(path_horizon[:, 0], path_horizon[:, 1], label='Horizon')

            ax.legend()
            ax.set_xlim(-2, 6)
            ax.set_ylim(-2, 6)
            ax.set_aspect('equal')
            ax.set_title('Model Simulation')
            plt.pause(dt)
            cbar.remove()

        plt.ioff()
        plt.show()
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
        sys.exit(0)

    print("Target reached!")