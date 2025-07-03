import casadi as cas
from matplotlib import pyplot as plt
import numpy as np

def create_solver(N, T, Q, R, path):
    # Model variables
    x = cas.SX.sym('x') # type: ignore
    y = cas.SX.sym('y') # type: ignore
    theta = cas.SX.sym('theta') # type: ignore
    states = cas.vertcat(
        x,
        y,
        theta
    )
    n_states = states.numel()

    v = cas.SX.sym('v') # type: ignore
    omega = cas.SX.sym('omega') # type: ignore
    controls = cas.vertcat(
        v,
        omega
    )
    n_controls = controls.numel()

    # matrix containing all states over all time steps +1 (each column is a state vector)
    X = cas.SX.sym('X', n_states, N + 1) # type: ignore

    # matrix containing all control actions over all time steps (each column is an action vector)
    U = cas.SX.sym('U', n_controls, N) # type: ignore

    # column vector for storing initial state and path. We assume that each waypoint is approximately dt apart.
    P = cas.SX.sym('P', n_states + (N*n_states)) # type: ignore

    # Model
    jacobian = cas.vertcat(
        cas.horzcat(cas.cos(theta), 0.0),
        cas.horzcat(cas.sin(theta), 0.0),
        cas.horzcat(0.0, 1.0)
    )
    rhs = jacobian@controls
    f = cas.Function('f', [states, controls], [rhs])

    # Discretization + objective function
    obj = 0
    X[:, 0] = P[:n_states]
    for i in range(N):
        state = X[:, i]
        control = U[:, i]
        
        point_ref = P[n_states+i*n_states:n_states+i*n_states+n_states]
        point = cas.vertcat(state[0], state[1], state[2])
        state_err = point - point_ref
        state_err[2] = cas.atan2(cas.sin(state_err[2]), cas.cos(state_err[2]))

        # Integration
        X[:, i+1] = state + T*f(state, control)

        # Objective
        obj += state_err.T @ Q @ state_err + control.T @ R @ control

    # this function to get the optimal trajectory knowing the optimal solution
    ff=cas.Function('ff',[U,P],[X])

    # Defining NLP
    g = [] # Constaints
    OPT_variables = cas.vertcat(
        U.reshape((-1, 1))
    )

    nlp_prob = {'f': obj, 'x': OPT_variables, 'g': g, 'p': P}

    # Creating solver
    solver_options = {
        "ipopt": {
            "max_iter": 100,
            "print_level": 0,
            "acceptable_tol": 1e-8,
            "acceptable_obj_change_tol": 1e-8
        }
    }

    return cas.nlpsol('solver', 'ipopt', nlp_prob, solver_options), ff


if __name__ == "__main__":
    path = np.loadtxt("traj.np")

    x_0 = path[0]
    v_max = 0.5
    v_omega_max = 0.5

    N = 50 # Horizon
    T = 1.0/10.0 # Delta time
    n_controls = 2
    Q = np.diag([1.0, 1.0, 1.0]) # State weighting
    R = np.diag([0.05, 0.1]) # Input weighting

    solver, ff = create_solver(N, T, Q, R, path)

    # Solving
    lbx = np.zeros(n_controls*N)
    lbx[0::n_controls] = -v_max
    lbx[1::n_controls] = -v_omega_max

    ubx = np.zeros(n_controls*N)
    ubx[0::n_controls] = v_max
    ubx[1::n_controls] = v_omega_max

    params = cas.vertcat(*x_0, *path[:N].flatten())

    solution = solver(
        x0=np.zeros(n_controls*N), # Initial guess
        lbx=lbx, # Lower bound x
        ubx=ubx, # Upper bound x
        lbg=-cas.inf, # Lower bound g
        ubg=cas.inf, # Upper bound g
        p=params # Parameters
    )

    print(f"Solution: {solution['x']}")

    sol_U = cas.reshape(solution['x'], n_controls, N)
    sol_X = ff(sol_U, params)

    x_pos = np.array(sol_X[0, :-1]) # type: ignore
    y_pos = np.array(sol_X[1, :-1]) # type: ignore
    theta = np.array(sol_X[2, :-1]) # type: ignore
    v = np.array(sol_U[0, :])

    uu = np.cos(theta)
    vv = np.sin(theta)

    # Plot
    plt.figure(figsize=(8, 6))
    quiver = plt.quiver(x_pos, y_pos, uu, vv, v, angles='xy', scale_units='xy', scale=10, cmap='viridis')
    cbar = plt.colorbar(quiver)
    cbar.set_label('Linear Speed')
    plt.plot(path[:, 0], path[:, 1], label='Path')
    # plt.quiver(x_ref[0], x_ref[1], np.cos(x_ref[2]), np.sin(x_ref[2]), angles='xy', scale_units='xy', scale=10, color='red', label='Target')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.grid(True)
    plt.title('Control Arrows along Trajectory')
    plt.show()
