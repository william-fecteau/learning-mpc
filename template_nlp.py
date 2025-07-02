import casadi as cas
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Data
data_x = [0,45,90,135,180]
data_y = [667,661,757,871,1210]

# Decision variables
m = cas.SX.sym("m") # type: ignore
b = cas.SX.sym("b") # type: ignore
x = cas.vertcat(m, b)

# Objective function
obj = 0
for pt_x, pt_y in zip(data_x, data_y):
    obj += (pt_y - (m*pt_x+b))**2 

obj_fun = cas.Function('obj_fun', [m,b], [obj])

# Defining NLP
g = [] # Constaints
p = [] # Parameters

nlp_prob = {'f': obj, 'x': x, 'g': g, 'p': p}

# Creating solver
solver_options = {
    "ipopt": {
        "max_iter": 100,
        "print_level": 0,
        "acceptable_tol": 1e-8,
        "acceptable_obj_change_tol": 1e-8
    }
}

solver = cas.nlpsol('solver', 'ipopt', nlp_prob, solver_options)

# Solving
solution = solver(
    x0=[0.5, 1], # Initial guess
    lbx=-cas.inf, # Lower bound x
    ubx=cas.inf, # Upper bound x
    lbg=-cas.inf, # Lower bound g
    ubg=cas.inf, # Upper bound g
    p=p # Parameters
)

print(f"Solution: {solution['x']}")

# Visualization
fig = plt.figure(figsize=(12, 6))

ax0 = fig.add_subplot(122)
solution_m = solution['x'][0]
solution_b = solution['x'][1]

sampled_x = np.linspace(np.min(data_x), np.max(data_x), 1000)
sampled_y = solution_m*sampled_x+solution_b

ax0.plot(sampled_x, sampled_y, 'black', label=f'Best fit m={solution_m}, b={solution_b}')
ax0.scatter(data_x, data_y, label='Data points')
ax0.set_title("Solution found")
ax0.legend()

ax1 = fig.add_subplot(121, projection='3d')
sampled_m = np.linspace(-10, 10, 100)
sampled_b = np.linspace(-1000, 1000, 100)

M, B = np.meshgrid(sampled_m, sampled_b)
Z = np.array(obj_fun(M, B))
ax1.plot_surface(M, B, Z, cmap='viridis') # type: ignore
ax1.set_title("Objective function")

plt.show()