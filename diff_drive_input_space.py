from matplotlib import pyplot as plt
import numpy as np


wheel_radius = 1.0
wheelbase = 1.0
max_wheel_speed = 1.0

jacobian = wheel_radius * np.array([
    [1/2., 1/2.],
    [0., 0.],
    [-1./wheelbase, 1./wheelbase]
])

u_left = np.linspace(-max_wheel_speed, max_wheel_speed, 25)
u_right = np.linspace(-max_wheel_speed, max_wheel_speed, 25)

U_left, U_right = np.meshgrid(u_left, u_right)

v_x = []
v_omega = []
colors = [] 

for i in range(U_left.shape[0]):
    for j in range(U_left.shape[1]):
        v_l = U_left[i, j]
        v_r = U_right[i, j]

        u = np.array([v_l, v_r]).T
        dstate = jacobian @ u

        v_x.append(dstate[0])
        v_omega.append(dstate[2])
        colors.append(dstate[0])
    

fig, axs =  plt.subplots(2, 1)

# Actuator frame
scat1 = axs[0].scatter(U_left, U_right, c=colors, cmap='viridis')
axs[0].set_title("Commands in actuator frame")
axs[0].set_xlabel("Left wheel speed $u_{left}$ (m/s)")
axs[0].set_ylabel("Right wheel speed $u_{right}$ (m/s)")

# Body frame
scat2 = axs[1].scatter(v_omega, v_x, c=colors, cmap='viridis')
axs[1].set_title("Commands in body frame")
axs[1].set_xlabel("Angular velocity $\\omega$ (rad/s)")
axs[1].set_ylabel("Linear velocity $v$ (m/s)")

plt.show()



