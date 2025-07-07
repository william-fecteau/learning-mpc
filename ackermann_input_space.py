from matplotlib import pyplot as plt
import numpy as np


wheel_radius = 1.0
wheelbase = 1.0

max_wheel_speed = 1.0
max_delta = np.deg2rad(30)

def jacobian(command):
    v_x, delta = command
    return np.array([
        [1.0, 0.],
        [0., 0.],
        [np.tan(delta)/wheelbase, (v_x/wheelbase)*(1 / np.cos(delta))**2]
    ])

u_v = np.linspace(-max_wheel_speed, max_wheel_speed, 25)
u_delta = np.linspace(-max_delta, max_delta, 25)

U_v, U_delta = np.meshgrid(u_v, u_delta)

v_x = []
v_omega = []
colors = [] 

for i in range(U_v.shape[0]):
    for j in range(U_v.shape[1]):
        u_v = U_v[i, j]
        u_delta = U_delta[i, j]

        u = np.array([u_v, u_delta]).T
        dstate = jacobian(u) @ u

        v_x.append(dstate[0])
        v_omega.append(dstate[2])
        colors.append(dstate[0])
    

fig, axs =  plt.subplots(2, 1)

# Actuator frame
scat1 = axs[0].scatter(U_delta, U_v, c=colors, cmap='viridis')
axs[0].set_title("Commands in actuator frame")
axs[0].set_xlabel("Steering angle $u_\\delta$ (rad)")
axs[0].set_ylabel("Wheel speed $u_{v}$ (m/s)")
fig.colorbar(scat1, ax=axs[0], label="Wheel speed $u_{v}$ (m/s)")  # Add colorbar

# Body frame
scat2 = axs[1].scatter(v_omega, v_x, c=colors, cmap='viridis')
axs[1].set_title("Commands in body frame")
axs[1].set_xlabel("Angular velocity $\\omega$ (rad/s)")
axs[1].set_ylabel("Linear velocity $v$ (m/s)")
fig.colorbar(scat2, ax=axs[1], label="Wheel speed $u_{v}$ (m/s)")  # Add colorbar

plt.show()



