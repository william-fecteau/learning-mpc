"""Retime a path subject to kinematic constraints
=================================================

In this example, we will see how can we retime a generic spline-based
path subject to kinematic constraints. This is very simple to do with
`toppra`, as we shall see below. First import the library.

"""

import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo
import numpy as np
import matplotlib.pyplot as plt
import time

ta.setup_logging("INFO")

################################################################################
# We generate a path with some random waypoints.

dt = 0.1
v_max = 0.5

theta = np.linspace(0, 2*np.pi, 100)
x = np.cos(theta)
y = np.sin(theta)
phi = theta + np.pi/2.0
way_pts = np.column_stack((x, y, phi))
vlims = np.array([[-v_max, v_max],
                  [-v_max, v_max],
                  [-v_max, v_max]])

ss = np.linspace(0, 1, 100)

################################################################################
# Define the geometric path and two constraints.
path = ta.SplineInterpolator(ss, way_pts)
pc_vel = constraint.JointVelocityConstraint(vlims)
# pc_acc = constraint.JointAccelerationConstraint(alims)

################################################################################
# We solve the parametrization problem using the
# `ParametrizeConstAccel` parametrizer. This parametrizer is the
# classical solution, guarantee constraint and boundary conditions
# satisfaction.
instance = algo.TOPPRA([pc_vel], path, parametrizer="ParametrizeConstAccel")
jnt_traj = instance.compute_trajectory()

################################################################################
# The output trajectory is an instance of
# :class:`toppra.interpolator.AbstractGeometricPath`.
ts_sample = np.linspace(0, jnt_traj.duration, 100)
qs_sample = jnt_traj(ts_sample)
qds_sample = jnt_traj(ts_sample, 1)
qdds_sample = jnt_traj(ts_sample, 2)
# fig, axs = plt.subplots(3, 1, sharex=True)
# for i in range(path.dof):
#     # plot the i-th joint trajectory
#     axs[0].plot(ts_sample, qs_sample[:, i], c="C{:d}".format(i))
#     axs[1].plot(ts_sample, qds_sample[:, i], c="C{:d}".format(i))
#     axs[2].plot(ts_sample, qdds_sample[:, i], c="C{:d}".format(i))
# axs[2].set_xlabel("Time (s)")
# axs[0].set_ylabel("Position (rad)")
# axs[1].set_ylabel("Velocity (rad/s)")
# axs[2].set_ylabel("Acceleration (rad/s2)")
ts_sample = np.arange(0, jnt_traj.duration, dt)

points = []
for t in ts_sample:
    q = jnt_traj(t)
    points.append([q[0], q[1], np.atan2(np.sin(q[2]), np.cos(q[2]))])

points = np.array(points)
np.savetxt("traj.np", points)

plt.plot(way_pts[:, 0], way_pts[:, 1])
# plt.scatter(points[:, 0], points[:, 1])
uu = np.cos(points[:, 2])
vv = np.sin(points[:, 2])
quiver = plt.quiver(points[:, 0], points[:, 1], uu, vv, angles='xy', scale_units='xy', scale=10)
plt.show()


################################################################################
# Optionally, we can inspect the output.
# instance.compute_feasible_sets()
# instance.inspect()