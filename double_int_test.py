import numpy as np
from planarBox import PlanarBox
from iLQR import iLQR
import matplotlib.pyplot as plt


# Test box object and dynamics
print("Define planar box object and contact points")
box = PlanarBox()

# TODO: Add gravity force. For now, we have 0 external forces

# Test iLQR for box with contact force inputs
Q = np.eye(box.m) * 2
R = np.eye(box.n)
Qf = np.eye(box.m) * 10

start_state = np.array([[0, 0, 0, 0, 0, 0]]).T
goal_state = np.array([[0, 0.5, 1, 0, 0, 0]]).T

num_steps = 1000
dt = 0.01

controller = iLQR(Q, R, Qf, start_state, goal_state, num_steps, dt, box)

print("\nRun iLQR")
s_bar, u_bar, l_arr, L_arr = controller.run_iLQR()

# Simulate system
ep_length = 1000

# Arrays to hold state and control
s = np.zeros((box.m, ep_length + 1))
s[:, 0] = start_state[:, 0]
u = np.zeros((box.n, ep_length))

for t in range(ep_length):
  u[:,t] = u_bar[:,t] + l_arr[:,t] + L_arr[:,:,t] @ (s[:,t] - s_bar[:,t])
  s[:,t+1] = np.squeeze(s[:,t] + dt * box.dyn_fun(s[:,t], u[:,t]))


# Plot trajectory
# State
times = np.linspace(0, ep_length*dt, ep_length+1)
labels = ["x", "y", "theta", "dx", "dy", "dtheta"]
plt.figure()
for i in range(box.m):
  plt.plot(times, s[i,:], label=labels[i])
plt.legend()
plt.title("Box state - goal state: {}".format(goal_state.T))
plt.xlabel("Time")

# Contact forces
plt.figure()
times = np.linspace(0, ep_length*dt, ep_length)
labels = ["cp1_x", "cp1_y", "cp2_x", "cp2_y"]
for i in range(box.n):
  plt.plot(times, u[i,:], label=labels[i])
plt.legend()
plt.title("Control input")
plt.xlabel("Time")
plt.show()
