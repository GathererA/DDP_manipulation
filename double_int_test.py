import numpy as np
from planarBox import PlanarBox
from iLQR import iLQR


# Test box object and dynamics
print("Define planar box object and contact points")
box = PlanarBox()

# TODO: Add gravity force. For now, we have 0 external forces

# Test iLQR for box with contact force inputs
Q = np.eye(box.m) * 2
R = np.eye(box.n)
Qf = np.eye(box.m) * 10

start_state = np.array([[0, 0, 0, 0, 0, 0]]).T
goal_state = np.array([[0, 0.5, 0, 0, 0, 0]]).T

num_steps = 100
dt = 0.01

controller = iLQR(Q, R, Qf, start_state, goal_state, num_steps, dt, box)

print("\nRun iLQR")
controller.run_iLQR()

