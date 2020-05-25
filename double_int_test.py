import numpy as np
from planarBox import PlanarBox

start_state = np.array([[0, 0, 0, 0, 0, 0]]).T
box = PlanarBox(start_state)
s = box.get_s()
print(s)
G = box.get_G(s, box.cp_list)
print(G.shape)

u = box.get_u()
box.dynamics(s, u, box.cp_list)
dfds, dfdu = box.dynamics_jacobians(s,u,box.cp_list)

dt = 0.1

s_ref = np.zeros((box.m,1))
u_ref = np.zeros((box.n,1))
box.linearized_dynamics(s_ref, u_ref, dt, dfds, dfdu)

