import numpy as np
from planarBox import *
from limits import *
from iLQR_costbased import *
import matplotlib.pyplot as plt

"""
Augmented Lagrangian Controller (Outer Loop)
"""

class aug_lag():

	def __init__(self,mu0,tau0,x0,lambda0,constaints,ilqr,obj):
		self.mu = mu0
		self.tau = tau0
		self.x = x0
		self.lambda0 = lambda0
		self.constraints = constraints

		# implment the "vanilla" ilqr
		self.ilqr = ilqr

		# find the number of states/controls
		self.m = self.ilqr.Q.shape[0] # State dimension
		self.n = self.ilqr.R.shape[0] # Control dimension
		self.num_steps = self.ilqr.num_steps # number of steps

	    # Handle box dynamics
		f, self.dyn_fun = obj.dynamics(obj.s, obj.u, obj.cp_list)
		self.obj = obj		


	def minimizer(self):
		# at each iteration, we need to call a minimizer, 
		# which in this case is a iLQR. This just minimizes
		# the cost function given the lambda/mu params

		# we have the initial iLQR so extract those params
		Q = self.ilqr.Q
		R = self.ilqr.R
		Qf = self.ilqr.Qf
		start_state = self.ilqr.start_state
		goal_state = self.ilqr.goal_state
		dt = self.ilqr.dt

		# change the dynamics to include constraints
		obj = PlanarBoxConst(self.constraints)

		# generate a new iLQR with modified dynamics
		return 0

print("Run iLQR on box with params:")
width = 0.2
height = 0.3
mass = 0.1
cp_params = [-1, -0.7, 1, 0.7]
box = PlanarBox(width, height, mass, cp_params)
print("Width: {}, Height: {}, Mass: {}".format(width, height, mass))
print("Contact point frames w.r.t. object frame: {}".format(box.cp_list)) 

# Define iLQR
Q = np.eye(box.m) * 2
R = np.eye(box.n)
Qf = np.eye(box.m) * 10

start_state = np.array([[0., 0., 0., 0, 0, 0]]).T
goal_state = np.array([[0, 0.5, 1, 0, 0, 0]]).T

num_steps = 100
dt = 0.1

controller = iLQR(Q, R, Qf, start_state, goal_state, num_steps, dt, box)

print("\nRunning iLQR...\n")
s_bar, u_bar, l_arr, L_arr = controller.run_iLQR()

####################### Visualize optimal trajectory ##########################
# State
s_times = np.linspace(0, num_steps*dt, num_steps+1)
labels = ["x", "y", "theta", "dx", "dy", "dtheta"]
plt.figure()
for i in range(box.m):
  plt.plot(s_times, s_bar[i,:], label=labels[i])
plt.legend()
plt.title("Box state - goal state: {}".format(goal_state.T))
plt.xlabel("Time")
plt.show()

# Contact forces
plt.figure()
times = np.linspace(0, num_steps*dt, num_steps)
labels = ["cp1_x", "cp1_y", "cp2_x", "cp2_y"]
for i in range(box.n):
  plt.plot(times, u_bar[i,:], label=labels[i])
plt.legend()
plt.title("Control input")
plt.xlabel("Time")
plt.show()

# ######################## Simulate system ##############################
# print("Simulating box with params:")
# sim_width = 0.2
# sim_height = 0.3
# sim_mass = 0.1
# sim_cp_params = [-1, -0.7, 1, 0.7]
# sim_box = PlanarBox(sim_width, sim_height, sim_mass, sim_cp_params)
# print("Width: {}, Height: {}, Mass: {}".format(sim_width, sim_height, sim_mass))
# print("Contact point frames w.r.t. object frame: {}".format(sim_box.cp_list)) 

# ep_length = num_steps
# # Change the process noise variance here. 0 for no noise.
# noise_variance = 0
# #noise_variance = 0.001

# # Arrays to hold state and control
# s = np.zeros((sim_box.m, ep_length + 1))
# s[:, 0] = start_state[:, 0]
# u = np.zeros((sim_box.n, ep_length))

# for t in range(ep_length):
#   u[:,t] = u_bar[:,t] + l_arr[:,t] + L_arr[:,:,t] @ (s[:,t] - s_bar[:,t])
#   w = noise_variance * np.random.rand(sim_box.m)
#   s[:,t+1] = np.squeeze(s[:,t] + dt * sim_box.dyn_fun(s[:,t], u[:,t])) + w

# ####################### Visualize trajectory ##########################
# # State
# s_times = np.linspace(0, ep_length*dt, ep_length+1)
# labels = ["x", "y", "theta", "dx", "dy", "dtheta"]
# plt.figure()
# for i in range(box.m):
#   plt.plot(s_times, s[i,:], label=labels[i])
# plt.legend()
# plt.title("Box state - goal state: {}".format(goal_state.T))
# plt.xlabel("Time")
# plt.show()

# # Contact forces
# plt.figure()
# times = np.linspace(0, ep_length*dt, ep_length)
# labels = ["cp1_x", "cp1_y", "cp2_x", "cp2_y"]
# for i in range(box.n):
#   plt.plot(times, u[i,:], label=labels[i])
# plt.legend()
# plt.title("Control input")
# plt.xlabel("Time")
# plt.show()


# # # Testing
# # mu0 = 1
# # tau0 = 1
# # x0 = np.array([[0, 0, 0, 0, 0, 0]]).T
# # lambda0 = 1


