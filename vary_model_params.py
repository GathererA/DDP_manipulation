import numpy as np
from planarBox import *
from limits import *
from iLQR_costbased import *
from animateSystem import AnimateSystem
import matplotlib.pyplot as plt

"""
Augmented Lagrangian Controller (Outer Loop)
"""

class aug_lag():

	def __init__(self,mu0,tau0,lambda0,constraint_obj,ilqr,obj):
		self.mu = mu0
		self.tau = tau0
		self.lambda0 = lambda0
		self.constraint_obj = constraint_obj

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
		s_bar = self.ilqr.s_bar
		u_bar = self.ilqr.u_bar
		goal_state = self.ilqr.goal_s
		dt = self.ilqr.dt

		# while constraints aren't met
		satisfied = False
		while not satisfied:

			# generate a new iLQR with modified dynamics
			ilqr_mod = iLQR(Q, R, Qf, s_bar, u_bar, goal_state, num_steps, dt, self.constraint_obj, box)
			s_bar, u_bar, l_arr, L_arr,satisfied = ilqr_mod.run_iLQR(self.mu,self.lambda0)
			# update parameters
			self.mu *= 1.5
			print("mu: {}".format(self.mu[0,0]))
			for t in range(self.num_steps):
				s = s_bar[:,t]
				u = u_bar[:,t]
				constraints = ilqr_mod.get_constraints(s,u)
				for i in range(constraints.size):
					if constraints[i] > 0:
						self.lambda0[i,t] = self.lambda0[i,t] + self.mu[i,t] * constraints[i]
					if constraints[i] <= 0:
						self.lambda0[i,t] = max(0,self.lambda0[i,t] + self.mu[i,t] * constraints[i])
				#print(self.lambda0)

		return s_bar, u_bar, l_arr, L_arr


print("Run iLQR on box with params:")
width = 0.2
height = 0.3
mass = 0.1
cp_params = [-1, -0.7, 1, 0.7]
box = PlanarBox(width, height, mass, cp_params)
print("Width: {}, Height: {}, Mass: {}".format(width, height, mass))
print("Contact point frames w.r.t. object frame: {}".format(box.cp_list))


############################# iLQR ####################################
Q = np.eye(box.m) * 2
R = np.eye(box.n)
Qf = np.eye(box.m) * 10

start_state = np.array([[0, 0, 0, 0, 0, 0]]).T
goal_state = np.array([[0.5, 0.5, 1, 0, 0, 0]]).T

num_steps = 100
dt = 0.1

####################### Run the Augmented Lagrangian ##########################
# define constraints
control_limits = np.array([[-.6,-.6,-.6,-.6],[.6,.6,.6,.6]])
#state_limits = np.array([[-1,-1,-1,-1,-1,-1],[.8,.8,.8,.8,.8,.8]])
state_limits = np.array([[-1,-1,-1,-1,-1,-1],[1,1,1,1,1,1]])

c = constraints(6,4)
c.add_input_constraint(control_limits)
c.add_state_constraint(state_limits)

# define constants
mu0 = np.ones((c.num_constraints,num_steps))
tau0 = 1
lambda0 = np.zeros((c.num_constraints,num_steps))

# define aug_lag
# calculate initial trajectory
f,dyn_fun = box.dynamics(box.s, box.u, box.cp_list)
s_bar = np.zeros((box.m, num_steps + 1))
s_bar[:, 0] = start_state[:,0]
u_bar = np.zeros((box.n, num_steps))
for t in range(num_steps):
	s_curr = s_bar[:, t]
	u_curr = u_bar[:, t]
	s_bar[:,t+1] = np.squeeze(s_curr + dt * dyn_fun(s_curr, u_curr))

controller = iLQR(Q, R, Qf, s_bar, u_bar, goal_state, num_steps, dt, c, box)

AL = aug_lag(mu0,tau0,lambda0,c,controller,box)

s_bar, u_bar, l_arr, L_arr = AL.minimizer()

######################## Simulate system ##############################
print("Simulating box with params:")
sim_width = 0.2
sim_height = 0.3
#sim_mass = 0.1
#sim_mass = 0.105
sim_mass = 0.095
sim_cp_params = [-1, -0.7, 1, 0.7]
sim_box = PlanarBox(sim_width, sim_height, sim_mass, sim_cp_params)
print("Width: {}, Height: {}, Mass: {}".format(sim_width, sim_height, sim_mass))
print("Contact point frames w.r.t. object frame: {}".format(sim_box.cp_list))

ep_length = num_steps
# Change the process noise variance here. 0 for no noise.
noise_variance = 0
#noise_variance = 0.001
#noise_variance = 0.01
#noise_variance = 0.1

# Arrays to hold state and control
s = np.zeros((sim_box.m, ep_length + 1))
s[:, 0] = start_state[:, 0]
u = np.zeros((sim_box.n, ep_length))

for t in range(ep_length):
  u[:,t] = u_bar[:,t] + l_arr[:,t] + L_arr[:,:,t] @ (s[:,t] - s_bar[:,t])
  w = noise_variance * np.random.rand(sim_box.m)
  s[:,t+1] = np.squeeze(s[:,t] + dt * sim_box.dyn_fun(s[:,t], u[:,t])) + w

####################### Visualize trajectory ##########################
# State
s_times = np.linspace(0, ep_length*dt, ep_length+1)
labels = ["x", "y", "theta", "dx", "dy", "dtheta"]
plt.figure()
for i in range(box.m):
  plt.plot(s_times, s[i,:], label=labels[i])
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

animator = AnimateSystem(sim_box, range(ep_length+1), s)
animator.draw_box(goal_state, "r")
animator.play(False, "animation")
plt.show()
