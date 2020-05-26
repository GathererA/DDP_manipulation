import numpy as np
from iLQR import *
from planarBox import *
from planarboxconst import *
from limits import *
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

	def log_barrier(self,s_bar,u_bar):
		# let's use the log-barrier function to enforce
		# constraints

		# takes in the class as well as a control and state
		# trajectory

		cost = 0

		temp = self.constraints.g(self.constraints,s_bar,u_bar)

		return temp


# Testing
mu0 = 1
tau0 = 1
x0 = np.array([[0, 0, 0, 0, 0, 0]]).T
lambda0 = 1
# Test box object and dynamics
print("Define planar box object and contact points")
box = PlanarBox()

# Define iLQR
Q = np.eye(box.m) * 2
R = np.eye(box.n)
Qf = np.eye(box.m) * 10

start_state = np.array([[0, 0, 0, 0, 0, 0]]).T
goal_state = np.array([[0, 0.5, 1, 0, 0, 0]]).T

num_steps = 1000
dt = 0.01

controller = iLQR(Q, R, Qf, start_state, goal_state, num_steps, dt, box)

# define limits
# testing
control_limits = np.array([[-1,-1,-1,-1],[1,1,1,1]])
state_limits = np.array([[-1,-1,-1,-1,-1,-1],[1,1,1,1,1,1]])

c = constraints(control_limits,state_limits)

AL = aug_lag(mu0,tau0,x0,lambda0,c,controller,box)
