import numpy as np

class constraints():

	def __init__(self,control_limits,state_limits):
		self.ulimits = control_limits
		self.slimits = state_limits

		# extract sizes
		self.n = np.size(self.slimits[0])
		self.m = np.size(self.ulimits[0])
		
	def g(self,x,u):
		# takes in a single time-specific state and control

		return x - self.slimits[0],x - self.slimits[1], u - self.ulimits[0], u - self.ulimits[1]

# # testing
# control_limits = np.array([[-1,-1,-1,-1],[1,1,1,1]])
# state_limits = np.array([[-1,-1,-1,-1,-1,-1],[1,1,1,1,1,1]])

# c = constraints(control_limits,state_limits)

# x = np.array([0,0,0,0,0,0])
# u = np.array([0,0,0,0])

# print(c.g(x,u))
