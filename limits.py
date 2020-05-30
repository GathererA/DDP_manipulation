import numpy as np

# each constraint is a type (input, state, others (still to come))
class constraints():

	def __init__(self,n,m):
		self.n = n # number of states
		self.m = m # number of inputs

		self.constraint_num = 0
		self.constraint_list = []

	def add_constraint(self,constraint_type,limits):
		if (constraint_type == 0):
			self.constraint_num += 1
			self.constraint_list.append(control_constraint(limits))

		if (constraint_type == 1):
			self.constraint_num += 1
			self.constraint_list.append(state_constraint(limits))


	def g(self,x,u):
		# takes in a single time-specific state and control

		state_violations = np.zeros((self.n,2))
		control_violations = np.zeros((self.m,2))
		for i in range(self.n):
			for j in range(self.constraint_num):
				if self.constraint_list[j].constraint_type == 1:
					state_violations[i,0] = x[i]-self.constraint_list[j].limits[0,i]
					state_violations[i,1] = x[i]-self.constraint_list[j].limits[1,i]
		
		for i in range(self.m):
			for j in range(self.constraint_num):
				if self.constraint_list[j].constraint_type == 0:
					control_violations[i,0] = x[i]-self.constraint_list[j].limits[0,i]
					control_violations[i,1] = x[i]-self.constraint_list[j].limits[1,i]

		return state_violations,control_violations		

class control_constraint():

	def __init__(self,limits):
		self.constraint_type = 0
		self.limits = limits

class state_constraint():

	def __init__(self,limits):
		self.constraint_type = 1
		self.limits = limits

# testing
control_limits = np.array([[-1,-1,-1,-1],[1,1,1,1]])
state_limits = np.array([[-1,-1,-1,-1,-1,-1],[1,1,1,1,1,1]])

c = constraints(6,4)
c.add_constraint(0,control_limits)
c.add_constraint(1,state_limits)

x = np.array([0,0,0,0,0,0])
u = np.array([0,0,0,0])

# print(c.constraint_list[0].limits)

state_violations, control_violations = c.g(x,u)

# print(control_violations)
