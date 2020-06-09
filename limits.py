import numpy as np

# each constraint is a type (input, state, others (still to come))
class constraints():

	def __init__(self,n,m):
		self.n = n # number of states
		self.m = m # number of inputs

		self.num_constraints = 0
		self.constraint_num = 0
		self.constraint_list = []

	def add_input_constraint(self,limits):
			self.constraint_num += 1
			self.num_constraints += limits.size
			self.constraint_list.append(control_constraint(limits))

	def add_state_constraint(self,limits):
			self.constraint_num += 1
			self.num_constraints += limits.size
			self.constraint_list.append(state_constraint(limits))

	def add_friction_constraint(self,mu):
			self.constraint_num += 1
			self.num_constraints += self.m
			self.constraint_list.append(friction_constraint(mu))

	def evaluate_constraints(self,s,u):

		violations = np.array([])

		for k in range(np.size(self.constraint_list)):
			# print(self.constraint_list[k])
			if self.constraint_list[k].constraint_type == 0:
				for i in range(self.m):
					violations = np.append(violations, self.constraint_list[k].limits[0,i] - u[i])
					violations = np.append(violations, u[i] - self.constraint_list[k].limits[1,i])

			if self.constraint_list[k].constraint_type == 1:
				for i in range(self.n):
					violations = np.append(violations, self.constraint_list[k].limits[0,i] - s[i])
					violations = np.append(violations, s[i] - self.constraint_list[k].limits[1,i])

			if self.constraint_list[k].constraint_type == 2:
				num_pairs = self.m // 2
				for i in range(num_pairs):
					violations = np.append(violations, -u[num_pairs*i+1] - self.constraint_list[k].mu * u[num_pairs*i])
					violations = np.append(violations, u[num_pairs*i+1] - self.constraint_list[k].mu * u[num_pairs*i])
					#violations = np.append(violations, -u[num_pairs*i])
		
		return violations


class control_constraint():

	def __init__(self,limits):
		self.constraint_type = 0
		self.limits = limits

class state_constraint():

	def __init__(self,limits):
		self.constraint_type = 1
		self.limits = limits

class friction_constraint():

	def __init__(self,mu):
		self.constraint_type = 2
		self.mu = mu


# testing
control_limits = np.array([[-1,-1,-1,-1],[1,1,1,1]])
state_limits = np.array([[-1,-1,-1,-1,-1,-1],[1,1,1,1,1,1]])

c = constraints(6,4)
c.add_input_constraint(control_limits)
c.add_state_constraint(state_limits)
c.add_friction_constraint(1)

x = np.array([0,0,0,0,0,0])
u = np.array([0,0,0,0])

# print(c.constraint_list[0].limits)

#print(c.evaluate_constraints(x,u))

