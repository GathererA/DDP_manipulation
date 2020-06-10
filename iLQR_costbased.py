import numpy as np
import matplotlib.pyplot as plt

class iLQR():

	def __init__(self, Q, R, Qf, s_bar, u_bar, goal_s, num_steps, dt, constraint_obj, obj):

		# add aug lag for now
		self.gamma = 1
		# these are the weighting matrixes for unconstrained goals
		self.Q = Q
		self.Qf = Qf
		self.R = R

		self.s_bar = s_bar
		self.u_bar = u_bar
		self.goal_s = goal_s

		self.num_steps = num_steps
		self.dt = dt

		self.m = Q.shape[0] # State dimension
		self.n = R.shape[0] # Control dimension

		self.epsilon = 0.001 # Termination threshold for iLQR

		# Handle box dynamics
		f, self.dyn_fun = obj.dynamics(obj.s, obj.u, obj.cp_list)
		self.obj = obj

		# initial constraint variables
		self.mu = 0
		self.lambda0 = 0

		self.constraint_obj = constraint_obj

	def stage_cost(self, s, u):
		c = 0

		# add state weights
		c += 1/2*(s.reshape((self.m,1)) - self.goal_s).T @ self.Q @ (s.reshape((self.m,1)) - self.goal_s)

		# add input weights
		c += 1/2*(u).T @ self.R @ (u)

		return c

	def get_constraints(self,s,u):

		# constraints = np.array([])
		# print(self.constraints)
		# for i in range(self.n):
		# 	constraints[i] = (self.input_constraints[0,i] - u[i])
		# 	constraints[i+self.n] = (u[i] - self.input_constraints[1,i])
		# 	constraints[i + self.n * 2] = (self.state_constraints[0,i] - s[i])
		# 	constraints[i + self.n * 2 + self.m] = (s[i] - self.state_constraints[1,i])

		return self.constraint_obj.evaluate_constraints(s,u)

	def get_penalty_multiplier(self,s,u,t,lambda0):

		constraints = self.get_constraints(s,u)

		I = np.zeros((np.size(constraints),np.size(constraints)))

		for i in range(np.size(constraints)):
			if constraints[i] > 0 and lambda0[i] != 0:
				I[i,i] = self.mu[i,t]

		return I

	def get_constraint_jacobians(self,s,u):

		delta = .001

		constraints = self.get_constraints(s,u)

		num_const = np.size(constraints)

		constraints_x = np.zeros((num_const,self.m))
		for i in range(self.m):
			s_diff = np.copy(s)
			s_diff[i] += delta
			constraints_x[:,i] = ((self.get_constraints(s_diff,u) \
				- self.get_constraints(s,u))/delta).flatten()

		constraints_u = np.zeros((num_const,self.n))
		for i in range(self.n):
			# create the delta
			u_diff = np.copy(u)
			u_diff[i] += delta

			constraints_u[:,i] = ((self.get_constraints(s,u_diff) \
				- self.get_constraints(s,u))/delta).flatten()

		return constraints_x, constraints_u

	def cost_jacobians(self, s, u):
		# calculates a gradient by taking small steps and 
		# then performing finite difference

		delta = .001

		c = self.stage_cost(s,u)

		c_x = np.zeros((self.m,1))
		for i in range(self.m):
			# create the delta
			s_diff = np.copy(s)
			s_diff[i] += delta

			c_x[i] = (self.stage_cost(s_diff,u) - self.stage_cost(s,u))/delta

		c_u = np.zeros((self.n,1))
		for i in range(self.n):
			# create the delta
			u_diff = np.copy(u)
			u_diff[i] += delta

			c_u[i] = (self.stage_cost(s,u_diff) - self.stage_cost(s,u))/delta
		
		c_xx = np.zeros((self.m,self.m))
		for i in range(self.m):
			for j in range(self.m):
				s_diff1 = np.zeros(s.shape)
				s_diff1[i] += delta
				s_diff2 = np.zeros(s.shape)
				s_diff2[j] += delta

				c_xx[i,j] = (self.stage_cost(s + s_diff1 + s_diff2,u) - \
					self.stage_cost(s + s_diff1,u) - \
					self.stage_cost(s + s_diff2,u) + \
					self.stage_cost(s,u))/delta/delta

		c_ux = np.zeros((self.n,self.m))
		for i in range(self.m):
			for j in range(self.n):
				s_diff = np.zeros(s.shape)
				s_diff1[i] += delta
				u_diff = np.zeros(u.shape)
				u_diff[j] += delta

				c_ux[j,i] = (self.stage_cost(s + s_diff, u + u_diff) - \
					self.stage_cost(s + s_diff,u) - \
					self.stage_cost(s, u + u_diff) + \
					self.stage_cost(s,u))/delta/delta

		c_uu = np.zeros((self.n,self.n))
		for i in range(self.n):
			for j in range(self.n):
				u_diff1 = np.zeros(u.shape)
				u_diff1[i] += delta
				u_diff2 = np.zeros(u.shape)
				u_diff2[j] += delta

				c_uu[i,j] = (self.stage_cost(s, u + u_diff1 + u_diff2) - \
					self.stage_cost(s, u + u_diff1) - \
					self.stage_cost(s, u + u_diff2) + \
					self.stage_cost(s,u))/delta/delta

		return c,c_x,c_u,c_xx,c_ux,c_uu

	def total_cost(self,s_bar,u_bar):
		# find the total cost for a trajectory
		cost = 0
		for t in range(self.num_steps):
			s = s_bar[:,t]
			u = u_bar[:,t]
			cost += self.stage_cost(s, u)

		return cost


	def forward_pass(self,u_bar_prev,s_bar_prev,l_arr,L_arr,Qu,Quu):

		s_bar = np.copy(s_bar_prev)
		u_bar = np.copy(u_bar_prev)
		
		for t in range(self.num_steps):
			# Update control input
			alpha = .5 # scaling term
			u_bar[:, t] = np.squeeze(u_bar[:, t] + L_arr[:,:,t] @ (s_bar[:,t] - s_bar_prev[:,t]) + alpha * l_arr[:,t])
			# Update state
			s_curr = s_bar[:, t]
			u_curr = u_bar[:, t]
			s_bar[:,t+1] = np.squeeze(s_curr + self.dt * self.dyn_fun(s_curr, u_curr))

		regularized = True

		"""
		z = 0
		alpha = 1
		count = 0
		converged = False
		regularized = True
		while not converged and count < 10:

			# calculate forward trajectory
			for t in range(self.num_steps):
				# Update state
				u_bar[:,t] = np.squeeze(u_bar_prev[:, t] + L_arr[:,:,t] @ (s_bar[:,t] - s_bar_prev[:,t]) + alpha * l_arr[:,t])
				s_curr = s_bar[:, t]
				u_curr = u_bar[:,t]
				s_bar[:,t+1] = np.squeeze(s_curr + self.dt * self.dyn_fun(s_curr, u_curr))

		
			# find the delV term
			delV = 0
			for t in range(self.num_steps-1):

				delV += alpha * l_arr[:,t].T @ Qu[:,t] + alpha ** 2 * .5 * l_arr[:,t].T @ Quu[:,:,t] @ l_arr[:,t]

			z = (self.total_cost(s_bar,u_bar) - self.total_cost(s_bar_prev,u_bar_prev))/ (delV)
			#print(alpha)
			#print(z)
	
			if z > 1e-4 and z < 10:
				converged = True

			alpha *= .5
			count += 1


		if count == 10:
			regularized = False
		"""
		return u_bar,s_bar, regularized

	def run_iLQR(self,mu,lambda0):

		# update mu
		self.mu = mu
		self.lambda0 = lambda0

		# are constraints satisfied? 
		satisfied = True

		u_bar_prev = np.ones(self.u_bar.shape) # Aribrary values that will not result in termination
		s_bar_prev = self.s_bar
		s_bar = self.s_bar
		u_bar = self.u_bar

		# Arrays to save gains
		l_arr = np.zeros((self.n, self.num_steps))
		L_arr = np.zeros((self.n, self.m, self.num_steps))
		v_const_arr = np.zeros((1,self.num_steps))
		v_arr = np.zeros((self.n, self.num_steps))
		V_arr = np.zeros((self.n, self.n, self.num_steps))

		# Box dynamics jacobians
		self.dfds, self.dfdu = self.obj.dynamics_jacobians(self.obj.s, self.obj.u, self.obj.cp_list)

		Q = self.Q
		R = self.R

		count = 0
	    # Loop until convergence 
		while (np.linalg.norm(u_bar - u_bar_prev) > self.epsilon and count < 10):
			delta = np.linalg.norm(u_bar - u_bar_prev) # for debugging
			print(delta)

			rho = 1e-6 # regularization parameter
			regularized = False
			while not regularized:
				V = self.Qf

				v = self.Qf @ (np.expand_dims(s_bar[:, -1],1) - self.goal_s)

				v_const = 1/2 * self.goal_s.T @ self.Qf @ self.goal_s

				Qu_vec = np.zeros((self.n,self.num_steps))
				Quu_vec = np.zeros((self.n,self.n,self.num_steps))

				for t in range(self.num_steps-1,-1,-1):
					s_curr = np.expand_dims(s_bar[:, t],1)
					u_curr = np.expand_dims(u_bar[:, t],1)
					l,L,v_const,v,V,Qu,Quu = self.backward_riccati_recursion(\
						s_curr,u_curr,V,v,v_const,rho,t)

					# Save l and L and Qu and Quu
					l_arr[:,t] = np.squeeze(l)
					L_arr[:,:,t] = L
					Qu_vec[:,t] = np.squeeze(Qu)
					Quu_vec[:,:,t] = Quu

				# Forward pass
				u_bar_prev = np.copy(u_bar)
				s_bar_prev = np.copy(s_bar)
				u_bar,s_bar,regularized = self.forward_pass(u_bar,s_bar,l_arr,L_arr,Qu_vec,Quu_vec)

				rho *= 10
				#print(rho)

		count += 1


		# check constraints
		for t in range(self.num_steps):
			s = s_bar[:,t]
			u = u_bar[:,t]
			constraints = self.get_constraints(s,u)
			#print(constraints)
			if np.any(constraints > .05):
				satisfied = False

		plt.figure()
		times = np.linspace(0, self.num_steps*self.dt, self.num_steps)
		labels = ["cp1_x", "cp1_y", "cp2_x", "cp2_y"]
		for i in range(self.n):
		  plt.plot(times, u_bar[i,:], label=labels[i])
		plt.legend()
		plt.title("Control input")
		plt.xlabel("Time")
		plt.show()

		s_times = np.linspace(0, self.num_steps*self.dt, self.num_steps+1)
		labels = ["x", "y", "theta", "dx", "dy", "dtheta"]
		plt.figure()
		for i in range(self.m):
		  plt.plot(s_times, s_bar[i,:], label=labels[i])
		plt.legend(loc='lower right')
		plt.title("Box state - goal state: {}".format(self.goal_s.T))
		plt.xlabel("Time")
		plt.show()

		print(satisfied)
		return s_bar, u_bar, l_arr, L_arr, satisfied

	def backward_riccati_recursion(self,s,u,V,v,v_const,rho,t):
		A, B = self.obj.linearized_dynamics(s, \
			u, self.dt, self.dfds, self.dfdu)

		c,c_x,c_u,c_xx,c_ux,c_uu = self.cost_jacobians(\
			s,u)

		constraints = self.get_constraints(s,u).reshape((self.constraint_obj.num_constraints,1))
		I = self.get_penalty_multiplier(s,u,t,self.lambda0[:,t])
		constraints_x,constraints_u = self.get_constraint_jacobians(s,u)

		Q = c + v_const
		lambda_spec = self.lambda0[:,t].reshape((constraints.size,1))
		Qx = c_x + np.array(A).T @ v + constraints_x.T @ (lambda_spec + I @ constraints)
		Qu = c_u + np.array(B).T @ v + constraints_u.T @ (lambda_spec + I @ constraints)
		Qxx = c_xx + np.array(A).T @ V @ np.array(A) + constraints_x.T @ I @ constraints_x
		Qux = c_ux + np.array(B).T @ V @ np.array(A) + constraints_u.T @ I @ constraints_x
		Quu = c_uu + np.array(B).T @ V @ np.array(B) + constraints_u.T @ I @ constraints_u

		l = -np.linalg.inv(Quu + rho * np.eye(self.n)) @ Qu 
		L = -np.linalg.inv(Quu + rho * np.eye(self.n)) @ Qux 

		v_const_new = Q - 1/2 * l.T @ Quu @ l 
		v_new = Qx - L.T @ Quu @ l 
		V_new = Qxx - L.T @ Quu @ L

		return l,L,v_const_new,v_new,V_new,Qu,Quu

