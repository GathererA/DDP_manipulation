import numpy as np
import matplotlib.pyplot as plt

class iLQR():

	def __init__(self, Q, R, Qf, start_s, goal_s, num_steps, dt, obj):

		# these are the weighting matrixes for unconstrained goals
		self.Q = Q
		self.Qf = Qf
		self.R = R

		self.start_s = start_s
		self.goal_s = goal_s

		self.num_steps = num_steps
		self.dt = dt

		self.m = Q.shape[0] # State dimension
		self.n = R.shape[0] # Control dimension

		self.epsilon = 0.001 # Termination threshold for iLQR

		# Handle box dynamics
		f, self.dyn_fun = obj.dynamics(obj.s, obj.u, obj.cp_list)
		self.obj = obj

		# hard code some input constraints
		self.input_constraints = np.array([[-.5,-.5,-.5,-.5],\
			[.5,.5,.5,.5]])

	def stage_cost(self, s, u):
		c = 0

		# add state weights
		c += 1/2*(s.reshape((self.m,1)) - self.goal_s).transpose() @ self.Q @ (s.reshape((self.m,1)) - self.goal_s)

		# add input weights
		c += 1/2*(u).transpose() @ self.R @ (u)

		# add constraints
		const = np.zeros((self.n,1))
		for i in range(self.n):
			if u[i] > self.input_constraints[1,i]:
				const[i] = (u[i] - self.input_constraints[1,i])
		c += np.sum(const)

		const = np.zeros((self.n,1))
		for i in range(self.n):
			if u[i] < self.input_constraints[0,i]:
				const[i] = (self.input_constraints[0,i] - u[i])
		c += np.sum(const)

		return c

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

			c_u[i] = (self.stage_cost(s,u_diff) - self.stage_cost(s,u_diff))/delta
		
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



	def run_iLQR(self):
		# Intialize nominal trajectory
		s_bar = np.zeros((self.m, self.num_steps + 1))
		s_bar[:, 0] = self.start_s[:,0]
		#u_bar = np.random.rand(self.n, self.num_steps) * 0.001 # random doesn't converge
		u_bar = np.zeros((self.n, self.num_steps))

		u_bar_prev = np.ones(u_bar.shape) # Aribrary values that will not result in termination
		s_bar_prev = s_bar

		# Arrays to save gains
		l_arr = np.zeros((self.n, self.num_steps))
		L_arr = np.zeros((self.n, self.m, self.num_steps))
		v_const_arr = np.zeros((1,self.num_steps))
		v_arr = np.zeros((self.n, self.num_steps))
		V_arr = np.zeros((self.n, self.n, self.num_steps))

		# Initial forward pass
		for t in range(self.num_steps):
			s_curr = s_bar[:, t]
			u_curr = u_bar[:, t]
			s_bar[:,t+1] = np.squeeze(s_curr + self.dt * self.dyn_fun(s_curr, u_curr))

		# Box dynamics jacobians
		self.dfds, self.dfdu = self.obj.dynamics_jacobians(self.obj.s, self.obj.u, self.obj.cp_list)

		Q = self.Q
		R = self.R

		count = 0
	    # Loop until convergence 
		while (np.linalg.norm(u_bar - u_bar_prev) > self.epsilon and count < 10):
			delta = np.linalg.norm(u_bar - u_bar_prev) # for debugging
			print(delta)


			V = self.Qf

			v = self.Qf @ (np.expand_dims(s_bar[:, -1],1) - self.goal_s)

			v_const = 1/2 * self.goal_s.transpose() @ self.Qf @ self.goal_s

			for t in range(self.num_steps-1,-1,-1):
				s_curr = np.expand_dims(s_bar[:, t],1)
				u_curr = np.expand_dims(u_bar[:, t],1)

				l,L,v_const,v,V = self.backward_riccati_recursion(\
					s_curr,u_curr,V,v,v_const)

				# Save l and L
				l_arr[:,t] = np.squeeze(l)
				L_arr[:,:,t] = L


			# Forward pass
			u_bar_prev = np.copy(u_bar)
			s_bar_prev = np.copy(s_bar)
			for t in range(self.num_steps):
				# Update control input
				u_bar[:, t] = np.squeeze(u_bar[:, t] + L_arr[:,:,t] @ (s_bar[:,t] - s_bar_prev[:,t]) + l_arr[:,t])
				# Update state
				s_curr = s_bar[:, t]
				u_curr = u_bar[:, t]
				s_bar[:,t+1] = np.squeeze(s_curr + self.dt * self.dyn_fun(s_curr, u_curr))

			count += 1

			# plt.figure()
			# times = np.linspace(0, self.num_steps*self.dt, self.num_steps)
			# labels = ["cp1_x", "cp1_y", "cp2_x", "cp2_y"]
			# for i in range(self.n):
			#   plt.plot(times, u_bar[i,:], label=labels[i])
			# plt.legend()
			# plt.title("Control input")
			# plt.xlabel("Time")
			# plt.show()


		return s_bar, u_bar, l_arr, L_arr

	def backward_riccati_recursion(self,s,u,V,v,v_const):
		A, B = self.obj.linearized_dynamics(s, \
			u, self.dt, self.dfds, self.dfdu)

		c,c_x,c_u,c_xx,c_ux,c_uu = self.cost_jacobians(\
			s,u)

		Q = c + v_const
		Qx = c_x + np.array(A).transpose() @ v 
		Qu = c_u + np.array(B).transpose() @ v 
		Qxx = c_xx + np.array(A).transpose() @ V @ np.array(A)
		Qux = c_ux + np.array(B).transpose() @ V @ np.array(A)
		Quu = c_uu + np.array(B).transpose() @ V @ np.array(B)

		l = -np.linalg.inv(Quu) @ Qu 
		L = -np.linalg.inv(Quu) @ Qux 

		v_const_new = Q - 1/2 * l.transpose() @ Quu @ l 
		v_new = Qx - L.transpose() @ Quu @ l 
		V_new = Qxx - L.transpose() @ Quu @ L

		return l,L,v_const_new,v_new,V_new

