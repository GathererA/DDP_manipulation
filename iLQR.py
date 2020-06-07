import numpy as np

"""
iLQR controller
"""
class iLQR():
  
  def __init__(self, Q, R, Qf, start_s, goal_s, num_steps, dt, obj):

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
 
  def run_iLQR(self):
    # Intialize nominal trajectory
    s_bar = np.zeros((self.m, self.num_steps + 1))
    s_bar[:, 0] = self.start_s[:,0]
    #u_bar = np.random.rand(self.n, self.num_steps) * 0.001 # random doesn't converge
    u_bar = np.zeros((self.n, self.num_steps))
    
    u_bar_prev = 10 * np.ones(u_bar.shape) # Aribrary values that will not result in termination
    s_bar_prev = s_bar

    # Arrays to save gains
    l_arr = np.zeros((self.n, self.num_steps))
    L_arr = np.zeros((self.n, self.m, self.num_steps))
  
    # Initial forward pass
    for t in range(self.num_steps):
      s_curr = s_bar[:, t]
      u_curr = u_bar[:, t]
      s_bar[:,t+1] = np.squeeze(s_curr + self.dt * self.dyn_fun(s_curr, u_curr))
  
    # Box dynamics jacobians
    dfds, dfdu = self.obj.dynamics_jacobians(self.obj.s, self.obj.u, self.obj.cp_list)

    Q = self.Q
    R = self.R

    # Loop until convergence 
    while (np.linalg.norm(u_bar - u_bar_prev) > self.epsilon):
      #delta = np.linalg.norm(u_bar - u_bar_prev) # for debugging
      #print(delta)

      # Cost function linear term in terminal cost
      qf = self.Qf @ (np.expand_dims(s_bar[:, -1],1) - self.goal_s)

      # Initial value terms at terminal cost
      P = self.Qf
      p = qf
      
      # Backwards pass
      for t in range(self.num_steps-1, -1, -1):
        s_curr = np.expand_dims(s_bar[:, t],1)
        u_curr = np.expand_dims(u_bar[:, t],1)
        # Linearized dynamics
        A, B = self.obj.linearized_dynamics(s_curr, u_curr, self.dt, dfds, dfdu)
        
        # Compute linear terms in cost function
        q = self.Q @ (np.expand_dims(s_bar[:, t],1) - self.goal_s)
        r = np.expand_dims(u_bar[:, t],1).T @ self.R
  
        # Riccati recursion
        l,L,p,P = self.backward_riccati_recursion(P,p,Q,q,R,r,A,B)
      
        # Save l and L
        l_arr[:,t] = np.squeeze(l)
        L_arr[:,:,t] = L

      # Forward pass
      u_bar_prev = np.copy(u_bar)
      s_bar_prev = np.copy(s_bar)
      for t in range(self.num_steps):
        # Update control input
        u_bar[:, t] = np.squeeze(u_bar[:, t] + L_arr[:,:,t] @ (s_bar[:,t] - s_bar_prev[:,t]) + l)
        # Update state
        s_curr = s_bar[:, t]
        u_curr = u_bar[:, t]
        s_bar[:,t+1] = np.squeeze(s_curr + self.dt * self.dyn_fun(s_curr, u_curr))

    return s_bar, u_bar, l_arr, L_arr

  def backward_riccati_recursion(self,P,p,Q,q,R,r,A,B):
    Su_k = r + p.T @ B
    Suu_k = R + B.T @ P @ B
    Sus_k = B.T @ P @ A

    L = -1 * np.linalg.inv(Suu_k) @ (Sus_k)
    l = -1 * np.linalg.inv(Suu_k) @ (Su_k.T)

    P_new = Q + A.T @ P @ A - L.T @ Suu_k @ L
    p_new = q + A.T @ p + Sus_k.T @ l

    P = P_new
    p = p_new

    return l,L,p,P

