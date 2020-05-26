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
    x_bar = np.zeros((self.m, self.num_steps + 1))
    x_bar[:, 0] = self.start_s[:,0]
    u_bar = np.random.rand(self.n, self.num_steps)
    
    # Hard code input to test dynamics
    # Seems to give expected forward pass results for object state
    u_bar = np.zeros(u_bar.shape)
    #u_bar[0, :] = -1
    #u_bar[2, :] = 1
    #u_bar[1, :] = 1
    #u_bar[3, :] = 1

    u_bar_prev = 10 * np.ones(u_bar.shape) # Aribrary values that will not result in termination
    x_bar_prev = x_bar
  
    # Initial forward pass
    for t in range(self.num_steps):
      x_curr = x_bar[:, t]
      u_curr = u_bar[:, t]
      x_bar[:,t+1] = np.squeeze(x_curr + self.dt * self.dyn_fun(x_curr, u_curr))
  
    #print(x_bar[:,-1])
    #quit()

    # Box dynamics jacobians
    dfds, dfdu = self.obj.dynamics_jacobians(self.obj.s, self.obj.u, self.obj.cp_list)

    Q = self.Q
    R = self.R

    # Loop until convergence 
    while (np.linalg.norm(u_bar - u_bar_prev) > self.epsilon):
      delta = np.linalg.norm(u_bar - u_bar_prev)
      print(delta)

      # Cost function linear term in terminal cost
      qf = self.Qf @ (np.expand_dims(x_bar[:, -1],1) - self.goal_s)

      # Initial value terms at terminal cost
      P = self.Qf
      p = qf
      
      # Backwards pass
      for t in range(self.num_steps-1, -1, -1):
        x_curr = np.expand_dims(x_bar[:, t],1)
        u_curr = np.expand_dims(u_bar[:, t],1)
        # Linearized dynamics
        A, B = self.obj.linearized_dynamics(x_curr, u_curr, self.dt, dfds, dfdu)
        
        # Compute linear terms in cost function
        q = self.Q @ (np.expand_dims(x_bar[:, t],1) - self.goal_s)
        r = np.expand_dims(u_bar[:, t],1).T @ self.R
  
        # Riccati recursion
        l,L,p,P = self.backward_riccati_recursion(P,p,Q,q,R,r,A,B)

      # Forward pass
      #print("ubar: {}".format(u_bar))
      #print("xbar: {}".format(x_bar))
      u_bar_prev = np.copy(u_bar)
      x_bar_prev = np.copy(x_bar)
      for t in range(self.num_steps):
        # Update control input
        u_bar[:, t] = np.squeeze(u_bar[:, t] + L @ (x_bar_prev[:,t] - x_bar[:,t]) + l)
        # Update state
        x_curr = x_bar[:, t]
        u_curr = u_bar[:, t]
        x_bar[:,t+1] = np.squeeze(x_curr + self.dt * self.dyn_fun(x_curr, u_curr))
      #print("new ubar: {}".format(u_bar))
      #print("new xbar: {}".format(x_bar))
      #quit()

  def backward_riccati_recursion(self,P,p,Q,q,R,r,A,B):
    Su_k = r + p.T @ B
    Suu_k = R + B.T @ P @ B
    Sux_k = B.T @ P @ A

    L = -1 * np.linalg.inv(Suu_k) @ (Sux_k)
    l = -1 * np.linalg.inv(Suu_k) @ (Su_k.T)

    P_new = Q + A.T @ P @ A - L.T @ Suu_k @ L
    p_new = q + A.T @ p + Sux_k.T @ l

    P = P_new
    p = p_new

    return l,L,p,P

