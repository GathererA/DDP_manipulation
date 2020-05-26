import numpy as np
from casadi import *

"""
2D box object
Full dynamics and dynamics linearized around a reference trajectory
start_state: [x0, y0, theta0, dx0, dy0, dtheta0]
goal_state: 
"""
class PlanarBox():
  def __init__(self, cp_params):

    # Define box parameters
    self.width = 0.2
    self.height = 0.3
    self.mass = 0.1
    self.M_obj = self.get_M_obj() # Mass matrix of box

    # Define contact points on box
    self.fnum = 2 # Number of fingers (or contact points)
    # Contact point positions
    self.cp_params = cp_params
    self.p = 100
    self.cp_list = self.set_cps(self.cp_params)
    print("Contact point frames w.r.t. object frame: {}".format(self.cp_list))
    
    # Matrix that determines which force components are transmitted through contact points
    self.H = np.array([
                      [1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                     ]) 

    self.m = 6 # State dimension
    self.n = self.fnum * 2 # Control dimension

    self.s = self.get_s()
    self.u = self.get_u()
    
    ds, self.dyn_fun = self.dynamics(self.s, self.u, self.cp_list)

  """
  Get state variables
  """
  def get_s(self):
    s = SX.sym("s",self.m,1) # 6x1 state vector
    return s
  
  """
  Unpack state vector into x, y, theta, dx, dy, dtheta
  s: 6x1 state vector
  """
  def s_unpack(self,s):
    x      = s[0,0]
    y      = s[1,0]
    theta  = s[2,0]
    dx     = s[3,0]
    dy     = s[4,0]
    dtheta = s[5,0]
    return x, y, theta, dx, dy, dtheta

  """
  Get control variables
  """
  def get_u(self):
    u = SX.sym("u",self.n,1) # 6x1 state vector
    return u

  """
  Get 3x3 object inertia matrix
  """
  def get_M_obj(self):
    M = np.diag([self.mass,self.mass, 1/12. * self.mass * (self.width ** 2 + self.height ** 2)]) 
    return M

  """
  Get grasp matrix given current object state
  s: state [x, y, theta, dx, dy, theta]
  """
  def get_G(self, s, cp_list):
    x,y,theta,dx,dy,dtheta = self.s_unpack(s)
    H_o_w = self.get_obj2world(x,y,theta)

    # Calculate G_i for each finger, and then stack
    G_iT_list = []
    for c in cp_list:
      cp_pos_of = c["p"] # Position of contact point in object frame
      cp_theta_of = c["theta"] # Orientation of contact point w.r.t. object frame
      cp_theta_wf = cp_theta_of + theta # Orientation of contact point w.r.t. world f

      P_i = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [-(cp_pos_of[1]), cp_pos_of[0], 1]])

      R_i = np.array([[np.cos(cp_theta_wf), -np.sin(cp_theta_wf), 0],
                      [np.sin(cp_theta_wf),  np.cos(cp_theta_wf), 0],
                      [0, 0, 1]])
      G_iT = R_i.T @ P_i.T
      G_iT_list.append(G_iT)

    GT_full = np.concatenate(G_iT_list)
    GT = self.H @ GT_full # Only take components of G affected by contact points

    return GT.T

  """
  Get object frame to world frame transformation matrix
  """
  def get_obj2world(self, obj_x, obj_y, obj_theta):
    H_o_w = np.array([
                   [np.cos(obj_theta), -np.sin(obj_theta), obj_x],
                   [np.sin(obj_theta), np.cos(obj_theta), obj_y],
                   [0, 0, 1]
                  ])
    return H_o_w

  """
  Define contact point position and ref frame in object reference frame
  Each contact point is parametrized with 2 floats: 1 float for x position, and 1 for y position
  in the format: [cp_x, cp_y]
  Top-right corner: [1, 1]
  Top-left corner: [-1, 1]
  Lower-right corner: [1, -1]
  Lower-left corner: [-1, -1]
  
  Input:
  cp_params: list of contact point paramters [cp1_x, cp1_y,...,cpn_x, cpn_y]
  
  Output: 
  c_list: List of contact point dicts
  Each contact point represented with a dict with the entries:
        "p": contact point position w.r.t. object reference frame
        "theta": contact point reference frame orientation w.r.t. object reference frame
  """
  def set_cps(self, cp_params):
    c_list = []
    p = self.p
    for i in range(self.fnum):
      # Parameters for x and y coord
      x_param = cp_params[i*2]
      y_param = cp_params[i*2+1]

      # x and y locations of contact point, scaled to object dimensions
      x = (-self.width / 2) + self.width * (x_param+1)/2
      y = (-self.height / 2) + self.height * (y_param+1)/2
  
      # Derivative of p-norm is norm
      pnorm_x = (fabs(x_param)**p + fabs(y_param)**p) ** (1/p)
      dx = (x_param * (fabs(x_param) ** (p - 2))) / (pnorm_x**(p-1))
      dy = (y_param * (fabs(y_param) ** (p - 2))) / (pnorm_x**(p-1))
      cp_theta = np.arctan2(-dy, -dx)

      cp_pos = [x, y, 1]
      c = {
            "p": cp_pos,
            "theta": cp_theta
          }

      c_list.append(c)

    return c_list  

  """
  Compute dynamics ds
  s: state (6x1 vector)
  u: contact force input (4x1 vector)
  Returns
  ds: derivative of state (6x1 vector)
  dyn_fun: Casadi function of dynamics, to be use for iLQR
  """
  def dynamics(self, s, u, cp_list):
    x,y,theta,dx,dy,dtheta = self.s_unpack(s)
    
    ds_list = []

    # Compute ddx, ddy, and ddtheta
    G = self.get_G(s, cp_list)
    dd = inv(self.M_obj) @ (G @ u)

    ds_list.append(dx)
    ds_list.append(dy)
    ds_list.append(dtheta)
    ds_list.append(dd[0])
    ds_list.append(dd[1])
    ds_list.append(dd[2])

    ds = vertcat(*ds_list)

    dyn_fun = Function("dyn", [s,u], [ds])
    return ds, dyn_fun

  """
  Use Casadi autodiff to compute jacobians of state dynamics w.r.t. s and u
  Returns
  dfds: Jacobian w.r.t. state
  dfdu: Jacobian w.r.t control input
  """
  def dynamics_jacobians(self,s,u,cp_list):
    f, dyn_fun = self.dynamics(s,u,cp_list)
    dfds = Function("dfds", [s,u], [jacobian(f, s)])
    dfdu = Function("dfdu", [s,u], [jacobian(f, u)])
    return dfds, dfdu

  """
  Linearize object dynamics around a reference trajectory
  """
  def linearized_dynamics(self, s_ref, u_ref, dt, dfds, dfdu):
    A = np.eye(self.m) + dt * dfds(s_ref,u_ref)
    B = dt * dfdu(s_ref,u_ref)
    return A, B

