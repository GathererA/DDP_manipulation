"""
single_integrator_dynamics.py

single integrator dynamics
state: [xo, yo, theta] => pose of object
control: [vcp1x, vcp1y, vcp2x, vcp2y] => velocity of two control points
"""

# CONSTANTS
# assume box dimensions of 4 units wide, 2 units height
# location of contact points
rho1 = np.array([-1,1])
rho2 = np.array([3,-1])

"""
f_single_integrator(control)
    This function contains the single integrator continuous dynamics and returns xdot
"""
def f_single_integrator(control):
    vcp1x = control[0]
    vcp1y = control[1]
    vcp2x = control[2]
    vcp2y = control[3]
    omega = (vcp2y-vcp1y)/(rho2[0]-rho1[0]) # eq2 and eq4
    vx = vcp1x + rho1[1]*omega # eq1
    vy = vcp1y - rho1[0]*omega # eq2
    xdot = np.array([vx,vy,omega])
    return xdot

"""
f_discrete_single_integrator(state_discrete,control)
    This function contains the discrete dynamcs of the single integrator dynamics model
    and returns the next state x(t+1)
"""
def f_discrete_single_integrator(state_discrete,control):
    vcp1x = control[0]
    vcp1y = control[1]
    vcp2x = control[2]
    vcp2y = control[3]
    xo = state_discrete[0]
    yo = state_discrete[1]
    theta = state_discrete[2]
    omega = (vcp2y-vcp1y)/(rho2[0]-rho1[0]) # eq2 and eq4
    xo_next = xo + (vcp1x + rho1[1]*omega)*dt # eq1
    yo_next = yo + (vcp1y - rho1[0]*omega)*dt # eq2
    theta_next = theta + ((vcp2y-vcp1y)/(rho2[0]-rho1[0]))*dt
    x_next = np.array([xo_next,yo_next,theta_next])
    return x_next

"""
derivatives_single_integrator()
    This function returns dfdx and dfdu of the continuous single integratory dynamics model
"""
def derivatives_single_integrator():
    dfdx = np.zeros((3))
    dfdu = np.array([[1,0,0,0],[0,1,0,0],[0, -1/(rho2[0]-rho1[0]), 0, 1/(rho2[0]-rho1[0])]])
    return dfdx,dfdu

"""
constraints_single_integrator(control)
    This function contains an equality constraint
"""
def constraints_single_integrator(control):
    vcp1x = control[0]
    vcp1y = control[1]
    vcp2x = control[2]
    vcp2y = control[3]
    omega = (vcp2y-vcp1y)/(rho2[0]-rho1[0]) # eq2 and eq4
    c = (vcp1x + rho1[1]*omega) - (vcp2x + rho2[1]*omega) # eq1 and eq3
    return c # if constraint is satisfied, c = 0
