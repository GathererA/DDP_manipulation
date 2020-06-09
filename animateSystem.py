import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib as mpl
import numpy as np

mpl.rcParams["animation.ffmpeg_path"] = "/Users/claire/miniconda3/envs/casadi/bin/ffmpeg" 

"""
Animate system trajectory
"""
class AnimateSystem():
  
  def __init__(self, obj, times, s):
    self.obj = obj
    self.times = times
    self.s = s
  
    self.fig = plt.figure()
    self.ax = self.fig.add_subplot(1,1,1)
    self.ax.set_aspect('equal')
    plt.xlim((-1, 1))
    plt.ylim((-0.5, 0.5))

    self.lines = []
    # Add lines for contact points
    for i in range(obj.fnum):
      lobj, = self.ax.plot([],[],'.',lw=1.0,markersize=15.0)
      self.lines.append(lobj)

    # Add line for object animation
    lobj, = self.ax.plot([],[],'g')
    self.lines.append(lobj)

  def play(self, save_mp4, save_string):
    # Set up formatting for the movie files
    Writer = anim.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
  
    a = anim.FuncAnimation(self.fig, self.animate_system, init_func = self.animate_init, frames = self.times, interval = 50, blit=True, repeat_delay=5000)
  
    if save_mp4: 
      a.save('{}.mp4'.format(save_string), writer=writer)

  def animate_init(self):
    for l in self.lines:
      l.set_data([],[])
    return self.lines

  def animate_system(self, t_ind):
    box_pose = self.s[0:3, t_ind]
    obj_x, obj_y = self.get_box_corners(box_pose)

    self.lines[-1].set_data(obj_x, obj_y)

    # Draw contact points
    center = np.array([box_pose[0].__float__(), box_pose[1].__float__(), 1])
    theta = box_pose[2].__float__()
    R = np.array([
                 [np.cos(theta), -np.sin(theta), 0],
                 [np.sin(theta), np.cos(theta), 0],
                 [0, 0, 1]])
    for f_i in range(self.obj.fnum):
      # x and y locations of contact point, scaled to object dimensions  
      x = self.obj.cp_list[f_i]["p"][0]
      y = self.obj.cp_list[f_i]["p"][1]
      x_obj = center + (R @  (np.array([center[0]+x, center[1]+y, 0]) - center))
      self.lines[f_i].set_data(x_obj[0],x_obj[1])

    return self.lines

  def get_box_corners(self, pos):
    l = self.obj.width
    w = self.obj.height
    center = np.array([pos[0].__float__(), pos[1].__float__(), 1])
    theta = pos[2].__float__()
    R = np.array([
                 [np.cos(theta), -np.sin(theta), 0],
                 [np.sin(theta), np.cos(theta), 0],
                 [0, 0, 1]])
    bl = center + (R @  (np.array([center[0]-l/2, center[1]-w/2, 0]) - center))
    br = center + (R @  (np.array([center[0]+l/2, center[1]-w/2, 0]) - center))
    tr = center + (R @  (np.array([center[0]+l/2, center[1]+w/2, 0]) - center))
    tl = center + (R @  (np.array([center[0]-l/2, center[1]+w/2, 0]) - center))
    bl = center + (R @  (np.array([center[0]-l/2, center[1]-w/2, 0]) - center))

    obj_x = [bl[0], br[0], tr[0], tl[0], bl[0]]
    obj_y = [bl[1], br[1], tr[1], tl[1], bl[1]]

    return obj_x, obj_y

  def draw_box(self, box_pose, color):
    obj_x, obj_y = self.get_box_corners(box_pose)
    self.ax.plot(obj_x, obj_y, c=color) 

    

