import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(precision=3)
pylab.rcParams['figure.figsize'] = 10, 10

def line_segment(t,    # Array of times for each position
                 t_now # Current time (float)
                ):
    """
    Returns the start and end index corresponding to the line
    segment in which t_now occurs.
    """
    i = np.argwhere(t>t_now)
    if len(i)>0: 
        if i[0]!=0: # if the current time is not less than the starting time 
            segment_starting_index = i[0][0]-1
        else: 
            segment_starting_index = 0

        segment_end_index = i[0][0]
        
    else: # if the current time is more than the last point (destination) time
        segment_starting_index = t.shape[0]
        segment_end_index = t.shape[0]
    
    return segment_starting_index, segment_end_index

#     index:  0    1    2    3    4
t = np.array([1.2, 3.4, 5.7, 6.2, 7.9])

t_now = 4.1 # this is between index 1 and 2

line_segment(t, t_now)

def commanded_values(x,     # Array of x positions 
                     t,     # Array of times for each position
                     t_now, # Current time 
                     segment_starting_index, 
                     segment_end_index
                     ):
    
    start_x = x[segment_starting_index]
    end_x   = x[segment_end_index]
    delta_x = end_x - start_x
    
    start_t = t[segment_starting_index]
    end_t   = t[segment_end_index]
    delta_t = end_t - start_t
    
    v     = delta_x / delta_t
    x_now = start_x + v * (t_now - start_t)
             
    return np.array([x_now, v])

# example usage

x = np.array([0,   1,   3,   6,   10])
t = np.array([1.2, 3.4, 5.7, 6.2, 7.9])

t_now = 2.3 # directly between t[0] and t[1]

start, end = line_segment(t, t_now)

commanded_values(x,t,t_now, start, end)

t = np.linspace(0,10,10)
x = np.zeros(t.shape)

omega_x = 0.8
x = np.sin(omega_x*t)
plt.plot(t,x)
plt.show()

t_now = 2.8

segment_starting_index, segment_end_index = line_segment(t,t_now)

target_parameters = commanded_values(x,
                                     t,
                                     t_now,
                                     segment_starting_index,
                                     segment_end_index)


print('Target position and velocity ',target_parameters)

plt.plot(t,x,marker='.')
plt.scatter(t_now,target_parameters[0],marker='o',color='red')
plt.title('Flight trajectory').set_fontsize(20)
plt.xlabel('$t$ [$s$]').set_fontsize(20)
plt.ylabel('$x$ [$m$]').set_fontsize(20)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.show()

def commanded_values_3d(p,     # Array of points
                        t,     # Array of time 
                        t_now, # Current time 
                        segment_starting_index,
                        segment_end_index,
                        previous_psi):
    
    x = p[0,:]
    y = p[1,:]
    z = p[2,:] 
    
    # calculating the velocities 
    v_x_now = (x[segment_end_index] - x[segment_starting_index]) \
            / (t[segment_end_index] - t[segment_starting_index])
    v_y_now = (y[segment_end_index] - y[segment_starting_index]) \
            / (t[segment_end_index] - t[segment_starting_index])
    v_z_now = (z[segment_end_index] - z[segment_starting_index]) \
            / (t[segment_end_index] - t[segment_starting_index])
    
    # calculating the position
    x_now = x[segment_starting_index] + v_x_now * (t_now - t[segment_starting_index])
    
    y_now = y[segment_starting_index] + v_y_now * (t_now - t[segment_starting_index])
            
    z_now = z[segment_starting_index] + v_z_now * (t_now - t[segment_starting_index])
    
    # If drone does not move vertically up. 
    if x[segment_end_index]!=x[segment_starting_index] and \
       y[segment_end_index]!=y[segment_starting_index]:
        
        psi_now = np.arctan2((y[segment_end_index] - y[segment_starting_index]),
                             (x[segment_end_index] - x[segment_starting_index]))
    else:
        psi_now = previous_psi
                           
    return np.array([x_now, y_now, z_now, v_x_now, v_y_now, v_z_now, psi_now])

t = np.linspace(0,10,10)
p = np.zeros((3,t.shape[0]))

omega_x = 0.8
omega_y = 0.4
omega_z = 0.4
x_amp = 1.0
y_amp = 1.0
z_amp = 1.0

p[0,:] = x_amp*np.sin(omega_x*t)
p[1,:] = y_amp*np.cos(omega_y*t)
p[2,:] = z_amp*np.cos(omega_z*t)

t_now = 2.8
previous_psi = 0

segment_starting_index, segment_end_index = line_segment(t,t_now)

target_parameters = commanded_values_3d(p,
                                       t,
                                       t_now,
                                       segment_starting_index,
                                       segment_end_index,
                                       previous_psi)

print('Target location (x, y, z) =', target_parameters[0:3])
print('Target velocities (v_x, v_y, v_z) = ', target_parameters[3:6])
print('Target yaw angle %5.3f'%target_parameters[6])

x_path=p[0,:]
y_path=p[1,:]
z_path=p[2,:]

x_now = target_parameters[0] 
y_now = target_parameters[1] 
z_now = target_parameters[2] 

v_x_now = target_parameters[3] 
v_y_now = target_parameters[4] 
v_z_now = target_parameters[5] 

psi_now = target_parameters[6] 

u = np.cos(psi_now) 
v = np.sin(psi_now) 
w = 0
    
fig = plt.figure() 
ax = fig.gca(projection='3d')

ax.plot(x_path, y_path, z_path, marker ='.')
ax.quiver(x_now, y_now, z_now, u, v, w, length=1.0, normalize=True,color='green')
ax.quiver(x_now, y_now, z_now, v_x_now, v_y_now, v_z_now,color='Red')
ax.scatter(x_now, y_now, z_now, marker ='o',color='red')

plt.title('Flight path').set_fontsize(20)
ax.set_xlabel('$x$ [$m$]').set_fontsize(20)
ax.set_ylabel('$y$ [$m$]').set_fontsize(20)
ax.set_zlabel('$z$ [$m$]').set_fontsize(20)
plt.legend(['Planned path','$\psi$','Velocity'],fontsize = 14)
ax.set_xlim(-1.0, 1.0)
ax.set_ylim(-1.0, 1.0)
ax.set_zlim(-1.0, 1.0)
plt.show()
