
import numpy as np 
import math
from math import sin, cos, tan, sqrt
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from mpl_toolkits.mplot3d import Axes3D
# import jdc
import random

from solution import UDACITYDroneIn3D, UDACITYController
import testing

pylab.rcParams['figure.figsize'] = 10, 10

class DroneIn3D(UDACITYDroneIn3D):
    
    def __init__(self,
                k_f = 1.0,
                k_m = 1.0,
                m = 0.5,
                L = 0.566, # full rotor to rotor distance
                i_x = 0.1,
                i_y = 0.1,
                i_z = 0.2):
        
        self.k_f = k_f
        self.k_m = k_m
        self.m = m
        self.l = L / (2*sqrt(2)) # perpendicular distance to axes
        self.i_x = i_x
        self.i_y = i_y
        self.i_z = i_z
        
        # x, y, y, phi, theta, psi, 
        # x_dot, y_dot, z_dot, p, q, r
        self.X=np.array([0.0,0.0,0.0,0.0,0.0,0.0,
                         0.0,0.0,0.0,0.0,0.0,0.0])
        self.omega = np.array([0.0,0.0,0.0,0.0])
        
        self.g = 9.81

# euler angles [rad] (in world / lab frame)
@property
def phi(self):
    return self.X[3]

@property
def theta(self):
    return self.X[4]

@property
def psi(self):
    return self.X[5]

# body rates [rad / s] (in body frame)
@property 
def p(self):
    return self.X[9]

@property
def q(self):
    return self.X[10]

@property 
def r(self):
    return self.X[11]


# forces from the four propellers [N]
@property
def f_1(self):
    f = self.k_f*self.omega[0]**2
    return f

@property 
def f_2(self):
    f = self.k_f*self.omega[1]**2
    return f

@property 
def f_3(self):
    f = self.k_f*self.omega[2]**2
    return f

@property 
def f_4(self):
    f = self.k_f*self.omega[3]**2
    return f

# collective force
@property
def f_total(self):
    f_t = self.f_1 + self.f_2 + self.f_3 + self.f_4
    return f_t

# reactive moments [N * m]
@property
def tau_1(self):
    tau = self.k_m * self.omega[0]**2
    return tau
    
@property
def tau_2(self):
    tau = -self.k_m * self.omega[1]**2
    return tau

@property
def tau_3(self):
    tau = self.k_m * self.omega[2]**2
    return tau

@property
def tau_4(self):
    tau = -self.k_m * self.omega[3]**2
    return tau

# moments about axes [N * m]
@property
def tau_x(self):
    tau = self.l*(self.f_1 + self.f_4 - self.f_2 - self.f_3)
    return tau

@property
def tau_y(self):
    tau = self.l*(self.f_1 + self.f_2 - self.f_3 - self.f_4)
    return tau

@property
def tau_z(self):
    tau = self.tau_1 + self.tau_2 + self.tau_3 + self.tau_4
    return tau

def set_propeller_angular_velocities(self,
                                    c,
                                    u_bar_p,
                                    u_bar_q,
                                    u_bar_r):
    
    # TODO replace with your own implementation.
    #   note that this function doesn't return anything
    #   it just sets self.omega
    # 
    # self.omega[0] = 
    # self.omega[1] =
    # self.omega[2] = 
    # self.omega[3] = 
    
    return super(DroneIn3D, self).set_propeller_angular_velocities(c,
                                                                   u_bar_p,
                                                                   u_bar_q,
                                                                   u_bar_r)

testing.test_exercise_1_1(DroneIn3D)

def R(self):
    
    # TODO replace with your own implementation 
    #   according to the math above
    # 
    # return rotation_matrix
    
    return super(DroneIn3D, self).R()

testing.test_exercise_1_2(DroneIn3D)


def linear_acceleration(self):
    
    # TODO replace with your own implementation
    #   This function should return a length 3 np.array
    #   with a_x, a_y, and a_z
    
    return super(DroneIn3D, self).linear_acceleration()

testing.test_exercise_1_3(DroneIn3D)

def get_omega_dot(self):
    
    # TODO replace with your own implementation
    # return np.array([p_dot, q_dot, r_dot])
    
    return super(DroneIn3D, self).get_omega_dot()

testing.test_exercise_2_1(DroneIn3D)

def get_euler_derivatives(self):
    
    # TODO - replace with your own implementation
    #   return np.array([phi_dot, theta_dot, psi_dot])
    return super(DroneIn3D, self).get_euler_derivatives()

testing.test_exercise_3_1(DroneIn3D)


def advance_state(self, dt):
    
    # TODO replace this with your own implementation
    # 
    #   make sure this function returns the new state! 
    #   Some of the code that calls this function will expect
    #   it to return the state, so simply updating self.X 
    #   is not enough (though you should do that in this
    #   method too.)
    
    return super(DroneIn3D, self).advance_state(dt)
    

testing.test_exercise_3_2(DroneIn3D)

class Controller(UDACITYController):
    
    def __init__(self,
                z_k_p=1.0,
                z_k_d=1.0,
                x_k_p=1.0,
                x_k_d=1.0,
                y_k_p=1.0,
                y_k_d=1.0,
                k_p_roll=1.0,
                k_p_pitch=1.0,
                k_p_yaw=1.0,
                k_p_p=1.0,
                k_p_q=1.0,
                k_p_r=1.0):
        
        
        self.z_k_p = z_k_p
        self.z_k_d = z_k_d
        self.x_k_p = x_k_p
        self.x_k_d = x_k_d
        self.y_k_p = y_k_p
        self.y_k_d = y_k_d
        self.k_p_roll = k_p_roll
        self.k_p_pitch = k_p_pitch
        self.k_p_yaw = k_p_yaw
        self.k_p_p = k_p_p
        self.k_p_q = k_p_q
        self.k_p_r = k_p_r
        
        print('x: delta = %5.3f'%(x_k_d/2/math.sqrt(x_k_p)), ' omega_n = %5.3f'%(math.sqrt(x_k_p)))
        print('y: delta = %5.3f'%(y_k_d/2/math.sqrt(y_k_p)), ' omega_n = %5.3f'%(math.sqrt(y_k_p)))
        print('z: delta = %5.3f'%(z_k_d/2/math.sqrt(z_k_p)), ' omega_n = %5.3f'%(math.sqrt(z_k_p)))
        
        self.g= 9.81
    

def lateral_controller(self,
                      x_target,
                      x_dot_target,
                      x_dot_dot_target,
                      x_actual,
                      x_dot_actual,
                      y_target,
                      y_dot_target,
                      y_dot_dot_target,
                      y_actual,
                      y_dot_actual,
                      c):
    
    # TODO replace with your own implementation
    # return b_x_c, b_y_c
    
    return super(Controller, self).lateral_controller(
        x_target,
        x_dot_target,
        x_dot_dot_target,
        x_actual,
        x_dot_actual,
        y_target,
        y_dot_target,
        y_dot_dot_target,
        y_actual,
        y_dot_actual,
        c)

testing.test_exercise_4_1(Controller)

def roll_pitch_controller(self,
                          b_x_c_target,
                          b_y_c_target,
                          rot_mat):
    
    # TODO replace with your own implementation
    # return p_c, q_c
    
    return super(Controller, self).roll_pitch_controller(b_x_c_target,
                                                          b_y_c_target,
                                                          rot_mat)

testing.test_exercise_4_2(Controller)

def body_rate_controller(self,
                         p_c,
                         q_c,
                         r_c,
                         p_actual,
                         q_actual,
                         r_actual):
    # TODO replace with your own implementation
    # return u_bar_p, u_bar_q, u_bar_r
    return super(Controller, self).body_rate_controller(p_c,
                                                         q_c,
                                                         r_c,
                                                         p_actual,
                                                         q_actual,
                                                         r_actual)

testing.test_exercise_5_1(Controller)

def yaw_controller(self,
                   psi_target,
                   psi_actual):
    
    # TODO replace with your own implementation
    # return r_c
    
    return super(Controller, self).yaw_controller(psi_target,
                                                  psi_actual)

testing.test_exercise_5_2(Controller)

def altitude_controller(self,
                       z_target,
                       z_dot_target,
                       z_dot_dot_target,
                       z_actual,
                       z_dot_actual,
                       rot_mat):
    
    # TODO replace with your own implementation
    # return c
    
    return super(Controller, self).altitude_controller(z_target,
                                                       z_dot_target,
                                                       z_dot_dot_target,
                                                       z_actual,
                                                       z_dot_actual,
                                                       rot_mat)

testing.test_exercise_5_3(Controller)

def attitude_controller(self,
                       b_x_c_target,
                       b_y_c_target,
                       psi_target,
                       psi_actual,
                       p_actual,
                       q_actual,
                       r_actual,
                       rot_mat):

    p_c, q_c = self.roll_pitch_controller(b_x_c_target,
                                          b_y_c_target,
                                          rot_mat)
    
    r_c = self.yaw_controller(psi_target, 
                              psi_actual)
    
    u_bar_p, u_bar_q, u_bar_r = self.body_rate_controller(
        p_c,
        q_c,
        r_c,
        p_actual,
        q_actual,
        r_actual)

    return u_bar_p, u_bar_q, u_bar_r

total_time = 20.0
dt = 0.01

t=np.linspace(0.0,total_time,int(total_time/dt))

omega_x = 0.8
omega_y = 0.4
omega_z = 0.4

a_x = 1.0 
a_y = 1.0
a_z = 1.0

x_path= a_x * np.sin(omega_x * t) 
x_dot_path= a_x * omega_x * np.cos(omega_x * t)
x_dot_dot_path= -a_x * omega_x**2 * np.sin(omega_x * t)

y_path= a_y * np.cos(omega_y * t)
y_dot_path= -a_y * omega_y * np.sin(omega_y * t)
y_dot_dot_path= -a_y * omega_y**2 * np.cos(omega_y * t)

z_path= a_z * np.cos(omega_z * t)
z_dot_path= -a_z * omega_z * np.sin(omega_z * t)
z_dot_dot_path= - a_z * omega_z**2 * np.cos(omega_z * t)

psi_path= np.arctan2(y_dot_path,x_dot_path)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(x_path, y_path, z_path)
plt.title('Flight path').set_fontsize(20)
ax.set_xlabel('$x$ [$m$]').set_fontsize(20)
ax.set_ylabel('$y$ [$m$]').set_fontsize(20)
ax.set_zlabel('$z$ [$m$]').set_fontsize(20)
plt.legend(['Planned path'],fontsize = 14)
plt.figure(figsize=(10,10))
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')

u = np.cos(psi_path)
v = np.sin(psi_path)
w = np.zeros(psi_path.shape)
for i in range(0,z_path.shape[0],20):
    ax.quiver(x_path[i], y_path[i], z_path[i], u[i], v[i], w[i], length=0.2, normalize=True,color='green')

plt.title('Flight path').set_fontsize(20)
ax.set_xlabel('$x$ [$m$]').set_fontsize(20)
ax.set_ylabel('$y$ [$m$]').set_fontsize(20)
ax.set_zlabel('$z$ [$m$]').set_fontsize(20)
plt.legend(['Planned yaw',],fontsize = 14)

plt.show()

# how fast the inner loop (Attitude controller) performs calculations 
# relative to the outer loops (altitude and position controllers).
inner_loop_relative_to_outer_loop = 10

# creating the drone object
drone = DroneIn3D()
 
# creating the control system object 

control_system = Controller(z_k_p=2.0, 
                            z_k_d=1.0, 
                            x_k_p=6.0,
                            x_k_d=4.0,
                            y_k_p=6.0,
                            y_k_d=4.0,
                            k_p_roll=8.0,
                            k_p_pitch=8.0,
                            k_p_yaw=8.0,
                            k_p_p=20.0,
                            k_p_q=20.0,
                            k_p_r=20.0)



# declaring the initial state of the drone with zero
# height and zero velocity 

drone.X = np.array([x_path[0],
                               y_path[0],
                               z_path[0],
                               0.0,
                               0.0,
                               psi_path[0],
                               x_dot_path[0],
                               y_dot_path[0],
                               z_dot_path[0],
                               0.0,
                               0.0,
                               0.0])

# arrays for recording the state history, 
# propeller angular velocities and linear accelerations
drone_state_history = drone.X
omega_history = drone.omega
accelerations = drone.linear_acceleration()
accelerations_history= accelerations
angular_vel_history = drone.get_euler_derivatives()

# executing the flight
for i in range(0,z_path.shape[0]):
    
    rot_mat = drone.R()

    c = control_system.altitude_controller(z_path[i],
                                           z_dot_path[i],
                                           z_dot_dot_path[i],
                                           drone.X[2],
                                           drone.X[8],
                                           rot_mat)
    
    b_x_c, b_y_c = control_system.lateral_controller(x_path[i],
                                                     x_dot_path[i],
                                                     x_dot_dot_path[i],
                                                     drone.X[0],
                                                     drone.X[6],
                                                     y_path[i],
                                                     y_dot_path[i],
                                                     y_dot_dot_path[i],
                                                     drone.X[1],
                                                     drone.X[7],
                                                     c) 
    
    for _ in range(inner_loop_relative_to_outer_loop):
        
        rot_mat = drone.R()
        
        angular_vel = drone.get_euler_derivatives()
        
        u_bar_p, u_bar_q, u_bar_r = control_system.attitude_controller(
            b_x_c,
            b_y_c,
            psi_path[i],
            drone.psi,
            drone.X[9],
            drone.X[10],
            drone.X[11],
            rot_mat)
        
        drone.set_propeller_angular_velocities(c, u_bar_p, u_bar_q, u_bar_r)
        
        drone_state = drone.advance_state(dt/inner_loop_relative_to_outer_loop)
        
    # generating a history of the state history, propeller angular velocities and linear accelerations
    drone_state_history = np.vstack((drone_state_history, drone_state))
    
    omega_history=np.vstack((omega_history,drone.omega))
    accelerations = drone.linear_acceleration()
    accelerations_history= np.vstack((accelerations_history,accelerations))
    angular_vel_history = np.vstack((angular_vel_history,drone.get_euler_derivatives()))
    


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(x_path, y_path, z_path,linestyle='-',marker='.',color='red')
ax.plot(drone_state_history[:,0],
         drone_state_history[:,1],
         drone_state_history[:,2],
         linestyle='-',color='blue')

plt.title('Flight path').set_fontsize(20)
ax.set_xlabel('$x$ [$m$]').set_fontsize(20)
ax.set_ylabel('$y$ [$m$]').set_fontsize(20)
ax.set_zlabel('$z$ [$m$]').set_fontsize(20)
plt.legend(['Planned path','Executed path'],fontsize = 14)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)

plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')

u = np.cos(psi_path)
v = np.sin(psi_path)
w = np.zeros(psi_path.shape)

drone_u = np.cos(drone_state_history[:,5])
drone_v = np.sin(drone_state_history[:,5])
drone_w = np.zeros(psi_path.shape)

for i in range(0,z_path.shape[0],20):
    ax.quiver(x_path[i], y_path[i], z_path[i], u[i], v[i], w[i], length=0.2, normalize=True,color='red')
    ax.quiver(drone_state_history[i,0], 
              drone_state_history[i,1], 
              drone_state_history[i,2], 
              drone_u[i], drone_v[i], drone_w[i], 
              length=0.2, normalize=True,color='blue')
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
plt.title('Flight path').set_fontsize(20)
ax.set_xlabel('$x$ [$m$]').set_fontsize(20)
ax.set_ylabel('$y$ [$m$]').set_fontsize(20)
ax.set_zlabel('$z$ [$m$]').set_fontsize(20)
plt.legend(['Planned yaw','Executed yaw'],fontsize = 14)

plt.show()

err= np.sqrt((x_path-drone_state_history[:-1,0])**2 
             +(y_path-drone_state_history[:-1,1])**2 
             +(y_path-drone_state_history[:-1,2])**2)


plt.plot(t,err)
plt.title('Error in flight position').set_fontsize(20)
plt.xlabel('$t$ [$s$]').set_fontsize(20)
plt.ylabel('$e$ [$m$]').set_fontsize(20)
plt.show()

plt.plot(t,-omega_history[:-1,0],color='blue')
plt.plot(t,omega_history[:-1,1],color='red')
plt.plot(t,-omega_history[:-1,2],color='green')
plt.plot(t,omega_history[:-1,3],color='black')

plt.title('Angular velocities').set_fontsize(20)
plt.xlabel('$t$ [$s$]').set_fontsize(20)
plt.ylabel('$\omega$ [$rad/s$]').set_fontsize(20)
plt.legend(['P 1','P 2','P 3','P 4' ],fontsize = 14)

plt.grid()
plt.show()

plt.plot(t,psi_path,marker='.')
plt.plot(t,drone_state_history[:-1,5])
plt.title('Yaw angle').set_fontsize(20)
plt.xlabel('$t$ [$s$]').set_fontsize(20)
plt.ylabel('$\psi$ [$rad$]').set_fontsize(20)
plt.legend(['Planned yaw','Executed yaw'],fontsize = 14)
plt.show()
