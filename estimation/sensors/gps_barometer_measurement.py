import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from Measurements import gps_and_barometer_measurements

pylab.rcParams['figure.figsize'] = 10, 10

# Loading and plotting the measurement data 
gps_t, gps_data, sigma_gps, baro_t, baro_data, sigma_baro = gps_and_barometer_measurements()

plt.plot(gps_t, gps_data)
plt.plot(baro_t, baro_data)
plt.legend(['GPS measurements','Barometer measuremet'],fontsize = 14)
plt.title('GPS and barometer measurements').set_fontsize(20)
plt.xlabel('$t$ [sec]').set_fontsize(20)
plt.ylabel('$h$ [m]').set_fontsize(20)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.grid()
plt.show()

Q = np.eye(1)   # Covariance 
H = np.eye(1)
h_hat = 0.5*(gps_data[0] + baro_data[0]) 
R = np.eye(1)*sigma_baro 

h_hat_history = np.array([h_hat[0]])

k_gps = 0  # an element from GPS measurement data 

for i in range(baro_t.shape[0]):
    if gps_t[k_gps] <= baro_t[i]: # If new GPS measurement has arrived 
        h_obs = gps_data[k_gps]   # Observed h value  
        R = np.eye(1)*sigma_gps   # measuring sigma is gps sigma
        k_gps+=1                  # advance to the next measurement point
        i-=1     # What this does is that it will update altitude based on the GPS and then 
                 # will come back and update the altitude estimation using the barometer measurement.
                 # This will be done without losing the proper count. 
    else:
        h_obs = baro_data[i]      # Observed h value using the barometer 
        R = np.eye(1)*sigma_baro  # Measuring sigma is barometer sigma

    # TODO: Use recursive estimator from lesson 1 to evalueat the altitude
    Q = None
    h_hat = None 
    
    # creating the historical data of the h_hat in time 
    h_hat_history = np.vstack((h_hat_history,h_hat))
   

plt.plot(gps_t, gps_data)
plt.plot(baro_t, baro_data)
plt.plot(baro_t, h_hat_history[1:], color='red',linewidth=5)
plt.legend(['GPS Measurements', 'Barometer Measuremet', 'Estimated Altitude'], fontsize = 14)
plt.title('Estimated Altitude').set_fontsize(20)
plt.xlabel('$t$ [sec]').set_fontsize(20)
plt.ylabel('$h$ [m]').set_fontsize(20)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.grid()
plt.show()

 
