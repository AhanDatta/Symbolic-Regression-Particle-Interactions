import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

#File path to the data input
FILENAME = "data6.csv"

#Reads in the raw data
raw_data = pd.read_csv(FILENAME)
raw_data_array = raw_data.to_numpy()
num_cols = len(raw_data)

#Reads in the data into individual np arrays
particle_one_position = np.empty([num_cols, 2])
particle_two_position = np.empty([num_cols, 2])
particle_three_position = np.empty([num_cols, 2])
time_array = np.empty(num_cols)

for i in range(num_cols):
    time_array[i] = raw_data_array[i][0]
    particle_one_position[i][0] = raw_data_array[i][1]
    particle_one_position[i][1] = raw_data_array[i][2]
    particle_two_position[i][0] = raw_data_array[i][3]
    particle_two_position[i][1] = raw_data_array[i][4]
    particle_three_position[i][0] = raw_data_array[i][5]
    particle_three_position[i][1] = raw_data_array[i][6]

#Now we find the velocity vectors for both particles
#v(t) = (x(t+dt) - x(t))/(dt)
particle_one_velocity = np.empty([num_cols, 2])
particle_two_velocity = np.empty([num_cols, 2])
particle_three_velocity = np.empty([num_cols, 2])
for i in range(num_cols - 2):
    dt_2 = time_array[i+2] - time_array[i]
    particle_one_velocity[i+1] = (particle_one_position[i+2] - particle_one_position[i])/dt_2
    particle_two_velocity[i+1] = (particle_two_position[i+2] - particle_two_position[i])/dt_2
    particle_three_velocity[i+1] = (particle_three_position[i+2] - particle_three_position[i])/dt_2
particle_one_velocity[0] = (particle_one_position[1] - particle_one_position[0])/(time_array[1] - time_array[0])
particle_two_velocity[0] = (particle_two_position[1] - particle_two_position[0])/(time_array[1] - time_array[0])
particle_three_velocity[0] = (particle_three_position[1] - particle_three_position[0])/(time_array[1] - time_array[0])
particle_one_velocity[-1] = (particle_one_position[-1] - particle_one_position[-2])/(time_array[-1] - time_array[-2])
particle_two_velocity[-1] = (particle_two_position[-1] - particle_two_position[-2])/(time_array[-1] - time_array[-2])
particle_three_velocity[-1] = (particle_three_position[-1] - particle_three_position[-2])/(time_array[-1] - time_array[-2])

#Creates the three distance arrays
distance_12 = np.empty(num_cols)
distance_13 = np.empty(num_cols)
distance_23 = np.empty(num_cols)
for i in range(num_cols): 
    delta_r_12 = particle_one_position[i] - particle_two_position[i]
    delta_r_13 = particle_one_position[i] - particle_three_position[i]
    delta_r_23 = particle_two_position[i] - particle_three_position[i]
    distance_12[i] = np.linalg.norm(delta_r_12)
    distance_13[i] = np.linalg.norm(delta_r_13)
    distance_23[i] = np.linalg.norm(delta_r_23)

#The Hamiltonian is H = p_1^2 + p_2^2 + p_3^2 + V(r_12) + V(r_13) + V(r_23)
#The masses are all 0.5 we can see
#We also assume that the potential is the same function for each pair
#Therefore, V(r_12) + V(r_13) + V(r_23) = -0.25 * (v_1^2 + v_2^2 + v_3^2)
total_potential_energy = np.empty(num_cols)
for i in range(num_cols):
    total_potential_energy[i] = -0.25 * (np.dot(particle_one_velocity[i], particle_one_velocity[i]) + np.dot(particle_two_velocity[i], particle_two_velocity[i])
                                          + np.dot(particle_three_velocity[i], particle_three_velocity[i]))
    
if __name__ == "__main__":
    plt.plot(time_array, particle_one_position)
    plt.show()