import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

#Constants derived from the form of the hamiltonian
MASS_1 = 2.5
MASS_2 = 0.5

#Takes a list of sublists and returns a list of the kth element in each sublist
def extract(lst, k):
    return [sub[k] for sub in lst]

#Reads data from csv and enters it into three np arrays: time and 2 position vectors arrays
raw_df = pd.read_csv("data5.csv")
raw_data_array = raw_df.to_numpy()

num_cols = len(raw_data_array)
particle_one_position = np.empty([num_cols, 2])
particle_two_position = np.empty([num_cols, 2])
time_array = np.empty(num_cols)

for i in range(num_cols):
    time_array[i] = raw_data_array[i][0]
    particle_one_position[i][0] = raw_data_array[i][1]
    particle_one_position[i][1] = raw_data_array[i][2]
    particle_two_position[i][0] = raw_data_array[i][3]
    particle_two_position[i][1] = raw_data_array[i][4]

#Now we find the velocity vectors for both particles
#v(t) = (x(t+dt) - x(t))/(dt)
particle_one_velocity = np.empty([num_cols, 2])
particle_two_velocity = np.empty([num_cols, 2])
for i in range(num_cols - 2):
    dt_2 = time_array[i+2] - time_array[i]
    particle_one_velocity[i+1] = (particle_one_position[i+2] - particle_one_position[i])/dt_2
    particle_two_velocity[i+1] = (particle_two_position[i+2] - particle_two_position[i])/dt_2
particle_one_velocity[0] = (particle_one_position[1] - particle_one_position[0])/(time_array[1] - time_array[0])
particle_two_velocity[0] = (particle_two_position[1] - particle_two_position[0])/(time_array[1] - time_array[0])
particle_one_velocity[-1] = (particle_one_position[-1] - particle_one_position[-2])/(time_array[-1] - time_array[-2])
particle_two_velocity[-1] = (particle_two_position[-1] - particle_two_position[-2])/(time_array[-1] - time_array[-2])

#Finding the distance between particles, r(t), as an np array
distance_array = np.empty(num_cols)
for i, pos_1 in enumerate(particle_one_position):
    pos_2 = particle_two_position[i]
    delta_r = pos_2 - pos_1
    distance_array[i] = np.linalg.norm(delta_r)

#The hamiltonian is H = (1/5)p_1^2 + p_2^2 + V(r)
#This implies that m_1 = 2.5 and m_2 = 0.5
#Since H is constant, we set it arbitrarily to 0
#Then, V = -1/2(m_1 v_1^2 + m_2 v_2^2) for all time
#We iterate through the time array to find V(t)
potential_vs_time = np.empty(num_cols)
for i in range(len(time_array)):
    particle_one_speed_sqr = np.dot(particle_one_velocity[i], particle_one_velocity[i])
    particle_two_speed_sqr = np.dot(particle_two_velocity[i], particle_two_velocity[i])
    potential_vs_time[i] = (-0.5 * (MASS_1 * particle_one_speed_sqr + MASS_2 * particle_two_speed_sqr))

if __name__ == "__main__":
    fig, ax = plt.subplots()
    ax.plot(extract(particle_one_position,0), extract(particle_one_position,1), label = "particle one")
    ax.plot(extract(particle_two_position,0), extract(particle_two_position,1), label = "particle two")
    fig.legend()
    plt.show()
