import numpy as np 
import pandas as pd 

#Set the input data
FILENAME = "data9.csv"

#Takes in a y-list and an x-list and finds dy/dx at each point
#Equations found here https://www.dam.brown.edu/people/alcyew/handouts/numdiff.pdf
def differentiate(y_list, x_list):
    #Checks both arrays are same size
    if np.shape(y_list)[0] != np.shape(x_list)[0]:
        raise Exception("SizeMismatch")

    #Fills in the middle with second order
    derivative_array = np.empty(y_list.shape)
    for i in range(num_cols - 2):
        dt_2 = x_list[i+2] - x_list[i]
        derivative_array[i+1] = (y_list[i+2] - y_list[i])/dt_2
    
    #Fills in edges with second order
    derivative_array[0] = (-y_list[2] + 4*y_list[1] - 3*y_list[0])/(x_list[2] - x_list[0])
    derivative_array[-1] = (3* y_list[-1] - 4*y_list[-2] + y_list[-3])/(x_list[-1] - x_list[-3])
    return derivative_array
    
#Takes in a y-list and an x-list and finds d^2y/dx^2
def differentiate_twice(y_list, x_list):
    #Checks both arrays are same size
    if np.shape(y_list)[0] != np.shape(x_list)[0]:
        raise Exception("SizeMismatch")
    
    #Fills in the middle with second order
    derivative_array = np.empty(y_list.shape)
    for i in range(num_cols - 2):
        dt = x_list[i+1] - x_list[i]
        derivative_array[i+1] = (y_list[i+1] + y_list[i-1] - 2*y_list[i])/(dt**2)

    #Fills in edges with second order
    derivative_array[0] = (2*y_list[0] - 5*y_list[1] + 4*y_list[2] - y_list[3])/((time_array[1]-time_array[0])**3)
    derivative_array[-1] = (2*y_list[-1] - 5*y_list[-2] + 4*y_list[-3] - y_list[-4])/((time_array[1]-time_array[0])**3)


#Reads in the raw data
raw_data = pd.read_csv(FILENAME)
raw_data_array = raw_data.to_numpy()
num_cols = len(raw_data)

#Reads in the data into individual np arrays
particle_one_position = np.empty([num_cols, 2])
particle_two_position = np.empty([num_cols, 2])
time_array = np.empty(num_cols)

for i in range(num_cols):
    time_array[i] = raw_data_array[i][0]
    particle_one_position[i][0] = raw_data_array[i][1]
    particle_one_position[i][1] = raw_data_array[i][2]
    particle_two_position[i][0] = raw_data_array[i][3]
    particle_two_position[i][1] = raw_data_array[i][4]

#Gets the velocities for each particle
particle_one_velocity = differentiate(particle_two_position, time_array)
particle_two_velocity = differentiate(particle_two_position, time_array)

#Gets the accelerations for each particle
particle_one_acceleration = differentiate_twice(particle_one_position, time_array)
particle_two_acceleration = differentiate_twice(particle_two_position, time_array)