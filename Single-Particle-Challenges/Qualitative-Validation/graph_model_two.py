import numpy as np 
from scipy import integrate
from matplotlib import pyplot as plt
import pandas as pd
import os

"""
Here we suppose a single particle in a one dimensional potential given by model two
We continue with the assumption that mass is one
H = p^2/2 + (x^2 + x)^2. From this, we find the resulting eom
dx/dt = p, and dp/dt = -(4x^3 + 6x^2 + 2x)
From this, we get a system of equations which we will approximate
"""

#Settings constants for the sim, using phase space (x, p) where both are scalar-like
END_TIME = 30.0
INITIAL_POSITION = 0.0
INITIAL_MOMENTUM = 3.182
INITIAL_STATE = [INITIAL_POSITION, INITIAL_MOMENTUM]

#Takes a point in phase space and return the time derivate as a vector (np array of size 2)
def eoms(t: float, phase_point):
    curr_position = phase_point[0]
    curr_momentum = phase_point[1]
    return np.array([curr_momentum, -(curr_position * (4 * (curr_position**2) + 6 * curr_position + 2))])

#Gets data and puts it into two lists
ex_time_data = []
ex_position_data = []
file = open(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Model-Generation','test2.txt')))
for line in file:
    unclean_nums = line.split(",")
    #Cleans the time component
    if '*' in unclean_nums[0]:
        ex_time_data.append(0)
    else:
        time = "".join(char for char in unclean_nums[0] if (char.isdecimal() or char == '.' or char == '-'))
        ex_time_data.append(float(time))
    
    #Cleans the position component
    if '*' in unclean_nums[1]:
        ex_position_data.append(0)
    else:
        position = ("".join(char for char in unclean_nums[1] if (char.isdecimal() or char == '.' or char == '-')))
        ex_position_data.append(float(position))
file.close()

#Finds velocity array using the difference quotent when possible
ex_velocity_data = []
ex_velocity_data.append((ex_position_data[1] - ex_position_data[0])/(ex_time_data[1] - ex_time_data[0]))
for i in range(len(ex_position_data) - 2):
    append_val = 0
    append_val = (ex_position_data[i+2] - ex_position_data[i])/(ex_time_data[i+2] - ex_time_data[i])
    ex_velocity_data.append(append_val)
ex_velocity_data.append((ex_position_data[-1] - ex_position_data[-2])/(ex_time_data[-1] - ex_time_data[-2]))

#Runs solution and stores solution arrays
solution = integrate.solve_ivp(eoms, (0, END_TIME), INITIAL_STATE, max_step = 0.01, vectorized=True)
time_data = solution.t
position_data = solution.y[0]
momentum_data = solution.y[1]

#Plots the position and momentum for the model potential and the actual data
ax = plt.figure().add_subplot()
ax.plot(position_data, momentum_data, label='Model')
ax.plot(ex_position_data, ex_velocity_data, label='Experiment')
ax.legend()
plt.ylabel("Momentum")
plt.xlabel("Position")
plt.show()

#Writes out data to csv
df = pd.DataFrame({'Time': time_data, 
                   'Position': position_data, 
                   'Momentum': momentum_data})
df.to_csv('second_trajectory.csv')