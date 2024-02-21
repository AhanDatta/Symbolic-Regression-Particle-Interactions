import matplotlib.pyplot as plt 
import numpy as np 
from math import *
import find_potential

#Getting useful data from previous file
potential_vs_time = find_potential.potential_vs_time
distance_vs_time = find_potential.distance_array
num_cols = find_potential.num_cols

#Input the model that is being plotted
def model(r):
    return r**2*(0.9930476*r**4 + 0.036601033) - 9.209152

#Plotting experimental data vs chosen model
max_dist = np.max(distance_vs_time)
min_dist = np.min(distance_vs_time)
distance = np.arange(min_dist, max_dist, (max_dist - min_dist)/num_cols)
output_array = np.empty(len(distance))
print(distance)
for i in range(len(distance)):
    output_array[i] = model(distance[i])
plt.plot(distance_vs_time, potential_vs_time, label = "Experiment")
plt.plot(distance, output_array, label = "Model")
plt.legend(loc="upper left")
plt.show()