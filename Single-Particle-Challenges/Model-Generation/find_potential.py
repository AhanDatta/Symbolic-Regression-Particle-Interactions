import pandas as pd 

#Mass of the particle, found from the hamiltonian
MASS = 1.0

#Gets data and puts it into two lists
time_data = []
position_data = []
file = open("test2.txt")
for line in file:
    unclean_nums = line.split(",")
    #Cleans the time component
    if '*' in unclean_nums[0]:
        time_data.append(0)
    else:
        time = "".join(char for char in unclean_nums[0] if (char.isdecimal() or char == '.' or char == '-'))
        time_data.append(float(time))
    
    #Cleans the position component
    if '*' in unclean_nums[1]:
        position_data.append(0)
    else:
        position = ("".join(char for char in unclean_nums[1] if (char.isdecimal() or char == '.' or char == '-')))
        position_data.append(float(position))
file.close()

#Finds velocity array using the difference quotent when possible
velocity_data = []
velocity_data.append((position_data[1] - position_data[0])/(time_data[1] - time_data[0]))
for i in range(len(position_data) - 2):
    append_val = 0
    append_val = (position_data[i+2] - position_data[i])/(time_data[i+2] - time_data[i])
    velocity_data.append(append_val)
velocity_data.append((position_data[-1] - position_data[-2])/(time_data[-1] - time_data[-2]))

#We know that H = p^2/2 + V(x). Arbitrarily we set H = 0.
#Then, V(x) = -p^2/2. The mass from this form is assumed to be one,
#this becomes V(x) = -m v^2/2
#Here we are finding V(t) and will sort it later
potential_data = []
for vel in velocity_data:
    potential_data.append(-MASS * (vel**2)/2)

potential_df = pd.DataFrame({"Time": time_data,
                             "Potential": potential_data})
potential_df.to_csv("potential.csv")