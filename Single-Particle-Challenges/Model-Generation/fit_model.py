import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import feyn 
import sympy
import find_potential

#How many threads you are willing to sacrifice
THREAD_COUNT = feyn.tools.infer_available_threads() 

#Takes a list of lists lst and returns a list of the ith element of each sublist
def extract(lst, i):
    answer = []
    for sub in lst:
        answer.append(sub[i])
    return answer

# Function to sort the list of tuples by its second item
def sort_Tuple(tup):
    # getting length of list of tuples
    lst = len(tup)
    for i in range(0, lst):
        for j in range(0, lst-i-1):
            if (tup[j][1] > tup[j + 1][1]):
                temp = tup[j]
                tup[j] = tup[j + 1]
                tup[j + 1] = temp
    return tup

#Main function
if __name__ == "__main__":
    #Gets potential data and seperates it into a time and potential np array
    potential_df = pd.read_csv('potential.csv')
    data_array = potential_df.to_numpy()
    time_array = extract(data_array, 1)
    potential_array = extract(data_array, 2)
    position_array = np.array(find_potential.position_data)

    #Graph potentiav vs position
    plt.plot(position_array, potential_array)
    plt.show()
    
    #Finding potential array against position
    potential_position_tuples = []
    for i in range(len(position_array)):
        potential_position_tuples.append((potential_array[i], position_array[i]))
    sorted_potential_position_tuples = sort_Tuple(potential_position_tuples)

    #Making the potential vs position dataframe
    potential_position_df = pd.DataFrame({"x": extract(sorted_potential_position_tuples, 1),
                                        "Potential": extract(sorted_potential_position_tuples, 0)})

    #Splitting the randomly data into a training and testing set with 50/50 split
    train, test = feyn.tools.split(potential_position_df, ratio =[0.5, 0.5], random_state = 42)

    #Training the symbolic model 
    q1 = feyn.QLattice()

    models = q1.auto_run(train, output_name = "Potential", kind = "regression", threads = THREAD_COUNT, loss_function = "absolute_error", max_complexity = 4, n_epochs = 25)

    #Outputs the models into html files and prints the latex
    for i, model in enumerate(models):
        sympy_model = model.sympify(signif = 3)
        sympy.print_latex(sympy_model)
        model.plot(train, compare_data = test, filename = "output_two_" + str((i + 1)) + ".html")
