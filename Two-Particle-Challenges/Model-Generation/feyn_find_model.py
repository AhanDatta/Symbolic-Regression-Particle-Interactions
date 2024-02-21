import pandas as pd 
import feyn
import numpy as np 
import sympy
import find_potential

#How many threads you are willing to sacrifice
THREAD_COUNT = feyn.tools.infer_available_threads()

#Getting useful data from previous file
potential_vs_time = find_potential.potential_vs_time
distance_vs_time = find_potential.distance_array
num_cols = find_potential.num_cols

#Creates the dataframe and splits it so that the model can be trained
samples_df = pd.DataFrame({"r": distance_vs_time, "Potential": potential_vs_time})

q1 = feyn.QLattice()

train, test = feyn.tools.split(samples_df, ratio = [0.5,0.5])

#Runs the model generation algorithm 
models = q1.auto_run(train, output_name = "Potential", n_epochs = 30, max_complexity = 5, loss_function = "absolute_error", threads = THREAD_COUNT)
sympy_models = []

#Outputs the models into html files and prints the latex
for i, model in enumerate(models):
    sympy_model = model.sympify(signif = 3)
    sympy_models.append(str(sympy.simplify(sympy_model)))
    model.plot(train, compare_data = test, filename = "output_" + str((i + 1)) + ".html")

#Outputs the expressions with constants to a single txt file
with open("expressions.txt", mode = "w+") as file:
    for expr in sympy_models:
        file.write(expr + "\n")