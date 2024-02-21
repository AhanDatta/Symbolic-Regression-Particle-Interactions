import numpy as np 
import pandas as pd
import sympy
from pysr import PySRRegressor
import find_potential

#Getting useful data from previous file
potential_vs_time = find_potential.potential_vs_time
distance_vs_time = find_potential.distance_array

#Creates the dataframe and splits it so that the model can be trained
input_data = pd.DataFrame({"r": distance_vs_time})
target_data = pd.DataFrame({"Potential": potential_vs_time})

#Creates model with changable parameters
model = PySRRegressor(
    niterations=40,  
    binary_operators=["+", "*", "-", "/"],
    unary_operators=[
        "cos",
        "sin",
        "exp",
        "inv(x) = 1/x",
    ],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    loss="loss(prediction, target) = (prediction - target)^2",
)

#Runs model 
best = model.fit(input_data, target_data).sympy()

print(sympy.simplify(best))