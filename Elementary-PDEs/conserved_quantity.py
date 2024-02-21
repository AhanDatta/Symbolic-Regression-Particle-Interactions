import numpy as np 
import pandas as pd 
import pysr 
import sympy
import get_data

#Gets important data from previous file
position_one = get_data.particle_one_position
position_two = get_data.particle_two_position
velocity_one = get_data.particle_one_velocity
velocity_two = get_data.particle_two_velocity
num_cols = get_data.num_cols

#Finds distance and speeds
distance = np.empty(num_cols)
speed_one = np.empty(num_cols)
speed_two = np.empty(num_cols)
for i in range(num_cols):
    distance[i] = np.linalg.norm(position_two[i] - position_one[i])
    speed_one[i] = np.linalg.norm(velocity_one[i])
    speed_two[i] = np.linalg.norm(velocity_two[i])

#Creates input and output such that the output is constant
input_df = pd.DataFrame({"r": distance,
                         "v1": speed_one, 
                         "v2": speed_two})
output_df = pd.DataFrame({"Constant": np.zeros(num_cols)})

#Creates the objective function to put constraints
objective = """
#Sets the objective 
function objective_function(tree, dataset::Dataset{T,L}, options) where {T,L} 
    #Gets the predictions and returns the error
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    !flag && return L(1e8)

    #Returns squared error
    diffs = (1e10 .* prediction) .- dataset.y
    return sum(diffs .^ 2) / length(diffs)
end
"""

#Sets up regressor
model = pysr.PySRRegressor(
    niterations = 40,
    model_selection = "accuracy",
    populations = 100,
    population_size = 15,
    binary_operators = ["+", "-", "*"],
    unary_operators = [],
    complexity_of_constants = 0,
    early_stop_condition = "f(loss, complexity) = (complexity > 30) && (loss < 1e-7)",
    equation_file = "ouput.csv",
    full_objective = objective,
    delete_tempfiles = True,
    turbo = True,
    should_simplify = True,
    maxsize = 50, 
    parsimony =  -1,
) 

#Running model
best = model.fit(input_df, output_df)

#Outputting results
print(best.latex())
