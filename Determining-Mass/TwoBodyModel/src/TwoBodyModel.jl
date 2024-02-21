include("GetTwoBodyData.jl")
include("EvalTwoBodyModel.jl")

module TwoBodyModel

using SymbolicRegression
using MLJ
using DataFrames
using Latexify

const FILENAME = "data12.csv"
const FILEPATH = joinpath("..", "data", FILENAME)

#Uses module defined in GetTwoBodyData.jl file to get the required data input from file one
#Columns: time, r, Δx, Δy, x_1, y_1, x_2, y_2, v_1x, v_1y, v_2x, v_2y, a_1x, a_1y, a_2x, a_2y
allData::DataFrame = Main.GetTwoBodyData.dataFromFile(FILEPATH)

#Creates the options for the model and fits to each desired output
model = MultitargetSRRegressor(
    niterations = 80,
    binary_operators = [+, -, *, /],
    unary_operators = [],
    constraints = [],
    elementwise_loss = L2DistLoss(),
    output_file = "equations",
    should_simplify = true,
    maxsize = 20,
    parsimony = 0.001,
)

#Puts inputs and outputs into the fitting 
#input = r, Δx, Δy
#output = a_1x, a_1y, a_2x, a_2y
mach = machine(model, select(allData, 2:4), select(allData, 13:16))

#Does the fitting and gets the output 
fit!(mach) 

r = report(mach) 
for (output_index, (eq, i)) in enumerate(zip(r.equation_strings, r.best_idx))
    println("Equation used for ", output_index, ": ", latexify(eq[i]))
end

#Plots the best model vs the original data
Main.EvalTwoBodyModel.evalData(allData, mach)

#To stop the program termination
readline()

end # module TwoBodyModel