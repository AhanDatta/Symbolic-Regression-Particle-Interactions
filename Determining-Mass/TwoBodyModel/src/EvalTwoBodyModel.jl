module EvalTwoBodyModel 

using DataFrames
using Plots; Plots.default(show = true)
using MLJ

#Constants for AB2 found here:
#https://en.wikipedia.org/wiki/Linear_multistep_method
const AB_ONE = 1.5
const AB_TWO = -0.5

#Evaluates a model vs the original data by graphing it 
#Structure of allData: time, r, Δx, Δy, x_1, y_1, x_2, y_2, v_1x, v_1y, v_2x, v_2y, a_1x, a_1y, a_2x, a_2y
function evalData(allData::DataFrame, machine) 
    #Gets the important data from allData Dataframe 
    #for plotting the raw data
    x_1 = allData.x_1 
    y_1 = allData.y_1 
    x_2 = allData.x_2 
    y_2 = allData.y_2 

    #Creates the new input dataframe
    inputDF::DataFrame = select(allData, 2:4)

    #Gets the predicted output from the model 
    outputPrediction::DataFrame = predict(machine, inputDF)

    #Solves the elementary diff eq system 
    #a = dv/dt and v = dx/dt with initial conditions
    positionPredDF::DataFrame = solveDiffEq(outputPrediction, allData)

    plt = plot(x_1, y_1, xlabel = "X Position", ylabel = "Y Position", label = "Particle One Experimental")
    plot!(plt, x_2, y_2, label = label = "Particle Two Experimental")
    plot!(plt, positionPredDF.x_1, positionPredDF.y_1, label = "Particle One Predicted")
    plot!(plt, positionPredDF.x_2, positionPredDF.y_2, label = "Particle Two Predicted")
end

#Uses AB Method diff eq solving 
function solveDiffEq(outputPrediction::DataFrame, rawData::DataFrame)::DataFrame 
    #Initializes the two dataframes with initial considitions
    velocityDF = DataFrame(
        "vx_1" => Vector{Float64}([rawData.vx_1[1]]),
        "vy_1" => Vector{Float64}([rawData.vy_1[1]]),
        "vx_2" => Vector{Float64}([rawData.vx_2[1]]),
        "vy_2" => Vector{Float64}([rawData.vy_2[1]]), 
    )

    positionDF = DataFrame(
        "x_1" => Vector{Float64}([rawData.x_1[1]]),
        "y_1" => Vector{Float64}([rawData.y_1[1]]),
        "x_2" => Vector{Float64}([rawData.x_2[1]]),
        "y_2" => Vector{Float64}([rawData.y_2[1]]),
        )

    #Integrates both data sets with euler
    for i in 1:(nrows(outputPrediction) - 1) 
        deltaT::Float64 = rawData.time[i+1] - rawData.time[i]
        push!(velocityDF.vx_1, eulerMethod(outputPrediction.ax_1, velocityDF.vx_1, i, deltaT))
        push!(velocityDF.vy_1, eulerMethod(outputPrediction.ay_1, velocityDF.vy_1, i, deltaT))
        push!(velocityDF.vx_2, eulerMethod(outputPrediction.ax_2, velocityDF.vx_2, i, deltaT))
        push!(velocityDF.vy_2, eulerMethod(outputPrediction.ay_2, velocityDF.vy_2, i, deltaT))

        push!(positionDF.x_1, eulerMethod(velocityDF.vx_1, positionDF.x_1, i, deltaT))
        push!(positionDF.y_1, eulerMethod(velocityDF.vy_1, positionDF.y_1, i, deltaT))
        push!(positionDF.x_2, eulerMethod(velocityDF.vx_2, positionDF.x_2, i, deltaT))
        push!(positionDF.y_2, eulerMethod(velocityDF.vy_2, positionDF.y_2, i, deltaT))
    end

    """
    #Now integrates with AB2 for the rest of the data 
    for i in 3:(nrows(outputPrediction) - 1) 
        deltaT::Float64 = rawData.time[i+1] - rawData.time[i]
        push!(velocityDF.vx_1, secondAdamsBashforth(outputPrediction.ax_1, velocityDF.vx_1, i, deltaT))
        push!(velocityDF.vy_1, secondAdamsBashforth(outputPrediction.ay_1, velocityDF.vy_1, i, deltaT))
        push!(velocityDF.vx_2, secondAdamsBashforth(outputPrediction.ax_2, velocityDF.vx_2, i, deltaT))
        push!(velocityDF.vy_2, secondAdamsBashforth(outputPrediction.ay_2, velocityDF.vy_2, i, deltaT))

        push!(positionDF.x_1, secondAdamsBashforth(velocityDF.vx_1, positionDF.x_1, i, deltaT))
        push!(positionDF.y_1, secondAdamsBashforth(velocityDF.vy_1, positionDF.y_1, i, deltaT))
        push!(positionDF.x_2, secondAdamsBashforth(velocityDF.vx_2, positionDF.x_2, i, deltaT))
        push!(positionDF.y_2, secondAdamsBashforth(velocityDF.vy_2, positionDF.y_2, i, deltaT))
    end
    """

    return positionDF;
end

#Euler's method 
function eulerMethod(derivVector::Vector{U}, outputVector::Vector{U}, i::Int , dt::U)::U where {U <: Real}
    return outputVector[i] + derivVector[i] * dt
end

#Fourth Order AB Method 
function secondAdamsBashforth(derivVector::Vector{U}, outputVector::Vector{U}, i::Int , dt::U)::U where {U <: Real}
    return outputVector[i] + dt * (AB_ONE * derivVector[i] + AB_TWO * derivVector[i-1])
end

#Runs only if this file is being run directly
if abspath(PROGRAM_FILE) == @__FILE__
    
end 

export evalData

end #Module EvalTwoBodyModel

