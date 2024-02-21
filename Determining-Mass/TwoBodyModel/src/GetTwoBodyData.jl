module GetTwoBodyData

using CSV 
using DataFrames 
import LinearAlgebra
using Plots; Plots.default(show = true)

#Creates a vector of the first element in each subvector
function extract(v::Vector{Vector{U}}, index::Int)::Vector{U} where U 
    retVec::Vector{U} = Vector{U}()
    sizehint!(retVec, size(v)[1])
    for i in 1:size(v)[1]
        push!(retVec, v[i][index])
    end

    return retVec
end

#Finds the forward derivative with accuracy of order dx^2
@inline
function forwardDiff(x::AbstractVector{T}, y::AbstractVector{U}, i::UInt)::U where {T <: Real, U}
    dx = x[i+1] - x[i]
    return (-y[i+2]  + 4*y[i+1] - 3*y[i])/(2*dx)
end

#Finds the backward derivative with accuracy of order dx^2
@inline
function backwardDiff(x::AbstractVector{T}, y::AbstractVector{U}, i::UInt)::U where {T <: Real, U}
    dx = x[i] - x[i-1]
    return (y[i-2]  - 4*y[i-1] + 3*y[i])/(2*dx)
end

#Finds the centered derivative with accuracy of order dx^2
@inline
function centeredDiff(x::AbstractVector{T}, y::AbstractVector{U}, i::UInt)::U where {T <: Real,U}
    twoDx = x[i+1] - x[i-1]
    return (y[i+1] - y[i-1])/(twoDx)
end

#Combines the other derivative methods to return a derivative array
function totalDiff(x::Vector{T}, y::Vector{U})::Vector{U} where {T <: Real, U}
    #Defines return vector and checks its size
    retVector::Vector{U} = Vector{U}()
    numElements::Int =  size(x, 1)

    #If size is not compatible, throws error
    if numElements != size(y,1)
        throw(error("The size of the 2 input arrays are incompatible"))
    end

    #Computes the derivative
    sizehint!(retVector, numElements)
    push!(retVector, forwardDiff(x,y,UInt(1)))
    for i in 2:(numElements - 1)
        push!(retVector, centeredDiff(x,y, UInt(i)))
    end
    push!(retVector, backwardDiff(x,y, UInt(numElements)))

    return retVector
end

#Finds the second derivative by differentiating the first derivative forward
@inline
function forwardDiffTwo(x::AbstractVector{T}, y::AbstractVector{U}, i::UInt)::U where {T <: Real, U}
    localDerivVector::Vector{U} = Vector{U}([forwardDiff(x,y, i), centeredDiff(x,y, i+1), centeredDiff(x, y, i+2)])
    return forwardDiff(x, localDerivVector, UInt(1))
end

#Finds the second derivative by differentiating the first derivative backward
@inline
function backwardDiffTwo(x::AbstractVector{T}, y::AbstractVector{U}, i::UInt)::U where {T <: Real, U}
    localDerivVector::Vector{U} = Vector{U}([centeredDiff(x,y, i-2), centeredDiff(x,y, i-1), backwardDiff(x, y, i)])
    return backwardDiff(x, localDerivVector, UInt(size(localDerivVector, 1)))
end

#Finds the second derivate of second order error in the center 
@inline
function centeredDiffTwo(x::AbstractVector{T}, y::AbstractVector{U}, i::UInt)::U where {T <: Real, U}
    return (y[i+1] - 2*y[i] + y[i-1])/((x[i+1] - x[i])^2)
end

#Computes the second derivative with the other second derivative methods
function totalDiffTwo(x::Vector{T}, y::Vector{U})::Vector{U} where {T <: Real, U}
    #Defines return vector and checks its size
    retVector::Vector{U} = Vector{U}()
    numElements::Int =  size(x, 1)

    #If size is not compatible, throws error
    if numElements != size(y,1)
        throw(error("The size of the 2 input arrays are incompatible"))
    end

    #Computes the derivative
    sizehint!(retVector, numElements)
    push!(retVector, forwardDiffTwo(x,y,UInt(1)))
    for i in 2:(numElements - 1)
        push!(retVector, centeredDiffTwo(x,y, UInt(i)))
    end
    push!(retVector, backwardDiffTwo(x,y, UInt(numElements)))

    return retVector
end

#Takes in the file path and returns the time array and position vector arrays
function getData(FILEPATH::String)::Tuple{Vector{Float64}, Vector{Vector{Float64}}, Vector{Vector{Float64}}}
    inputDataFrame = CSV.read(FILEPATH, DataFrame)
    numCols = size(inputDataFrame[:,1])[1]

    #Defines the pair of positions we want to bind
    colOne = Vector{Float64}(inputDataFrame[:,1])
    colTwoThree = Vector{Vector{Float64}}()
    colFourFive = Vector{Vector{Float64}}()

    #Binds each piece of data to a 2vector
    sizehint!(colTwoThree, numCols)
    sizehint!(colFourFive, numCols)
    for i in 1:numCols
        push!(colTwoThree, Vector{Float64}([inputDataFrame[i, 2], inputDataFrame[i, 3]]))
        push!(colFourFive, Vector{Float64}([inputDataFrame[i, 4], inputDataFrame[i, 5]]))
    end

    return colOne, colTwoThree, colFourFive
end

function dataFromFile(FILEPATH::String)::DataFrame
    #Reads in the data to the useful arrays 
    time, particleOnePosition, particleTwoPosition = getData(FILEPATH)

    #Gets the total number of elements
    numElements::Int = size(time, 1)

    #Finds the velocity vectors of each particle 
    #Then creates the vector of velocity norm squares to return
    particleOneVelocity = totalDiff(time, particleOnePosition)
    particleTwoVelocity = totalDiff(time, particleTwoPosition)

    #Finds the acceleration vectors of each particle 
    #Old Method:
    #particleOneAcceleration = totalDiff(time, particleOneVelocity)
    #particleTwoAcceleration = totalDiff(time, particleTwoVelocity)
    particleOneAcceleration = totalDiffTwo(time, particleOnePosition)
    particleTwoAcceleration = totalDiffTwo(time, particleTwoPosition)

    #Finds the Δx and Δy
    positionDifference::Vector{Vector{Float64}} = particleTwoPosition[:,1] - particleOnePosition[:,1]
    Δx::Vector{Float64} = extract(positionDifference, 1)
    Δy::Vector{Float64} = extract(positionDifference, 2)
    
    #Since we know the particles interact pairwise, 
    #We will compute the distance vectors 
    distanceBetweenParticles::Vector{Float64} = Vector{Float64}()
    sizehint!(distanceBetweenParticles, numElements)
    for i in 1:numElements
        push!(distanceBetweenParticles, LinearAlgebra.norm(positionDifference[i], 2))
    end
    
    #Creates return dataFrame of all important fitting data
    retData::DataFrame = DataFrame("time" => time,
                                    "r" => distanceBetweenParticles,
                                    "Δx" => Δx,
                                    "Δy" => Δy,
                                    "x_1" => extract(particleOnePosition, 1),
                                    "y_1" => extract(particleOnePosition, 2),
                                    "x_2" => extract(particleTwoPosition, 1),
                                    "y_2" => extract(particleTwoPosition, 2),
                                    "vx_1" => extract(particleOneVelocity, 1),
                                    "vy_1" => extract(particleOneVelocity, 2),
                                    "vx_2" => extract(particleTwoVelocity, 1),
                                    "vy_2" => extract(particleTwoVelocity, 2),
                                    "ax_1" => extract(particleOneAcceleration, 1),
                                    "ay_1" => extract(particleOneAcceleration, 2),
                                    "ax_2" => extract(particleTwoAcceleration, 1),
                                    "ay_2" => extract(particleTwoAcceleration, 2),
    )

    return retData
end

#Plots the data parametrically from a file 
function plotData(FILEPATH::String)
    #Reads in the data to the useful arrays 
    #Then frees the not used memory
    _, particleOnePosition, particleTwoPosition = getData(FILEPATH)

    #Plots the data parametrically 
    plot(extract(particleOnePosition, 1), extract(particleOnePosition, 2))
    plot!(extract(particleTwoPosition, 1), extract(particleTwoPosition, 2))
end

export dataFromFile, plotData

#Runs only if this file is being run directly
if abspath(PROGRAM_FILE) == @__FILE__
    const FILENAME = "data12.csv"
    const FILEPATH = joinpath("..", "data", FILENAME)
    allData::DataFrame = Main.GetTwoBodyData.dataFromFile(FILEPATH)
    println(select(allData, 13:16))
end 

end #Module GetTwoBodyData

