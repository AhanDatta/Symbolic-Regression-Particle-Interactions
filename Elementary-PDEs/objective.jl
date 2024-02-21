#Sets the objective 
function objective_function(tree, dataset::Dataset{T,L}, options) where {T,L} 
    #Gets the predictions and returns the error
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    !flag && return L(1e8)

    #Returns squared error
    diffs = (1e10 .* prediction) .- dataset.y
    return sum(diffs .^ 2) / length(diffs)
end