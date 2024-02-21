#Gives out f(x) which satisfies E = f(r12) + f(r13) + f(r23)
#Answer gotten here: https://github.com/MilesCranmer/PySR/discussions/528
function objective_function(tree, dataset::Dataset{T,L}, options) where {T,L} 
    # Want base tree to have x1 as only feature; any other feature node will return early:
    if any(node -> node.degree == 0 && !node.constant && node.feature != 1, tree)
        return L(1e9)
    end

    # Evaluate once with only the feature passed
    # which is like you are setting x1=x1, then x1=x2, then x1=x3.
    f_x1, flag = eval_tree_array(tree, (@view dataset.X[[1], :]), options)  # Or just `dataset.X` is good too as it will take the first col anyways
    !flag && return L(1e8)
    f_x2, flag = eval_tree_array(tree, (@view dataset.X[[2], :]), options)
    !flag && return L(1e8)
    f_x3, flag = eval_tree_array(tree, (@view dataset.X[[3], :]), options)
    !flag && return L(1e8)
    
    prediction = f_x1 .+ f_x2 .+ f_x3
    
    #Returns squared error
    diffs = prediction .- dataset.y
    return sum(diffs .^ 2) / length(diffs)
end