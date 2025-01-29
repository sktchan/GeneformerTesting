#= TODO:
1. take target_arr, labels
2. calculate std.dev b/n target_arr?
3. see if std.dev == labels (or similar?)
=#

using PyCall, Statistics, Random, LinearAlgebra, DataFrames, Plots

np = pyimport("numpy")
target_arr_py = np.load("data/target_arr.npy")
labels_py = np.load("data/labels.npy")
targets_py = np.load("data/targets.npy")

target_arr = Float32.(Array(target_arr_py))
labels = Float32.(Array(labels_py))
targets = Float32.(Array(targets_py))

std_devs = Float64[]

for gene in targets
    positions = Int[]
    for column in 1:10000
        position = findall(x -> x == gene, target_arr[:, column])
        append!(positions, position)
    end
    if !isempty(positions)
        push!(std_devs, std(positions))
    else
        push!(std_devs, NaN)
    end
end

println(targets)
println(std_devs)

# issue: intersect(targets, target_arr) only returns 28 values -- so there are only 28 tfs being checked out of the 244??
