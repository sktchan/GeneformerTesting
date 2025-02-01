# testing the standard deviation of gene expression ranks - to validate whether gene dosage sensitivity is predicted by std dev

using PyCall, Statistics, Random, LinearAlgebra, DataFrames, Plots

np = pyimport("numpy")
target_arr_py = np.load("data/target_arr.npy")
labels_py = np.load("data/labels.npy")
targets_py = np.load("data/targets.npy")

target_arr = Float32.(Array(target_arr_py))
labels = Float32.(Array(labels_py))
targets = Float32.(Array(targets_py))

len = [length(filter(x -> x > 0, row)) for row in eachrow(target_arr)]
stds = [std(filter(x -> x > 0, row)) for row in eachrow(target_arr)]
o = .!isnan.(stds)

o_1 = labels .== 1.0
o_0 = labels .== 0.0

r = roc(len[o_1], len[o_0])
auc(r)
using Plots
Plots.plot(r)

CairoMakie.boxplot(labels[o], stds[o])
CairoMakie.boxplot(labels, len / 100)

# std_devs = std(target_arr, dims=2)
# mat = [labels std_devs]

roc

println(targets)
println(std_devs)