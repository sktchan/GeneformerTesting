using PyCall, Flux, Statistics, Random, LinearAlgebra, DataFrames, Plots, CSV

### attempting to load in stuff directly from .py file
# # import data from python file
# py_sys = pyimport("sys")
# pushfirst!(py_sys["path"], "/home/muninn/scratch/chans/test_serena")

# pretraining = pyimport("pretraining") ### THIS IS WHERE RUNTIME EXCEEDS

# target_arr_py = pretraining.target_arr
# target_arr = Array(target_arr_py)

### giving up and saving stuff from .py file and loading it in manually
np = pyimport("numpy")
target_arr_py = np.load("target_arr.npy")
labels_py = np.load("labels.npy")
target_arr = Float32.(Array(target_arr_py))
labels = Float32.(Array(labels_py))

# model!!!
model = Chain(
    Dense(10000, 256, relu),
    BatchNorm(256),
    Dense(256, 64, relu),
    BatchNorm(64),
    Dense(64, 1, sigmoid),  # tried SkipConnection(Dense(input, input=output, relu), +), didn't do much
)

# partition
train_ratio = 0.8
n_samples = size(target_arr, 1)
train_size = Int(floor(train_ratio * n_samples))

# shuffle indices
indices = shuffle(1:n_samples)
train_indices = indices[1:train_size]
eval_indices = indices[train_size+1:end]

# split
targets_train = target_arr[train_indices, :]'
targets_eval = target_arr[eval_indices, :]'
labels_train = labels[train_indices]
labels_eval = labels[eval_indices]

println("train size: ", size(targets_train))
println("test size: ", size(targets_eval))

# for checking gradient norm
function gradient_norm(model, data, loss_function)
    grads = gradient(() -> loss_function(model, data...), Flux.params(model)) # fyi params() is deprecated
    return sum(norm(g) for g in grads)
end

# training
n_epochs = 60
n_batch = 128
loss(model, x, y) = Flux.logitbinarycrossentropy((model(x))', y)
opt = Flux.setup(Adam(0.005), model) # also tried ClipGrad(0.001), didn't do much

training_data = Flux.DataLoader((targets_train, labels_train), batchsize=n_batch, shuffle=true) # dataloader requires it to be x: (10k, 195), y: (195,)

grad_norms = []
epoch_losses = []
for epoch in 1:n_epochs
    total_loss = 0.0
    for (targets_batch, labels_batch) in training_data
        batch_loss = loss(model, targets_batch, labels_batch)
        total_loss += batch_loss

        grad_norm = gradient_norm(model, (targets_batch, labels_batch), loss)
        push!(grad_norms, grad_norm)

        Flux.train!(loss, model, [(targets_batch, labels_batch)], opt)
    end
    avg_loss = total_loss / length(training_data)
    push!(epoch_losses, avg_loss)
    @info "epoch $epoch average loss $(avg_loss)"
end

# evaluating
labels_pred = model(targets_eval) .> 0.5  # threshold predictions at 0.5 b/c binary

p1 = plot(1:n_epochs, epoch_losses, label="training loss", xlabel="epoch", ylabel="loss", title="loss across epochs")
p2 = plot(1:length(grad_norms), grad_norms, label="grad norm", xlabel="steps", ylabel="norm", title="gradient across steps") 
plot(p1, p2, layout=(1,2))

# export for python
CSV.write("labels_pred.csv", DataFrame(labels_pred=vec(labels_pred)))
CSV.write("labels_eval.csv", DataFrame(labels_eval=labels_eval))