using Flux
using Plots
using Distributions
using Turing
using StatsPlots

# Tutorial
#  https://dm13450.github.io/2020/12/19/EdwardAndFlux.html



## NN with FLUX

# generate the training DT
f(x) = cos(x) + rand(Normal(0, 0.1))

xTrain = collect(-3:0.1:3)
yTrain = f.(xTrain)
plot(xTrain, yTrain, seriestype=:scatter, label="Train Data")
plot!(xTrain, cos.(xTrain), label="Truth")

#NN
model = Chain(Dense(1, 2, tanh),
              Dense(2, 1))

loss(x, y) = Flux.Losses.mse(model(x), y)
optimiser = Descent(0.1);


x = rand(Normal(), 100)
y = f.(x)
train_data = Iterators.repeated((Array(x'), Array(y')), 100);

Flux.@epochs 10 Flux.train!(loss, Flux.params(model), train_data, optimiser)


yOut = zeros(length(xTrain))
for (i, x) in enumerate(xTrain)
    yOut[i] = model([xTrain[i]])[1]
end

ppp=plot(xTrain, yOut, label="Predicted")
plot!(xTrain, cos.(xTrain), label="True")
plot!(x, y, seriestype=:scatter, label="Data")
display(ppp)
