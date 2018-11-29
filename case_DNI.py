# DNI = Decoupled Neural Interface


'''
# synthetic models

    1. synthetic gradients - estimating gradients for some layer enabeling multi threaded backpropergation
    2. synthetic input - estimating input for some layer enabeling multi threaded input conputation

    both increase conputational cost and introduce noisy estimates


# syntetic gradients in dense and CNN models

for multi threaded and multi GPU acceleration this methode tries to predict
the derivatives of a layer at a given time (localising the error conputation) and training 
said predictive model on better derivative estimates (syntetic gradient model).

this can entail, for performance sake, that the gradient gets computed for a batch of layers
rather than one, increasing the conputations per GPU unit.

the synthetic gradient model can also be fed any other information helping with the gradient prediction:
for example:
    batch lables (called conditional DNI):
        wich reduces the loss of predictability over multiple layers
        and increases convergence

simple (x256 in 1 hidden) SGMs can be used, interestingly


# in RNNs

truncated backpropagation through time (TBPTT):
    conpute backprop for last n hidden states

TBPTT with Synthetic gradients:
    estimate gradients for some hidden state at t=n-1 [-1 because the layer at n should be untouched for the computation of its error in the future]
    propergate that estimated gradient
    walk another m steps into the future and repeat
    now update the estimated gradients at t=n with the estimated gradients at t=n+m backpropergated through the layers between n and m.

    (theoretically: you could update the SG model at t=n wrt. the gradient backpropergated from t=n+m
                    and then use the updated model to compute another gradient over the same period
                    and repeate this some x times "engraining" the gradient of one specific time frame [t=n to t=n+m in this case]
                    [has to be tested by me])


