using Flux

'''
Version 1:

Auto Segmented Conv structure
-> every T steps take T priors
-> convolute
-> 1. reconstruct latent space
-> 2. predict current label from latent space

'''

LABELS = ["inert", "thumb", "index", "middel", "ring", "little"]
T = 15
ENCODER_CHANNELS = EC = [1, 5, 10, 15, 20, 25]
ENCODER_KERNELS = EK = [(3,3), (3,3), (3,3), (3,3), (3,3)]
ENCODER_ACTIVATIONS = EA = [Flux.relu, Flux.relu, Flux.relu, Flux.relu, Flux.relu]
FC_HIDDENS = FCH = [20, 10, 5, length(LABELS)]
conv_net = Flux.Chain(
    [Flux.Conv(EK[i], EC[i]=>EC[i+1], EA[i]) for i in 1:length(EK)]...)
fc_net = Flux.Chain(
    [Flux.Dense(FCH[i], FCH[i+1]) for i in length(FCH)-1]...)

loss(x, y) = Flux.crossentropy(fc_net(conv_net(x)), y)
