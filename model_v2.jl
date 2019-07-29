using Flux

'''
Version 2:

Manual segmented RNN structure
-> if prediction is required take T prior steps
-> 1. RNN(s(t)) -> s(t+1) for each t element of T
-> 2. RNN(s(t)) -> label(t)

'''

INPUT_BUFFER = []


encoder = Flux.Chain(
    [Flux.Conv(EK[i], EC[i]=>EC[i+1]) for i in 1:length(EK)]...)
