from neural import *

print("<<<<<<<<<<<<<< XOR >>>>>>>>>>>>>>\n")

xor_training_data = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0])
]


network = NeuralNet(2, 2, 1)
network.train(xor_training_data)
print("\n")

network2 = NeuralNet(2, 2, 1)
network2.train(xor_training_data)
print("\n")

network3 = NeuralNet(2, 2, 1)
network3.train(xor_training_data)

# ERROR: 0.36 -> 0.01 -> 0.008

