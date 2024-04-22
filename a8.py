from neural import *

# print("<<<<<<<<<<<<<< XOR >>>>>>>>>>>>>>\n")

# xor_training_data = [
#     ([0, 0], [0]),
#     ([0, 1], [1]),
#     ([1, 0], [1]),
#     ([1, 1], [0])
# ]

# network = NeuralNet(2, 2, 1)
# network.train(xor_training_data)
# print("\n")

# network2 = NeuralNet(2, 8, 1)
# network2.train(xor_training_data)
# print("\n")

# network3 = NeuralNet(2, 1, 1)
# network3.train(xor_training_data)

print("<<<<<<<<<<<<<< VOTERS >>>>>>>>>>>>>>\n")

voting_training_data = [
    ([0.9, 0.6,	0.8, 0.3, 0.1], [1.0]),
    ([0.8, 0.8, 0.4, 0.6, 0.4], [1.0]),
    ([0.7, 0.2,	0.4, 0.6, 0.3],	[1.0]),
    ([0.5, 0.5,	0.8, 0.4, 0.8],	[0.0]),
    ([0.3, 0.1,	0.6, 0.8, 0.8],	[0.0]),
    ([0.6, 0.3,	0.4, 0.3, 0.6],	[0.0])
]

voting_network = NeuralNet(5, 8, 1)
voting_network.train(voting_training_data)

print(voting_network.evaluate([1.0, 1.0, 1.0, 0.1, 0.1])) # 1
print(voting_network.evaluate([0.5, 0.2, 0.1, 0.7, 0.7])) # 0
print(voting_network.evaluate([0.8, 0.3, 0.3, 0.3, 0.8])) # 0
print(voting_network.evaluate([0.8, 0.3, 0.3, 0.8, 0.3])) # 1
print(voting_network.evaluate([0.9, 0.8, 0.8, 0.3, 0.6])) # 0 A BIT UNCERTAIN

