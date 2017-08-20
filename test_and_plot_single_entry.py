# python notebook for Make Your Own Neural Network
# code for a 3-layer neural network, and code for learning the MNIST dataset
# (c) Tariq Rashid, 2016
# (c) Martin Haag, 2017
# license is GPLv2
import numpy
from neuralnetwork import NeuralNetwork
from random import randint
from matplotlib import pyplot


n = NeuralNetwork()

n.load("config.pickle")

# load the mnist test data CSV file into a list
test_data_file = open("mnist_dataset/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()



# go through all the records in the test data set
record = test_data_list[randint(0,len(test_data_list))]

# split the record by the ',' commas
all_values = record.split(',')
# correct answer is first value
correct_label = int(all_values[0])
# scale and shift the inputs
inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
# query the network
outputs = n.query(inputs)
# the index of the highest value corresponds to the label
label = numpy.argmax(outputs)

print("Detected:", label )
print("Stats:", outputs)

image_arry = numpy.asfarray(all_values[1:]).reshape((28,28))
pyplot.imshow(image_arry, cmap='Greys', interpolation='None')
pyplot.show()


