#!/usr/bin/env python2

"""
Banana Identification Network
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import Banana_Processor as banana
import Neural_Network_Class as nn


"""The following code loads the training set and testing set for banana classification."""


train_and_test = banana.create_full_train_and_test_sets()

train_input = train_and_test[0]
test_input = train_and_test[1]

num_train_examples = len(train_input[0])
num_test_examples = len(test_input[0])
            
train_output = np.concatenate((np.ones((1, num_train_examples/2)), np.zeros((1, num_train_examples/2))), axis = 1)
test_output = np.concatenate((np.ones((1, num_test_examples/2)), np.zeros((1, num_test_examples/2))), axis = 1)


"""The following code trains a neural network for banana classification.
Uncomment the final lines of this section to make it run."""


n = 100 * 100 * 3

connection_matrix_0 = nn.build_connection_matrix(100, 3, 10, 5, 5)
n1 = len(connection_matrix_0)
connection_matrix_1 = nn.build_connection_matrix(19, 5, 3, 2, 9)
n2 = len(connection_matrix_1)

connection_matrix_2 = np.ones((500, n2+1))
connection_matrix_3 = np.ones((10, 501))
connection_matrix_4 = np.ones((1, 11))
connection_matrices = [connection_matrix_0, connection_matrix_1, connection_matrix_2, connection_matrix_3, connection_matrix_4]

weight_matrix_0 = np.zeros((n1, n+1))
weight_matrix_1 = np.zeros((n2, n1+1))
weight_matrix_2 = np.zeros((500, n2+1))
weight_matrix_3 = np.zeros((10, 501))
weight_matrix_4 = np.zeros((1, 11))
weight_matrices = [weight_matrix_0, weight_matrix_1, weight_matrix_2, weight_matrix_3, weight_matrix_4]

banana_network_0 = nn.Neural_Network(weight_matrices, connection_matrices, num_train_examples)

#banana_network_0.gradient_descent(train_input, train_output, 0.00000001, 0.005, 4000, 0.00001, 0.01)

#banana_network_0.change_num_examples(len(test_input[0]))
#prediction = banana_network_0.forward_prop(test_input)
#print "Prediction on testing set:"
#print prediction
#print "Cost on testing set:"
#print banana_network_0.cost(prediction, test_output)


"""The following code loads a trained neural network for banana classification."""


connection_matrix_0 = nn.build_connection_matrix(100, 3, 10, 5, 5)
n1 = len(connection_matrix_0)
connection_matrix_1 = nn.build_connection_matrix(19, 5, 3, 2, 9)
n2 = len(connection_matrix_1)

connection_matrix_2 = np.ones((500, n2+1))
connection_matrix_3 = np.ones((10, 501))
connection_matrix_4 = np.ones((1, 11))
connection_matrices = [connection_matrix_0, connection_matrix_1, connection_matrix_2, connection_matrix_3, connection_matrix_4]

weight_matrix_0 = np.zeros((n1, n+1))
weight_matrix_1 = np.zeros((n2, n1+1))
weight_matrix_2 = np.zeros((500, n2+1))
weight_matrix_3 = np.zeros((10, 501))
weight_matrix_4 = np.zeros((1, 11))
weight_matrices = [weight_matrix_0, weight_matrix_1, weight_matrix_2, weight_matrix_3, weight_matrix_4]

banana_network_1 = nn.Neural_Network(weight_matrices, connection_matrices, num_train_examples)

layers = banana_network_1.layers
weight_path = "Weight_Matrices"
for index in range(5):
    current_layer = layers[index]
    current_weight_matrix = current_layer.weight_matrix
    current_filename = "weight_matrix_" + str(index) + ".txt"
    current_file = os.path.join(weight_path, current_filename)
    new_weights = nn.read_array(current_file)
    current_layer.weight_matrix = new_weights


"""The following code begins with a random image then uses the neural network to perform gradient descent
on that image, moving it in a banana-ish direction.
Uncomment the final lines of this section to make it run."""


random_input = 2*np.random.rand(n, 1) - 1

bananas = banana.create_banana_array()
mean = banana.array_mean(bananas)
st_dev = banana.array_standard_deviation(bananas, mean)
mean = np.reshape(mean, (len(mean), 1))
st_dev = np.reshape(st_dev, (len(mean), 1))
mean = (2*mean/255.0)-1
st_dev = 2*st_dev/255.0
st_dev = np.maximum(st_dev, 0.001)
bananas = (2*bananas/255.0) - 1

banana_network_1.change_num_examples(1)

#new_vector = banana_network_1.input_gradient_descent(random_input, mean, st_dev, 10.0, 0.001, 1000)

#new_image = banana.vector_to_image(new_vector)
#plt.imshow(new_image)
#plt.show()


"""The following code uses both the neural network and the multivariate Gaussian model of the input space
to classify a type of image the network hasn't seen before."""


apples = banana.create_other_fruit_array("Apple_Crimson_Snow")
apples = (2*apples/255.0) - 1

banana_network_1.change_num_examples(len(apples[0]))

prediction = banana_network_1.two_factor_prediction(apples, mean, st_dev, .8, 5.0)
print "There are " + str(np.sum(prediction)) + " false positives out of " + str(len(apples[0])) + " examples."
