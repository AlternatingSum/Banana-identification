#!/usr/bin/env python2

"""
Neural Network class
"""

import numpy as np
import math


class Layer:
    """Constructs a layer for use in a neural network.
    This layer may be fully or partially connected. It may be a convolutional layer, in particular."""
    
    
    def __init__(self, weight_matrix, connection_matrix, activation_function, num_examples):
        """Initializes the layer. The weight matrix consists of all weights and biases for the layer.
        The connection matrix consists of ones and zeros, and determines which inputs are connected to which outputs. 
        For a fully connected layer, all entries in the connection matrix are 1.0.
        The activation function for the layer may either be 'leaky_relu' or 'sigmoid'.
        For this implementation, 'sigmoid' should only be used for the final layer."""
        
        self.weight_matrix = weight_matrix
        self.connection_matrix = connection_matrix
        self.activation_function = activation_function
        self.height = len(weight_matrix)
        self.width = len(weight_matrix[0])
        self.input = np.zeros((self.width - 1, num_examples))
        self.output = np.zeros((self.height, num_examples))
        self.layer_gradient = np.zeros((self.width, self.height))
        self.input_gradient = np.zeros((self.width - 1, num_examples))
        self.output_gradient = np.zeros((self.height, num_examples))
        self.num_examples = num_examples
        
    
    def change_num_examples(self, num_examples):
        """Changes the number of examples which the layer expects to receive."""
        
        self.num_examples = num_examples
    
    
    def forward_prop(self, input_vector):
        """Feeds the input vector forward through the layer.
        This returns the output after applying the activation function, 
        and also stores the output before applying the activation function.
        The latter is needed for back propagation."""
        
        self.input = input_vector
        new_vector = np.concatenate((input_vector, np.ones((1, self.num_examples))))
        self.output = np.matmul(self.weight_matrix, new_vector)
        
        if self.activation_function == "leaky_relu":
            new_activation = np.maximum(self.output, 0.2*self.output)
        elif self.activation_function == "sigmoid":
            new_activation = ([[1.0]] + np.exp(-self.output))**-1.0
            
        return new_activation
    
    
    def backward_prop(self, change_vector):
        """Performs back propagation on the layer, given the gradient of the output vector.
        Stores the gradient for the weights and biases, and returns the gradient for the input vector, 
        which can then be fed to the previous layer."""
        
        if self.activation_function == "leaky_relu":
            output_change = ((self.output < 0) * 0.2 + (self.output >= 0)) * change_vector
        elif self.activation_function == "sigmoid":
            """This does not actually take the derivative of the sigmoid. This branch only applies for the final layer."""
            output_change = change_vector
            
        new_vector = np.concatenate((self.input, np.ones((1, self.num_examples))))
        layer_gradient = np.matmul(output_change, new_vector.transpose())
        self.layer_gradient = layer_gradient * self.connection_matrix
        self.input_gradient = np.matmul(self.weight_matrix.transpose(), output_change)[:self.width - 1, :]
        
        return self.input_gradient
    
    
    def gradient_descent_step(self, alpha):
        """Updates the matrix of weights and biases based on a single iteration of gradient descent.
        Uses the learning rate alpha."""
        
        self.weight_matrix = self.weight_matrix - alpha * (self.layer_gradient * self.connection_matrix)
    
    
    def random_perturbation(self, alpha):
        """Randomly alters the weights and biases, perturbing them by up to alpha.
        This is used for generating the initial neural network, and for escaping from saddle points."""
        
        change_matrix = (2.0 * np.random.rand(self.height, self.width) - 1.0) * (alpha * self.connection_matrix)
        self.weight_matrix = self.weight_matrix + change_matrix
    
    
    def copy(self):
        """Creates a copy of the layer."""
        
        weight_matrix = np.copy(self.weight_matrix)
        connection_matrix = np.copy(self.connection_matrix)
        activation_function = self.activation_function
        num_examples = self.num_examples
        
        return Layer(weight_matrix, connection_matrix, activation_function, num_examples)
    
    
    
class Neural_Network:
    """Constructs a neural network. Each layer may be fully or partially connected. 
    This supports convolutional layers."""
    
    
    def __init__(self, weight_matrices, connection_matrices, num_examples):
        """Initializes the network. 
        weight_matrices is a list of matrices containing the weights and biases for each layer. 
        connection_matrices is a list of the connection matrices for each layer.
        num_examples indicates the number of input vectors the neural network expects."""
        
        self.layers = []
        self.num_examples = num_examples
        
        for index in range(len(weight_matrices)):
            
            if index == len(weight_matrices) - 1:
                activation_function = "sigmoid"
            else:
                activation_function = "leaky_relu"
                
            new_layer = Layer(weight_matrices[index], connection_matrices[index], activation_function, num_examples)
            self.layers.append(new_layer)
    
    
    def change_num_examples(self, num_examples):
        """Changes the number of input vectors the neural network expects."""
        
        self.num_examples = num_examples
        layers = self.layers
        
        for layer in layers:
            layer.change_num_examples(num_examples)
    
    
    def forward_prop(self, input_vectors):
        """Feeds a matrix of inputs forwards through the network, arriving at a prediction for each input."""
        
        for layer in self.layers:
            input_vectors = layer.forward_prop(input_vectors)
            
        return input_vectors
    
    
    def backward_prop(self, change_vector):
        """Performs back propagation on the network, storing the gradients of all weight matrices and input vectors.
        Returns the gradient for the initial input."""
        
        for index in range(len(self.layers)):
            layer = self.layers[-1-index]
            change_vector = layer.backward_prop(change_vector)
            
        return change_vector
    
    
    def gradient_descent_step(self, alpha):
        """Updates the neural network by performing a single iteration of gradient descent, with learning rate alpha."""
        
        for layer in self.layers:
            layer.gradient_descent_step(alpha)
    
    
    def random_perturbation(self, alpha):
        """Randomly alters the weights and biases for the network, perturbing them by up to alpha.
        This is used for generating the initial neural network, and for escaping from saddle points."""
        
        for layer in self.layers:
            layer.random_perturbation(alpha)
    
    
    def copy(self):
        """Creates a copy of the neural network."""
        
        weight_matrices = []
        connection_matrices = []
        num_examples = self.num_examples
        
        for layer in self.layers:
            weight_matrix = np.copy(layer.weight_matrix)
            connection_matrix = np.copy(layer.connection_matrix)
            weight_matrices.append(weight_matrix)
            connection_matrices.append(connection_matrix)
            
        return Neural_Network(weight_matrices, connection_matrices, num_examples)

    
    def copy_weights(self, better_network):
        """Copies all weights and biases from another neural network to the current neural network."""
        
        for index in range(len(self.layers)):
            current_layer = self.layers[index]
            better_layer = better_network.layers[index]
            current_layer.weight_matrix = np.copy(better_layer.weight_matrix)
            
            
    def find_best_mutation(self, population, alpha, current_cost, input_matrix, output_vector):
        """Creates a population of randomly perturbed versions of the current neural network, 
        and updates the neural network to match the version with the lowest cost.
        This is used for generating the initial neural network, and for escaping from saddle points."""
        
        best = self
        best_cost = current_cost
        index = 0
        
        while index < population:
            
            index += 1
            new_member = self.copy()
            new_member.random_perturbation(alpha)
            new_prediction = new_member.forward_prop(input_matrix)
            new_cost = new_member.cost(new_prediction, output_vector)
            
            if new_cost < best_cost:
                best_cost = new_cost
                best = new_member
                
        self.copy_weights(best)
        
        return best_cost
    
    
    def cost(self, prediction, output_vector):
        """Calculates the cost of a neural network associated with a given prediction and desired output."""
        
        return np.sum((prediction - output_vector)**2)/self.num_examples

    
    def gradient_magnitude(self):
        """Calculates the magnitude of the gradient for the neural network's weight matrices."""
        
        num_weights = 0
        gradient_square_sum = 0.0
        
        for layer in self.layers:
            gradient = layer.layer_gradient
            connections = layer.connection_matrix
            num_weights += np.sum(connections)
            gradient_square_sum += np.sum(gradient**2)
            
        gradient_square_sum = gradient_square_sum/num_weights
        
        return gradient_square_sum**0.5            
    
    
    def gradient_descent(self, input_matrix, output_vector, alpha, mutation_alpha, num_iterations, gradient_min, cost_max):
        """Trains the neural network via gradient descent.
        The initial learning rate is alpha, however this algorithm updates the learning rate automatically.
        mutation_alpha determines how far the current network should be randomly perturbed when generating the initial network
        or when attempting to escape from a saddle point.
        num_iterations is the maximum number of iterations which will be performed, though the process will stop earlier
        if the desired cost is reached.
        If at any step the magnitude of the gradient is below gradient_min and the cost is above cost_max, 
        this algorithm concludes that it is stuck at a saddle point and uses random perturbations to escape."""
        
        index = 0
        num_mutations = 0
        prev_cost = 1000.0
        initial_state = self.copy()
        initial_alpha = alpha
        prev_state = self.copy()
        
        while index < num_iterations and prev_cost > 0.0001:
            
            index += 1
            print index
            prediction = self.forward_prop(input_matrix)
            print "Current cost:"
            current_cost = self.cost(prediction, output_vector)
            gradient_magnitude = self.gradient_magnitude()
            
            if current_cost > prev_cost:
                """If the cost has increased, return to the previous step and adjust the learning rate downwards."""
                alpha = 0.9 * alpha
                self.copy_weights(prev_state)
            elif current_cost > 0.99999 * prev_cost:
                """If the cost has stayed the same or decreased only very slightly, adjust the learning rate upwards."""
                alpha = 1.2*alpha
            else:
                """If the cost has decreased noticeably, then slightly adjust the learning rate upwards."""
                alpha = 1.01*alpha
                
            while gradient_magnitude * alpha > 10.0:
                """Do not allow the step size to be 10.0 or greater."""
                alpha = alpha/1.5
                
            print current_cost
            print "Current step size:"
            print alpha
            
            prev_state = self.copy()
            self.backward_prop(prediction - output_vector)
            self.gradient_descent_step(alpha)
            
            print "Gradient magnitude:"
            print self.gradient_magnitude()
            
            if self.gradient_magnitude() < gradient_min and current_cost > cost_max:
                """This step helps the algorithm escape from saddle points."""
                
                print "Mutation is happening"
                num_mutations += 1
                best_cost = self.find_best_mutation(20, mutation_alpha, current_cost, input_matrix, output_vector)
                
                if best_cost >= current_cost:
                    print "Mutation is happening again"
                    num_mutations += 2
                    self.find_best_mutation(20, 10*mutation_alpha, current_cost, input_matrix, output_vector)
                    
            prev_cost = current_cost
            
            if gradient_magnitude > 10000.0:
                """Starts over in the case of dramatic overshooting."""
                
                self.copy_weights(initial_state)
                prev_cost = 1000.0
                alpha = initial_alpha
                index = 0
                num_mutations = 0
                print "Starting over"
                
        print "Number of mutations:"
        print num_mutations
        
    
    """The following three methods are used to modify an input vector so as to maximize the neural network's prediction.
    For example, if the neural network classifies images as bananas (1.0) or not bananas (0.0), these methods can 
    be used to modify an image to increase its banana-ness, according to the neural network."""
    
    
    def input_gradient(self, input_vector):
        """Determines the gradient of an input vector, assuming the correct output is 1.0."""
        
        prediction = self.forward_prop(input_vector)
        return self.backward_prop(prediction - [[1.0]])

    
    def input_gradient_descent_step(self, input_vector, mean_vector, st_dev_vector, alpha, beta):
        """Modifies an input vector through one iteration of gradient descent. This uses both the gradient arising 
        from the neural network, and the gradient obtained from a multivariate Gaussian model of the input space.
        The learning rate used for the neural network gradient is alpha, and the learning rate used for the 
        Gaussian gradient is beta."""
        
        nn_gradient = self.input_gradient(input_vector)
        color_gradient = (input_vector - mean_vector)/st_dev_vector
        color_cost = math.sqrt(np.sum(((input_vector - mean_vector)/st_dev_vector)**2)/len(input_vector))
        
        if color_cost > 2.48:
            """The number above is the maximum color cost for a banana in the training set."""
            color_emphasis = 1.0
        elif color_cost > 1.22:
            """The number above is the mean color cost for bananas in the training set."""
            color_emphasis = 0.3
        else:
            color_emphasis = 0.1
            
        change = alpha*nn_gradient + beta*color_gradient*color_emphasis
        
        return input_vector - change
        
    
    def input_gradient_descent(self, input_vector, mean_vector, st_dev_vector, alpha, beta, num_iterations):
        """Performs gradient descent on an input vector to modify it so as to maximize the neural network's prediction."""
        
        index = 0
        
        while index < num_iterations:
            index += 1
            input_vector = self.input_gradient_descent_step(input_vector, mean_vector, st_dev_vector, alpha, beta)
            
        return input_vector
    
    
    """The following method uses both a neural network and a multivariate Gaussian model of the input space to 
    predict the category of an input vector."""
    
    
    def two_factor_prediction(self, input_vector, mean_vector, st_dev_vector, nn_min, color_cost_max):
        """Uses both the neural network and a multivariate Gaussian model of the input space to 
        predict the category of an input vector."""
    
        prediction = self.forward_prop(input_vector)
        nn_true = (prediction >= nn_min)
        input_color_cost = (np.sum(((input_vector - mean_vector)/st_dev_vector)**2, axis = 0)/len(input_vector))**0.5
        color_true = (input_color_cost <= color_cost_max)
        
        return np.multiply(nn_true, color_true)
        
    

def build_connection_matrix(image_width, image_depth, filter_width, stride, output_depth):
    """Builds the connection matrix for a convolutional layer in a neural network.
    Assumes the image is square"""
    
    num_filters_per_row = (image_width - (filter_width - stride)) // stride
    num_filters = num_filters_per_row**2
    output_size = num_filters * output_depth
    input_size = image_width**2 * image_depth
    connection_matrix = np.concatenate((np.zeros((output_size, input_size)), np.ones((output_size, 1))), axis = 1)
    
    for current_filter in range(num_filters):
        
        filter_row_start = (current_filter // num_filters_per_row) * stride
        filter_col_start = (current_filter % num_filters_per_row) * stride
        
        for current_depth in range(output_depth):
            
            output_number = current_filter * output_depth + current_depth
            
            for filter_row in range(filter_width):
                
                for filter_col in range(filter_width):
                    
                    for current_input_depth in range(image_depth):
                        
                        pixel_row = filter_row_start + filter_row
                        pixel_col = filter_col_start + filter_col
                        pixel_number = pixel_row * image_width + pixel_col
                        input_number = pixel_number * image_depth + current_input_depth
                        connection_matrix[output_number][input_number] = 1.0
                        
    return connection_matrix


def write_array(filename, array):
    """Writes the contents of a 2D array to a file"""
    
    text_file  = open(filename, 'w')
    height = len(array)
    width = len(array[0])
    
    for row in range(height):
        for col in range(width):
            
            text_file.write(str(array[row][col]))
            
            if col < width - 1:
                text_file.write(" ")
                
        if row < height - 1:
            text_file.write('\n')
            
    text_file.close()
    return


def read_array(filename):
    """Creates a 2D array of real numbers based on the content of a text file"""
    
    text_file  = open(filename, 'r')
    rows = text_file.readlines()
    height = len(rows)
    width = len(rows[0].split(" "))
    built_image = np.zeros((height, width))
    
    for row in range(height):
        read_row = rows[row].split(" ")
        
        for col in range(width):
            built_image[row][col] = float(read_row[col])
            
    text_file.close()
    return built_image
