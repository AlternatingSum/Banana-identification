#!/usr/bin/env python2

"""
Banana Processor
"""

from skimage import io
import os
import matplotlib.pyplot as plt
import numpy as np
import random


def create_banana_array():
    """Creates a 2d array representing the bananas in the training set."""
    
    path = "Banana_Lady_Finger"
    folder = os.listdir(path)

    m = len(folder)
    n = 100*100*3

    train = np.zeros((n, m), dtype = int)

    for image_index in range(len(folder)):
        image_name = folder[image_index]
        current_filename = os.path.join(path, image_name)
        current_image = io.imread(current_filename)
        unrolled_image = np.reshape(current_image, (100*100*3))
        train[:, image_index] = unrolled_image
    
    return train


def create_banana_test_array():
    """Creates a 2d array representing the bananas in the testing set."""
    
    path = "Banana_Lady_Finger_Test"
    folder = os.listdir(path)

    m = len(folder)
    n = 100*100*3

    test = np.zeros((n, m), dtype = int)

    for image_index in range(len(folder)):
        image_name = folder[image_index]
        current_filename = os.path.join(path, image_name)
        current_image = io.imread(current_filename)
        unrolled_image = np.reshape(current_image, (100*100*3))
        test[:, image_index] = unrolled_image
    
    return test


def create_other_fruit_array(folder_name):
    """Creates a 2d array representing the fruits in the testing set of a given type."""
    
    folder = os.listdir(folder_name)

    m = len(folder)
    n = 100*100*3

    test = np.zeros((n, m), dtype = int)

    for image_index in range(len(folder)):
        image_name = folder[image_index]
        current_filename = os.path.join(folder_name, image_name)
        current_image = io.imread(current_filename)
        unrolled_image = np.reshape(current_image, n)
        test[:, image_index] = unrolled_image
    
    return test


def array_mean(array):
    """Finds the mean for each row in an array."""
    
    width = len(array[0])
    return np.sum(array, axis = 1)/width


def array_standard_deviation(array, mean):
    """Finds the standard deviation for each row in an array, given the mean for each row."""
    
    height = len(array)
    width = len(array[0])
    mean_array = np.reshape(mean, (height, 1))
    standard_deviation = (np.sum((array - mean_array)**2, axis = 1)/width)**0.5
    
    return np.ndarray.astype(standard_deviation, int)


def random_banana(mean, standard_deviation):
    """Creates a random image based on the mean and standard deviation for a set of images"""
    
    new_image = np.random.normal(mean, standard_deviation)
    
    return np.ndarray.astype(new_image, int)


def color_cost(images, mean, st_dev):
    """Calculates the cost of an image based on a multivariate Gaussian model of correct images."""
    
    st_dev = np.maximum(st_dev, 0.001)
    return (np.sum(((images - mean)/st_dev)**2, axis = 0)/30000.0)**0.5


def vector_to_image(vector):
    """Transforms a vector into an image. Assumes the vector has values in [-1, 1]."""
    
    vector = 255*(vector + 1)/2
    vector = np.reshape(vector, (100, 100, 3))
    vector = np.maximum(vector, np.zeros((100, 100, 3)))
    vector = np.minimum(vector, 255*np.ones((100, 100, 3)))
    return np.ndarray.astype(vector, int)


def find_all_costs():
    """Finds the color costs of all bananas in the training set."""
    
    full_train = create_banana_array()
    full_train.astype(float)
    mean = array_mean(full_train)
    standard_deviation = array_standard_deviation(full_train, mean)
    height = len(mean)
    mean_array = np.reshape(mean, (height, 1))
    st_dev_array = np.reshape(standard_deviation, (height, 1))
    full_train = (2*full_train/255.0) - 1
    mean_array = (2*mean_array/255.0) - 1
    st_dev_array = 2*st_dev_array/255.0
    
    return color_cost(full_train, mean_array, st_dev_array)


def find_not_banana_costs(number):
    """Finds the color costs of a collection of randomly generated non-bananas.
    These non-bananas are constructed from the multivariate Gaussian distribution which models the bananas."""
    
    full_train = create_banana_array()
    full_train.astype(float)
    mean = array_mean(full_train)
    standard_deviation = array_standard_deviation(full_train, mean)
    small_training_set = create_training_set(number)
    not_bananas = small_training_set[:, number:2*number]
    not_bananas.astype(float)
    height = len(mean)
    st_dev_array = np.reshape(standard_deviation, (height, 1))
    mean_array = np.reshape(mean, (height, 1))
    mean_array = (2*mean_array/255.0) - 1
    st_dev_array = 2*st_dev_array/255.0
    
    return color_cost(not_bananas, mean_array, st_dev_array)


def find_random_image_costs(number):
    """Finds the color costs of a collection of randomly generated images.
    These images are constructed from a uniform distribution on [-1, 1) for each parameter."""
    
    full_train = create_banana_array()
    full_train.astype(float)
    mean = array_mean(full_train)
    standard_deviation = array_standard_deviation(full_train, mean)
    height = len(mean)
    mean_array = np.reshape(mean, (height, 1))
    mean_array = (2*mean_array/255.0) - 1
    st_dev_array = np.reshape(standard_deviation, (height, 1))
    st_dev_array = 2*st_dev_array/255.0
    new_array = np.random.rand(height, number)
    new_array = 2*new_array - 1
    
    return color_cost(new_array, mean_array, st_dev_array)
    

def create_training_set(num):
    """Creates a training set with num bananas and num non-bananas"""
    
    full_train = create_banana_array()
    mean = array_mean(full_train)
    standard_deviation = array_standard_deviation(full_train, mean)
    small_train = np.zeros((len(full_train), 2*num), dtype = float)
    
    for index in range(num):
        banana_index = random.randrange(len(full_train[0]))
        small_train[:, index] = full_train[:, banana_index]
        
    for index in range(num):
        new_non_banana = random_banana(mean, standard_deviation)
        small_train[:, index + num] = new_non_banana
        
    return (2*small_train/255.0) - 1


def create_full_train_and_test_sets():
    """Creates the full training set."""
    
    full_train = create_banana_array()
    mean = array_mean(full_train)
    standard_deviation = array_standard_deviation(full_train, mean)
    train = np.zeros((len(full_train), 2*len(full_train[0])), dtype = float)
    
    for index in range(len(full_train[0])):
        train[:, index] = full_train[:, index]
        new_non_banana = random_banana(mean, standard_deviation)
        train[:, index + len(full_train[0])] = new_non_banana
        
    full_test = create_banana_test_array()
    test = np.zeros((len(full_test), 2*len(full_test[0])), dtype = float)
    
    for index in range(len(full_test[0])):
        test[:, index] = full_test[:, index]
        new_non_banana = random_banana(mean, standard_deviation)
        test[:, index + len(full_test[0])] = new_non_banana
        
    train = (2*train/255.0) - 1
    test = (2*test/255.0) - 1
    
    return [train, test]


"""The rest of the code in this file creates images showing the mean for each parameter for the bananas in the training set, 
the standard deviation for that same data, and a randomly generated image made using that mean and standard deviation."""

bananas = create_banana_array()
banana_mean = array_mean(bananas)
banana_st_dev = array_standard_deviation(bananas, banana_mean)
not_banana = random_banana(banana_mean, banana_st_dev)

"""Uncomment one of the following three lines to see the relevant image."""

#image = np.reshape(banana_mean, (100, 100, 3))
#image = np.reshape(banana_st_dev, (100, 100, 3))
#image = np.reshape(not_banana, (100, 100, 3))

"""Uncomment the next two lines to see one of the above images."""

#plt.imshow(image)
#plt.show()
