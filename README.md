# Banana-identification
This program constructs a class to implement neural networks with both convolutional layers and fully connected layers. It then trains a convolutional neural network to distinguish images of bananas from randomly generated images which are statistically similar to bananas. This neural network performs perfectly (304/304 correct predictions on the testing set) for types of images which are similar to (but distinct from) those used in its training. 

## Neural network class
The file Neural_Network_Class.py contains an implementation of neural networks with both partially and fully connected layers, including convolutional layers. To accomplish this each neural networks has a list of weight matrices (containing the weights and biases for each layer), as well as a list of "connection matrices".  A connection matrix must be the same size as the weight matrix for its layer, but its entries are all ones and zeros, indicating which inputs are connected to which outputs. For a fully connected layer all the entries in its connection matrix are 1.0. 

To create a convolutional layer, one uses the function build_connection_matrix. This function creates the appropriate connection matrix once the user specifies the sizes of the (square) image and filter, the stride, and the depth for both the input and output. 

## Statistical bananas
The file Banana_Processor.py loads images of bananas (and sometimes other fruits). These images all come from Kaggle's [Fruits 360 dataset](https://www.kaggle.com/moltean/fruits), and can be downloaded there. 

After loading the training set of banana images, this file calculates the mean and standard deviation of this dataset for each pixel/color parameter combination. For example, the parameter "red" at the pixel (17, 36) takes 450 values (including repeats) in the 450 images in the dataset, and so we can calculate the mean and standard deviation of these values. Since each pixel/parameter combination has its own mean, we can create an image representing these means, and do the same for the standard deviations: 

![Banana training set mean and standard deviation](https://imgur.com/dTOd1iH)
