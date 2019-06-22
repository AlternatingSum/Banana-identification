# Banana-identification
This program constructs a class to implement neural networks with both convolutional layers and fully connected layers. It then trains a convolutional neural network to distinguish images of bananas from randomly generated images which are statistically similar to bananas. This neural network performs perfectly (304/304 correct predictions on the testing set) for types of images which are similar to (but distinct from) those used in its training. 

## Neural network class
The file Neural_Network_Class.py contains an implementation of neural networks with both partially and fully connected layers, including convolutional layers. To accomplish this each neural network has a list of weight matrices (containing the weights and biases for each layer), as well as a list of "connection matrices".  A connection matrix must be the same size as the weight matrix for its layer, but its entries are all ones and zeros, indicating which inputs are connected to which outputs. For a fully connected layer all the entries in its connection matrix are 1.0. 

To create a convolutional layer, one uses the function build_connection_matrix. This function creates the appropriate connection matrix once the user specifies the sizes of the (square) image and filter, the stride, and the depth for both the input and output. 

## Statistical bananas
The file Banana_Processor.py loads images of bananas (and sometimes other fruits). These images all come from Kaggle's [Fruits 360 dataset](https://www.kaggle.com/moltean/fruits), and can be downloaded there. 

After loading the training set of banana images, this file calculates the mean and standard deviation of this dataset for each pixel/color parameter combination. For example, the parameter "red" at the pixel (17, 36) takes 450 values (including repeats) in the 450 images in the dataset, and so we can calculate the mean and standard deviation of these values. Since each pixel/parameter combination has its own mean, we can create an image representing these means, and do the same for the standard deviations: 

![Banana training set mean and standard deviation](https://i.imgur.com/dTOd1iH.png)

Once we have these means and standard deviations, for each pixel/parameter combination we can choose a random value (from 0 to 255) based on the normal distribution associated with that pixel/parameter. In this way we can create random images which are statistically similar to banana images: 

![Three statistical bananas](https://i.imgur.com/eSt6qqd.png)

We then use these randomly generated "statistical bananas" as non-banana images to train a neural network. As as result, all of the training images will come from the same ellipsoid in the 30,000 dimensional input space. Determining whether an image falls in this banana-centric ellipsoid is easy, and we don't need a neural network to do it. But distinguishing a banana from a non-banana within this ellipsoid is more challenging, and this is a more appropriate task for a convolutional neural network. 

## Training the banana identification network
The class Neural_Network includes a method to perform gradient descent. This method automatically adjusts the learning rate - lowering it significantly if the cost increases, increasing it significantly if the cost barely decreases, and increasing it slightly if the cost decreases noticeably. If the neural network appears to be stuck at a saddle point (based on a small gradient but high cost), the program searches random perturbations of the current neural network for the one with the lowest cost, and resumes gradient descent there. 

The file Banana_Identification_Network.py performs gradient descent to train a convolutional neural network which can distinguish banana images from random statistical bananas. This neural network has five layers: 

1. A convolutional layer with 10x10 filters, stride 5, and depth 5. 
2. A convolutional layer with 3x3 filters, stride 2, and depth 9. 
3. A fully connected layer with 500 neurons. 
4. A fully connected layer with 10 neurons. 
5. A fully connected layer with 1 neuron. 

Once trained, this neural network performed perfectly (304/304) on the testing set. 

## Using the neural network and Gaussian model to classify images
The neural network described above never saw a non-banana fruit in its training. 
