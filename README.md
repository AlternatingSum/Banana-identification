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
The neural network described above never saw a non-banana fruit in its training, and can't necessarily be expected to label such an image as a non-banana on its own. However, we can use the neural network together with the multivariate Gaussian model of the banana images to obtain more information about new types of images. 

The class Neural_Network includes a method called two_factor_prediction. This method asks two questions of a given input: 

1. Is the neural network's prediction for this input above a certain cutoff? 
2. Is the distance from the image to the banana mean (measured after scaling each parameter by the standard deviation for that parameter) below a certain cutoff? In other words, does this image come from the banana-centric ellipsoid described above? 

If the answer to both questions is yes, this method labels the image a banana. If the answer to either question is no, this method labels the image a non-banana. 

The performance of this method was mixed. We used a cutoff of 0.8 for the neural network, and a cutoff of 5.0 for the Gaussian distance. With these cutoffs the method labeled only 85% of bananas in the testing set as bananas. (In fairness, the testing set and training set weren't chosen randomly from the same batch of images, and have some noticeable differences as a result.) 

The method did well on apples, including apples with mixed red and yellow coloration. For apples from both the "Apple Crimson Snow" folder and the "Apple Red Yellow 1" folder it correctly classified more than 99% of images as non-bananas. For pears, it classified 79% of images as non-bananas. Carambulas (yellow fruits which often appear elongated) were more challenging, and the method classified only 61% as non-bananas. 

## Future approaches
It may be possible to improve the approach described in the last section, to create a system which can reliably classify types of images it hasn't seen before. Here are two possible tactics: 

1. Instead of using a multivariate Gaussian distribution to model the data in the training set, use the covariance matrix of this 30,000 dimensional dataset. This would allow us to fit a rotated ellipsoid to the data, rather than one whose axes are parallel to the axes of the space, and would probably result in a better fit. 

2. Given an image with randomly colored pixels, we can use both the Gaussian model and the neural network to modify this image to make it more banana-ish. The class Neural_Network has a method, input_gradient_descent, to implement this. The images generated by this method do not look very much like bananas, but the method doesn't know that. Here's an image generated by 3000 iterations, which this method considers to be a banana: 

![Neural network constructed banana](https://i.imgur.com/sqmqMNJ.png)

If we created and stored many such images, we could use them to train a second neural network to distinguish them from bananas. This neural network could then work together with the first neural network and the Gaussian model, and hopefully acheive more accurate labelling on new types of images as a result. We could iterate on this process, with each system creating the non-banana training images for the next neural network, until we arrive at a system which has the accuracy we want. This system may even be capable of drawing a convincing banana on its own, but that remains to be seen. 
