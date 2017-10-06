# Autonomous Driving Simulation
### David Rose

### Using TF/Keras to teach an agent to drive a car using my driving as a training set.

![Driving Car](https://github.com/cipher982/Autonomous-Driving-Simulation/blob/master/media/car_driving.gif?raw=True)

* See Processing.ipynb for image processing and driving data
* See training.ipynb for network code

# Behavioral Cloning

# Using convolutional nets to model human driving in a simulation

David Rose

**Introduction**

## A broad overview of the steps taken:

 * Use the simulator to collect data of good driving behavior (a bit trickier than it seems)
 * Build, a convolution neural network in Keras that predicts steering angles from images (each image is frame of video)
 * Train and validate the model with a split set of data collected
 * Tune model until car drives around the track on its own, with no mistakes!

**Model Basics and Training Strategy**

1. An appropriate model architecture has been employed

The model includes a ReLu layer to introduce nonlinearity (code line 48), and the data is normalized in the model using a Keras lambda layer (code line 43). In the future, I may use more activations but this sufficiently drives the track as is.

1. Attempts to reduce overfitting in the model

The model contains 2 dropout layers to reduce overfitting (model.py lines 50,55).

The model was trained and validated on different data sets to ensure that the model was not overfitting (Validation split on line 63). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. Good news, it does! (Though it swerves a bit)

1. Model parameter tuning

The model used an adam optimizer, but I did change the learning rate to 0.0003,

1. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road (extra cameras), and reversed images.

 * More details in the next section!

**Solution Design Approach**

I first just added a couple convolutional layers and testing the driving performance, going back and forth with modifying the network and autonomous driving attempts.

Quickly I realized that was not working, car just kept driving in circles!!

Next approach is using validation and training loss in the model to track my progress from a high-level perspective. I ended up getting fairly low numbers, but the model became stuck on going straight, it was basically overfitting on all my straight frames where I was not turning. The solution was to re-run the training processing step, this time appending only 30% of frames with a steering angle of less than 10%. I added a couple extra epochs of training to help it overcome the loss in training data.

It then ran perfectly on my first attempt. It LOVES to swerve around on the road a lot (I think due to altering the steering angle in training on the left-right frames too much (20%), but at least it is very strict about staying away from the edges.

**Final Model Architecture**

My model is a convolutional neural network (lines xxxx) with the following layers:

| **Layer** | **Details** |
| --- | --- |
| Average Pooling | Strides = 2 |
| Lambda value normalizing | x: x/255 - .5) |
| Cropping | Remove 25px on top, 15 on bottom |
| Convolution | 16 filters, kernel size 1, strides 1 |
| Activation | ReLu |
| MaxPooling | Strides = 2 |
| Dropout | 50% |
| Flatten |   |
| Fully Connected (Dense) | 16 neurons |
| Dropout | 70% |
| Fully Connected (Dense) | 10 neurons |
| Fully Connected (Dense) | 1 neuron for predicting steering angle |

**Creation of the Training Set &amp; Training Process**

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

 ![img 1](https://github.com/cipher982/Autonomous-Driving-Simulation/blob/master/media/1.jpg?raw=True)

I also have images of the left/right cameras, to train a situation in which it was serving off the road

 ![img left](https://github.com/cipher982/Autonomous-Driving-Simulation/blob/master/media/2.jpg?raw=True)
 ![img right](https://github.com/cipher982/Autonomous-Driving-Simulation/blob/master/media/3.jpg?raw=True)


Images were also augmented by flipping them:

 ![img flip1](https://github.com/cipher982/Autonomous-Driving-Simulation/blob/master/media/4.jpg?raw=True)
 ![img flip2](https://github.com/cipher982/Autonomous-Driving-Simulation/blob/master/media/5.jpg?raw=True)

After the collection process, I had 29,464 total data points. The data is preprocessed in the model by ways of:

 * **Normalizing** – So that differences in lighting do not skew the model
 * **Resizing** – Trains much faster without a significant loss in performance
 * **Cropping** – Remove un-needed information such as the hood of car and the sky

I finally randomly shuffled the data set and put 30% of the data into a validation set. This step is done within the model.fit step, using the built-in validation\_split argument.

Total number of epochs was 8, as you can see with the image below the loss begins to level out after 3-4.

 ![img epochs](https://github.com/cipher982/Autonomous-Driving-Simulation/blob/master/media/6.jpg?raw=True)

# Conclusion

The car ends up navigating the first track fairly well, with no run-offs or other dramatic events. It does swerve a bit but I think it is due to having too much steering correction on the left-right cameras.

I would also introduce more training data in the future, including a lot of corrections from when the car is off the road. As it stands, the car loses all hope when off the track, is there is not sufficient training from that aspect. And due to the cropping of the images, it does not perform well on track #2, so I would need to re-work some of the model for that.
