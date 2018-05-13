# **Behavioral Cloning** 

## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[example0]: ./examples/example0-bridge.png "Driving on the bridge"
[example1]: ./examples/example1-opposite-direction.png "Driving in the opposite direction 1"
[example2]: ./examples/example2-opposite-direction.png "Driving in the opposite direction 2"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* models/model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py models/model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My solution uses a modified version of the NVidia architecture presented in the lectures. 

#### 2. Attempts to reduce overfitting in the model

I added dropout and batch normalization to reduce overfitting. I found this sufficient to implement a working model, so I didn't look at any further teachniques (e.g. adding L2 regularization in the loss function)

#### 3. Model parameter tuning

I found that most of the default values as well as my initial guesses worked so I didn't need to change them. This was true in particular for the learning rate (1e-2), batch size (64) and dropout probability (0.2). I did however play around with the number of epochs and found 15 to be an appropriate compromise between speed and accuracy.

#### 4. Appropriate training data

I started with the data provided by Udacity and added a few laps of driving of my own, half of which were in the opposite direction.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with the NVidia architecture presented in the lectures as it was designed specifically for self-driving cars and add custom layers to reduce overfitting. I then trained the network until for a few epochs (until the loss stopped decreasing). I found that the car was driving mostly fine with the exception of a few specific spots (first the curves, then the bridge, then the left turn after the bridge etc.)

I then proceeded to add more data in order to mitigate those problems as I thought that the network itself was good enough. I successively added:

1. Flipped images (with the steering angle multiplied by -1)
2. Images from all 2 cameras with angle correction 0.2
3. I recorded a couple of runs on the bridge, specifically recording recovery from the sides
4. I recorded two more laps of custom data (regular driving in the middle of the road)
5. I recorded two (or three) more laps of custom data driving in the opposite direction (again - regular driving in the middle of the road)

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

My model uses the NVidia architecture as presented here: https://devblogs.nvidia.com/deep-learning-self-driving-cars/ with a few modifications to preprocess the data and reduce overfitting:

1. A normalization lambda layer to move all values in the range [-0.5, 0.5]
2. A cropping layer as suggested in the lectures
3. BatchNormalization layers before each RELU
4. Dropout layers after each RELU (except for the last layer) - the value I used during training for the dropout probability was 0.2. It was the first value I tried and as the model successfully drives the car around the track I found no need to change it.

Here is a complete overview of the model as output by the keras summary() method:
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 158, 24)       1824      
_________________________________________________________________
batch_normalization_1 (Batch (None, 31, 158, 24)       96        
_________________________________________________________________
activation_1 (Activation)    (None, 31, 158, 24)       0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 31, 158, 24)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 77, 36)        21636     
_________________________________________________________________
batch_normalization_2 (Batch (None, 14, 77, 36)        144       
_________________________________________________________________
activation_2 (Activation)    (None, 14, 77, 36)        0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 14, 77, 36)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 37, 48)         43248     
_________________________________________________________________
batch_normalization_3 (Batch (None, 5, 37, 48)         192       
_________________________________________________________________
activation_3 (Activation)    (None, 5, 37, 48)         0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 5, 37, 48)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 35, 64)         27712     
_________________________________________________________________
batch_normalization_4 (Batch (None, 3, 35, 64)         256       
_________________________________________________________________
activation_4 (Activation)    (None, 3, 35, 64)         0         
_________________________________________________________________
dropout_4 (Dropout)          (None, 3, 35, 64)         0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 33, 64)         36928     
_________________________________________________________________
batch_normalization_5 (Batch (None, 1, 33, 64)         256       
_________________________________________________________________
activation_5 (Activation)    (None, 1, 33, 64)         0         
_________________________________________________________________
dropout_5 (Dropout)          (None, 1, 33, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 2112)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               211300    
_________________________________________________________________
batch_normalization_6 (Batch (None, 100)               400       
_________________________________________________________________
activation_6 (Activation)    (None, 100)               0         
_________________________________________________________________
dropout_6 (Dropout)          (None, 100)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
batch_normalization_7 (Batch (None, 50)                200       
_________________________________________________________________
activation_7 (Activation)    (None, 50)                0         
_________________________________________________________________
dropout_7 (Dropout)          (None, 50)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
batch_normalization_8 (Batch (None, 10)                40        
_________________________________________________________________
activation_8 (Activation)    (None, 10)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 349,803
Trainable params: 349,011
Non-trainable params: 792
_________________________________________________________________
```


#### 3. Creation of the Training Set & Training Process

I started off collecting custom driving in the middle of the lane but found that the provided sample data worked better so I used that instead. 

Additionallу I added a few runs of driving on the bridge as that's where the car was failing. Here is an example image:

![Driving on the bridge][example0]

At this point I found out that no further changes to the model architecture or hyperparameters improved the driving behaviour, which was having problems navigating the left turn after the bridge. That's why I decided to record a couple of laps driving in the opposite direction in order to train a more general model. Here are a couple of images from the run:

![Driving in the opposite direction 1][example1]
![Driving in the opposite direction 2][example2]

