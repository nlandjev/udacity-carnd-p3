import csv
import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import sklearn


def read_image(src_path):
    filename = os.path.basename(src_path)
    target_path = "data/IMG/{0}".format(filename)

    return cv2.imread(target_path)


def batch_generator(samples, batch_size=64, angle_correction=0.2, shuffle=False):
    num_samples = len(samples)
    while True:
        # shuffle samples (only in training)
        if shuffle:
            shuffled_samples = sklearn.utils.shuffle(samples)
        else:
            shuffled_samples = samples

        for offset in range(0, num_samples, batch_size):
            batch_samples = shuffled_samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # three images per frame
                center_image = read_image(batch_sample[0])
                left_image = read_image(batch_sample[1])
                right_image = read_image(batch_sample[2])

                # flipped images
                center_image_flipped = np.fliplr(center_image)
                left_image_flipped = np.fliplr(left_image)
                right_image_flipped = np.fliplr(right_image)

                images.extend([center_image, left_image, right_image,
                               center_image_flipped, left_image_flipped, right_image_flipped])

                # calculate angles for the three images (multiplied by -1 for the flipped images)
                angle = float(batch_sample[3])
                angle_left = angle + angle_correction
                angle_right = angle - angle_correction

                angles.extend([angle, angle_left, angle_right,
                               -angle, -angle_left, -angle_right])

            X_train = np.array(images)
            y_train = np.array(angles)

            yield (X_train, y_train)


def lenet():
    """
    LeNet architecture NOTE: it is not actually used below
    :return: a sequential keras model of LeNet
    """
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Conv2D(6, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

    return model


def nvidia_net(dropout=None):
    """
    NVidia network architecture for self-driving cars as defined here: https://devblogs.nvidia.com/deep-learning-self-driving-cars/
    with added BatchNorm and optional dropout
    :param dropout: Dropout probability for activations during training
    :return: Sequential Keras model of the NVidia architecture
    """
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    if dropout:
        model.add(Dropout(dropout))
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    if dropout:
        model.add(Dropout(dropout))
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    if dropout:
        model.add(Dropout(dropout))
    model.add(Conv2D(64, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    if dropout:
        model.add(Dropout(dropout))
    model.add(Conv2D(64, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    if dropout:
        model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    if dropout:
        model.add(Dropout(dropout))
    model.add(Dense(50))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    if dropout:
        model.add(Dropout(dropout))
    model.add(Dense(10))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(1))
    return model


# read all data in the data/ folder
samples = []
with open('data/driving_log.csv') as driving_log:
    reader = csv.reader(driving_log)
    for line in reader:
        samples.append(line)

batch_size = 64
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
# generate batches
# NOTE: this implementation generates augmented samples as well.
# it would be better if we only used non-augmented samples for validation -
train_generator = batch_generator(train_samples, batch_size=batch_size, shuffle=True)
validation_generator = batch_generator(validation_samples, batch_size=batch_size)


# 2 samples per image - one regular and one flipped
# 3 camera images per frame
steps_per_epoch = 2 * 3 * len(train_samples) / batch_size
val_steps_per_epoch = 2 * 3 * len(validation_samples) / batch_size
print("Training samples: {0}, Validation samples: {1}, Steps per Epoch: {2}, Validation Steps per Epoch: {3}"
      .format(len(train_samples), len(validation_samples), steps_per_epoch, val_steps_per_epoch))

# define network
model = nvidia_net(dropout=0.2)  # lenet() <- doesn't work, use only if you want to drive into the lake

# no need to change learning rate - this one works fine
optimizer = Adam(lr=0.001)

# train and validate for 15 epochs <- arrived at through trial and error
model.compile(loss='mse', optimizer=optimizer)
model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch,
                    validation_data=validation_generator,  validation_steps=val_steps_per_epoch,
                    epochs=15, shuffle=True)

# save model
model.save('models/model.h5')
