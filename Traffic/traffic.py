# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 19:59:06 2020

@author: sakbv
"""



import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )
    

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
  
    """
   
    
    images = []
    labels = []
    
    filepath = os.path.dirname(os.path.abspath(data_dir))

    for i in range(0,43):
        print(f"Loading files from {i}...")
        os.chdir(os.path.join(filepath, data_dir, str(i)))
        for img in os.listdir(os.getcwd()):
            img = cv2.imread(img, cv2.IMREAD_COLOR)
            if img.size != 0:
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), 3)
            images.append(img)
            labels.append(str(i))
        os.chdir(os.path.join(filepath, data_dir))
    
    return (images, labels)

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    
    # Initialising the CNN
    cnn = tf.keras.models.Sequential()

    

    # Step 1 - Convolution
    cnn.add(tf.keras.layers.Conv2D(32,(3,3), input_shape=[30, 30, 3],activation='relu'))

    # Step 2 - Poolin
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=(3,3),padding='same'))

    # Adding convolutional layer
    cnn.add(tf.keras.layers.Conv2D(32, (3,3),activation="relu"))
    cnn.add(tf.keras.layers.Conv2D(32, (3,3),activation="relu"))
   
    
    #cnn.add(tf.keras.layers.Conv2D(32, (3,3),activation="relu"))
    
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2,padding='same'))

    # Step 3 - Flattening
    cnn.add(tf.keras.layers.Flatten())
    cnn.add(tf.keras.layers.Dropout(0.2))

    # Step 4 - Full Connection
    
    cnn.add(tf.keras.layers.Dense(NUM_CATEGORIES*64, activation='relu'))
    #cnn.add(tf.keras.layers.Dense(NUM_CATEGORIES*32, activation='relu'))
    
    cnn.add(tf.keras.layers.Dropout(0.2))
    cnn.add(tf.keras.layers.Dense(NUM_CATEGORIES*32, activation='relu'))
    cnn.add(tf.keras.layers.Dropout(0.2))
     
    cnn.add(tf.keras.layers.Dense(NUM_CATEGORIES*4, activation='relu'))
    
    cnn.add(tf.keras.layers.Dropout(0.05))
    
 
    
    # Step 5 - Output Layer
    cnn.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax'))

    # Part 3 - Training the CNN

    # Compiling the CNN
    cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return cnn

    
    
if __name__ == "__main__":
    main()
