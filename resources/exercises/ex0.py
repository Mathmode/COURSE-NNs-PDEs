#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: curiarteb
"""

import numpy as np
import os
import tensorflow as tf
os.environ['KERAS_BACKEND'] = "tensorflow"
import keras

# EXECRISE 0
# Given a sorted vector, create a function that returns the vector of mid points
# Do not use for loops! Treat its lenght (dimension) as it was 'None' (undetermined)

# To check the above, consider the following:
    
# SOLUTION
@tf.function
def mid_points_sol(x):
    return

# Define the function as:
@tf.function
def mid_points(x):
    out = mid_points_sol(x)
    return out

# Thereafter, execute the following:
# This is a custom Model
class mid_points_model(keras.Model):

    def __init__(self, **kwargs):
        (super(mid_points_model, self).__init__)(**kwargs)

    def call(self, inputs):
        out = mid_points(inputs)
        return out

# This is a custom loss
# Needed to compile the model without errors
class dummy_loss(keras.losses.Loss):
    
    def call(self, y_true, y_pred):
        return tf.reduce_mean(y_pred, axis=-1)

# Generate data
x = tf.constant(np.array([1.,2.,5.,8.,10.]))
print("x: ",x)

# Initialize the model
model = mid_points_model()
print("x mid: ", model(x))

# Compile the model (add the optimizer and the loss)
model.compile(optimizer="SGD", loss=dummy_loss())

# Fit the model (only for checking graph-mode compatibility)
history = model.fit(x=x, y=x, epochs=2)

# If the run did not give errors, your implementation is suitable for graph-mode execution!



