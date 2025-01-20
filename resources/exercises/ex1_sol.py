#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: curiarteb
"""

import os
import tensorflow as tf
os.environ['KERAS_BACKEND'] = "tensorflow"
import keras

# EXECRISE 1
# Given two vectors, x and y, create a function that generates the cartesian product vector
# Do not use for loops! Treat its lenght (dimension) as it was 'None' (undetermined)

# To check the above, consider the following:
    
# SOLUTIONS (two possibilities)
# OPTION 1
@tf.function
def cartesian_product1(x,y):
    x_new, y_new = x[None,:,None], y[ :, None, None]
    out = tf.concat([x_new + tf.zeros_like(y_new), tf.zeros_like(x_new) + y_new], axis = 2)
    out = tf.reshape(out, shape=(-1,2))
    return out

# OPTION 2
@tf.function
def cartesian_product2(x,y):
    out = tf.stack(tf.meshgrid(x,y),axis=-1)
    out = tf.reshape(out, shape=(-1,2))
    return out

# Define the function as:
@tf.function
def cartesian_product(x,y):
    out = cartesian_product1(x,y)
    # out = cartesian_product2(x,y)
    return out

# Thereafter, execute the following:
# This is a custom Model
class cartesian_prod_model(keras.Model):

    def __init__(self, **kwargs):
        (super(cartesian_prod_model, self).__init__)(**kwargs)

    def call(self, inputs):
        x,y = inputs
        out = cartesian_product(x,y)
        return out

# This is a custom loss
# Needed to compile the model without errors
class dummy_loss(keras.losses.Loss):
    
    def call(self, y_true, y_pred):
        return tf.reduce_mean(y_pred, axis=-1)

# Generate data
x = tf.random.normal(shape=(3,))
y = tf.random.normal(shape=(3,))
print("x: ",x)
print("\ny: ",y)
data=[x,y]

# Initialize the model
model = cartesian_prod_model()
print("\nx times y: ", model(data))

# Compile the model (add the optimizer and the loss)
model.compile(optimizer="SGD", loss=dummy_loss())

# Fit the model (only for checking graph-mode compatibility)
history = model.fit(x=data, y=y, epochs=2)

# If the run did not given errors, your implementation is suitable for graph-mode execution!



