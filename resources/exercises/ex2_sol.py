#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: curiarteb
"""

import os
import tensorflow as tf
os.environ['KERAS_BACKEND'] = "tensorflow"
import keras

# EXECRISE 2
# Solve a parametric system of linear equations with NNs
# Warning! Treat the first dimension of the Tensors as 'None' (undetermined)

# Consider the matrix
# A = [a,a**2,a**3,a**4;
#      a**2,a**3,a**4,a**3;
#      a**3,a**4,a**3,a**2;
#      a**4,a**3,a**2,a]
# and load vector f =[1;2;3;4]
# Objective: Solve x in Ax=f, with a in (2,5). x is approximated by a NN.

# Treat A as the following matrices combination:
A1 = tf.constant([[1.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,1.]])
A2 = tf.constant([[0.,1.,0.,0.],[1.,0.,0.,0.],[0.,0.,0.,1.],[0.,0.,1.,0.]])
A3 = tf.constant([[0.,0.,1.,0.],[0.,1.,0.,1.],[1.,0.,1.,0.],[0.,1.,0.,0.]])
A4 = tf.constant([[0.,0.,0.,1.],[0.,0.,1.,0.],[0.,1.,0.,0.],[1.,0.,0.,0.]])
A_list = [A1, A2, A3, A4]
# Namely, A = a*A1+a**2*A2+a**3*A3+a**4*A4

f = tf.constant([1.,2.,3.,4.])

# STEP 1 (define the trainable NN architecture):
# input: batch_size x 1 (a parameter)
# output: batch_size x 4 (approximation to solution x=[x1;x2;x3;x4])
class x_net(keras.Model):

    def __init__(self, depth=2, width=20, activation='tanh', **kwargs):
        (super(x_net, self).__init__)(**kwargs)
        self.depth = depth
        self.width = width
        self.activation = activation
        self.layers_list = list()

        for _ in range(self.depth):
            self.layers_list.append(keras.layers.Dense(units=self.width, activation=self.activation, use_bias=True))

        self.layers_list.append(keras.layers.Dense(units=4, activation=None, use_bias=False))

    def call(self, inputs):

        out = inputs
        for layer in self.layers_list:
            out = layer(out)

        return out

# STEP 2 (define the residual computation):
# input: batch_size x 1/4 (a parameter/approximation to solution x=[x1;x2;x3;x4])
# output: 1 (mean square computation of the residual Ax-f)
class residual(keras.Model):

    def __init__(self, A_list, f, **kwargs):
        (super(residual, self).__init__)(**kwargs)
        self.A_tensor = tf.stack(A_list, axis=0)
        self.f = tf.constant(f)
        
    def call(self, inputs):

        a, out = inputs
        
        a_square = tf.square(a)
        a_cube = tf.pow(a,3)
        a_fourth = tf.pow(a,4)
        a_powers = tf.concat([a,a_square,a_cube,a_fourth], axis=1)
        
        A_batch = tf.einsum("bt,tij->bij",a_powers,self.A_tensor)
        
        A_times_out = tf.einsum("bij,bj->bi", A_batch, out)
        
        residual_batch = A_times_out - self.f
        
        norm_batch = tf.sqrt(tf.reduce_sum(tf.square(residual_batch), axis=1))
        
        loss = tf.reduce_mean(norm_batch, axis=-1)

        return loss


# STEP 3 (main model):
# input: batch_size x 1 (a parameter)
# output: 1 (mean square computation of the residual Ax-f)
class main_model(keras.Model):

    def __init__(self, net, residual, **kwargs):
        (super(main_model, self).__init__)(**kwargs)
        self.x = net
        self.residual = residual

    def call(self, inputs):
        
        a = inputs
        out = self.x(a)
        residual = self.residual([a, out])

        return residual

# STEP 4 (dummy loss)
class dummy_loss(keras.losses.Loss):
    
    def __init__(self, **kwargs):
        (super(dummy_loss, self).__init__)(**kwargs)
    
    def call(self, y_true, y_pred):
        return y_pred
    
    
# STEP 5 (data creation)
a_train = tf.random.uniform(minval=2.0, maxval=5.0, shape=(10**4, 1))
print("\na for training:\n", a_train)

# STEP 6 (Architectures assembling)
x = x_net(depth=5, width=30)
norm = residual(A_list, f)
model = main_model(x, norm)

# STEP 7 (Compilation and fitting)
model.compile(optimizer="Adam", loss=dummy_loss())
training = model.fit(x=a_train, y=a_train, batch_size=32, epochs=200)
#y=a_train is never used (because y_true in dummy_loss is never used)
#However, try to run it without invoking y=... in .fit

# STEP 8 (Testing)
a_test = tf.range(start=2.0, limit=5.1, delta=3/5)[:, None]

# Analytic solution of x=[x1;x2;x3;x4]
def analytic_sol(a):
    x1 = (4*a-3)/(a**3*(a**2-1))
    x2 = (3*(a-1))/(a**4*(a+1))
    x3 = (2*(a-1))/(a**4*(a+1))
    x4 = (a-2)/(a**3*(a**2-1))
    return [x1,x2,x3,x4]

# Vectorization of above solution
def analytic_sol_vect(a):
    out = tf.vectorized_map(analytic_sol, a)
    out = tf.concat(out, axis=1)
    return out

print("\na for testing:\n", a_test)

print("\nerrors (x1, x2, x3, x4):\n", (x(a_test)-analytic_sol_vect(a_test)).numpy())

# STEP 9 (access training data)
loss = training.history["loss"]

# You can plot or do whatever yo want to do with that long vector

