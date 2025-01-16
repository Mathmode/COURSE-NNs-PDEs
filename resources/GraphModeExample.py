# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 10:55:34 2024

@author: jamie.taylor
         Ángel Javier Omella

Updated on Jan 2025 by:
    Ángel Javier Omella
"""
import time
import numpy as np

import tensorflow as tf

#we enforce to run in CPU
# tf.config.set_visible_devices([], 'GPU')

#display information about the device used
tf.debugging.set_log_device_placement(True)

#we force to run in eagerly mode
# tf.config.run_functions_eagerly(True)

import jax
#To enable float64 in jax
jax.config.update("jax_enable_x64", True)
# jax.config.update('jax_platform_name', 'cpu')
# jax.devices()
import jax.numpy as jnp

# Create a random tensor of size 10^8, which will be used for testing the functions
x = np.random.uniform(size=10**6)

#TENSORFLOW
###############################################################################
print()
print('Tensorflow')
print('************')

# Define a function to compute the derivative of sin(x) using TensorFlow's GradientTape in Eager mode.
def deriv_sin_tf(x):
    # GradientTape is used to record the computation for automatic differentiation
    with tf.GradientTape() as t1:
        # Watch the input tensor x to track its gradients
        t1.watch(x)
        # Compute sin(x)
        s = tf.math.sin(x)
    # Compute the gradient of sin(x) with respect to x
    ds = t1.gradient(s, x)
    return ds

# Define a function to compute the derivative of sin(x) in Graph mode
deriv_sin_graph_tf = tf.function(deriv_sin_tf)

# Is the same with the decorator @tf.function

# @tf.function
# def deriv_sin_graph(x):
#     # Same computation as deriv_sin, but this function will be executed in Graph mode
#     with tf.GradientTape() as t1:
#         # Watch the input tensor x to track its gradients
#         t1.watch(x)
#         # Compute sin(x)
#         s = tf.math.sin(x)
#     # Compute the gradient of sin(x) with respect to x
#     ds = t1.gradient(s, x)
#     return ds


#We convert the numpy array to a tf tensor
x_tf = tf.convert_to_tensor(x)

# Measure the time it takes to compute the derivative using the Eager execution mode
t0 = time.time()  # Start time before executing the function
ds = deriv_sin_tf(x_tf)  # Compute derivative in Eager mode
time_eager = time.time() - t0  # Calculate elapsed time for Eager execution
print(ds)

# Measure the time for the first execution of the graph-optimized function
t0 = time.time()  # Start time before executing the function
ds = deriv_sin_graph_tf(x_tf)  # Compute derivative in Graph mode (first call)
time_graph_1 = time.time() - t0  # Calculate elapsed time for the first Graph execution
print(ds)

# Measure the time for the second execution of the graph-optimized function
t0 = time.time()  # Start time before executing the function
ds = deriv_sin_graph_tf(x_tf)  # Compute derivative in Graph mode (second call)
time_graph_2 = time.time() - t0  # Calculate elapsed time for the second Graph execution
print(ds)

# Print the results for comparison between Eager and Graph mode executions
print("Time_tf (Eager)", time_eager)  # Time for Eager mode (immediate execution)
print("Time_tf (Graph, first call)", time_graph_1)  # Time for the first call in Graph mode (includes graph compilation overhead)
print("Time_tf (Graph, second call)", time_graph_2)  # Time for the second call in Graph mode (faster since graph is reused)

print(tf.config.list_logical_devices())

#JAX
###############################################################################
print()
print('JAX')
print('***')
print(jax.devices())
def f(x):
    return jnp.sin(x)


df_jax = jax.grad(f)
#grad is a scalar output
print(df_jax(x[0]))

#returns the value of the function and the gradient
value_and_grad_jax = jax.value_and_grad(f)
#is also a scalar function
print(value_and_grad_jax(x[0]))

# Vectorizing the functions
df_jax_vmap = jax.vmap(df_jax)
value_and_grad_vmap = jax.vmap(value_and_grad_jax)

#we jit the functions
f_jit = jax.jit(f)
df_jax_vmap_jit = jax.jit(df_jax_vmap)
value_and_grad_vmap_jit = jax.jit(value_and_grad_vmap)


# Measure the time it takes to compute the derivative using the Eager execution mode
t0 = time.time()  # Start time before executing the function
df = df_jax_vmap(x) # Compute derivative in Eager mode
time_eager = time.time() - t0  # Calculate elapsed time for Eager execution
print(df)

# Measure the time for the first execution of the graph-optimized function
t0 = time.time()  # Start time before executing the function
df = df_jax_vmap_jit(x)  # Compute derivative in Graph mode (first call)
time_graph_1 = time.time() - t0  # Calculate elapsed time for the first Graph execution
print(df)

# Measure the time for the second execution of the graph-optimized function
t0 = time.time()  # Start time before executing the function
df = df_jax_vmap_jit(x)  # Compute derivative in Graph mode (second call)
time_graph_2 = time.time() - t0  # Calculate elapsed time for the second Graph execution
print(df)


# Print the results for comparison between Eager and Graph mode executions
print("Time_jax (Eager)", time_eager)  # Time for Eager mode (immediate execution)
print("Time_jax (Graph, first call)", time_graph_1)  # Time for the first call in Graph mode (includes graph compilation overhead)
print("Time_jax (Graph, second call)", time_graph_2)  # Time for the second call in Graph mode (faster since graph is reused)



# We test here the value_and_grad function
#we call the functions to have the jit compilation
f = f_jit(x)
f_df = value_and_grad_vmap_jit(x)
#
t0 = time.time()  # Start time before executing the function
f = f_jit(x) # Compute value
df = df_jax_vmap(x) #compute grad
time_separated = time.time() - t0  # Calculate elapsed time
print (f)
print (df)

t0 = time.time()  # Start time before executing the function
f_df = value_and_grad_vmap_jit(x)
time_joint = time.time() - t0  # Calculate elapsed time
print (f_df)
print("Time call f and grad", time_separated)
print("Time call value_and_grad", time_joint)


print('Device:', df.device)