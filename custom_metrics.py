import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

def mae_from_dist(y_true, y_pred):  
    # Cast to TF float 64
    y_true = tf.cast(y_true[...,0], dtype=tf.float64)
    y_pred = tf.cast(y_pred, dtype=tf.float64)
    if 1 < y_true.shape[-1] < 32: # If running with multiple timesteps
        ntimesteps = y_true.shape[-1]
        y_pred = tf.reshape(y_pred,(tf.shape(y_true)[0],32,32,ntimesteps,4))
 
    root_power = tf.constant(1.,tf.float64)/tf.math.multiply(tf.constant(10.,tf.float64),tf.cast(tf.math.exp(1.),tf.float64))
        
    # Split distribution
    mu = y_pred[...,0]       
    sigma = tf.math.pow(tf.math.exp(y_pred[...,1]),root_power)
    gamma = y_pred[..., 2] 
    tau = tf.math.pow(tf.math.exp(y_pred[...,3]),root_power)

    dist = tfp.distributions.SinhArcsinh(loc=mu, scale=sigma, skewness=gamma, tailweight=tau)

    y_out = dist.quantile(0.5) 

    # MAE
    mae = tf.reduce_mean(tf.abs(y_true - y_out))
    return mae
