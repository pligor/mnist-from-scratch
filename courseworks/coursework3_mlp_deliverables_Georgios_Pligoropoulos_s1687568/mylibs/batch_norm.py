# -*- coding: utf-8 -*-
"""BatchNormalization Helper"""

import tensorflow as tf


def batchNormWrapper_byExponentialMovingAvg(bnId, ins, training, epsilon = 1e-3, bo=None, sg=None):
    outputDim = ins.get_shape()[-1]
    
    pop_mean = tf.Variable(tf.zeros(outputDim), trainable=False, name='pm%d' % bnId)
    pop_var = tf.Variable(tf.ones(outputDim), trainable=False, name='pv%d' % bnId)
    
    beta_offset = tf.Variable(tf.zeros(outputDim) if bo is None else bo, name='bo%d' % bnId)    
    scale_gamma = tf.Variable(tf.ones(outputDim) if sg is None else sg, name='sg%d' % bnId)
    
    #given that on axis=0 is where the batches extend (we want mean and var for each attribute)
    batch_mean, batch_var = tf.nn.moments(ins,[0])
    
    decay = 0.999 # use numbers closer to 1 if you have more data
    mean_of_train = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay)) #we just want to use the 
    var_of_train = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
    
    with tf.control_dependencies([mean_of_train, var_of_train]):
        normalized = tf.nn.batch_normalization(ins,
                                               tf.cond(training, lambda: batch_mean, lambda: pop_mean),
                                               tf.cond(training, lambda: batch_var, lambda: pop_var),
                                               beta_offset, scale_gamma, epsilon)

    return normalized


def fully_connected_layer_with_batch_norm(fcId, inputs, input_dim, output_dim, training, nonlinearity=tf.nn.relu, avoidDeadNeurons=0.,
                          w=None, b=None, bo = None, sg = None):
    weights = tf.Variable(
        tf.truncated_normal([input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5) if w is None else w,
            name = 'weights%d' % fcId
    )
    
    biases = tf.Variable(avoidDeadNeurons * tf.ones([output_dim]) if b is None else b, name = 'biases%d' % fcId)
    
    out_affine = tf.matmul(inputs, weights) + biases
    
    batchNorm = batchNormWrapper_byExponentialMovingAvg(fcId, out_affine, training, bo = bo, sg = sg)
    
    outputs = nonlinearity(batchNorm)
    return outputs


