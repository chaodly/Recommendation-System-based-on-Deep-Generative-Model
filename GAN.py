# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 15:00:22 2019

@author: nizhe
"""

import tensorflow as tf

class GAN(object):
    
    def __init__(self, g_units, d_units, alpha):
        
        self.g_units = g_units
        self.d_units = d_units
        self.alpha = alpha
        
    def get_inputs(real_size, fake_size):
    
        real = tf.placeholder(tf.float32, [None, real_size], name='real_matrix')
        fake = tf.placeholder(tf.float32, [None, fake_size], name='fake_matrix')
        
        return real, fake   
        
    def generator(self, rate = 0.2):
        
        hidden0 = tf.layers.dense(noise_img, self.g_units)
        hidden1 = tf.maximum(self.alpha * hidden0, hidden0)
        hidden2 = tf.layers.dropout(hidden1, rate)
        logits = tf.layers.dense(hidden2, out_dim)
        outputs = tf.tanh(logits)
        
        return logits, outputs
            

    def discriminator(self):
        
        hidden0 = tf.layers.dense(img, self.d_units)
        hidden1 = tf.maximum(self.alpha * hidden0, hidden0)
        logits = tf.layers.dense(hidden1, 1)
        outputs = tf.sigmoid(logits)
        
        return logits, outputs
    
    def build_graph(self):
        
        return 